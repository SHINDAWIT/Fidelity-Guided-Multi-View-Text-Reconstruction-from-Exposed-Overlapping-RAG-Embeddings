"""Stage 5A practical no-overlap chunking-policy comparison.

The same frozen 136-token source spans are repartitioned into five disjoint raw
substrings. Each substring is independently tokenized and must fit at most 31
content IDs plus EOS. A nominal 31-source-token boundary is shortened only when
standalone tokenization would exceed the 32-position checkpoint contract; the
deferred source token is carried into the next substring. This module implements
no overlap-aware method. It only constructs, strongly inverts, concatenates, and
evaluates the no-overlap baseline against the frozen Stage 4 overlap results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median
import time
from typing import Any, Callable, Mapping, Sequence

import yaml

from .chunking import _prepare_standalone_window, _raw_token_span
from .data import NaturalQuestionsLoader, Passage, PassageLoader
from .diagnostic_experiment import (
    _cuda_telemetry_end,
    _cuda_telemetry_start,
    load_diagnostic_config,
)
from .diagnostics import align_token_ids
from .reconstruction import Candidate, Vec2TextBackend
from .stage1_strong_saturation import _candidate_from_dict, _write_json_atomic
from .stage3_longdoc_strong import _distribution, _percentile
from .strong_pool_preflight import _environment


_IMPLEMENTATION_VERSION = "stage5a-nooverlap-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stage5a_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        stage5a = yaml.safe_load(handle)
    if not isinstance(stage5a, dict):
        raise ValueError("Stage 5A config must be a YAML mapping")
    base_candidate = Path(str(stage5a["base_config"]))
    base_path = (
        base_candidate
        if base_candidate.is_absolute()
        else config_path.parent / base_candidate
    )
    resolved = {
        "base": load_diagnostic_config(base_path),
        "stage5a": stage5a,
        "resolved_base_config": str(base_path.resolve()),
        "resolved_stage5a_config": str(config_path.resolve()),
    }
    for key in (
        "cohort_artifact",
        "stage3_strong_cache",
        "stage4_result",
        "stage4_trajectories",
    ):
        candidate = Path(str(stage5a[key]))
        resolved[f"resolved_{key}"] = str(
            (candidate if candidate.is_absolute() else config_path.parent / candidate)
            .resolve()
        )
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _validate_frozen_inputs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage5a = dict(config["stage5a"])
    paths = {
        key: Path(str(config[f"resolved_{key}"]))
        for key in (
            "cohort_artifact",
            "stage3_strong_cache",
            "stage4_result",
            "stage4_trajectories",
        )
    }
    hash_keys = {
        "cohort_artifact": "cohort_sha256",
        "stage3_strong_cache": "stage3_strong_cache_sha256",
        "stage4_result": "stage4_result_sha256",
        "stage4_trajectories": "stage4_trajectories_sha256",
    }
    expected = {key: str(stage5a[hash_keys[key]]) for key in paths}
    actual = {key: _sha256(path) for key, path in paths.items()}
    if actual != expected:
        raise RuntimeError(
            f"Stage 5A frozen artifact hash mismatch: expected={expected} actual={actual}"
        )
    cohort = _load_json(paths["cohort_artifact"])
    strong_cache = _load_json(paths["stage3_strong_cache"])
    stage4 = _load_json(paths["stage4_result"])
    trajectories = _load_json(paths["stage4_trajectories"])
    manifest = list(cohort["manifest"])
    if len(manifest) != int(stage5a["expected_documents"]):
        raise RuntimeError("Stage 5A frozen cohort size changed")
    if [document["passage_id"] for document in cohort["documents"]] != manifest:
        raise RuntimeError("Stage 5A frozen cohort order changed")
    if stage4["gate"]["state"] != "DOCUMENT_CORRECTION_GO":
        raise RuntimeError("Stage 5A requires the frozen Stage 4 GO artifact")
    if [row["passage_id"] for row in trajectories["trajectories"]] != manifest:
        raise RuntimeError("Stage 4 trajectory manifest changed")
    if not bool(strong_cache["public_top1_exact_id_match"]):
        raise RuntimeError("Stage 3 cache public/instrumented Top-1 mismatch")
    if strong_cache.get("strong_reproduction_setting") != "correction50+sbeam8":
        raise RuntimeError("Stage 3 cache strong-search setting changed")
    if bool(strong_cache.get("original_paper_beam_width_claimed")):
        raise RuntimeError("repository sbeam8 must not be labeled as a paper claim")
    expected_stage3_keys = [
        f"{passage_id}:{window_index}"
        for passage_id in manifest
        for window_index in range(8)
    ]
    if list(strong_cache["view_keys"]) != expected_stage3_keys:
        raise RuntimeError("Stage 3 strong-cache view order changed")
    if len(strong_cache["top1_candidates"]) != len(expected_stage3_keys):
        raise RuntimeError("Stage 3 strong-cache candidate count changed")
    return cohort, strong_cache, stage4, trajectories


def _passages_by_source_row(passages: Sequence[Passage]) -> dict[int, Passage]:
    result: dict[int, Passage] = {}
    for fallback_index, passage in enumerate(passages):
        source_row = int(passage.metadata.get("row_index", fallback_index))
        if source_row in result:
            raise RuntimeError("pinned scan returned a duplicate source row")
        result[source_row] = passage
    return result


def _build_nooverlap_documents(
    cohort: Mapping[str, Any],
    passages: Sequence[Passage],
    tokenizer: Any,
    *,
    max_standalone_content_tokens: int,
    expected_vectors_per_document: int,
    model_positions: int,
) -> list[dict[str, Any]]:
    if max_standalone_content_tokens != 31 or model_positions != 32:
        raise ValueError("Stage 5A must remain at 31 content IDs + EOS")
    by_source_row = _passages_by_source_row(passages)
    documents: list[dict[str, Any]] = []
    for frozen in cohort["documents"]:
        source_row = int(frozen["source_row_index"])
        passage = by_source_row.get(source_row)
        if passage is None or passage.passage_id != frozen["passage_id"]:
            raise RuntimeError(
                f"pinned source row no longer resolves to {frozen['passage_id']}"
            )
        encoded = tokenizer(
            passage.text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        document_ids = tuple(int(value) for value in encoded["input_ids"])
        offsets = tuple(
            (int(start), int(end)) for start, end in encoded["offset_mapping"]
        )
        selected_start = int(frozen["selected_start"])
        reference = tuple(
            int(value) for value in frozen["prefixes"]["8"]["reference_ids"]
        )
        if document_ids[selected_start : selected_start + len(reference)] != reference:
            raise RuntimeError("pinned passage tokenization no longer matches frozen span")
        windows: list[dict[str, Any]] = []
        relative_start = 0
        while relative_start < len(reference):
            window_index = len(windows)
            nominal_source_length = min(
                max_standalone_content_tokens,
                len(reference) - relative_start,
            )
            source_content_length = nominal_source_length
            token_start = selected_start + relative_start
            while source_content_length > 0:
                source_ids = reference[
                    relative_start : relative_start + source_content_length
                ]
                raw_span = _raw_token_span(
                    passage.text,
                    offsets,
                    token_start,
                    int(source_content_length),
                )
                if raw_span is None:
                    raise RuntimeError("no-overlap raw token span is empty")
                char_start, char_end, raw_text = raw_span
                independent_ids = tuple(
                    int(value)
                    for value in tokenizer(
                        raw_text,
                        add_special_tokens=False,
                        truncation=False,
                    )["input_ids"]
                )
                if len(independent_ids) <= max_standalone_content_tokens:
                    break
                source_content_length -= 1
            if source_content_length < 1:
                raise RuntimeError("no-overlap boundary cannot fit one source token")
            model_input_ids, attention_mask = _prepare_standalone_window(
                raw_text,
                independent_ids,
                tokenizer,
                model_positions=model_positions,
                require_full_content=False,
            )
            windows.append(
                {
                    "window_index": window_index,
                    "text": raw_text,
                    "content_ids": list(independent_ids),
                    "content_length": len(independent_ids),
                    "source_ids": list(source_ids),
                    "source_content_length": int(source_content_length),
                    "nominal_source_content_length": int(nominal_source_length),
                    "source_tokens_deferred_for_capacity": (
                        int(nominal_source_length) - int(source_content_length)
                    ),
                    "source_ids_equal_standalone_ids": source_ids == independent_ids,
                    "model_input_ids": list(model_input_ids),
                    "attention_mask": list(attention_mask),
                    "document_token_start": token_start,
                    "document_token_end": token_start + int(source_content_length),
                    "relative_token_start": relative_start,
                    "relative_token_end": relative_start
                    + int(source_content_length),
                    "char_start": char_start,
                    "char_end": char_end,
                    "is_final_tail": relative_start + source_content_length
                    == len(reference),
                }
            )
            relative_start += int(source_content_length)
        if relative_start != len(reference):
            raise AssertionError("no-overlap windows do not cover exactly 136 tokens")
        if tuple(
            token for window in windows for token in window["source_ids"]
        ) != reference:
            raise AssertionError("no-overlap windows changed source token order")
        if len(windows) != expected_vectors_per_document:
            raise RuntimeError(
                f"capacity-safe no-overlap produced {len(windows)} views for "
                f"{frozen['passage_id']}, expected {expected_vectors_per_document}"
            )
        frozen_first = frozen["windows"][0]
        first = windows[0]
        if (
            first["text"] != frozen_first["text"]
            or first["content_ids"] != frozen_first["content_ids"]
            or first["model_input_ids"] != frozen_first["model_input_ids"]
        ):
            raise RuntimeError("reusable first window differs from Stage 3")
        documents.append(
            {
                "passage_id": frozen["passage_id"],
                "source_row_index": source_row,
                "selected_start": selected_start,
                "reference_ids": list(reference),
                "windows": windows,
            }
        )
    return documents


def _manifest_path(config: Mapping[str, Any]) -> Path:
    stage5a = dict(config["stage5a"])
    return Path(str(stage5a["output_dir"])) / str(stage5a["manifest_filename"])


def _prepare_manifest(
    config: Mapping[str, Any],
    cohort: Mapping[str, Any],
    *,
    loader: PassageLoader | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    base = dict(config["base"])
    stage5a = dict(config["stage5a"])
    dataset = dict(base["dataset"])
    models = dict(base["models"])
    path = _manifest_path(config)
    if loader is None and tokenizer is None and path.exists():
        manifest = _load_json(path)
        if manifest.get("implementation_version") != _IMPLEMENTATION_VERSION:
            raise RuntimeError("existing Stage 5A manifest implementation changed")
        if manifest.get("source_cohort_sha256") != str(stage5a["cohort_sha256"]):
            raise RuntimeError("existing Stage 5A manifest cohort hash changed")
        if list(manifest.get("manifest", [])) != list(cohort["manifest"]):
            raise RuntimeError("existing Stage 5A manifest document order changed")
        geometry = dict(manifest.get("geometry", {}))
        expected_geometry = {
            "source_content_tokens": int(stage5a["source_content_tokens"]),
            "max_standalone_content_tokens": int(
                stage5a["max_standalone_content_tokens"]
            ),
            "nominal_source_stride_tokens": int(
                stage5a["nominal_source_stride_tokens"]
            ),
            "model_positions": int(stage5a["model_positions"]),
            "vectors_per_document": int(stage5a["expected_vectors_per_document"]),
            "boundary_policy": (
                "greedily take up to 31 whole-document source IDs; if the raw "
                "substring independently tokenizes above 31 content IDs, defer "
                "right-edge source IDs to the next disjoint window"
            ),
        }
        if geometry != expected_geometry:
            raise RuntimeError("existing Stage 5A manifest geometry changed")
        frozen_by_id = {
            document["passage_id"]: document for document in cohort["documents"]
        }
        for document in manifest.get("documents", []):
            frozen = frozen_by_id.get(document["passage_id"])
            if frozen is None:
                raise RuntimeError("existing Stage 5A manifest has an unknown document")
            reference = list(frozen["prefixes"]["8"]["reference_ids"])
            if list(document["reference_ids"]) != reference:
                raise RuntimeError("existing Stage 5A source IDs changed")
            windows = list(document["windows"])
            if len(windows) != int(stage5a["expected_vectors_per_document"]):
                raise RuntimeError("existing Stage 5A vector count changed")
            if any(
                len(window["content_ids"])
                > int(stage5a["max_standalone_content_tokens"])
                for window in windows
            ):
                raise RuntimeError("existing Stage 5A input exceeds checkpoint capacity")
            if [token for window in windows for token in window["source_ids"]] != reference:
                raise RuntimeError("existing Stage 5A windows no longer cover the source")
        if len(manifest.get("documents", [])) != int(stage5a["expected_documents"]):
            raise RuntimeError("existing Stage 5A manifest cohort size changed")
        return manifest

    from transformers import AutoTokenizer

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        models["embedder_name"],
        revision=models.get("embedder_revision"),
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Stage 5A raw-substring mode requires a fast tokenizer")
    loader = loader or NaturalQuestionsLoader(
        dataset_name=dataset["name"],
        dataset_config=dataset.get("config"),
        split=dataset["split"],
        revision=dataset["revision"],
        max_words=None,
        streaming=bool(dataset.get("streaming", True)),
    )
    passages = loader.load(int(dataset["scan_limit"]))
    documents = _build_nooverlap_documents(
        cohort,
        passages,
        tokenizer,
        max_standalone_content_tokens=int(
            stage5a["max_standalone_content_tokens"]
        ),
        expected_vectors_per_document=int(
            stage5a["expected_vectors_per_document"]
        ),
        model_positions=int(stage5a["model_positions"]),
    )
    all_windows = [
        window for document in documents for window in document["windows"]
    ]
    manifest = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "stage": "STAGE5A_NO_OVERLAP_FROZEN_SOURCE_MANIFEST",
        "dataset": dataset,
        "source_cohort_sha256": str(stage5a["cohort_sha256"]),
        "comparison_type": "practical_chunking_policy_comparison",
        "geometry": {
            "source_content_tokens": int(stage5a["source_content_tokens"]),
            "max_standalone_content_tokens": int(
                stage5a["max_standalone_content_tokens"]
            ),
            "nominal_source_stride_tokens": int(
                stage5a["nominal_source_stride_tokens"]
            ),
            "model_positions": int(stage5a["model_positions"]),
            "vectors_per_document": int(stage5a["expected_vectors_per_document"]),
            "boundary_policy": (
                "greedily take up to 31 whole-document source IDs; if the raw "
                "substring independently tokenizes above 31 content IDs, defer "
                "right-edge source IDs to the next disjoint window"
            ),
        },
        "manifest": list(cohort["manifest"]),
        "documents": documents,
        "acceptance": {
            "requested_documents": int(stage5a["expected_documents"]),
            "accepted_documents": len(documents),
            "yield": len(documents) / int(stage5a["expected_documents"]),
            "source_spans_reselected": False,
            "all_raw_inputs_fit_without_truncation": True,
            "source_ids_preserved_exactly": True,
            "standalone_ids_may_differ_at_chunk_boundaries": True,
            "boundary_adjusted_window_count": sum(
                int(window["source_tokens_deferred_for_capacity"]) > 0
                for window in all_windows
            ),
            "standalone_ids_differ_from_source_slice_count": sum(
                not bool(window["source_ids_equal_standalone_ids"])
                for window in all_windows
            ),
            "source_tokens_deferred_total": sum(
                int(window["source_tokens_deferred_for_capacity"])
                for window in all_windows
            ),
            "source_content_lengths": sorted(
                {int(window["source_content_length"]) for window in all_windows}
            ),
            "standalone_content_lengths": sorted(
                {int(window["content_length"]) for window in all_windows}
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, manifest)
    return manifest


def _view_keys(manifest: Mapping[str, Any]) -> list[str]:
    return [
        f"{document['passage_id']}:nooverlap:{window['window_index']}"
        for document in manifest["documents"]
        for window in document["windows"]
    ]


def _fingerprint(
    *,
    manifest_sha256: str,
    embedding_sha256: str,
    view_keys: Sequence[str],
    model_config: Mapping[str, Any],
    strong_setting: Mapping[str, Any],
    reused_cache_sha256: str,
) -> str:
    frozen = {
        "implementation_version": _IMPLEMENTATION_VERSION,
        "manifest_sha256": manifest_sha256,
        "embedding_sha256": embedding_sha256,
        "view_keys": list(view_keys),
        "models": dict(model_config),
        "strong_setting": dict(strong_setting),
        "reused_cache_sha256": reused_cache_sha256,
    }
    return hashlib.sha256(
        json.dumps(
            frozen, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _load_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    view_keys: Sequence[str],
) -> tuple[list[Candidate | None], list[dict[str, Any] | None]]:
    if not path.exists():
        return [None] * len(view_keys), [None] * len(view_keys)
    state = _load_json(path)
    if state.get("fingerprint") != fingerprint:
        raise RuntimeError("Stage 5A checkpoint belongs to another frozen run")
    if list(state.get("view_keys", [])) != list(view_keys):
        raise RuntimeError("Stage 5A checkpoint view order changed")
    raw_candidates = list(state.get("candidates", []))
    records = list(state.get("records", []))
    if len(raw_candidates) != len(view_keys) or len(records) != len(view_keys):
        raise RuntimeError("Stage 5A checkpoint shape changed")
    candidates = [
        None if raw is None else _candidate_from_dict(raw)
        for raw in raw_candidates
    ]
    return candidates, records


def _save_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    manifest_sha256: str,
    embedding_sha256: str,
    view_keys: Sequence[str],
    candidates: Sequence[Candidate | None],
    records: Sequence[Mapping[str, Any] | None],
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "manifest_sha256": manifest_sha256,
            "embedding_sha256": embedding_sha256,
            "view_keys": list(view_keys),
            "completed_views": sum(candidate is not None for candidate in candidates),
            "candidates": [
                None if candidate is None else candidate.to_dict()
                for candidate in candidates
            ],
            "records": list(records),
        },
    )


def _evaluate_nooverlap_document(
    document: Mapping[str, Any], candidates: Sequence[Candidate], tokenizer: Any
) -> dict[str, Any]:
    reference = tuple(int(value) for value in document["reference_ids"])
    windows = list(document["windows"])
    if len(candidates) != len(windows):
        raise ValueError("no-overlap candidate count differs from window count")
    window_rows = []
    for window, candidate in zip(windows, candidates):
        target = tuple(int(value) for value in window["content_ids"])
        alignment = align_token_ids(candidate.content_ids, target)
        window_rows.append(
            {
                "window_index": int(window["window_index"]),
                "source_content_length": int(window["source_content_length"]),
                "target_content_length": len(target),
                "prediction_content_length": len(candidate.content_ids),
                "is_final_tail": bool(window["is_final_tail"]),
                "source_ids_equal_standalone_ids": bool(
                    window["source_ids_equal_standalone_ids"]
                ),
                "source_tokens_deferred_for_capacity": int(
                    window["source_tokens_deferred_for_capacity"]
                ),
                "edit_distance": alignment.distance,
                "normalized_edit_similarity": max(
                    0.0, 1.0 - alignment.distance / len(target)
                ),
                "exact_matched_token_recovery": len(alignment.exact_matches)
                / len(target),
                "exact": tuple(candidate.content_ids) == target,
                "embedding_fidelity": float(candidate.local_score),
            }
        )
    id_concatenated_prediction = tuple(
        token for candidate in candidates for token in candidate.content_ids
    )
    prediction_text = " ".join(candidate.text.strip() for candidate in candidates)
    prediction = tuple(
        int(token)
        for token in tokenizer(
            prediction_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )
    standalone_targets = tuple(
        int(token)
        for window in windows
        for token in window["content_ids"]
    )
    id_boundary_alignment = align_token_ids(standalone_targets, reference)
    raw_text_reassembly = " ".join(
        str(window["text"]).strip() for window in windows
    )
    raw_text_reassembly_ids = tuple(
        int(token)
        for token in tokenizer(
            raw_text_reassembly,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )
    text_boundary_alignment = align_token_ids(raw_text_reassembly_ids, reference)
    id_prediction_alignment = align_token_ids(id_concatenated_prediction, reference)
    alignment = align_token_ids(prediction, reference)
    source_errors = len(reference) - len(alignment.exact_matches)
    return {
        "passage_id": document["passage_id"],
        "prediction_text": prediction_text,
        "prediction_ids": list(prediction),
        "window_metrics": window_rows,
        "document_edit_distance": alignment.distance,
        "document_normalized_edit_similarity": max(
            0.0, 1.0 - alignment.distance / len(reference)
        ),
        "document_exact_matched_token_recovery": len(alignment.exact_matches)
        / len(reference),
        "document_source_position_error_count": source_errors,
        "document_prediction_insertion_count": len(alignment.prediction_insertions),
        "document_exact": prediction == reference,
        "raw_text_reassembly_floor": {
            "definition": (
                "Raw ground-truth chunk strings joined by one fixed space and then "
                "tokenized once versus the frozen 136-ID reference; no inversion "
                "prediction is used."
            ),
            "edit_distance": text_boundary_alignment.distance,
            "normalized_edit_similarity": max(
                0.0, 1.0 - text_boundary_alignment.distance / len(reference)
            ),
            "exact_matched_source_token_recovery": len(
                text_boundary_alignment.exact_matches
            )
            / len(reference),
            "source_position_error_count": len(reference)
            - len(text_boundary_alignment.exact_matches),
            "prediction_insertion_count": len(
                text_boundary_alignment.prediction_insertions
            ),
            "exact": raw_text_reassembly_ids == reference,
        },
        "standalone_id_concatenation_diagnostic": {
            "definition": (
                "Standalone ground-truth chunk IDs concatenated without text-level "
                "re-tokenization versus the frozen 136-ID reference."
            ),
            "edit_distance": id_boundary_alignment.distance,
            "source_position_error_count": len(reference)
            - len(id_boundary_alignment.exact_matches),
            "prediction_insertion_count": len(
                id_boundary_alignment.prediction_insertions
            ),
        },
        "candidate_id_concatenation_diagnostic": {
            "edit_distance": id_prediction_alignment.distance,
            "source_position_error_count": len(reference)
            - len(id_prediction_alignment.exact_matches),
            "prediction_insertion_count": len(
                id_prediction_alignment.prediction_insertions
            ),
        },
    }


def _aggregate_nooverlap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    windows = [window for row in rows for window in row["window_metrics"]]
    nonfinal = [window for window in windows if not window["is_final_tail"]]
    tails = [window for window in windows if window["is_final_tail"]]
    reference_tokens = 136 * len(rows)
    source_errors = sum(int(row["document_source_position_error_count"]) for row in rows)
    insertions = sum(int(row["document_prediction_insertion_count"]) for row in rows)
    edits = sum(int(row["document_edit_distance"]) for row in rows)
    boundary_edits = sum(
        int(row["raw_text_reassembly_floor"]["edit_distance"]) for row in rows
    )
    boundary_source_errors = sum(
        int(row["raw_text_reassembly_floor"]["source_position_error_count"])
        for row in rows
    )
    boundary_insertions = sum(
        int(row["raw_text_reassembly_floor"]["prediction_insertion_count"])
        for row in rows
    )

    def window_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(values),
            "normalized_edit_similarity": _distribution(
                [float(value["normalized_edit_similarity"]) for value in values]
            ),
            "exact_matched_token_recovery": _distribution(
                [float(value["exact_matched_token_recovery"]) for value in values]
            ),
            "exact_fraction": mean(bool(value["exact"]) for value in values),
            "edit_distance": _distribution(
                [float(value["edit_distance"]) for value in values]
            ),
            "prediction_content_length": _distribution(
                [float(value["prediction_content_length"]) for value in values]
            ),
        }

    return {
        "documents": len(rows),
        "vectors_per_document": 5,
        "window_level": {
            "all": window_summary(windows),
            "nonfinal": window_summary(nonfinal),
            "final_tail": window_summary(tails),
        },
        "document_level": {
            "normalized_edit_similarity": _distribution(
                [float(row["document_normalized_edit_similarity"]) for row in rows]
            ),
            "pooled_token_error_rate": edits / reference_tokens,
            "total_edit_operations": edits,
            "exact_matched_source_token_recovery": 1.0
            - source_errors / reference_tokens,
            "source_position_error_count": source_errors,
            "prediction_insertion_count": insertions,
            "document_exact_fraction": mean(bool(row["document_exact"]) for row in rows),
        },
        "boundary_diagnostics": {
            "adjusted_window_count": sum(
                int(window["source_tokens_deferred_for_capacity"]) > 0
                for window in windows
            ),
            "standalone_ids_differ_from_source_slice_count": sum(
                not bool(window["source_ids_equal_standalone_ids"])
                for window in windows
            ),
            "source_tokens_deferred_total": sum(
                int(window["source_tokens_deferred_for_capacity"])
                for window in windows
            ),
            "tokenization_only_floor": {
                "definition": (
                    "Raw source chunk strings joined by one fixed space, tokenized "
                    "once, and aligned to the frozen whole-document token reference."
                ),
                "total_edit_operations": boundary_edits,
                "pooled_token_error_rate": boundary_edits / reference_tokens,
                "normalized_edit_similarity": _distribution(
                    [
                        float(
                            row["raw_text_reassembly_floor"][
                                "normalized_edit_similarity"
                            ]
                        )
                        for row in rows
                    ]
                ),
                "exact_matched_source_token_recovery": 1.0
                - boundary_source_errors / reference_tokens,
                "source_position_error_count": boundary_source_errors,
                "prediction_insertion_count": boundary_insertions,
                "document_exact_fraction": mean(
                    bool(row["raw_text_reassembly_floor"]["exact"])
                    for row in rows
                ),
            },
            "standalone_id_concatenation": {
                "total_edit_operations": sum(
                    int(
                        row["standalone_id_concatenation_diagnostic"][
                            "edit_distance"
                        ]
                    )
                    for row in rows
                ),
                "source_position_error_count": sum(
                    int(
                        row["standalone_id_concatenation_diagnostic"][
                            "source_position_error_count"
                        ]
                    )
                    for row in rows
                ),
                "prediction_insertion_count": sum(
                    int(
                        row["standalone_id_concatenation_diagnostic"][
                            "prediction_insertion_count"
                        ]
                    )
                    for row in rows
                ),
            },
        },
    }


def _paired_comparisons(
    nooverlap_rows: Sequence[Mapping[str, Any]],
    stage4_trajectories: Mapping[str, Any],
) -> list[dict[str, Any]]:
    overlap_by_id = {
        row["passage_id"]: row["k_prefixes"]["8"]
        for row in stage4_trajectories["trajectories"]
    }
    comparisons = []
    for nooverlap in nooverlap_rows:
        overlap = overlap_by_id[nooverlap["passage_id"]]
        improved = overlap["improved"]
        oracle = overlap["diagnostic"]["oracle"]
        nooverlap_recovery = 1.0 - float(
            nooverlap["document_source_position_error_count"]
        ) / 136.0
        overlap_improved_recovery = 1.0 - float(
            improved["final_source_token_error_count"]
        ) / 136.0
        overlap_oracle_recovery = float(
            oracle["multiview_only_oracle_source_token_recovery"]
        )
        comparisons.append(
            {
                "passage_id": nooverlap["passage_id"],
                "nooverlap_document_similarity": float(
                    nooverlap["document_normalized_edit_similarity"]
                ),
                "overlap_improved_document_similarity": float(
                    improved["document_normalized_edit_similarity"]
                ),
                "nooverlap_source_recovery": nooverlap_recovery,
                "overlap_improved_source_recovery": overlap_improved_recovery,
                "overlap_multiview_oracle_source_recovery": overlap_oracle_recovery,
                "nooverlap_minus_overlap_improved_similarity": float(
                    nooverlap["document_normalized_edit_similarity"]
                )
                - float(improved["document_normalized_edit_similarity"]),
                "nooverlap_minus_overlap_improved_source_recovery": (
                    nooverlap_recovery - overlap_improved_recovery
                ),
                "overlap_multiview_oracle_minus_nooverlap_source_recovery": (
                    overlap_oracle_recovery - nooverlap_recovery
                ),
            }
        )
    return comparisons


def _bootstrap_comparisons(
    comparisons: Sequence[Mapping[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    metrics = (
        "nooverlap_minus_overlap_improved_similarity",
        "nooverlap_minus_overlap_improved_source_recovery",
        "overlap_multiview_oracle_minus_nooverlap_source_recovery",
    )
    rng = random.Random(seed)
    samples = {metric: [] for metric in metrics}
    for _ in range(iterations):
        selected = [
            comparisons[rng.randrange(len(comparisons))]
            for _ in range(len(comparisons))
        ]
        for metric in metrics:
            samples[metric].append(mean(float(row[metric]) for row in selected))
    return {
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "frozen document",
        "mean_delta": {
            metric: mean(float(row[metric]) for row in comparisons)
            for metric in metrics
        },
        "ci95": {
            metric: [_percentile(values, 0.025), _percentile(values, 0.975)]
            for metric, values in samples.items()
        },
    }


def _sensitivity_subgroups(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rows_by_id = {str(row["passage_id"]): row for row in rows}
    comparisons_by_id = {
        str(row["passage_id"]): row for row in comparisons
    }
    document_sets: dict[str, list[str]] = {
        "literal_fixed_roundtrip_stable": [],
        "literal_fixed_capacity_safe": [],
        "capacity_adjusted": [],
    }
    for document in manifest["documents"]:
        passage_id = str(document["passage_id"])
        windows = list(document["windows"])
        adjusted = any(
            int(window["source_tokens_deferred_for_capacity"]) > 0
            for window in windows
        )
        if adjusted:
            document_sets["capacity_adjusted"].append(passage_id)
        else:
            document_sets["literal_fixed_capacity_safe"].append(passage_id)
            if all(
                bool(window["source_ids_equal_standalone_ids"])
                for window in windows
            ):
                document_sets["literal_fixed_roundtrip_stable"].append(
                    passage_id
                )

    result: dict[str, Any] = {}
    for offset, (name, passage_ids) in enumerate(document_sets.items()):
        selected_rows = [rows_by_id[passage_id] for passage_id in passage_ids]
        selected_comparisons = [
            comparisons_by_id[passage_id] for passage_id in passage_ids
        ]
        result[name] = {
            "documents": len(passage_ids),
            "passage_ids": passage_ids,
            "nooverlap_summary": _aggregate_nooverlap(selected_rows),
            "paired_comparison": _bootstrap_comparisons(
                selected_comparisons,
                iterations=iterations,
                seed=seed + offset + 1,
            ),
            "descriptive_only_not_a_gate": True,
        }
    return result


def _classify_gate(
    bootstrap: Mapping[str, Any],
    sensitivity_subgroups: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metric = "overlap_multiview_oracle_minus_nooverlap_source_recovery"
    estimate = float(bootstrap["mean_delta"][metric])
    lower, upper = bootstrap["ci95"][metric]
    if upper is not None and float(upper) < 0.0:
        state = "NO_OVERLAP_DOMINATES_ORACLE_STOP"
        reason = (
            "No-overlap source recovery exceeds even the overlap multi-view "
            "source-position oracle with a paired-bootstrap interval below zero."
        )
    elif lower is not None and float(lower) > 0.0:
        state = "OVERLAP_INFORMATION_CEILING_GO"
        reason = (
            "The overlap multi-view source-position oracle exceeds no-overlap "
            "source recovery with a paired-bootstrap interval above zero."
        )
    else:
        state = "NO_OVERLAP_COMPARISON_INCONCLUSIVE"
        reason = (
            "The paired-bootstrap interval does not distinguish no-overlap source "
            "recovery from the overlap multi-view source-position oracle."
        )
    boundary_sensitive = False
    qualification = None
    next_stage = None
    if state == "OVERLAP_INFORMATION_CEILING_GO":
        next_stage = "STAGE5B_HOLDOUT_REPRODUCTION_ONLY"
        if sensitivity_subgroups is not None:
            fixed = sensitivity_subgroups["literal_fixed_capacity_safe"][
                "paired_comparison"
            ]
            adjusted = sensitivity_subgroups["capacity_adjusted"][
                "paired_comparison"
            ]
            fixed_lower, fixed_upper = fixed["ci95"][metric]
            adjusted_lower, _ = adjusted["ci95"][metric]
            boundary_sensitive = (
                fixed_lower is not None
                and fixed_upper is not None
                and float(fixed_lower) <= 0.0 <= float(fixed_upper)
                and adjusted_lower is not None
                and float(adjusted_lower) > 0.0
            )
            if boundary_sensitive:
                qualification = (
                    "The prespecified all-document gate passes, but the positive "
                    "oracle gap is concentrated in documents requiring capacity-safe "
                    "boundary adjustment. The literal fixed-boundary capacity-safe "
                    "subgroup does not distinguish overlap oracle from no-overlap."
                )
    return {
        "state": state,
        "reason": reason,
        "primary_metric": metric,
        "mean_delta_overlap_minus_nooverlap": estimate,
        "paired_bootstrap_ci95": [lower, upper],
        "method_implementation_authorized": False,
        "next_stage_authorized": next_stage,
        "boundary_sensitive": boundary_sensitive,
        "qualification": qualification,
        "interpretation": "conditional Stage 5A policy-comparison gate",
    }


def _write_report(output_dir: Path, payload: Mapping[str, Any]) -> None:
    stage5a = dict(payload["config"]["stage5a"])
    result_path = output_dir / str(stage5a["result_filename"])
    _write_json_atomic(result_path, payload)
    nooverlap = payload["nooverlap_summary"]
    overlap = payload["overlap_stage4_summary"]
    overlap_improved = overlap["improved"]["document_level"]
    overlap_oracle = overlap["oracle"]
    comparisons = payload["paired_comparison"]
    sensitivities = payload["sensitivity_subgroups"]
    lines = [
        "# Stage 5A — no-overlap strong practical policy comparison",
        "",
        f"- Status: `{payload['status']}`",
        f"- Gate: `{payload['gate']['state']}`",
        f"- Reason: {payload['gate']['reason']}",
        f"- Qualification: {payload['gate']['qualification']}",
        f"- Next stage authorized: `{payload['gate']['next_stage_authorized']}`",
        "- Same frozen 30 documents and same 136 source content IDs: `true`.",
        "- No-overlap policy: five disjoint raw substrings/document, each independently tokenized to at most 31 content IDs.",
        "- Nominal source boundary is 31 IDs; a right-edge source ID is deferred when standalone tokenization would exceed checkpoint capacity.",
        "- Overlap policy: `31`, stride `15`, eight vectors/document.",
        "- Comparison type: practical chunking-policy comparison, not a pure causal overlap effect.",
        "- New document-assembly method implemented: `false`.",
        "",
        "## No-overlap strong reconstruction",
        "",
        f"- Document similarity mean: {nooverlap['document_level']['normalized_edit_similarity']['mean']:.4f}",
        f"- Source-position recovery: {nooverlap['document_level']['exact_matched_source_token_recovery']:.4f}",
        f"- Total edits: {nooverlap['document_level']['total_edit_operations']}",
        f"- Source errors / insertions: {nooverlap['document_level']['source_position_error_count']} / {nooverlap['document_level']['prediction_insertion_count']}",
        f"- Document exact: {nooverlap['document_level']['document_exact_fraction']:.4f}",
        f"- Final-tail token recovery: {nooverlap['window_level']['final_tail']['exact_matched_token_recovery']['mean']:.4f}",
        f"- Capacity-adjusted windows: {nooverlap['boundary_diagnostics']['adjusted_window_count']} / 150",
        f"- Standalone/source-slice ID mismatches: {nooverlap['boundary_diagnostics']['standalone_ids_differ_from_source_slice_count']} / 150",
        f"- Tokenization-only boundary floor edits: {nooverlap['boundary_diagnostics']['tokenization_only_floor']['total_edit_operations']}",
        f"- Tokenization-only boundary source recovery: {nooverlap['boundary_diagnostics']['tokenization_only_floor']['exact_matched_source_token_recovery']:.4f}",
        "",
        "## Main same-source comparison",
        "",
        "| Policy | Vectors/doc | Mean document similarity | Pooled edit error | Source-position recovery | Document exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| No-overlap strong | 5 | "
            f"{nooverlap['document_level']['normalized_edit_similarity']['mean']:.4f} | "
            f"{nooverlap['document_level']['pooled_token_error_rate']:.4f} | "
            f"{nooverlap['document_level']['exact_matched_source_token_recovery']:.4f} | "
            f"{nooverlap['document_level']['document_exact_fraction']:.4f} |"
        ),
        (
            f"| Overlap strong + improved stitch | 8 | "
            f"{overlap_improved['normalized_edit_similarity']['mean']:.4f} | "
            f"{overlap_improved['pooled_token_error_rate']:.4f} | "
            f"{overlap_improved['pooled_exact_matched_token_recovery']:.4f} | "
            f"{overlap_improved['document_exact_fraction']:.4f} |"
        ),
        (
            "| Overlap GT-aligned multi-view oracle | 8 | n/a | n/a | "
            f"{overlap_oracle['multiview_only_oracle_source_token_recovery']:.4f} | "
            f"{overlap_oracle['multiview_only_oracle_source_position_document_exact_fraction']:.4f} |"
        ),
        "",
        "## Paired policy comparison",
        "",
        f"- Mean no-overlap minus overlap-improved similarity: {comparisons['mean_delta']['nooverlap_minus_overlap_improved_similarity']:.4f}",
        f"- Mean no-overlap minus overlap-improved source recovery: {comparisons['mean_delta']['nooverlap_minus_overlap_improved_source_recovery']:.4f}",
        f"- Mean overlap-oracle minus no-overlap source recovery: {comparisons['mean_delta']['overlap_multiview_oracle_minus_nooverlap_source_recovery']:.4f}",
        f"- Primary paired CI95: {payload['gate']['paired_bootstrap_ci95']}",
        "",
        "## Boundary-policy sensitivity (descriptive)",
        "",
        *[
            (
                f"- {name}: n={value['documents']}, overlap-oracle minus "
                "no-overlap source recovery="
                f"{value['paired_comparison']['mean_delta']['overlap_multiview_oracle_minus_nooverlap_source_recovery']:.4f}, "
                f"CI95={value['paired_comparison']['ci95']['overlap_multiview_oracle_minus_nooverlap_source_recovery']}"
            )
            for name, value in sensitivities.items()
        ],
        "",
        "## Compute",
        "",
        f"- Logical strong-search queries, no-overlap: {payload['telemetry']['logical_search_embedding_queries']}",
        f"- Logical strong-search queries, overlap: {payload['telemetry']['overlap_logical_search_embedding_queries']}",
        f"- No-overlap/overlap vector and query ratio: {payload['telemetry']['logical_vector_ratio_nooverlap_over_overlap']:.3f}",
        f"- Actual new queries in this run after 30-view reuse: {payload['telemetry']['actual_new_search_embedding_queries']}",
        f"- Runtime/new view: {payload['telemetry']['new_runtime_seconds_per_view']:.2f}s",
        f"- Peak allocated GPU memory: {payload['telemetry']['new_peak_allocated_bytes']} bytes",
        "",
        "The overlap oracle is source-position-only and does not construct a string or remove prediction insertions. It is therefore compared to no-overlap source-position recovery, not directly to Levenshtein document similarity.",
    ]
    (output_dir / "stage5a_nooverlap_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_stage5a(
    config: Mapping[str, Any],
    *,
    manifest_only: bool = False,
    preflight_only: bool = False,
    loader: PassageLoader | None = None,
    tokenizer: Any | None = None,
    backend_factory: Callable[..., Vec2TextBackend] = Vec2TextBackend,
) -> dict[str, Any]:
    import torch

    started_all = time.perf_counter()
    cohort, stage3_cache, stage4, stage4_trajectories = _validate_frozen_inputs(
        config
    )
    stage5a = dict(config["stage5a"])
    base = dict(config["base"])
    output_dir = Path(str(stage5a["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _prepare_manifest(
        config, cohort, loader=loader, tokenizer=tokenizer
    )
    manifest_path = _manifest_path(config)
    manifest_sha256 = _sha256(manifest_path)
    if manifest_only:
        return {
            "status": "STAGE5A_MANIFEST_READY",
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "acceptance": manifest["acceptance"],
            "boundary_policy": manifest["geometry"]["boundary_policy"],
            "vec2text_executed": False,
        }

    model_config = dict(base["models"])
    requested_device = torch.device(model_config.get("device", "cuda"))
    if requested_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(
            requested_device.index if requested_device.index is not None else 0
        )
    backend = backend_factory(
        max_length=int(stage5a["model_positions"]),
        device=model_config.get("device", "cuda"),
        inversion_checkpoint=model_config["inversion_checkpoint"],
        corrector_checkpoint=model_config["corrector_checkpoint"],
        inversion_revision=model_config.get("inversion_revision"),
        corrector_revision=model_config.get("corrector_revision"),
    )
    windows = [
        window for document in manifest["documents"] for window in document["windows"]
    ]
    texts = [str(window["text"]) for window in windows]
    expected_input_ids = [window["model_input_ids"] for window in windows]
    embedding_started = _cuda_telemetry_start(torch)
    embeddings = backend.embed_raw_windows(texts, expected_input_ids=expected_input_ids)
    embedding_telemetry = _cuda_telemetry_end(torch, embedding_started, len(windows))
    if embeddings.dtype != torch.float32:
        raise AssertionError("Stage 5A target embeddings must be FP32")
    embedding_sha256 = hashlib.sha256(
        embeddings.contiguous().numpy().tobytes()
    ).hexdigest()
    view_keys = _view_keys(manifest)
    strong_setting = dict(stage5a["strong_setting"])
    if (
        int(strong_setting["correction_steps"]),
        int(strong_setting["sequence_beam_width"]),
        int(strong_setting["top_n"]),
    ) != (50, 8, 1):
        raise ValueError("Stage 5A strong setting changed")
    fingerprint = _fingerprint(
        manifest_sha256=manifest_sha256,
        embedding_sha256=embedding_sha256,
        view_keys=view_keys,
        model_config=model_config,
        strong_setting=strong_setting,
        reused_cache_sha256=str(stage5a["stage3_strong_cache_sha256"]),
    )
    checkpoint_path = output_dir / str(stage5a["checkpoint_filename"])
    candidates, records = _load_checkpoint(
        checkpoint_path, fingerprint=fingerprint, view_keys=view_keys
    )

    stage3_candidates = [
        _candidate_from_dict(raw) for raw in stage3_cache["top1_candidates"]
    ]
    for document_index, document in enumerate(manifest["documents"]):
        target_index = document_index * 5
        if candidates[target_index] is None:
            candidate = stage3_candidates[document_index * 8]
            candidates[target_index] = candidate
            records[target_index] = {
                "view_index": target_index,
                "view_key": view_keys[target_index],
                "passage_id": document["passage_id"],
                "window_index": 0,
                "reused_from_stage3_strong_cache": True,
                "source_stage3_view_index": document_index * 8,
                "public_top1_exact_id_match": True,
                "embed_call_count": 51,
                "search_hypothesis_rows_embedded": 3145,
                "telemetry": {"elapsed_seconds": 0.0},
            }
    _save_checkpoint(
        checkpoint_path,
        fingerprint=fingerprint,
        manifest_sha256=manifest_sha256,
        embedding_sha256=embedding_sha256,
        view_keys=view_keys,
        candidates=candidates,
        records=records,
    )

    def generate_view(view_index: int) -> None:
        if candidates[view_index] is not None:
            return
        telemetry_start = _cuda_telemetry_start(torch)
        pools, generated_records = backend.generate_final_sbeam_candidate_pools(
            embeddings[view_index : view_index + 1],
            correction_steps=50,
            sequence_beam_width=8,
            top_n=1,
        )
        telemetry = _cuda_telemetry_end(torch, telemetry_start, 1)
        record = dict(generated_records[0])
        record.update(
            {
                "view_index": view_index,
                "view_key": view_keys[view_index],
                "passage_id": manifest["documents"][view_index // 5]["passage_id"],
                "window_index": view_index % 5,
                "is_final_tail": bool(windows[view_index]["is_final_tail"]),
                "reused_from_stage3_strong_cache": False,
                "telemetry": telemetry,
            }
        )
        if not bool(record["public_top1_exact_id_match"]):
            raise RuntimeError("Stage 5A public/instrumented Top-1 mismatch")
        candidates[view_index] = pools[0][0]
        records[view_index] = record
        _save_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            manifest_sha256=manifest_sha256,
            embedding_sha256=embedding_sha256,
            view_keys=view_keys,
            candidates=candidates,
            records=records,
        )

    first_tail_index = 4
    generate_view(first_tail_index)
    tail_candidate = candidates[first_tail_index]
    tail_record = records[first_tail_index]
    if tail_candidate is None or tail_record is None:
        raise AssertionError("Stage 5A tail preflight did not produce a candidate")
    preflight = {
        "status": "PASS",
        "view_key": view_keys[first_tail_index],
        "source_tokens": int(windows[first_tail_index]["source_content_length"]),
        "standalone_content_tokens": int(
            windows[first_tail_index]["content_length"]
        ),
        "model_positions": 32,
        "attention_tokens": sum(windows[first_tail_index]["attention_mask"]),
        "padding_tokens": windows[first_tail_index]["attention_mask"].count(0),
        "candidate_content_tokens": len(tail_candidate.content_ids),
        "public_top1_exact_id_match": bool(
            tail_record["public_top1_exact_id_match"]
        ),
        "correction_steps": 50,
        "sequence_beam_width": 8,
    }
    _write_json_atomic(output_dir / "stage5a_tail_preflight.json", preflight)
    if preflight_only:
        return {
            "status": "STAGE5A_TAIL_PREFLIGHT_PASS",
            "manifest_sha256": manifest_sha256,
            "embedding_sha256": embedding_sha256,
            "preflight": preflight,
            "completed_checkpoint_views": sum(
                candidate is not None for candidate in candidates
            ),
            "full_baseline_executed": False,
        }

    generation_started = time.perf_counter()
    initial_completed = sum(candidate is not None for candidate in candidates)
    progress_every = int(stage5a["progress_every_views"])
    for view_index in range(len(view_keys)):
        if candidates[view_index] is not None:
            continue
        generate_view(view_index)
        completed = sum(candidate is not None for candidate in candidates)
        newly_completed = completed - initial_completed
        if newly_completed % progress_every == 0 or completed == len(view_keys):
            elapsed = time.perf_counter() - generation_started
            remaining_new = sum(candidate is None for candidate in candidates)
            seconds_per_new = elapsed / newly_completed if newly_completed else 0.0
            print(
                f"stage5a_progress={completed}/{len(view_keys)} "
                f"last={view_keys[view_index]} "
                f"eta_minutes={seconds_per_new * remaining_new / 60:.1f}",
                flush=True,
            )
    if any(candidate is None for candidate in candidates) or any(
        record is None for record in records
    ):
        raise AssertionError("Stage 5A strong cache is incomplete")
    completed_candidates = [candidate for candidate in candidates if candidate is not None]
    completed_records = [record for record in records if record is not None]
    if len(completed_candidates) != 150:
        raise AssertionError("Stage 5A expected exactly 150 logical views")

    cache_path = output_dir / str(stage5a["cache_filename"])
    cache_payload = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "manifest_sha256": manifest_sha256,
        "embedding_sha256": embedding_sha256,
        "fingerprint": fingerprint,
        "view_keys": view_keys,
        "top1_candidates": [candidate.to_dict() for candidate in completed_candidates],
        "records": completed_records,
        "logical_views": 150,
        "reused_views": sum(
            bool(record["reused_from_stage3_strong_cache"])
            for record in completed_records
        ),
        "public_top1_exact_id_match": all(
            bool(record["public_top1_exact_id_match"])
            for record in completed_records
        ),
        "strong_reproduction_setting": "correction50+sbeam8",
    }
    _write_json_atomic(cache_path, cache_payload)

    rows = []
    for document_index, document in enumerate(manifest["documents"]):
        offset = document_index * 5
        rows.append(
            _evaluate_nooverlap_document(
                document,
                completed_candidates[offset : offset + 5],
                backend.embedder_tokenizer,
            )
        )
    summary = _aggregate_nooverlap(rows)
    comparisons = _paired_comparisons(rows, stage4_trajectories)
    bootstrap = _bootstrap_comparisons(
        comparisons,
        iterations=int(stage5a["bootstrap_iterations"]),
        seed=int(stage5a["bootstrap_seed"]),
    )
    sensitivity_subgroups = _sensitivity_subgroups(
        manifest,
        rows,
        comparisons,
        iterations=int(stage5a["bootstrap_iterations"]),
        seed=int(stage5a["bootstrap_seed"]),
    )
    gate = _classify_gate(bootstrap, sensitivity_subgroups)
    generated_records = [
        record
        for record in completed_records
        if not bool(record["reused_from_stage3_strong_cache"])
    ]
    elapsed_values = [
        float(record["telemetry"]["elapsed_seconds"]) for record in generated_records
    ]
    peak_allocated = [
        int(record["telemetry"]["peak_allocated_bytes"])
        for record in generated_records
        if record["telemetry"].get("peak_allocated_bytes") is not None
    ]
    payload = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE5A_COMPLETE",
        "stage": "NO_OVERLAP_STRONG_PRACTICAL_POLICY_COMPARISON",
        "config": dict(config),
        "environment": _environment(),
        "frozen_validation": {
            "cohort_sha256": str(stage5a["cohort_sha256"]),
            "stage3_strong_cache_sha256": str(
                stage5a["stage3_strong_cache_sha256"]
            ),
            "stage4_result_sha256": str(stage5a["stage4_result_sha256"]),
            "stage4_trajectories_sha256": str(
                stage5a["stage4_trajectories_sha256"]
            ),
            "manifest_sha256": manifest_sha256,
            "same_document_order": True,
            "same_136_source_ids": True,
            "source_spans_reselected": False,
        },
        "estimand": (
            "The same frozen 30 round-trip-stable 136-token NQ source spans under "
            "two practical chunking policies with different vector counts."
        ),
        "comparison_warning": (
            "Overlap uses eight vectors and no-overlap uses five; this is a practical "
            "policy comparison, not a pure causal estimate of overlap. No-overlap "
            "boundaries are adjusted only when standalone tokenization would exceed "
            "31 content IDs, and standalone boundary token IDs may differ from the "
            "whole-document source-token slices."
        ),
        "tail_preflight": preflight,
        "nooverlap_summary": summary,
        "nooverlap_rows": rows,
        "overlap_stage4_summary": stage4["summaries"]["8"],
        "paired_rows": comparisons,
        "paired_comparison": bootstrap,
        "sensitivity_subgroups": sensitivity_subgroups,
        "gate": gate,
        "telemetry": {
            "stage5a_wall_seconds": time.perf_counter() - started_all,
            "embedding": {
                **embedding_telemetry,
                "embedding_sha256": embedding_sha256,
                "logical_views": 150,
            },
            "logical_nooverlap_views": 150,
            "reused_stage3_views": 30,
            "new_strong_views": len(generated_records),
            "new_search_runtime_seconds": sum(elapsed_values),
            "new_runtime_seconds_per_view": mean(elapsed_values),
            "new_runtime_seconds_per_view_median": median(elapsed_values),
            "new_peak_allocated_bytes": max(peak_allocated)
            if peak_allocated
            else None,
            "logical_search_embedding_queries": 150 * 51,
            "actual_new_search_embedding_queries": len(generated_records) * 51,
            "overlap_logical_search_embedding_queries": 240 * 51,
            "logical_vector_ratio_nooverlap_over_overlap": 5 / 8,
            "resume_note": (
                "Candidate search resumes from the atomic checkpoint; the 150 target "
                "embeddings are recomputed before checkpoint validation."
            ),
        },
        "cache": {
            "path": str(cache_path),
            "sha256": _sha256(cache_path),
            "checkpoint_path": str(checkpoint_path),
            "all_public_instrumented_top1_match": True,
        },
        "stage5b_executed": False,
        "stage5c_executed": False,
        "document_assembly_method_implemented": False,
        "joint_reranker_implemented": False,
    }
    _write_report(output_dir, payload)
    return payload
