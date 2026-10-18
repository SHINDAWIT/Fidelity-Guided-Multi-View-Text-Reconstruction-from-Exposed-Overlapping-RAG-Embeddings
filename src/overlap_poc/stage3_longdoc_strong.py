"""Stage 3B/C strong Independent long-document diagnostics.

The frozen Stage 3A cohort is consumed directly. Each of its 240 windows is
inverted exactly once with the repository's correction50+sbeam8 reproduction
setting and cached. Evaluation reuses only the public-equivalent Top-1; this
module intentionally contains no candidate reranking, Joint score, voting, or
path decoder.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median
import time
from typing import Any, Callable, Mapping, Sequence

import yaml

from .diagnostic_experiment import (
    _cuda_telemetry_end,
    _cuda_telemetry_start,
    load_diagnostic_config,
)
from .diagnostics import (
    StitchedSequenceIDs,
    TokenAlignment,
    align_token_ids,
    stitch_candidate_sequence,
)
from .reconstruction import Candidate, Vec2TextBackend
from .stage1_strong_saturation import _candidate_from_dict, _write_json_atomic
from .strong_pool_preflight import _environment


_K_VALUES = (2, 3, 4, 6, 8)
_IMPLEMENTATION_VERSION = "stage3bc-strong-v2"


def load_stage3bc_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        stage3bc = yaml.safe_load(handle)
    if not isinstance(stage3bc, dict):
        raise ValueError("Stage 3B/C config must be a YAML mapping")
    raw_base = Path(str(stage3bc["base_config"]))
    base_path = raw_base if raw_base.is_absolute() else config_path.parent / raw_base
    raw_cohort = Path(str(stage3bc["cohort_artifact"]))
    cohort_path = (
        raw_cohort if raw_cohort.is_absolute() else config_path.parent / raw_cohort
    )
    return {
        "base": load_diagnostic_config(base_path),
        "stage3bc": stage3bc,
        "resolved_base_config": str(base_path.resolve()),
        "resolved_stage3bc_config": str(config_path.resolve()),
        "resolved_cohort_artifact": str(cohort_path.resolve()),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def _validate_cohort(
    cohort: Mapping[str, Any],
    *,
    expected_sha256: str,
    actual_sha256: str,
    expected_documents: int,
    content_tokens: int,
    overlap_tokens: int,
    stride: int,
    model_positions: int,
) -> dict[str, Any]:
    if actual_sha256 != expected_sha256:
        raise RuntimeError("Stage 3A cohort SHA-256 changed")
    if tuple(int(k) for k in cohort["geometry"]["k_values"]) != _K_VALUES:
        raise RuntimeError("Stage 3A cohort K grid changed")
    documents = list(cohort["documents"])
    manifest = list(cohort["manifest"])
    if len(documents) != expected_documents or len(manifest) != expected_documents:
        raise RuntimeError("Stage 3A cohort size changed")
    if len(set(manifest)) != len(manifest):
        raise RuntimeError("Stage 3A cohort manifest contains duplicates")
    if [document["passage_id"] for document in documents] != manifest:
        raise RuntimeError("Stage 3A cohort document order differs from manifest")

    checked_prefixes = 0
    checked_windows = 0
    for document in documents:
        selected_start = int(document["selected_start"])
        windows = list(document["windows"])
        if len(windows) != max(_K_VALUES):
            raise RuntimeError("every frozen document must contain exactly eight windows")
        for window_index, window in enumerate(windows):
            content_ids = tuple(int(value) for value in window["content_ids"])
            model_ids = tuple(int(value) for value in window["model_input_ids"])
            attention = tuple(int(value) for value in window["attention_mask"])
            if len(content_ids) != content_tokens:
                raise RuntimeError("frozen window content length changed")
            if len(model_ids) != model_positions or attention != (1,) * model_positions:
                raise RuntimeError("frozen model-position contract changed")
            expected_start = selected_start + window_index * stride
            if int(window["document_token_start"]) != expected_start:
                raise RuntimeError("frozen window stride changed")
            if int(window["document_token_end"]) != expected_start + content_tokens:
                raise RuntimeError("frozen window end changed")
            if window_index:
                previous_ids = tuple(
                    int(value) for value in windows[window_index - 1]["content_ids"]
                )
                if previous_ids[-overlap_tokens:] != content_ids[:overlap_tokens]:
                    raise RuntimeError("frozen adjacent overlap changed")
            checked_windows += 1

        for k in _K_VALUES:
            prefix = document["prefixes"][str(k)]
            latent_length = content_tokens + stride * (k - 1)
            reference = tuple(int(value) for value in prefix["reference_ids"])
            multiplicity = tuple(
                int(value) for value in prefix["observation_multiplicity"]
            )
            if len(reference) != latent_length or len(multiplicity) != latent_length:
                raise RuntimeError("frozen prefix length changed")
            expected_reference = tuple(
                int(value) for value in windows[0]["content_ids"]
            ) + tuple(
                token
                for window in windows[1:k]
                for token in tuple(int(value) for value in window["content_ids"])[
                    overlap_tokens:
                ]
            )
            if reference != expected_reference:
                raise RuntimeError("frozen prefix union IDs changed")
            recomputed_multiplicity = tuple(
                sum(
                    int(window["document_token_start"])
                    <= selected_start + position
                    < int(window["document_token_end"])
                    for window in windows[:k]
                )
                for position in range(latent_length)
            )
            if multiplicity != recomputed_multiplicity:
                raise RuntimeError("frozen source-token multiplicity changed")
            checked_prefixes += 1
    return {
        "sha256_match": True,
        "manifest_unique": True,
        "document_order_match": True,
        "checked_documents": len(documents),
        "checked_windows": checked_windows,
        "checked_prefixes": checked_prefixes,
        "adjacent_overlap_tokens": overlap_tokens,
        "distance_two_overlap_tokens": max(0, content_tokens - 2 * stride),
    }


def _view_keys(cohort: Mapping[str, Any]) -> list[str]:
    return [
        f"{document['passage_id']}:{window['window_index']}"
        for document in cohort["documents"]
        for window in document["windows"]
    ]


def _checkpoint_fingerprint(
    *,
    cohort_sha256: str,
    view_keys: Sequence[str],
    model_config: Mapping[str, Any],
    strong_setting: Mapping[str, Any],
) -> str:
    frozen = {
        "implementation_version": _IMPLEMENTATION_VERSION,
        "cohort_sha256": cohort_sha256,
        "view_keys": list(view_keys),
        "models": dict(model_config),
        "strong_setting": dict(strong_setting),
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
    embedding_sha256: str,
    view_keys: Sequence[str],
    strong_setting: Mapping[str, Any],
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("fingerprint") != fingerprint:
        raise RuntimeError("Stage 3B/C checkpoint belongs to another frozen run")
    if checkpoint.get("embedding_sha256") != embedding_sha256:
        raise RuntimeError("Stage 3B/C checkpoint target embeddings changed")
    candidates = [
        _candidate_from_dict(candidate)
        for candidate in checkpoint.get("top1_candidates", [])
    ]
    records = list(checkpoint.get("instrumentation_records", []))
    completed_keys = list(checkpoint.get("completed_view_keys", []))
    if len(candidates) != len(records) or len(candidates) != len(completed_keys):
        raise RuntimeError("Stage 3B/C checkpoint progress is inconsistent")
    if completed_keys != list(view_keys[: len(completed_keys)]):
        raise RuntimeError("Stage 3B/C checkpoint view order changed")
    if len(candidates) > len(view_keys):
        raise RuntimeError("Stage 3B/C checkpoint has too many views")
    expected_steps = int(strong_setting["correction_steps"])
    expected_beam = int(strong_setting["sequence_beam_width"])
    for index, (candidate, record) in enumerate(zip(candidates, records)):
        if record.get("view_key") != view_keys[index]:
            raise RuntimeError("Stage 3B/C checkpoint record key changed")
        if int(record.get("view_index", -1)) != index:
            raise RuntimeError("Stage 3B/C checkpoint record index changed")
        if (
            candidate.correction_steps != expected_steps
            or candidate.sequence_beam_width != expected_beam
        ):
            raise RuntimeError("Stage 3B/C checkpoint candidate setting changed")
        if int(record.get("embed_call_count", -1)) != expected_steps + 1:
            raise RuntimeError("Stage 3B/C checkpoint search budget changed")
    return candidates, records


def _save_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    cohort_sha256: str,
    embedding_sha256: str,
    view_keys: Sequence[str],
    candidates: Sequence[Candidate],
    records: Sequence[Mapping[str, Any]],
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "cohort_sha256": cohort_sha256,
            "embedding_sha256": embedding_sha256,
            "expected_views": len(view_keys),
            "completed_views": len(candidates),
            "completed_view_keys": list(view_keys[: len(candidates)]),
            "top1_candidates": [candidate.to_dict() for candidate in candidates],
            "instrumentation_records": list(records),
        },
    )


def _prediction_to_global_matches(
    alignment: TokenAlignment,
    *,
    relative_window_start: int,
) -> tuple[dict[int, int], set[int]]:
    prediction_to_global = {
        prediction_index: relative_window_start + reference_index
        for prediction_index, reference_index in alignment.exact_matches
    }
    return prediction_to_global, set(prediction_to_global.values())


def _evaluate_document_prefix(
    document: Mapping[str, Any],
    candidates: Sequence[Candidate],
    records: Sequence[Mapping[str, Any]],
    *,
    k: int,
    content_tokens: int,
    overlap_tokens: int,
    stride: int,
    minimum_alignment: float,
    stitch_sequence: Callable[[Sequence[Candidate]], StitchedSequenceIDs] | None = None,
) -> dict[str, Any]:
    prefix = document["prefixes"][str(k)]
    reference = tuple(int(value) for value in prefix["reference_ids"])
    multiplicity = tuple(
        int(value) for value in prefix["observation_multiplicity"]
    )
    windows = list(document["windows"][:k])
    candidates = tuple(candidates[:k])
    records = tuple(records[:k])
    if len(candidates) != k or len(records) != k:
        raise ValueError("document prefix candidate cache is incomplete")

    observations: list[list[bool]] = [[] for _ in reference]
    correct_views: list[set[int]] = [set() for _ in reference]
    prediction_global_maps: list[dict[int, int]] = []
    window_rows: list[dict[str, Any]] = []
    for view_index, (window, candidate) in enumerate(zip(windows, candidates)):
        window_reference = tuple(int(value) for value in window["content_ids"])
        alignment = align_token_ids(candidate.content_ids, window_reference)
        relative_start = int(window["document_token_start"]) - int(
            document["selected_start"]
        )
        prediction_map, matched_global = _prediction_to_global_matches(
            alignment, relative_window_start=relative_start
        )
        prediction_global_maps.append(prediction_map)
        for local_position in range(content_tokens):
            global_position = relative_start + local_position
            is_correct = global_position in matched_global
            observations[global_position].append(is_correct)
            if is_correct:
                correct_views[global_position].add(view_index)
        window_rows.append(
            {
                "window_index": view_index,
                "prediction_length": len(candidate.content_ids),
                "edit_distance": alignment.distance,
                "token_recovery": max(0.0, 1.0 - alignment.distance / content_tokens),
                "exact_matched_token_recovery": (
                    len(alignment.exact_matches) / content_tokens
                ),
                "exact": tuple(candidate.content_ids) == window_reference,
            }
        )
    if tuple(len(values) for values in observations) != multiplicity:
        raise AssertionError("evaluated observation multiplicity differs from cohort")

    states: list[str] = []
    for values in observations:
        if all(values):
            states.append("ALL_CORRECT")
        elif any(values):
            states.append("MIXED")
        else:
            states.append("ALL_WRONG")

    stitched = (
        stitch_sequence(candidates)
        if stitch_sequence is not None
        else stitch_candidate_sequence(
            candidates,
            overlap_tokens=overlap_tokens,
            minimum_alignment=minimum_alignment,
        )
    )
    final_alignment = align_token_ids(stitched.content_ids, reference)
    final_matched_positions = {
        reference_index for _, reference_index in final_alignment.exact_matches
    }
    final_error_positions = set(range(len(reference))) - final_matched_positions
    retained_correct_views: list[set[int]] = [set() for _ in reference]
    retained_correct_output_indices: list[list[int]] = [
        [] for _ in reference
    ]
    for output_index, (view_index, prediction_index) in enumerate(stitched.origins):
        if output_index >= len(stitched.content_ids):
            raise AssertionError("stitch provenance exceeds output length")
        global_position = prediction_global_maps[view_index].get(prediction_index)
        if global_position is not None:
            retained_correct_views[global_position].add(view_index)
            retained_correct_output_indices[global_position].append(output_index)

    attribution = Counter()
    attribution_by_position: list[dict[str, Any]] = []
    for position in sorted(final_error_positions):
        if not correct_views[position]:
            cause = "SOURCE_VIEW_WRONG"
        elif retained_correct_views[position]:
            cause = "ALIGNMENT_AMBIGUITY"
        else:
            cause = "STITCH_ONLY_ERROR"
        attribution[cause] += 1
        attribution_by_position.append(
            {
                "source_position": position,
                "multiplicity": multiplicity[position],
                "cause": cause,
                "correct_view_indices": sorted(correct_views[position]),
                "retained_correct_view_indices": sorted(
                    retained_correct_views[position]
                ),
            }
        )
    insertion_attribution: list[dict[str, Any]] = []
    for prediction_index in final_alignment.prediction_insertions:
        view_index, local_prediction_index = stitched.origins[prediction_index]
        global_position = prediction_global_maps[view_index].get(
            local_prediction_index
        )
        if global_position is None:
            cause = "SOURCE_VIEW_WRONG"
        elif len(retained_correct_output_indices[global_position]) > 1:
            cause = "STITCH_ONLY_ERROR"
        else:
            cause = "ALIGNMENT_AMBIGUITY"
        attribution[cause] += 1
        insertion_attribution.append(
            {
                "stitched_prediction_index": prediction_index,
                "token_id": stitched.content_ids[prediction_index],
                "origin_view_index": view_index,
                "origin_local_prediction_index": local_prediction_index,
                "locally_aligned_source_position": global_position,
                "retained_outputs_for_source_position": (
                    []
                    if global_position is None
                    else retained_correct_output_indices[global_position]
                ),
                "cause": cause,
            }
        )
    if sum(attribution.values()) != final_alignment.distance:
        raise AssertionError("final edit attribution does not sum to edit distance")

    state_counts_by_multiplicity: dict[str, Any] = {}
    for group in sorted(set(multiplicity)):
        positions = [
            position for position, value in enumerate(multiplicity) if value == group
        ]
        state_counts = Counter(states[position] for position in positions)
        view_observations = sum(len(observations[position]) for position in positions)
        correct_observations = sum(
            sum(observations[position]) for position in positions
        )
        final_correct = sum(position in final_matched_positions for position in positions)
        state_counts_by_multiplicity[str(group)] = {
            "source_token_count": len(positions),
            "final_correct_count": final_correct,
            "final_error_count": len(positions) - final_correct,
            "final_token_accuracy": _safe_ratio(final_correct, len(positions)),
            "view_observation_count": view_observations,
            "correct_view_observation_count": correct_observations,
            "view_observation_accuracy": _safe_ratio(
                correct_observations, view_observations
            ),
            "ALL_CORRECT_count": state_counts["ALL_CORRECT"],
            "MIXED_count": state_counts["MIXED"],
            "ALL_WRONG_count": state_counts["ALL_WRONG"],
        }

    multiply_observed_positions = [
        position for position, value in enumerate(multiplicity) if value >= 2
    ]
    mixed_count = sum(states[position] == "MIXED" for position in multiply_observed_positions)
    all_wrong_count = sum(
        states[position] == "ALL_WRONG" for position in multiply_observed_positions
    )
    all_correct_count = sum(
        states[position] == "ALL_CORRECT" for position in multiply_observed_positions
    )
    wrong_view_observations = sum(
        sum(not value for value in observations[position])
        for position in multiply_observed_positions
    )
    recoverable_wrong_view_observations = sum(
        sum(not value for value in observations[position])
        for position in multiply_observed_positions
        if states[position] == "MIXED"
    )
    final_recoverable_upper_denominator_positions = [
        position
        for position in final_error_positions
        if multiplicity[position] >= 2
    ]
    final_recoverable_upper_numerator = sum(
        bool(correct_views[position])
        for position in final_recoverable_upper_denominator_positions
    )
    final_recoverable_denominator_positions = [
        position
        for position in final_recoverable_upper_denominator_positions
        if not retained_correct_views[position]
    ]
    final_recoverable_numerator = sum(
        bool(correct_views[position])
        for position in final_recoverable_denominator_positions
    )
    exact_match_count = len(final_alignment.exact_matches)
    error_rate = final_alignment.distance / len(reference)
    return {
        "passage_id": document["passage_id"],
        "k": k,
        "latent_content_length": len(reference),
        "window_metrics": window_rows,
        "incorrect_window_count": sum(not row["exact"] for row in window_rows),
        "has_incorrect_window": any(not row["exact"] for row in window_rows),
        "stitched_prediction_ids": list(stitched.content_ids),
        "stitch_consumed_right_tokens": list(stitched.consumed_right_tokens),
        "stitch_aligned_left_tokens": list(stitched.aligned_left_tokens),
        "stitch_alignment_scores": list(stitched.alignment_scores),
        "stitch_noncanonical_boundary_count": sum(
            consumed != overlap_tokens
            for consumed in stitched.consumed_right_tokens
        ),
        "document_edit_distance": final_alignment.distance,
        "document_normalized_edit_similarity": max(0.0, 1.0 - error_rate),
        "document_token_recovery": exact_match_count / len(reference),
        "document_token_error_rate": error_rate,
        "document_error_tokens_per_100": 100.0 * error_rate,
        "document_exact": tuple(stitched.content_ids) == reference,
        "final_exact_matched_source_tokens": exact_match_count,
        "final_source_token_error_count": len(final_error_positions),
        "final_prediction_insertion_count": len(
            final_alignment.prediction_insertions
        ),
        "multiplicity": state_counts_by_multiplicity,
        "multiply_observed_source_tokens": len(multiply_observed_positions),
        "ALL_CORRECT_multiply_count": all_correct_count,
        "MIXED_multiply_count": mixed_count,
        "ALL_WRONG_multiply_count": all_wrong_count,
        "token_level_recoverable_redundancy": _safe_ratio(
            mixed_count, len(multiply_observed_positions)
        ),
        "view_recoverable_numerator": recoverable_wrong_view_observations,
        "view_recoverable_denominator": wrong_view_observations,
        "view_recoverable_error_rate": _safe_ratio(
            recoverable_wrong_view_observations, wrong_view_observations
        ),
        "final_recoverable_numerator": final_recoverable_numerator,
        "final_recoverable_denominator": len(
            final_recoverable_denominator_positions
        ),
        "final_recoverable_error_rate": _safe_ratio(
            final_recoverable_numerator,
            len(final_recoverable_denominator_positions),
        ),
        "final_recoverable_upper_bound_numerator": (
            final_recoverable_upper_numerator
        ),
        "final_recoverable_upper_bound_denominator": len(
            final_recoverable_upper_denominator_positions
        ),
        "final_recoverable_upper_bound_rate": _safe_ratio(
            final_recoverable_upper_numerator,
            len(final_recoverable_upper_denominator_positions),
        ),
        "final_recoverable_definition": (
            "Primary conservative R_final excludes alignment-ambiguous source "
            "positions and counts only final errors where locally correct token "
            "evidence exists in a view but no such correct token survived the "
            "deterministic stitch. The upper bound includes ambiguous positions."
        ),
        "final_error_attribution": {
            "SOURCE_VIEW_WRONG": attribution["SOURCE_VIEW_WRONG"],
            "STITCH_ONLY_ERROR": attribution["STITCH_ONLY_ERROR"],
            "ALIGNMENT_AMBIGUITY": attribution["ALIGNMENT_AMBIGUITY"],
            "total_edit_operations": final_alignment.distance,
            "stitch_only_fraction": _safe_ratio(
                attribution["STITCH_ONLY_ERROR"], final_alignment.distance
            ),
            "alignment_ambiguity_fraction": _safe_ratio(
                attribution["ALIGNMENT_AMBIGUITY"], final_alignment.distance
            ),
            "operational_definition": (
                "SOURCE_VIEW_WRONG: no local view has an exact GT-aligned token; "
                "STITCH_ONLY_ERROR: a locally correct token exists but was dropped, "
                "or stitching retains duplicate locally correct mappings and one "
                "becomes a global insertion; "
                "ALIGNMENT_AMBIGUITY: a locally correct token survives stitching but "
                "the deterministic global alignment does not match that GT position, "
                "including a non-duplicate locally correct global insertion"
            ),
        },
        "final_error_attribution_by_source_position": attribution_by_position,
        "final_prediction_insertion_attribution": insertion_attribution,
        "cached_prefix_search_seconds": sum(
            float(record["telemetry"]["elapsed_seconds"]) for record in records
        ),
    }


def _aggregate_prefix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty K prefix")
    window_rows = [window for row in rows for window in row["window_metrics"]]
    latent_length = int(rows[0]["latent_content_length"])
    if any(int(row["latent_content_length"]) != latent_length for row in rows):
        raise AssertionError("paired prefix lengths differ within K")
    total_reference_tokens = latent_length * len(rows)
    total_edit_distance = sum(int(row["document_edit_distance"]) for row in rows)
    total_exact_matches = sum(
        int(row["final_exact_matched_source_tokens"]) for row in rows
    )

    multiplicity: dict[str, Any] = {}
    for group in ("1", "2", "3"):
        available = [row["multiplicity"][group] for row in rows if group in row["multiplicity"]]
        source_tokens = sum(int(value["source_token_count"]) for value in available)
        final_correct = sum(int(value["final_correct_count"]) for value in available)
        view_observations = sum(int(value["view_observation_count"]) for value in available)
        correct_view_observations = sum(
            int(value["correct_view_observation_count"]) for value in available
        )
        state_counts = {
            state: sum(int(value[f"{state}_count"]) for value in available)
            for state in ("ALL_CORRECT", "MIXED", "ALL_WRONG")
        }
        multiplicity[group] = {
            "source_token_count": source_tokens,
            "final_correct_count": final_correct,
            "final_error_count": source_tokens - final_correct,
            "final_token_accuracy": _safe_ratio(final_correct, source_tokens),
            "final_token_error_rate": _safe_ratio(
                source_tokens - final_correct, source_tokens
            ),
            "view_observation_count": view_observations,
            "correct_view_observation_count": correct_view_observations,
            "view_observation_accuracy": _safe_ratio(
                correct_view_observations, view_observations
            ),
            **{
                f"{state}_count": count for state, count in state_counts.items()
            },
            **{
                f"{state}_fraction": _safe_ratio(count, source_tokens)
                for state, count in state_counts.items()
            },
        }

    multiply_total = sum(int(row["multiply_observed_source_tokens"]) for row in rows)
    state_totals = {
        state: sum(int(row[f"{state}_multiply_count"]) for row in rows)
        for state in ("ALL_CORRECT", "MIXED", "ALL_WRONG")
    }
    view_numerator = sum(int(row["view_recoverable_numerator"]) for row in rows)
    view_denominator = sum(int(row["view_recoverable_denominator"]) for row in rows)
    final_numerator = sum(int(row["final_recoverable_numerator"]) for row in rows)
    final_denominator = sum(
        int(row["final_recoverable_denominator"]) for row in rows
    )
    final_upper_numerator = sum(
        int(row["final_recoverable_upper_bound_numerator"]) for row in rows
    )
    final_upper_denominator = sum(
        int(row["final_recoverable_upper_bound_denominator"]) for row in rows
    )
    attribution = {
        cause: sum(int(row["final_error_attribution"][cause]) for row in rows)
        for cause in (
            "SOURCE_VIEW_WRONG",
            "STITCH_ONLY_ERROR",
            "ALIGNMENT_AMBIGUITY",
        )
    }
    if sum(attribution.values()) != total_edit_distance:
        raise AssertionError("aggregate edit attribution differs from edit distance")
    return {
        "documents": len(rows),
        "latent_content_length": latent_length,
        "window_level": {
            "window_count": len(window_rows),
            "token_recovery": _distribution(
                [float(row["token_recovery"]) for row in window_rows]
            ),
            "exact_matched_token_recovery": _distribution(
                [float(row["exact_matched_token_recovery"]) for row in window_rows]
            ),
            "exact_recovery_fraction": mean(bool(row["exact"]) for row in window_rows),
            "incorrect_window_count": sum(not row["exact"] for row in window_rows),
            "incorrect_windows_per_document": _distribution(
                [float(row["incorrect_window_count"]) for row in rows]
            ),
            "documents_with_incorrect_window_fraction": mean(
                bool(row["has_incorrect_window"]) for row in rows
            ),
        },
        "document_level": {
            "normalized_edit_similarity": _distribution(
                [float(row["document_normalized_edit_similarity"]) for row in rows]
            ),
            "token_recovery": _distribution(
                [float(row["document_token_recovery"]) for row in rows]
            ),
            "pooled_exact_matched_token_recovery": (
                total_exact_matches / total_reference_tokens
            ),
            "token_error_rate": _distribution(
                [float(row["document_token_error_rate"]) for row in rows]
            ),
            "pooled_token_error_rate": total_edit_distance / total_reference_tokens,
            "error_tokens_per_100": 100.0 * total_edit_distance / total_reference_tokens,
            "document_exact_fraction": mean(bool(row["document_exact"]) for row in rows),
            "absolute_error_token_count": total_edit_distance,
            "absolute_error_tokens_per_document": _distribution(
                [float(row["document_edit_distance"]) for row in rows]
            ),
            "final_source_token_error_count": sum(
                int(row["final_source_token_error_count"]) for row in rows
            ),
            "final_prediction_insertion_count": sum(
                int(row["final_prediction_insertion_count"]) for row in rows
            ),
        },
        "multiplicity": multiplicity,
        "view_state_on_multiply_observed_tokens": {
            "source_token_count": multiply_total,
            **{
                f"{state}_count": count for state, count in state_totals.items()
            },
            **{
                f"{state}_fraction": _safe_ratio(count, multiply_total)
                for state, count in state_totals.items()
            },
        },
        "recoverable_redundancy": {
            "view_level": {
                "numerator": view_numerator,
                "denominator": view_denominator,
                "rate": _safe_ratio(view_numerator, view_denominator),
            },
            "token_level": {
                "definition": "MIXED / all multiply-observed source tokens",
                "numerator": state_totals["MIXED"],
                "denominator": multiply_total,
                "rate": _safe_ratio(state_totals["MIXED"], multiply_total),
            },
            "final_document": {
                "definition": (
                    "Conservative primary R_final excludes ALIGNMENT_AMBIGUITY; "
                    "numerator requires locally correct evidence that did not "
                    "survive deterministic stitching."
                ),
                "numerator": final_numerator,
                "denominator": final_denominator,
                "rate": _safe_ratio(final_numerator, final_denominator),
                "ambiguity_inclusive_upper_bound": {
                    "numerator": final_upper_numerator,
                    "denominator": final_upper_denominator,
                    "rate": _safe_ratio(
                        final_upper_numerator, final_upper_denominator
                    ),
                },
            },
        },
        "final_error_attribution": {
            **attribution,
            "total_edit_operations": total_edit_distance,
            "SOURCE_VIEW_WRONG_fraction": _safe_ratio(
                attribution["SOURCE_VIEW_WRONG"], total_edit_distance
            ),
            "STITCH_ONLY_ERROR_fraction": _safe_ratio(
                attribution["STITCH_ONLY_ERROR"], total_edit_distance
            ),
            "ALIGNMENT_AMBIGUITY_fraction": _safe_ratio(
                attribution["ALIGNMENT_AMBIGUITY"], total_edit_distance
            ),
            "noncanonical_stitch_boundary_count": sum(
                int(row["stitch_noncanonical_boundary_count"]) for row in rows
            ),
            "total_stitch_boundaries": len(rows) * (int(rows[0]["k"]) - 1),
        },
        "cached_compute": {
            "total_prefix_search_seconds": sum(
                float(row["cached_prefix_search_seconds"]) for row in rows
            ),
            "mean_prefix_search_seconds_per_document": mean(
                float(row["cached_prefix_search_seconds"]) for row in rows
            ),
            "is_reused_prefix_accounting_not_reexecution": True,
        },
    }


def _bootstrap_stat(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> float | None:
    if metric in {
        "document_normalized_edit_similarity",
        "document_token_recovery",
        "document_token_error_rate",
        "document_exact",
        "document_edit_distance",
    }:
        return mean(float(row[metric]) for row in rows)
    numerator_key, denominator_key = {
        "mixed_rate": ("MIXED_multiply_count", "multiply_observed_source_tokens"),
        "all_wrong_rate": (
            "ALL_WRONG_multiply_count",
            "multiply_observed_source_tokens",
        ),
        "r_view": ("view_recoverable_numerator", "view_recoverable_denominator"),
        "r_final": (
            "final_recoverable_numerator",
            "final_recoverable_denominator",
        ),
        "stitch_only_fraction": (None, None),
    }[metric]
    if metric == "stitch_only_fraction":
        numerator = sum(
            int(row["final_error_attribution"]["STITCH_ONLY_ERROR"])
            for row in rows
        )
        denominator = sum(int(row["document_edit_distance"]) for row in rows)
    else:
        numerator = sum(int(row[numerator_key]) for row in rows)
        denominator = sum(int(row[denominator_key]) for row in rows)
    return _safe_ratio(numerator, denominator)


def _paired_bootstrap(
    rows_by_k: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    document_count = len(rows_by_k[min(rows_by_k)])
    if any(len(rows) != document_count for rows in rows_by_k.values()):
        raise ValueError("paired bootstrap requires identical document counts")
    metrics = (
        "document_normalized_edit_similarity",
        "document_token_recovery",
        "document_token_error_rate",
        "document_exact",
        "document_edit_distance",
        "mixed_rate",
        "all_wrong_rate",
        "r_view",
        "r_final",
        "stitch_only_fraction",
    )
    samples = {
        k: {metric: [] for metric in metrics} for k in rows_by_k
    }
    deltas_from_k2 = {
        k: {metric: [] for metric in metrics} for k in rows_by_k if k != 2
    }
    rng = random.Random(seed)
    for _ in range(iterations):
        indices = [rng.randrange(document_count) for _ in range(document_count)]
        selected_by_k = {
            k: [rows[index] for index in indices] for k, rows in rows_by_k.items()
        }
        k2_values = {
            metric: _bootstrap_stat(selected_by_k[2], metric) for metric in metrics
        }
        for k, selected in selected_by_k.items():
            for metric in metrics:
                value = _bootstrap_stat(selected, metric)
                if value is None:
                    continue
                samples[k][metric].append(value)
                if k != 2 and k2_values[metric] is not None:
                    deltas_from_k2[k][metric].append(value - k2_values[metric])

    def intervals(values_by_metric: Mapping[str, Sequence[float]]) -> dict[str, Any]:
        return {
            metric: [
                _percentile(values, 0.025),
                _percentile(values, 0.975),
            ]
            for metric, values in values_by_metric.items()
        }

    return {
        "iterations": iterations,
        "seed": seed,
        "paired_resampling_unit": "frozen document manifest index",
        "by_k": {str(k): intervals(values) for k, values in samples.items()},
        "delta_from_k2": {
            str(k): intervals(values) for k, values in deltas_from_k2.items()
        },
    }


def classify_stage3bc_gate(
    summaries: Mapping[int, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen long-document error plus redundancy gate at K=8."""

    k8 = summaries[8]
    document = k8["document_level"]
    redundancy = k8["recoverable_redundancy"]
    states = k8["view_state_on_multiply_observed_tokens"]
    absolute_delta_ci = bootstrap["delta_from_k2"]["8"][
        "document_edit_distance"
    ]
    meaningful_residual_error = (
        float(document["pooled_token_error_rate"])
        >= float(thresholds["minimum_k8_token_error_rate"])
        and float(document["absolute_error_tokens_per_document"]["mean"])
        >= float(thresholds["minimum_k8_mean_absolute_errors"])
    )
    recoverable_redundancy = (
        float(states["MIXED_fraction"] or 0.0)
        >= float(thresholds["minimum_k8_mixed_rate"])
        and float(redundancy["view_level"]["rate"] or 0.0)
        >= float(thresholds["minimum_k8_r_view"])
        and float(redundancy["final_document"]["rate"] or 0.0)
        >= float(thresholds["minimum_k8_r_final"])
        and int(redundancy["final_document"]["denominator"])
        >= int(thresholds["minimum_k8_final_recoverable_denominator"])
    )
    if not meaningful_residual_error:
        state = "LONGDOC_NO_MEANINGFUL_ERROR"
        reason = (
            "K=8 does not clear the frozen exploratory normalized and absolute "
            "residual-error criteria."
        )
    elif not recoverable_redundancy:
        state = "LONGDOC_ERROR_BUT_NO_REDUNDANCY"
        reason = (
            "K=8 has meaningful residual error, but MIXED, R_view, conservative "
            "R_final, and denominator requirements do not all establish usable "
            "overlap redundancy."
        )
    else:
        state = "LONGDOC_RECOVERABLE_REDUNDANCY_GO"
        reason = (
            "K=8 has meaningful residual error and independently reconstructed "
            "overlapping views retain conservative recoverable token evidence."
        )
    normalized_delta_ci = bootstrap["delta_from_k2"]["8"][
        "document_token_error_rate"
    ]
    return {
        "state": state,
        "reason": reason,
        "meaningful_document_error": meaningful_residual_error,
        "recoverable_redundancy": recoverable_redundancy,
        "absolute_error_burden_increase_established": (
            absolute_delta_ci[0] is not None
            and float(absolute_delta_ci[0]) > 0.0
        ),
        "normalized_error_rate_worsening_established": (
            normalized_delta_ci[0] is not None
            and float(normalized_delta_ci[0]) > 0.0
        ),
        "thresholds": dict(thresholds),
        "observed_k8": {
            "pooled_token_error_rate": document["pooled_token_error_rate"],
            "mean_absolute_error_tokens": document[
                "absolute_error_tokens_per_document"
            ]["mean"],
            "paired_absolute_error_delta_from_k2_ci95": absolute_delta_ci,
            "paired_token_error_rate_delta_from_k2_ci95": normalized_delta_ci,
            "MIXED_fraction": states["MIXED_fraction"],
            "ALL_WRONG_fraction": states["ALL_WRONG_fraction"],
            "R_view": redundancy["view_level"]["rate"],
            "R_final": redundancy["final_document"]["rate"],
            "R_final_ambiguity_inclusive_upper_bound": redundancy[
                "final_document"
            ]["ambiguity_inclusive_upper_bound"]["rate"],
            "R_final_denominator": redundancy["final_document"]["denominator"],
            "stitch_only_error_fraction": k8["final_error_attribution"][
                "STITCH_ONLY_ERROR_fraction"
            ],
        },
        "joint_implementation_authorized_by_gate": (
            state == "LONGDOC_RECOVERABLE_REDUNDANCY_GO"
        ),
    }


def _write_report(output_dir: Path, payload: Mapping[str, Any]) -> None:
    def metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    _write_json_atomic(output_dir / "stage3bc_longdoc_strong.json", payload)
    lines = [
        "# Stage 3B/C — strong Independent long-document diagnostics",
        "",
        f"- Status: `{payload['status']}`",
        f"- Gate: `{payload['gate']['state']}`",
        f"- Reason: {payload['gate']['reason']}",
        "- Estimand: the frozen 30-document NQ cohort with a round-trip-stable "
        "K=8 / 136-content-ID segment, not the full pinned NQ scan frame.",
        "- Strong setting: repository reproduction `correction50+sbeam8`; no "
        "claim that beam width 8 is specified by the original Vec2Text paper.",
        "- Joint/reranking/voting/path decoding implemented: `false`",
        "",
        "| K | L | Window exact | Doc similarity | Token recovery | Token error | Error/100 | Doc exact | MIXED | ALL_WRONG | R_view | R_final | Cached sec/doc |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in payload["k_values"]:
        summary = payload["summaries"][str(k)]
        window = summary["window_level"]
        document = summary["document_level"]
        states = summary["view_state_on_multiply_observed_tokens"]
        redundancy = summary["recoverable_redundancy"]
        lines.append(
            f"| {k} | {summary['latent_content_length']} | "
            f"{window['exact_recovery_fraction']:.4f} | "
            f"{document['normalized_edit_similarity']['mean']:.4f} | "
            f"{document['token_recovery']['mean']:.4f} | "
            f"{document['pooled_token_error_rate']:.4f} | "
            f"{document['error_tokens_per_100']:.2f} | "
            f"{document['document_exact_fraction']:.4f} | "
            f"{metric(states['MIXED_fraction'])} | "
            f"{metric(states['ALL_WRONG_fraction'])} | "
            f"{metric(redundancy['view_level']['rate'])} | "
            f"{metric(redundancy['final_document']['rate'])} | "
            f"{summary['cached_compute']['mean_prefix_search_seconds_per_document']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Final edit attribution",
            "",
            "| K | Total edits | Source-view wrong | Stitch-only | Alignment ambiguity | Noncanonical stitch boundaries |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for k in payload["k_values"]:
        attribution = payload["summaries"][str(k)]["final_error_attribution"]
        lines.append(
            f"| {k} | {attribution['total_edit_operations']} | "
            f"{attribution['SOURCE_VIEW_WRONG']} "
            f"({metric(attribution['SOURCE_VIEW_WRONG_fraction'])}) | "
            f"{attribution['STITCH_ONLY_ERROR']} "
            f"({metric(attribution['STITCH_ONLY_ERROR_fraction'])}) | "
            f"{attribution['ALIGNMENT_AMBIGUITY']} "
            f"({metric(attribution['ALIGNMENT_AMBIGUITY_fraction'])}) | "
            f"{attribution['noncanonical_stitch_boundary_count']} / "
            f"{attribution['total_stitch_boundaries']} |"
        )
    lines.extend(
        [
            "",
            "All K rows reuse the same cached public-equivalent Top-1 views. Ground "
            "truth enters only alignment and evaluation after the cache is frozen.",
        ]
    )
    (output_dir / "stage3bc_longdoc_strong.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_stage3bc_strong(
    config: Mapping[str, Any],
    *,
    backend_factory: Callable[..., Vec2TextBackend] = Vec2TextBackend,
) -> dict[str, Any]:
    """Invert the frozen max-K views once, then evaluate every paired prefix."""

    import torch

    end_to_end_started = time.perf_counter()

    base = dict(config["base"])
    stage3bc = dict(config["stage3bc"])
    geometry = dict(base["geometry"])
    model_config = dict(base["models"])
    cohort_path = Path(str(config["resolved_cohort_artifact"]))
    cohort_sha256 = _sha256(cohort_path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_validation = _validate_cohort(
        cohort,
        expected_sha256=str(stage3bc["cohort_sha256"]),
        actual_sha256=cohort_sha256,
        expected_documents=int(stage3bc["expected_documents"]),
        content_tokens=int(geometry["content_tokens"]),
        overlap_tokens=int(geometry["overlap_tokens"]),
        stride=int(geometry["stride_tokens"]),
        model_positions=int(geometry["model_positions"]),
    )
    documents = list(cohort["documents"])
    view_keys = _view_keys(cohort)
    expected_views = int(stage3bc["expected_views"])
    if len(view_keys) != expected_views:
        raise RuntimeError("frozen max-K view count changed")

    # On unified-memory GPUs, loading both checkpoint shards before the first
    # CUDA context can leave too little headroom for Trainer's lazy context
    # initialization. Establish only the requested device context up front;
    # model dtype, weights, generation, and ranking remain unchanged.
    requested_device = torch.device(model_config.get("device", "cuda"))
    if requested_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(
            requested_device.index if requested_device.index is not None else 0
        )
    backend = backend_factory(
        max_length=int(geometry["model_positions"]),
        device=model_config.get("device", "cuda"),
        inversion_checkpoint=model_config["inversion_checkpoint"],
        corrector_checkpoint=model_config["corrector_checkpoint"],
        inversion_revision=model_config.get("inversion_revision"),
        corrector_revision=model_config.get("corrector_revision"),
    )
    windows = [window for document in documents for window in document["windows"]]
    texts = [str(window["text"]) for window in windows]
    expected_input_ids = [window["model_input_ids"] for window in windows]
    embedding_started = _cuda_telemetry_start(torch)
    embeddings = backend.embed_raw_windows(
        texts, expected_input_ids=expected_input_ids
    )
    embedding_telemetry = _cuda_telemetry_end(
        torch, embedding_started, expected_views
    )
    embedding_sha256 = hashlib.sha256(
        embeddings.contiguous().numpy().tobytes()
    ).hexdigest()
    embedding_telemetry.update(
        {
            "observations": expected_views,
            "embedding_dtype": str(embeddings.dtype),
            "embedding_shape": list(embeddings.shape),
            "embedding_sha256": embedding_sha256,
            "generated_once_for_all_k_prefixes": True,
        }
    )
    if embeddings.dtype != torch.float32:
        raise AssertionError("Stage 3B target embeddings are not frozen FP32")

    strong_setting = dict(stage3bc["strong_setting"])
    if (
        int(strong_setting["correction_steps"]),
        int(strong_setting["sequence_beam_width"]),
        int(strong_setting["top_n"]),
    ) != (50, 8, 1):
        raise ValueError("Stage 3B/C strong Top-1 setting changed")
    fingerprint = _checkpoint_fingerprint(
        cohort_sha256=cohort_sha256,
        view_keys=view_keys,
        model_config=model_config,
        strong_setting=strong_setting,
    )
    output_dir = Path(stage3bc["output_dir"])
    checkpoint_path = output_dir / str(stage3bc["checkpoint_filename"])
    candidates, records = _load_checkpoint(
        checkpoint_path,
        fingerprint=fingerprint,
        embedding_sha256=embedding_sha256,
        view_keys=view_keys,
        strong_setting=strong_setting,
    )
    resumed_views = len(candidates)
    run_started = time.perf_counter()
    progress_every = int(stage3bc.get("progress_every_views", 1))
    for view_index in range(resumed_views, expected_views):
        started = _cuda_telemetry_start(torch)
        pools, generated_records = backend.generate_final_sbeam_candidate_pools(
            embeddings[view_index : view_index + 1],
            correction_steps=50,
            sequence_beam_width=8,
            top_n=1,
        )
        telemetry = _cuda_telemetry_end(torch, started, 1)
        record = dict(generated_records[0])
        record.update(
            {
                "view_index": view_index,
                "view_key": view_keys[view_index],
                "document_index": view_index // 8,
                "window_index": view_index % 8,
                "passage_id": documents[view_index // 8]["passage_id"],
                "telemetry": telemetry,
            }
        )
        candidates.append(pools[0][0])
        records.append(record)
        _save_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            cohort_sha256=cohort_sha256,
            embedding_sha256=embedding_sha256,
            view_keys=view_keys,
            candidates=candidates,
            records=records,
        )
        completed = view_index + 1
        if completed % progress_every == 0 or completed == expected_views:
            elapsed = time.perf_counter() - run_started
            new_views = completed - resumed_views
            seconds_per_new_view = elapsed / new_views if new_views else 0.0
            eta_seconds = seconds_per_new_view * (expected_views - completed)
            print(
                f"stage3bc_progress={completed}/{expected_views} "
                f"last={view_keys[view_index]} eta_minutes={eta_seconds / 60:.1f}",
                flush=True,
            )

    if len(candidates) != expected_views or len(records) != expected_views:
        raise AssertionError("strong Top-1 cache is incomplete")
    if not all(bool(record["public_top1_exact_id_match"]) for record in records):
        raise AssertionError("instrumented strong Top-1 differs from public output")
    for record in records:
        if int(record["embed_call_count"]) != 51:
            raise AssertionError("strong search embedding query count changed")
        if int(record["search_hypothesis_rows_embedded"]) != 3145:
            raise AssertionError("strong search embedded-row budget changed")

    cache_path = output_dir / "stage3bc_strong_top1_cache.json"
    cache_payload = {
        "schema_version": 1,
        "cohort_sha256": cohort_sha256,
        "embedding_sha256": embedding_sha256,
        "fingerprint": fingerprint,
        "view_keys": view_keys,
        "top1_candidates": [candidate.to_dict() for candidate in candidates],
        "public_top1_exact_id_match": True,
        "strong_reproduction_setting": "correction50+sbeam8",
        "original_paper_beam_width_claimed": False,
    }
    _write_json_atomic(cache_path, cache_payload)
    cache_sha256 = _sha256(cache_path)

    rows_by_k: dict[int, list[dict[str, Any]]] = {k: [] for k in _K_VALUES}
    trajectories: list[dict[str, Any]] = []
    for document_index, document in enumerate(documents):
        offset = document_index * 8
        document_candidates = candidates[offset : offset + 8]
        document_records = records[offset : offset + 8]
        trajectory = {
            "passage_id": document["passage_id"],
            "source_row_index": document["source_row_index"],
            "selected_start": document["selected_start"],
            "k_prefixes": {},
        }
        for k in _K_VALUES:
            row = _evaluate_document_prefix(
                document,
                document_candidates,
                document_records,
                k=k,
                content_tokens=int(geometry["content_tokens"]),
                overlap_tokens=int(geometry["overlap_tokens"]),
                stride=int(geometry["stride_tokens"]),
                minimum_alignment=float(
                    base["diagnostic"]["stitch_minimum_alignment"]
                ),
            )
            rows_by_k[k].append(row)
            trajectory["k_prefixes"][str(k)] = row
        trajectories.append(trajectory)

    summaries = {k: _aggregate_prefix(rows) for k, rows in rows_by_k.items()}
    bootstrap = _paired_bootstrap(
        rows_by_k,
        iterations=int(stage3bc["bootstrap_iterations"]),
        seed=int(stage3bc["bootstrap_seed"]),
    )
    gate = classify_stage3bc_gate(
        summaries,
        bootstrap,
        dict(stage3bc["gate"]),
    )

    elapsed_values = [
        float(record["telemetry"]["elapsed_seconds"]) for record in records
    ]
    peak_allocated = [
        int(record["telemetry"]["peak_allocated_bytes"])
        for record in records
        if record["telemetry"].get("peak_allocated_bytes") is not None
    ]
    peak_reserved = [
        int(record["telemetry"]["peak_reserved_bytes"])
        for record in records
        if record["telemetry"].get("peak_reserved_bytes") is not None
    ]
    telemetry = {
        "views": expected_views,
        "resumed_views": resumed_views,
        "newly_generated_views": expected_views - resumed_views,
        "total_search_runtime_seconds": sum(elapsed_values),
        "runtime_seconds_per_view": mean(elapsed_values),
        "runtime_seconds_per_view_median": median(elapsed_values),
        "peak_allocated_bytes": max(peak_allocated) if peak_allocated else None,
        "peak_reserved_bytes": max(peak_reserved) if peak_reserved else None,
        "inversion_generation_calls_total": expected_views,
        "correction_generation_calls_total": 50 * expected_views,
        "search_embedding_queries_per_view": 51,
        "search_embedding_queries_total": 51 * expected_views,
        "search_hypothesis_rows_embedded_per_view": 3145,
        "search_hypothesis_rows_embedded_total": 3145 * expected_views,
        "all_public_instrumented_top1_match": True,
        "target_embedding": embedding_telemetry,
        "end_to_end_wall_seconds_before_final_serialization": (
            time.perf_counter() - end_to_end_started
        ),
    }
    trajectories_path = output_dir / "stage3bc_paired_trajectories.json"
    _write_json_atomic(
        trajectories_path,
        {
            "schema_version": 1,
            "cohort_sha256": cohort_sha256,
            "top1_cache_sha256": cache_sha256,
            "k_values": list(_K_VALUES),
            "trajectories": trajectories,
        },
    )
    payload = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE3BC_COMPLETE",
        "stage": "STRONG_INDEPENDENT_K_SCALING_AND_RECOVERABLE_REDUNDANCY",
        "config": config,
        "environment": _environment(),
        "estimand": (
            "The frozen 30-document subset of the pinned NQ scan frame for which "
            "a contiguous 136-content-ID K=8 chain is raw-substring round-trip stable"
        ),
        "selection_limitation": (
            "This is a tokenizer-stability-selected segment cohort and must not be "
            "generalized to all pinned NQ passages without external validation."
        ),
        "k_values": list(_K_VALUES),
        "cohort": {
            "path": str(cohort_path),
            "sha256": cohort_sha256,
            "validation": cohort_validation,
            "manifest": cohort["manifest"],
            "reselected": False,
        },
        "strong_top1_cache": {
            "path": str(cache_path),
            "sha256": cache_sha256,
            "candidate_count": len(candidates),
            "one_generation_per_view": True,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "fingerprint": fingerprint,
            "completed_views": len(candidates),
        },
        "telemetry": telemetry,
        "summaries": {str(k): summaries[k] for k in _K_VALUES},
        "paired_bootstrap_95": bootstrap,
        "paired_trajectories": {
            "path": str(trajectories_path),
            "sha256": _sha256(trajectories_path),
            "documents": len(trajectories),
        },
        "gate": gate,
        "joint_selector_implemented": False,
        "lambda_tuned": False,
        "candidate_reranking_implemented": False,
        "consensus_voting_implemented": False,
        "path_decoder_implemented": False,
        "scope_exclusions": [
            "candidate_level_joint_reranking",
            "lambda_tuning",
            "search_time_joint_vec2text",
            "consensus_or_majority_voting",
            "dynamic_programming_path_decoding",
            "llm_based_correction",
            "new_inversion_decoder",
            "pii",
            "vector_db_hnsw_ivf_pq_int8",
            "noise_dp_late_chunking",
        ],
    }
    _write_report(output_dir, payload)
    return payload
