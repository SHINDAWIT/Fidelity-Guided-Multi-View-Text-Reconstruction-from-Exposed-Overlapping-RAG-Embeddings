"""Frozen Stage 5B strong Top-1 generation with cross-policy view deduplication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from statistics import mean, median
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from .diagnostic_experiment import (
    _cuda_telemetry_end,
    _cuda_telemetry_start,
    load_diagnostic_config,
)
from .reconstruction import Candidate, Vec2TextBackend
from .stage1_strong_saturation import _candidate_from_dict, _write_json_atomic
from .stage5b_protocol import validate_stage5b_protocol


_IMPLEMENTATION_VERSION = "stage5b-strong-top1-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def load_stage5b_strong_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 5B strong config must be a YAML mapping")

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return (
            candidate if candidate.is_absolute() else config_path.parent / candidate
        ).resolve()

    base_path = resolve(str(raw["base_config"]))
    protocol_path = resolve(str(raw["evaluation_protocol"]))
    snapshot_path = resolve(str(raw["evaluation_snapshot"]))
    return {
        "base": load_diagnostic_config(base_path),
        "stage5b": raw,
        "resolved_config": str(config_path.resolve()),
        "resolved_base_config": str(base_path),
        "resolved_evaluation_protocol": str(protocol_path),
        "resolved_evaluation_snapshot": str(snapshot_path),
    }


def _canonical_payload(
    *,
    text: str,
    content_ids: Sequence[int],
    model_input_ids: Sequence[int],
    attention_mask: Sequence[int],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "text": text,
        "content_ids": [int(value) for value in content_ids],
        "model_input_ids": [int(value) for value in model_input_ids],
        "attention_mask": [int(value) for value in attention_mask],
        "embedder_name": str(model_config["embedder_name"]),
        "embedder_revision": str(model_config.get("embedder_revision")),
    }


def _canonical_key(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_canonical_view_manifest(
    overlap_cohort: Mapping[str, Any],
    nooverlap_manifest: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Deduplicate only byte-equivalent model observations across policies."""

    overlap_documents = list(overlap_cohort["documents"])
    nooverlap_documents = list(nooverlap_manifest["documents"])
    if len(overlap_documents) != len(nooverlap_documents):
        raise RuntimeError("Stage 5B policy cohorts have different sizes")
    entries: list[dict[str, Any]] = []
    index_by_key: dict[str, int] = {}
    alias_to_index: dict[str, int] = {}

    def add(window: Mapping[str, Any], alias: str) -> None:
        payload = _canonical_payload(
            text=str(window["text"]),
            content_ids=window["content_ids"],
            model_input_ids=window["model_input_ids"],
            attention_mask=window["attention_mask"],
            model_config=model_config,
        )
        key = _canonical_key(payload)
        index = index_by_key.get(key)
        if index is None:
            index = len(entries)
            index_by_key[key] = index
            entries.append(
                {
                    "canonical_index": index,
                    "canonical_key": key,
                    **payload,
                    "aliases": [alias],
                }
            )
        else:
            incumbent = entries[index]
            for field, value in payload.items():
                if incumbent[field] != value:
                    raise RuntimeError("canonical view hash collision")
            incumbent["aliases"].append(alias)
        if alias in alias_to_index:
            raise RuntimeError(f"duplicate logical Stage 5B alias: {alias}")
        alias_to_index[alias] = index

    for document in overlap_documents:
        passage_id = str(document["passage_id"])
        for window in document["windows"]:
            add(window, f"{passage_id}:overlap:{window['window_index']}")
    for document in nooverlap_documents:
        passage_id = str(document["passage_id"])
        for window in document["windows"]:
            add(window, f"{passage_id}:nooverlap:{window['window_index']}")

    shared = [entry for entry in entries if len(entry["aliases"]) > 1]
    if any(len(entry["aliases"]) != 2 for entry in shared):
        raise RuntimeError("unexpected Stage 5B cross-policy dedup multiplicity")
    expected_shared_aliases = {
        tuple(
            sorted(
                (
                    f"{document['passage_id']}:overlap:0",
                    f"{document['passage_id']}:nooverlap:0",
                )
            )
        )
        for document in overlap_documents
    }
    actual_shared_aliases = {
        tuple(sorted(str(alias) for alias in entry["aliases"])) for entry in shared
    }
    if actual_shared_aliases != expected_shared_aliases:
        raise RuntimeError("Stage 5B dedup is not exactly the 60 shared W0 views")
    return {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "documents": len(overlap_documents),
        "logical_overlap_views": sum(
            len(document["windows"]) for document in overlap_documents
        ),
        "logical_nooverlap_views": sum(
            len(document["windows"]) for document in nooverlap_documents
        ),
        "shared_cross_policy_views": len(shared),
        "unique_views": len(entries),
        "entries": entries,
        "alias_to_canonical_index": alias_to_index,
        "dedup_definition": (
            "Exact equality of raw text, content IDs, padded model input IDs, "
            "attention mask, embedder name, and embedder revision."
        ),
    }


def _fingerprint(
    *,
    protocol_sha256: str,
    canonical_manifest_sha256: str,
    embedding_sha256: str,
    model_config: Mapping[str, Any],
    strong_setting: Mapping[str, Any],
) -> str:
    value = {
        "implementation_version": _IMPLEMENTATION_VERSION,
        "protocol_sha256": protocol_sha256,
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "embedding_sha256": embedding_sha256,
        "model_config": dict(model_config),
        "strong_setting": dict(strong_setting),
    }
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path, *, fingerprint: str, expected_keys: Sequence[str]
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    value = _load_object(path)
    if value.get("fingerprint") != fingerprint:
        raise RuntimeError("Stage 5B checkpoint belongs to another frozen run")
    candidates = [
        _candidate_from_dict(candidate) for candidate in value["top1_candidates"]
    ]
    records = list(value["records"])
    completed_keys = list(value["completed_canonical_keys"])
    if len(candidates) != len(records) or len(candidates) != len(completed_keys):
        raise RuntimeError("Stage 5B checkpoint lengths disagree")
    if completed_keys != list(expected_keys[: len(completed_keys)]):
        raise RuntimeError("Stage 5B checkpoint canonical order changed")
    return candidates, records


def _save_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    manifest_sha256: str,
    embedding_sha256: str,
    canonical_keys: Sequence[str],
    candidates: Sequence[Candidate],
    records: Sequence[Mapping[str, Any]],
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "implementation_version": _IMPLEMENTATION_VERSION,
            "fingerprint": fingerprint,
            "canonical_manifest_sha256": manifest_sha256,
            "embedding_sha256": embedding_sha256,
            "expected_views": len(canonical_keys),
            "completed_views": len(candidates),
            "completed_canonical_keys": list(canonical_keys[: len(candidates)]),
            "top1_candidates": [candidate.to_dict() for candidate in candidates],
            "records": list(records),
        },
    )


def run_stage5b_strong_generation(
    config: Mapping[str, Any],
    *,
    backend_factory: Callable[..., Vec2TextBackend] = Vec2TextBackend,
) -> dict[str, Any]:
    import torch

    stage5b = dict(config["stage5b"])
    protocol_path = Path(str(config["resolved_evaluation_protocol"]))
    snapshot_path = Path(str(config["resolved_evaluation_snapshot"]))
    if _sha256(protocol_path) != str(stage5b["evaluation_protocol_sha256"]):
        raise RuntimeError("Stage 5B evaluation protocol changed")
    if _sha256(snapshot_path) != str(stage5b["evaluation_snapshot_sha256"]):
        raise RuntimeError("Stage 5B evaluation snapshot changed")
    protocol = validate_stage5b_protocol(protocol_path)
    if protocol["protocol_sha256"] != str(stage5b["evaluation_protocol_sha256"]):
        raise RuntimeError("validated Stage 5B protocol hash changed")
    artifact_paths = {
        name: Path(value["path"])
        for name, value in protocol["validated_artifacts"].items()
    }
    overlap_cohort = _load_object(artifact_paths["overlap_cohort"])
    nooverlap_manifest = _load_object(artifact_paths["nooverlap_manifest"])
    model_config = dict(config["base"]["models"])
    strong_setting = dict(stage5b["strong_setting"])
    if strong_setting != dict(protocol["strong_reproduction_setting"]):
        raise RuntimeError("Stage 5B strong setting differs from frozen protocol")
    if (
        int(strong_setting["correction_steps"]),
        int(strong_setting["sequence_beam_width"]),
        int(strong_setting["top_n"]),
    ) != (50, 8, 1):
        raise RuntimeError("Stage 5B strong setting must remain correction50+sbeam8")

    canonical = build_canonical_view_manifest(
        overlap_cohort, nooverlap_manifest, model_config
    )
    expected_counts = {
        "documents": int(stage5b["expected_documents"]),
        "logical_overlap_views": int(stage5b["expected_overlap_views"]),
        "logical_nooverlap_views": int(stage5b["expected_nooverlap_views"]),
        "shared_cross_policy_views": int(stage5b["expected_shared_views"]),
        "unique_views": int(stage5b["expected_unique_views"]),
    }
    actual_counts = {key: int(canonical[key]) for key in expected_counts}
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Stage 5B canonical view counts changed: {actual_counts}"
        )
    output_dir = Path(str(stage5b["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / str(stage5b["canonical_manifest_filename"])
    _write_json_atomic(canonical_path, canonical)
    canonical_sha256 = _sha256(canonical_path)
    canonical_keys = [str(entry["canonical_key"]) for entry in canonical["entries"]]

    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    minimum_free = float(stage5b["minimum_cuda_free_gib_before_load"])
    if free_bytes / 2**30 < minimum_free:
        raise RuntimeError(
            f"Stage 5B requires at least {minimum_free:.1f} GiB free CUDA memory "
            f"before model load; observed {free_bytes / 2**30:.2f} GiB"
        )
    requested_device = torch.device(model_config.get("device", "cuda"))
    if requested_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(
            requested_device.index if requested_device.index is not None else 0
        )
    backend = backend_factory(
        max_length=32,
        device=model_config.get("device", "cuda"),
        inversion_checkpoint=model_config["inversion_checkpoint"],
        corrector_checkpoint=model_config["corrector_checkpoint"],
        inversion_revision=model_config.get("inversion_revision"),
        corrector_revision=model_config.get("corrector_revision"),
    )

    embedding_path = output_dir / str(stage5b["embedding_cache_filename"])
    embedding_metadata_path = output_dir / str(
        stage5b["embedding_metadata_filename"]
    )
    embedding_started = _cuda_telemetry_start(torch)
    if embedding_path.exists() and embedding_metadata_path.exists():
        embedding_metadata = _load_object(embedding_metadata_path)
        if embedding_metadata["canonical_manifest_sha256"] != canonical_sha256:
            raise RuntimeError("Stage 5B embedding cache manifest changed")
        values = np.load(embedding_path, allow_pickle=False)
        if str(values.dtype) != "float32" or list(values.shape) != [720, 768]:
            raise RuntimeError("Stage 5B embedding cache shape/dtype changed")
        if _sha256(embedding_path) != embedding_metadata["file_sha256"]:
            raise RuntimeError("Stage 5B embedding cache bytes changed")
        embeddings = torch.from_numpy(values.copy())
        embedding_reused = True
    else:
        if embedding_path.exists() != embedding_metadata_path.exists():
            raise RuntimeError("Stage 5B embedding cache is incomplete")
        batches = []
        batch_size = int(stage5b["embedding_batch_size"])
        entries = list(canonical["entries"])
        for offset in range(0, len(entries), batch_size):
            batch = entries[offset : offset + batch_size]
            batches.append(
                backend.embed_raw_windows(
                    [str(entry["text"]) for entry in batch],
                    expected_input_ids=[entry["model_input_ids"] for entry in batch],
                )
            )
        embeddings = torch.cat(batches, dim=0).to(dtype=torch.float32, device="cpu")
        if list(embeddings.shape) != [720, 768]:
            raise RuntimeError(f"unexpected Stage 5B embedding shape {embeddings.shape}")
        _write_npy_atomic(embedding_path, embeddings.numpy())
        embedding_metadata = {
            "schema_version": 1,
            "canonical_manifest_sha256": canonical_sha256,
            "canonical_keys": canonical_keys,
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.numpy().dtype),
            "file_sha256": _sha256(embedding_path),
        }
        _write_json_atomic(embedding_metadata_path, embedding_metadata)
        embedding_reused = False
    embedding_telemetry = _cuda_telemetry_end(
        torch, embedding_started, len(embeddings)
    )
    embedding_sha256 = hashlib.sha256(
        embeddings.contiguous().numpy().tobytes()
    ).hexdigest()
    fingerprint = _fingerprint(
        protocol_sha256=protocol["protocol_sha256"],
        canonical_manifest_sha256=canonical_sha256,
        embedding_sha256=embedding_sha256,
        model_config=model_config,
        strong_setting=strong_setting,
    )
    checkpoint_path = output_dir / str(stage5b["checkpoint_filename"])
    candidates, records = _load_checkpoint(
        checkpoint_path, fingerprint=fingerprint, expected_keys=canonical_keys
    )
    resumed_views = len(candidates)
    run_started = time.perf_counter()
    progress_every = int(stage5b["progress_every_views"])
    for view_index in range(resumed_views, len(canonical_keys)):
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
                "canonical_index": view_index,
                "canonical_key": canonical_keys[view_index],
                "aliases": list(canonical["entries"][view_index]["aliases"]),
                "telemetry": telemetry,
            }
        )
        candidate = pools[0][0]
        if not bool(record["public_top1_exact_id_match"]):
            raise RuntimeError("Stage 5B public/instrumented Top-1 mismatch")
        if int(record["embed_call_count"]) != 51:
            raise RuntimeError("Stage 5B embedding-query budget changed")
        if int(record["search_hypothesis_rows_embedded"]) != 3145:
            raise RuntimeError("Stage 5B hypothesis-row budget changed")
        candidates.append(candidate)
        records.append(record)
        _save_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            manifest_sha256=canonical_sha256,
            embedding_sha256=embedding_sha256,
            canonical_keys=canonical_keys,
            candidates=candidates,
            records=records,
        )
        completed = view_index + 1
        if completed % progress_every == 0 or completed == len(canonical_keys):
            elapsed = time.perf_counter() - run_started
            new_views = completed - resumed_views
            seconds_per_view = elapsed / new_views if new_views else 0.0
            eta_seconds = seconds_per_view * (len(canonical_keys) - completed)
            print(
                f"stage5b_progress={completed}/{len(canonical_keys)} "
                f"last={canonical_keys[view_index][:12]} "
                f"sec_per_view={seconds_per_view:.1f} "
                f"eta_minutes={eta_seconds / 60:.1f}",
                flush=True,
            )

    elapsed_values = [
        float(record["telemetry"]["elapsed_seconds"]) for record in records
    ]
    peak_allocated = [
        int(record["telemetry"]["peak_allocated_bytes"])
        for record in records
        if record["telemetry"].get("peak_allocated_bytes") is not None
    ]
    cache_path = output_dir / str(stage5b["cache_filename"])
    payload = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE5B_STRONG_TOP1_CACHE_COMPLETE",
        "protocol_sha256": protocol["protocol_sha256"],
        "canonical_manifest": {
            "path": str(canonical_path),
            "sha256": canonical_sha256,
        },
        "embedding_cache": {
            "path": str(embedding_path),
            "metadata_path": str(embedding_metadata_path),
            "tensor_sha256": embedding_sha256,
            "file_sha256": _sha256(embedding_path),
            "reused": embedding_reused,
            "telemetry": embedding_telemetry,
        },
        "fingerprint": fingerprint,
        "canonical_keys": canonical_keys,
        "alias_to_canonical_index": canonical["alias_to_canonical_index"],
        "top1_candidates": [candidate.to_dict() for candidate in candidates],
        "records": records,
        "strong_reproduction_setting": "correction50+sbeam8",
        "telemetry": {
            "unique_views": len(candidates),
            "resumed_views": resumed_views,
            "new_views": len(candidates) - resumed_views,
            "total_search_runtime_seconds": sum(elapsed_values),
            "runtime_seconds_per_view": mean(elapsed_values),
            "runtime_seconds_per_view_median": median(elapsed_values),
            "peak_allocated_bytes": max(peak_allocated) if peak_allocated else None,
            "correction_calls": 50 * len(candidates),
            "embedding_queries": 51 * len(candidates),
            "hypothesis_rows": 3145 * len(candidates),
        },
        "evaluation_executed": False,
        "method_implemented": False,
    }
    _write_json_atomic(cache_path, payload)
    return payload
