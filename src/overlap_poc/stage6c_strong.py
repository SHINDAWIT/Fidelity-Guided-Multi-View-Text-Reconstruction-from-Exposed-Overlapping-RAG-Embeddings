"""Frozen correction50+sbeam8 generation for the Stage 6C final cohort."""

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
from .stage5b_strong import build_canonical_view_manifest


_IMPLEMENTATION_VERSION = "stage6c-final-strong-top1-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _validate_frozen_code(freeze: Mapping[str, Any]) -> None:
    for group, entries in freeze.get("code_hashes", {}).items():
        for name, entry in entries.items():
            path = Path(str(entry["path"]))
            if not path.is_file() or _sha256(path) != str(entry["sha256"]):
                raise RuntimeError(
                    f"Stage 6B frozen {group} code changed before inversion: {name}"
                )


def _validate_frozen_protocols(freeze: Mapping[str, Any]) -> None:
    for name, entry in freeze.get("protocol_files", {}).items():
        path = Path(str(entry["path"]))
        if not path.is_file() or _sha256(path) != str(entry["sha256"]):
            raise RuntimeError(f"Stage 6B frozen protocol changed before inversion: {name}")


def _validate_cohort_links(
    cohort_freeze: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
) -> None:
    if cohort_freeze.get("algorithm_freeze", {}).get("sha256") != str(
        inputs["algorithm_freeze"]["sha256"]
    ):
        raise RuntimeError("Stage 6C cohort points to another algorithm freeze")
    for name in ("overlap_cohort", "nooverlap_manifest"):
        if cohort_freeze.get("artifacts", {}).get(name, {}).get("sha256") != str(
            inputs[name]["sha256"]
        ):
            raise RuntimeError(f"Stage 6C cohort artifact linkage changed: {name}")


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def load_stage6c_strong_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 6C strong config must be a YAML mapping")

    def resolve(value: str) -> str:
        candidate = Path(value)
        return str(
            (candidate if candidate.is_absolute() else config_path.parent / candidate)
            .resolve()
        )

    resolved = dict(raw)
    base_entry = dict(raw["base_config"])
    resolved["base"] = load_diagnostic_config(resolve(str(base_entry["path"])))
    resolved["resolved_base_config"] = resolve(str(base_entry["path"]))
    resolved["resolved_config"] = str(config_path.resolve())
    resolved["resolved_inputs"] = {
        name: {**entry, "path": resolve(str(entry["path"]))}
        for name, entry in raw["inputs"].items()
    }
    return resolved


def _validate_inputs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if config.get("stage") != "stage6c_final_strong_top1_v1":
        raise RuntimeError("Stage 6C strong stage identity changed")
    if config.get("frozen_after_stage6b") is not True:
        raise RuntimeError("Stage 6C strong run was not frozen after Stage 6B")
    base_path = Path(str(config["resolved_base_config"]))
    if _sha256(base_path) != str(config["base_config"]["sha256"]):
        raise RuntimeError("Stage 6C strong base config changed")
    loaded: dict[str, dict[str, Any]] = {}
    for name, entry in config["resolved_inputs"].items():
        path = Path(str(entry["path"]))
        if "stage5d_assembly" in str(path):
            raise RuntimeError("Stage 6C cannot reuse Stage 5D inversion artifacts")
        if _sha256(path) != str(entry["sha256"]):
            raise RuntimeError(f"Stage 6C frozen input changed: {name}")
        loaded[name] = _load_json(path)
    freeze = loaded["algorithm_freeze"]
    cohort_freeze = loaded["cohort_freeze"]
    overlap = loaded["overlap_cohort"]
    nooverlap = loaded["nooverlap_manifest"]
    if (
        freeze.get("status") != "STAGE6B_ALGORITHM_FROZEN"
        or freeze.get("freeze_closed") is not True
    ):
        raise RuntimeError("Stage 6B algorithm freeze is not closed")
    _validate_frozen_code(freeze)
    _validate_frozen_protocols(freeze)
    if cohort_freeze.get("status") != "STAGE6C_FINAL_COHORT_FROZEN":
        raise RuntimeError("Stage 6C final cohort is not frozen")
    _validate_cohort_links(cohort_freeze, config["resolved_inputs"])
    if not all(bool(value) for value in cohort_freeze["invariants"].values()):
        raise RuntimeError("Stage 6C cohort invariant failed")
    if overlap.get("stage") != "STAGE6C_PLAN2_OVERLAP_FINAL_UNSEEN_COHORT":
        raise RuntimeError("Stage 6C overlap cohort identity changed")
    if nooverlap.get("stage") != "STAGE6C_PLAN2_NOOVERLAP_FINAL_UNSEEN_MANIFEST":
        raise RuntimeError("Stage 6C no-overlap manifest identity changed")
    if overlap["manifest"] != nooverlap["manifest"]:
        raise RuntimeError("Stage 6C policy manifests differ")
    if len(overlap["manifest"]) != 60:
        raise RuntimeError("Stage 6C final cohort must contain 60 documents")
    return overlap, nooverlap, cohort_freeze


def _fingerprint(
    *,
    freeze_sha256: str,
    cohort_freeze_sha256: str,
    canonical_manifest_sha256: str,
    embedding_sha256: str,
    model_config: Mapping[str, Any],
    strong_setting: Mapping[str, Any],
) -> str:
    value = {
        "implementation_version": _IMPLEMENTATION_VERSION,
        "freeze_sha256": freeze_sha256,
        "cohort_freeze_sha256": cohort_freeze_sha256,
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


def _load_checkpoint(
    path: Path, *, fingerprint: str, expected_keys: Sequence[str]
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    value = _load_json(path)
    if value.get("fingerprint") != fingerprint:
        raise RuntimeError("Stage 6C checkpoint belongs to another frozen run")
    candidates = [
        _candidate_from_dict(candidate) for candidate in value["top1_candidates"]
    ]
    records = list(value["records"])
    completed_keys = list(value["completed_canonical_keys"])
    if len(candidates) != len(records) or len(candidates) != len(completed_keys):
        raise RuntimeError("Stage 6C checkpoint lengths disagree")
    if completed_keys != list(expected_keys[: len(completed_keys)]):
        raise RuntimeError("Stage 6C checkpoint canonical order changed")
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


def run_stage6c_strong_generation(
    config: Mapping[str, Any],
    *,
    backend_factory: Callable[..., Vec2TextBackend] = Vec2TextBackend,
) -> dict[str, Any]:
    """Generate every unique final-cohort view once, with atomic resume."""

    import torch

    overlap, nooverlap, _cohort_freeze = _validate_inputs(config)
    model_config = dict(config["base"]["models"])
    strong = dict(config["strong_setting"])
    if (
        int(strong["correction_steps"]),
        int(strong["sequence_beam_width"]),
        int(strong["top_n"]),
    ) != (50, 8, 1):
        raise RuntimeError("Stage 6C must remain correction50+sbeam8 Top-1")
    canonical = build_canonical_view_manifest(overlap, nooverlap, model_config)
    expected = {
        "documents": int(config["expected_documents"]),
        "logical_overlap_views": int(config["expected_overlap_views"]),
        "logical_nooverlap_views": int(config["expected_nooverlap_views"]),
        "shared_cross_policy_views": int(config["expected_shared_views"]),
        "unique_views": int(config["expected_unique_views"]),
    }
    actual = {name: int(canonical[name]) for name in expected}
    if actual != expected or actual != {
        "documents": 60,
        "logical_overlap_views": 480,
        "logical_nooverlap_views": 300,
        "shared_cross_policy_views": 60,
        "unique_views": 720,
    }:
        raise RuntimeError(f"Stage 6C canonical counts changed: {actual}")

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / str(config["checkpoint_filename"])
    forbidden = {
        Path(value).resolve() for value in config.get("forbidden_reuse", [])
    }
    if checkpoint_path.resolve() in forbidden:
        raise RuntimeError("Stage 6C checkpoint path aliases a forbidden Stage 5D path")
    canonical_path = output_dir / str(config["canonical_manifest_filename"])
    _write_json_atomic(canonical_path, canonical)
    canonical_sha = _sha256(canonical_path)
    canonical_keys = [str(entry["canonical_key"]) for entry in canonical["entries"]]

    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    minimum_free = float(config["minimum_cuda_free_gib_before_load"])
    if free_bytes / 2**30 < minimum_free:
        raise RuntimeError(
            f"Stage 6C requires {minimum_free:.1f} GiB free CUDA; observed "
            f"{free_bytes / 2**30:.2f} GiB"
        )
    backend = backend_factory(
        max_length=32,
        device=model_config.get("device", "cuda"),
        inversion_checkpoint=model_config["inversion_checkpoint"],
        corrector_checkpoint=model_config["corrector_checkpoint"],
        inversion_revision=model_config.get("inversion_revision"),
        corrector_revision=model_config.get("corrector_revision"),
    )

    embedding_path = output_dir / str(config["embedding_cache_filename"])
    metadata_path = output_dir / str(config["embedding_metadata_filename"])
    embedding_started = _cuda_telemetry_start(torch)
    if embedding_path.exists() and metadata_path.exists():
        metadata = _load_json(metadata_path)
        if metadata.get("canonical_manifest_sha256") != canonical_sha:
            raise RuntimeError("Stage 6C embedding manifest changed")
        values = np.load(embedding_path, allow_pickle=False)
        if str(values.dtype) != "float32" or list(values.shape) != [720, 768]:
            raise RuntimeError("Stage 6C embedding cache shape/dtype changed")
        if _sha256(embedding_path) != metadata.get("file_sha256"):
            raise RuntimeError("Stage 6C embedding cache bytes changed")
        embeddings = torch.from_numpy(values.copy())
        embedding_reused = True
    else:
        if embedding_path.exists() != metadata_path.exists():
            raise RuntimeError("Stage 6C embedding cache is incomplete")
        batches = []
        entries = list(canonical["entries"])
        batch_size = int(config["embedding_batch_size"])
        for offset in range(0, len(entries), batch_size):
            batch = entries[offset : offset + batch_size]
            batches.append(
                backend.embed_raw_windows(
                    [str(entry["text"]) for entry in batch],
                    expected_input_ids=[entry["model_input_ids"] for entry in batch],
                )
            )
        embeddings = torch.cat(batches).to(dtype=torch.float32, device="cpu")
        if list(embeddings.shape) != [720, 768]:
            raise RuntimeError(f"unexpected Stage 6C embedding shape {embeddings.shape}")
        _write_npy_atomic(embedding_path, embeddings.numpy())
        metadata = {
            "schema_version": 1,
            "canonical_manifest_sha256": canonical_sha,
            "canonical_keys": canonical_keys,
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.numpy().dtype),
            "file_sha256": _sha256(embedding_path),
        }
        _write_json_atomic(metadata_path, metadata)
        embedding_reused = False
    embedding_telemetry = _cuda_telemetry_end(
        torch, embedding_started, len(embeddings)
    )
    embedding_sha = hashlib.sha256(
        embeddings.contiguous().numpy().tobytes()
    ).hexdigest()
    freeze_sha = str(config["resolved_inputs"]["algorithm_freeze"]["sha256"])
    cohort_freeze_sha = str(
        config["resolved_inputs"]["cohort_freeze"]["sha256"]
    )
    fingerprint = _fingerprint(
        freeze_sha256=freeze_sha,
        cohort_freeze_sha256=cohort_freeze_sha,
        canonical_manifest_sha256=canonical_sha,
        embedding_sha256=embedding_sha,
        model_config=model_config,
        strong_setting=strong,
    )
    candidates, records = _load_checkpoint(
        checkpoint_path, fingerprint=fingerprint, expected_keys=canonical_keys
    )
    resumed = len(candidates)
    started_run = time.perf_counter()
    progress_every = int(config["progress_every_views"])
    for view_index in range(resumed, len(canonical_keys)):
        started = _cuda_telemetry_start(torch)
        pools, generated = backend.generate_final_sbeam_candidate_pools(
            embeddings[view_index : view_index + 1],
            correction_steps=50,
            sequence_beam_width=8,
            top_n=1,
        )
        record = dict(generated[0])
        record.update(
            {
                "canonical_index": view_index,
                "canonical_key": canonical_keys[view_index],
                "aliases": list(canonical["entries"][view_index]["aliases"]),
                "telemetry": _cuda_telemetry_end(torch, started, 1),
            }
        )
        candidate = pools[0][0]
        if not bool(record["public_top1_exact_id_match"]):
            raise RuntimeError("Stage 6C public/instrumented Top-1 mismatch")
        if int(record["embed_call_count"]) != 51:
            raise RuntimeError("Stage 6C embedding-query budget changed")
        if int(record["search_hypothesis_rows_embedded"]) != 3145:
            raise RuntimeError("Stage 6C hypothesis-row budget changed")
        candidates.append(candidate)
        records.append(record)
        _save_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            manifest_sha256=canonical_sha,
            embedding_sha256=embedding_sha,
            canonical_keys=canonical_keys,
            candidates=candidates,
            records=records,
        )
        completed = view_index + 1
        if completed % progress_every == 0 or completed == len(canonical_keys):
            elapsed = time.perf_counter() - started_run
            new_views = completed - resumed
            seconds_per_view = elapsed / new_views if new_views else 0.0
            print(
                f"stage6c_progress={completed}/{len(canonical_keys)} "
                f"sec_per_view={seconds_per_view:.1f} "
                f"eta_minutes={seconds_per_view * (len(canonical_keys) - completed) / 60:.1f}",
                flush=True,
            )

    elapsed_values = [float(record["telemetry"]["elapsed_seconds"]) for record in records]
    peak_values = [
        int(record["telemetry"]["peak_allocated_bytes"])
        for record in records
        if record["telemetry"].get("peak_allocated_bytes") is not None
    ]
    payload = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE6C_STRONG_TOP1_CACHE_COMPLETE",
        "algorithm_freeze_sha256": freeze_sha,
        "cohort_freeze_sha256": cohort_freeze_sha,
        "config": {
            "path": str(config["resolved_config"]),
            "sha256": _sha256(Path(str(config["resolved_config"]))),
        },
        "canonical_manifest": {"path": str(canonical_path), "sha256": canonical_sha},
        "embedding_cache": {
            "path": str(embedding_path),
            "metadata_path": str(metadata_path),
            "tensor_sha256": embedding_sha,
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
            "resumed_views": resumed,
            "new_views": len(candidates) - resumed,
            "total_search_runtime_seconds": sum(elapsed_values),
            "runtime_seconds_per_view": mean(elapsed_values),
            "runtime_seconds_per_view_median": median(elapsed_values),
            "peak_allocated_bytes": max(peak_values) if peak_values else None,
            "correction_calls": 50 * len(candidates),
            "embedding_queries": 51 * len(candidates),
            "hypothesis_rows": 3145 * len(candidates),
        },
        "evaluation_executed": False,
    }
    _write_json_atomic(output_dir / str(config["cache_filename"]), payload)
    return payload
