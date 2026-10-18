"""End-to-end K=2 diagnostic gate; intentionally no Joint selector."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml

from .chunking import StableTokenPair, find_roundtrip_stable_pair
from .data import NaturalQuestionsLoader, PassageLoader
from .diagnostics import (
    classify_diagnostic_gate,
    evaluate_candidate_pair,
    summarize_diagnostics,
)
from .reconstruction import Vec2TextBackend


def load_diagnostic_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("diagnostic config must be a YAML mapping")
    return config


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _normalized_pair_key(pair: StableTokenPair) -> str:
    left, right = pair.windows
    return " ".join(left.text.casefold().split()) + "\0" + " ".join(
        right.text.casefold().split()
    )


def _deduplicate_pairs(
    pairs: Sequence[StableTokenPair],
) -> tuple[list[StableTokenPair], dict[str, Any]]:
    by_passage_id: dict[str, StableTokenPair] = {}
    duplicate_ids: list[str] = []
    for pair in sorted(pairs, key=lambda item: (item.source_row_index, item.passage_id)):
        if pair.passage_id in by_passage_id:
            duplicate_ids.append(pair.passage_id)
            continue
        by_passage_id[pair.passage_id] = pair

    by_text: dict[str, StableTokenPair] = {}
    duplicate_text_ids: list[str] = []
    for pair in sorted(
        by_passage_id.values(),
        key=lambda item: (item.passage_id, item.source_row_index),
    ):
        key = _normalized_pair_key(pair)
        incumbent = by_text.get(key)
        if incumbent is None:
            by_text[key] = pair
            continue
        survivor = min(
            (incumbent, pair),
            key=lambda item: (item.passage_id, item.source_row_index),
        )
        removed = pair if survivor is incumbent else incumbent
        by_text[key] = survivor
        duplicate_text_ids.append(removed.passage_id)

    deduplicated = sorted(
        by_text.values(), key=lambda item: (item.source_row_index, item.passage_id)
    )
    return deduplicated, {
        "raw_stable_pairs": len(pairs),
        "duplicate_passage_id_count": len(duplicate_ids),
        "duplicate_passage_ids": duplicate_ids,
        "duplicate_normalized_pair_count": len(duplicate_text_ids),
        "duplicate_normalized_pair_passage_ids": duplicate_text_ids,
        "deduplicated_stable_pairs": len(deduplicated),
    }


def _partition_pairs(
    pairs: Sequence[StableTokenPair],
    *,
    dataset_revision: str,
    partition_seed: int,
    sizes: Mapping[str, int],
) -> dict[str, list[StableTokenPair]]:
    required = sum(int(value) for value in sizes.values())
    if len(pairs) < required:
        raise ValueError(f"need {required} deduplicated pairs, found {len(pairs)}")

    def digest(pair: StableTokenPair) -> bytes:
        value = (
            f"{dataset_revision}\0{partition_seed}\0{pair.passage_id}"
        ).encode("utf-8")
        return hashlib.sha256(value).digest()

    ordered = sorted(pairs, key=lambda pair: (digest(pair), pair.passage_id))
    partitions: dict[str, list[StableTokenPair]] = {}
    offset = 0
    for name, raw_size in sizes.items():
        size = int(raw_size)
        partitions[name] = ordered[offset : offset + size]
        offset += size
    return partitions


def _scan_pairs(
    passages,
    tokenizer,
    *,
    content_tokens: int,
    overlap_tokens: int,
    model_positions: int,
) -> tuple[list[StableTokenPair], dict[str, Any]]:
    stable_pairs: list[StableTokenPair] = []
    length_eligible = 0
    fixed_start_stable = 0
    candidate_starts = 0
    stable_starts = 0
    rejected: dict[str, int] = {}
    selected_starts: list[int] = []
    for fallback_index, passage in enumerate(passages):
        source_row_index = int(passage.metadata.get("row_index", fallback_index))
        search = find_roundtrip_stable_pair(
            passage.text,
            tokenizer,
            passage_id=passage.passage_id,
            source_row_index=source_row_index,
            content_tokens=content_tokens,
            overlap_tokens=overlap_tokens,
            model_positions=model_positions,
        )
        candidate_starts += search.candidate_starts
        stable_starts += search.stable_starts
        fixed_start_stable += int(search.fixed_start_stable)
        if search.content_token_count >= content_tokens + (
            content_tokens - overlap_tokens
        ):
            length_eligible += 1
        if search.pair is None:
            reason = search.rejection_reason or "unknown"
            rejected[reason] = rejected.get(reason, 0) + 1
        else:
            stable_pairs.append(search.pair)
            selected_starts.append(search.pair.selected_start)

    acceptance = len(stable_pairs) / length_eligible if length_eligible else 0.0
    wilson = (
        _wilson_interval(len(stable_pairs), length_eligible)
        if length_eligible
        else (0.0, 0.0)
    )
    return stable_pairs, {
        "scanned_rows": len(passages),
        "length_eligible_rows": length_eligible,
        "raw_stable_pairs": len(stable_pairs),
        "passage_acceptance_yield": acceptance,
        "passage_acceptance_wilson95": list(wilson),
        "fixed_start_stable_pairs": fixed_start_stable,
        "fixed_start_stability_rate": (
            fixed_start_stable / length_eligible if length_eligible else 0.0
        ),
        "candidate_starts": candidate_starts,
        "stable_starts": stable_starts,
        "position_stability_rate": (
            stable_starts / candidate_starts if candidate_starts else 0.0
        ),
        "selected_start_mean": mean(selected_starts) if selected_starts else None,
        "selected_start_max": max(selected_starts) if selected_starts else None,
        "rejection_counts": rejected,
    }


def _cuda_telemetry_start(torch_module) -> tuple[float, int | None]:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        torch_module.cuda.reset_peak_memory_stats()
        return time.perf_counter(), torch_module.cuda.memory_allocated()
    return time.perf_counter(), None


def _cuda_telemetry_end(
    torch_module, started: tuple[float, int | None], pair_count: int
) -> dict[str, Any]:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    elapsed = time.perf_counter() - started[0]
    telemetry: dict[str, Any] = {
        "elapsed_seconds": elapsed,
        "seconds_per_pair": elapsed / pair_count if pair_count else None,
    }
    if torch_module.cuda.is_available():
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        telemetry.update(
            {
                "start_allocated_bytes": started[1],
                "peak_allocated_bytes": torch_module.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch_module.cuda.max_memory_reserved(),
                "device_free_bytes_after": free_bytes,
                "device_total_bytes": total_bytes,
            }
        )
    return telemetry


def _run_generation_stage(
    name: str,
    pairs: Sequence[StableTokenPair],
    backend: Vec2TextBackend,
    candidate_config: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    texts = [window.text for pair in pairs for window in pair.windows]
    expected = [
        window.model_input_ids for pair in pairs for window in pair.windows
    ]
    started = _cuda_telemetry_start(torch)
    embeddings = backend.embed_raw_windows(texts, expected_input_ids=expected)
    if embeddings.dtype != torch.float32:
        raise AssertionError("diagnostic target embeddings are not FP32")
    pools = backend.generate_frozen_candidate_pools(
        embeddings,
        candidate_config["settings"],
        top_n=int(candidate_config["top_n"]),
        batch_size=int(candidate_config.get("batch_size", 4)),
    )
    telemetry = _cuda_telemetry_end(torch, started, len(pairs))
    telemetry.update(
        {
            "stage": name,
            "pairs": len(pairs),
            "observations": len(texts),
            "embedding_dtype": str(embeddings.dtype),
            "embedding_shape": list(embeddings.shape),
            "embedding_sha256": hashlib.sha256(
                embeddings.contiguous().numpy().tobytes()
            ).hexdigest(),
            "candidate_pool_hash": backend.candidate_pool_hash(pools),
            "candidate_counts": [len(pool) for pool in pools],
        }
    )
    return embeddings, pools, telemetry


def _environment() -> dict[str, Any]:
    import datasets
    import torch
    import transformers

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report["cuda_device"] = torch.cuda.get_device_name(0)
        report["cuda_runtime"] = torch.version.cuda
    return report


def _write_artifacts(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic_gate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    scan = payload["dataset_scan"]
    lines = [
        "# K=2 Vec2Text diagnostic gate",
        "",
        f"- Run status: `{payload['status']}`",
        f"- Dataset acceptance: {scan['passage_acceptance_yield']:.4f}",
        f"- Deduplicated stable pairs: {scan['deduplication']['deduplicated_stable_pairs']}",
    ]
    if payload.get("gate"):
        lines.append(f"- Gate: `{payload['gate']['state']}`")
        lines.append(f"- Gate reason: {payload['gate']['reason']}")
    if payload.get("summary"):
        summary = payload["summary"]
        lines.extend(
            [
                "",
                "| Metric | Mean | Median |",
                "| --- | ---: | ---: |",
            ]
        )
        for name in ("p_ind", "p_oracle", "h_pair", "c_ind", "c_oracle", "delta_c"):
            lines.append(
                f"| {name} | {summary[name]['mean']:.4f} | {summary[name]['median']:.4f} |"
            )
        lines.extend(
            [
                "",
                f"- H_pair > 0 fraction: {summary['h_pair_positive_fraction']:.4f}",
                f"- Delta C > 0 fraction: {summary['delta_c_positive_fraction']:.4f}",
                "- Delta C > 0 among positive-headroom pairs: "
                f"{summary['delta_c_positive_fraction_when_headroom']:.4f}",
                f"- H/Delta-C Pearson: {summary['h_pair_delta_c_pearson']}",
                f"- H/Delta-C Spearman: {summary['h_pair_delta_c_spearman']}",
                "- Independent shared recovery mean: "
                f"{summary['independent_shared_recovery_mean']['mean']:.4f}",
                "- Per-view oracle shared recovery mean: "
                f"{summary['oracle_shared_recovery_mean']['mean']:.4f}",
                f"- Error complementarity mean: {summary['error_complementarity']['mean']:.4f}",
                f"- Pooled error phi: {summary['pooled_error_phi']}",
            ]
        )
    if payload.get("candidate_generation"):
        lines.extend(
            [
                "",
                "## Candidate generation",
                "",
                f"- Pool: {payload['candidate_generation']['pool_semantics']}",
                f"- Top-N cap: {payload['candidate_generation']['top_n_cap']}",
            ]
        )
        for setting in payload["candidate_generation"]["settings"]:
            lines.append(
                "- "
                f"{setting['id']}: correction_steps={setting['correction_steps']}, "
                f"sequence_beam_width={setting['sequence_beam_width']}"
            )
    (output_dir / "diagnostic_gate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_diagnostic_gate(
    config: Mapping[str, Any],
    *,
    loader: PassageLoader | None = None,
    tokenizer: Any | None = None,
    backend: Vec2TextBackend | None = None,
    scan_only: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Run scan, preflight, 30-pair diagnostics, and the pre-Joint gate."""

    from transformers import AutoTokenizer

    dataset_config = dict(config["dataset"])
    geometry = dict(config["geometry"])
    model_config = dict(config["models"])
    candidate_config = dict(config["candidate_generation"])
    diagnostic_config = dict(config["diagnostic"])
    partition_config = dict(config["partitions"])
    output_dir = Path(config.get("output_dir", "results/diagnostic_gate"))

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        model_config["embedder_name"],
        revision=model_config.get("embedder_revision"),
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("round-trip raw-substring mode requires a fast tokenizer")
    loader = loader or NaturalQuestionsLoader(
        dataset_name=dataset_config["name"],
        dataset_config=dataset_config.get("config"),
        split=dataset_config["split"],
        revision=dataset_config["revision"],
        max_words=None,
        streaming=bool(dataset_config.get("streaming", True)),
    )
    passages = loader.load(int(dataset_config["scan_limit"]))
    stable_pairs, scan = _scan_pairs(
        passages,
        tokenizer,
        content_tokens=int(geometry["content_tokens"]),
        overlap_tokens=int(geometry["overlap_tokens"]),
        model_positions=int(geometry["model_positions"]),
    )
    deduplicated, deduplication = _deduplicate_pairs(stable_pairs)
    scan["deduplication"] = deduplication
    required = sum(int(value) for value in partition_config["sizes"].values())
    acceptance_floor = float(dataset_config["acceptance_yield_floor"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "SCANNED",
        "config": config,
        "environment": _environment(),
        "dataset_scan": scan,
        "scope_exclusions": [
            "Vector DB",
            "ANN/HNSW/IVF",
            "PQ/INT8/quantization",
            "noise/DP",
            "Late Chunking",
        ],
    }
    if (
        scan["passage_acceptance_yield"] < acceptance_floor
        or len(deduplicated) < required
    ):
        payload.update(
            {
                "status": "FALLBACK_REQUIRED",
                "gate": {
                    "state": "POOL_LIMITED_STOP",
                    "reason": "Raw-substring eligibility failed before model calls.",
                },
            }
        )
        _write_artifacts(output_dir, payload)
        return payload

    partitions = _partition_pairs(
        deduplicated,
        dataset_revision=dataset_config["revision"],
        partition_seed=int(partition_config["seed"]),
        sizes=partition_config["sizes"],
    )
    payload["partition_manifest"] = {
        name: [pair.passage_id for pair in pairs]
        for name, pairs in partitions.items()
    }
    if scan_only:
        payload["status"] = "SCAN_COMPLETE"
        _write_artifacts(output_dir, payload)
        return payload

    backend = backend or Vec2TextBackend(
        max_length=int(geometry["model_positions"]),
        device=model_config.get("device", "cuda"),
        inversion_checkpoint=model_config["inversion_checkpoint"],
        corrector_checkpoint=model_config["corrector_checkpoint"],
        inversion_revision=model_config.get("inversion_revision"),
        corrector_revision=model_config.get("corrector_revision"),
    )
    if (
        backend.embedder_tokenizer.eos_token_id != tokenizer.eos_token_id
        or backend.embedder_tokenizer.pad_token_id != tokenizer.pad_token_id
        or backend.embedder_tokenizer.vocab_size != tokenizer.vocab_size
    ):
        raise RuntimeError("scan tokenizer and loaded Vec2Text embedder tokenizer differ")

    _, preflight_pools, preflight_telemetry = _run_generation_stage(
        "reserved_preflight",
        partitions["preflight"],
        backend,
        candidate_config,
    )
    payload["preflight"] = {
        "telemetry": preflight_telemetry,
        "pool_sizes": [len(pool) for pool in preflight_pools],
        "predictions_used_for_parameters": False,
    }
    if preflight_only:
        payload["status"] = "PREFLIGHT_COMPLETE"
        _write_artifacts(output_dir, payload)
        return payload

    _, diagnostic_pools, diagnostic_telemetry = _run_generation_stage(
        "diagnostic",
        partitions["diagnostic"],
        backend,
        candidate_config,
    )
    rows: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(partitions["diagnostic"]):
        pools = diagnostic_pools[pair_index * 2 : pair_index * 2 + 2]
        row = evaluate_candidate_pair(
            pools,
            reference_ids=pair.reference_ids,
            shared_ids=pair.shared_ids,
            overlap_tokens=int(geometry["overlap_tokens"]),
            minimum_alignment=float(diagnostic_config["stitch_minimum_alignment"]),
        )
        row.update(
            {
                "pair_id": pair.passage_id,
                "source_row_index": pair.source_row_index,
                "selected_start": pair.selected_start,
                "reference_ids": list(pair.reference_ids),
                "shared_ids": list(pair.shared_ids),
                "candidate_pools": [
                    [candidate.to_dict() for candidate in pool] for pool in pools
                ],
            }
        )
        rows.append(row)

    try:
        summary = summarize_diagnostics(
            rows,
            bootstrap_seed=int(diagnostic_config["bootstrap_seed"]),
            bootstrap_iterations=int(diagnostic_config["bootstrap_iterations"]),
        )
        gate = classify_diagnostic_gate(
            summary,
            oracle_floor=float(diagnostic_config["oracle_floor"]),
            independent_saturation=float(
                diagnostic_config["independent_saturation"]
            ),
            minimum_headroom=float(diagnostic_config["minimum_headroom"]),
            minimum_signal_fraction=float(
                diagnostic_config["minimum_signal_fraction"]
            ),
        )
        status = "DIAGNOSTIC_COMPLETE"
    except (ValueError, FloatingPointError) as exc:
        summary = None
        gate = {"state": "POOL_LIMITED_STOP", "reason": f"INVALID_STOP: {exc}"}
        status = "INVALID_STOP"

    payload.update(
        {
            "status": status,
            "candidate_generation": {
                "pool_semantics": (
                    "Top-N unique final endpoints from the configured Vec2Text "
                    "profiles, reranked by frozen target-embedding cosine fidelity; "
                    "not the internal sequence beam"
                ),
                "settings": candidate_config["settings"],
                "top_n_cap": int(candidate_config["top_n"]),
            },
            "diagnostic_telemetry": diagnostic_telemetry,
            "diagnostic_rows": rows,
            "summary": summary,
            "gate": gate,
            "joint_selector_implemented": False,
            "lambda_tuned": False,
        }
    )
    _write_artifacts(output_dir, payload)
    return payload
