"""Stage 1: diagnostic-30 K=2 strong Independent saturation check.

No Joint selector, consistency weight, compute sweep, or K-scaling code belongs
in this module. Ground truth is used only after the strong candidate pools have
been frozen, through the existing oracle diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import yaml

from .data import NaturalQuestionsLoader, PassageLoader
from .diagnostic_experiment import (
    _cuda_telemetry_end,
    _cuda_telemetry_start,
    _deduplicate_pairs,
    _partition_pairs,
    _scan_pairs,
    load_diagnostic_config,
)
from .reconstruction import Candidate, Vec2TextBackend
from .strong_pool_preflight import _environment, _pool_profile


def load_stage1_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        stage1 = yaml.safe_load(handle)
    if not isinstance(stage1, dict):
        raise ValueError("Stage 1 config must be a YAML mapping")
    raw_base = Path(str(stage1["base_config"]))
    base_path = raw_base if raw_base.is_absolute() else config_path.parent / raw_base
    return {
        "base": load_diagnostic_config(base_path),
        "stage1": stage1,
        "resolved_base_config": str(base_path.resolve()),
        "resolved_stage1_config": str(config_path.resolve()),
    }


def classify_stage1_saturation(
    summary: Mapping[str, Any],
    *,
    independent_lower95_threshold: float = 0.90,
    headroom_upper95_threshold: float = 2 / 46,
) -> dict[str, Any]:
    """Decide whether strong K=2 post-hoc reranking has usable headroom."""

    intervals = summary["bootstrap_95"]
    p_ind_lower, p_ind_upper = intervals["p_ind"]
    h_lower, h_upper = intervals["h_pair"]
    absolute_saturation_established = (
        float(p_ind_lower) >= independent_lower95_threshold
    )
    posthoc_headroom_absent = float(h_upper) < headroom_upper95_threshold
    if posthoc_headroom_absent:
        state = "K2_STRONG_POSTHOC_STOP"
        if absolute_saturation_established:
            interpretation = (
                "Strong Independent quality is robustly high and the frozen "
                "final-beam pool has less than two-token upper-bound mean headroom."
            )
        else:
            interpretation = (
                "Absolute strong-quality saturation is not established, but the "
                "frozen final-beam alternatives still have less than two-token "
                "upper-bound mean headroom, so post-hoc reranking cannot address "
                "the remaining hard cases."
            )
        next_step = "PIVOT_TO_STAGE2_COMPUTE_BUDGET_DIAGNOSTIC"
    else:
        state = "K2_STRONG_AMBIGUITY_REMAINS"
        interpretation = (
            "Diagnostic 30 does not establish both strong Independent saturation "
            "and negligible final-beam oracle headroom."
        )
        next_step = "REVIEW_STAGE1_BEFORE_ANY_NEXT_IMPLEMENTATION"
    return {
        "state": state,
        "interpretation": interpretation,
        "next_step": next_step,
        "thresholds": {
            "independent_lower95_threshold": independent_lower95_threshold,
            "headroom_upper95_threshold": headroom_upper95_threshold,
        },
        "observed": {
            "p_ind_lower95": float(p_ind_lower),
            "p_ind_upper95": float(p_ind_upper),
            "h_pair_lower95": float(h_lower),
            "h_pair_upper95": float(h_upper),
            "absolute_saturation_established": absolute_saturation_established,
            "posthoc_headroom_absent": posthoc_headroom_absent,
        },
    }


def _candidate_from_dict(raw: Mapping[str, Any]) -> Candidate:
    values = dict(raw)
    values["content_ids"] = tuple(int(value) for value in values["content_ids"])
    return Candidate(**values)


def _checkpoint_fingerprint(
    base: Mapping[str, Any], stage1: Mapping[str, Any], manifest: list[str]
) -> str:
    frozen = {
        "dataset": base["dataset"],
        "geometry": base["geometry"],
        "models": base["models"],
        "partition_seed": base["partitions"]["seed"],
        "partition_name": stage1["partition_name"],
        "expected_pairs": stage1["expected_pairs"],
        "candidate_generation": stage1["candidate_generation"],
        "manifest": manifest,
    }
    encoded = json.dumps(
        frozen, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    embedding_sha256: str,
    expected_views: int,
) -> tuple[list[tuple[Candidate, ...]], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("fingerprint") != fingerprint:
        raise RuntimeError("Stage 1 checkpoint belongs to a different frozen config")
    if checkpoint.get("embedding_sha256") != embedding_sha256:
        raise RuntimeError("Stage 1 checkpoint target embeddings do not match")
    pools = [
        tuple(_candidate_from_dict(candidate) for candidate in pool)
        for pool in checkpoint.get("candidate_pools", [])
    ]
    records = list(checkpoint.get("instrumentation_records", []))
    if len(pools) != len(records) or len(pools) > expected_views:
        raise RuntimeError("Stage 1 checkpoint has inconsistent progress")
    return pools, records


def _save_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    embedding_sha256: str,
    manifest: list[str],
    pools: list[tuple[Candidate, ...]],
    records: list[dict[str, Any]],
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "embedding_sha256": embedding_sha256,
            "pair_manifest": manifest,
            "completed_views": len(pools),
            "candidate_pools": [
                [candidate.to_dict() for candidate in pool] for pool in pools
            ],
            "instrumentation_records": records,
        },
    )


def _write_stage1_artifacts(output_dir: Path, payload: Mapping[str, Any]) -> None:
    _write_json_atomic(output_dir / "stage1_strong_saturation.json", payload)
    profile = payload["profile"]
    summary = profile["summary"]
    diversity = profile["pairwise_diversity"]
    runtime = profile["runtime"]
    peak = runtime.get("peak_reserved_bytes")
    peak_gib = peak / 2**30 if peak is not None else float("nan")
    lines = [
        "# Stage 1 — K=2 strong saturation on diagnostic 30",
        "",
        f"- Status: `{payload['status']}`",
        f"- Gate: `{payload['gate']['state']}`",
        f"- Interpretation: {payload['gate']['interpretation']}",
        f"- Next: `{payload['gate']['next_step']}`",
        "- Joint selector implemented: `false`",
        "- Lambda tuned: `false`",
        "",
        "| Metric | Mean | Median | Bootstrap 95% where available |",
        "| --- | ---: | ---: | --- |",
    ]
    for name in ("p_ind", "p_oracle", "h_pair", "c_ind", "c_oracle", "delta_c"):
        interval = summary["bootstrap_95"].get(name)
        interval_text = (
            f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "—"
        )
        lines.append(
            f"| {name} | {summary[name]['mean']:.4f} | "
            f"{summary[name]['median']:.4f} | {interval_text} |"
        )
    lines.extend(
        [
            "",
            "## Window and shared-span recovery",
            "",
            f"- Independent per-window token recovery: {summary['independent_window_token_recovery']['mean']:.4f}",
            f"- Independent per-window exact fraction: {summary['independent_window_exact_fraction']:.4f}",
            f"- Per-view oracle token recovery: {summary['per_view_oracle_window_token_recovery']['mean']:.4f}",
            f"- Per-view oracle exact fraction: {summary['per_view_oracle_window_exact_fraction']:.4f}",
            f"- Independent shared-span recovery: {summary['independent_shared_recovery_mean']['mean']:.4f}",
            f"- Per-view oracle shared-span recovery: {summary['oracle_shared_recovery_mean']['mean']:.4f}",
            f"- Independent document exact fraction: {profile['independent_document_exact_fraction']:.4f}",
            f"- Pair-oracle document exact fraction: {profile['pair_oracle_document_exact_fraction']:.4f}",
            "",
            "## Final-beam diversity and cost",
            "",
            f"- Returned unique candidates/view: mean {profile['unique_candidate_count']['mean']:.3f}",
            f"- Unique normalized strings before Top-8 cap: mean {profile['unique_before_cap']['mean']:.3f}",
            f"- Raw-final duplicate rate: mean {profile['duplicate_rate']['mean']:.4f}",
            f"- Pairwise token diversity: mean {diversity['token_edit_diversity']['mean']:.4f}",
            f"- Pairwise embedding cosine: mean {diversity['embedding_cosine_similarity']['mean']:.4f}",
            f"- Pairwise embedding diversity: mean {diversity['embedding_cosine_diversity']['mean']:.4f}",
            f"- Generation runtime/view: {runtime['seconds_per_view']:.3f} seconds",
            f"- Peak PyTorch reserved GPU memory: {peak_gib:.3f} GiB",
            "",
            "`sbeam8` is this repository's strong reproduction setting, not a "
            "beam width claimed to be fixed by the original Vec2Text paper.",
        ]
    )
    (output_dir / "stage1_strong_saturation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_stage1_strong_saturation(
    config: Mapping[str, Any],
    *,
    loader: PassageLoader | None = None,
    tokenizer: Any | None = None,
    backend: Vec2TextBackend | None = None,
) -> dict[str, Any]:
    """Run only Stage 1 on the frozen diagnostic-30 K=2 partition."""

    import torch
    from transformers import AutoTokenizer

    base = dict(config["base"])
    stage1 = dict(config["stage1"])
    dataset_config = dict(base["dataset"])
    geometry = dict(base["geometry"])
    model_config = dict(base["models"])
    partitions_config = dict(base["partitions"])
    diagnostic_config = dict(base["diagnostic"])
    output_dir = Path(stage1["output_dir"])
    checkpoint_path = output_dir / str(stage1["checkpoint_filename"])
    partition_name = str(stage1["partition_name"])
    expected_pairs = int(stage1["expected_pairs"])

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
    partitions = _partition_pairs(
        deduplicated,
        dataset_revision=dataset_config["revision"],
        partition_seed=int(partitions_config["seed"]),
        sizes=partitions_config["sizes"],
    )
    pairs = partitions[partition_name]
    if len(pairs) != expected_pairs:
        raise RuntimeError(
            f"Stage 1 expected {expected_pairs} pairs, found {len(pairs)}"
        )
    manifest = [pair.passage_id for pair in pairs]

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
        raise RuntimeError("scan tokenizer and Vec2Text embedder tokenizer differ")

    texts = [window.text for pair in pairs for window in pair.windows]
    expected_input_ids = [
        window.model_input_ids for pair in pairs for window in pair.windows
    ]
    embedding_started = _cuda_telemetry_start(torch)
    embeddings = backend.embed_raw_windows(
        texts, expected_input_ids=expected_input_ids
    )
    embedding_telemetry = _cuda_telemetry_end(torch, embedding_started, len(pairs))
    embedding_sha256 = hashlib.sha256(
        embeddings.contiguous().numpy().tobytes()
    ).hexdigest()
    embedding_telemetry.update(
        {
            "stage": "diagnostic30_shared_fp32_target_embeddings",
            "observations": len(texts),
            "embedding_dtype": str(embeddings.dtype),
            "embedding_shape": list(embeddings.shape),
            "embedding_sha256": embedding_sha256,
        }
    )

    fingerprint = _checkpoint_fingerprint(base, stage1, manifest)
    pools, records = _load_checkpoint(
        checkpoint_path,
        fingerprint=fingerprint,
        embedding_sha256=embedding_sha256,
        expected_views=len(texts),
    )
    resumed_views = len(pools)
    candidate_config = dict(stage1["candidate_generation"])
    for view_index in range(resumed_views, len(texts)):
        started = _cuda_telemetry_start(torch)
        generated_pools, generated_records = (
            backend.generate_final_sbeam_candidate_pools(
                embeddings[view_index : view_index + 1],
                correction_steps=int(candidate_config["correction_steps"]),
                sequence_beam_width=int(
                    candidate_config["sequence_beam_width"]
                ),
                top_n=int(candidate_config["top_n"]),
            )
        )
        telemetry = _cuda_telemetry_end(torch, started, 1)
        record = dict(generated_records[0])
        record["observation_index"] = view_index
        record["pair_index"] = view_index // 2
        record["window_index"] = view_index % 2
        record["pair_id"] = manifest[view_index // 2]
        record["telemetry"] = telemetry
        pools.append(generated_pools[0])
        records.append(record)
        _save_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            embedding_sha256=embedding_sha256,
            manifest=manifest,
            pools=pools,
            records=records,
        )

    analysis_started = _cuda_telemetry_start(torch)
    profile = _pool_profile(
        "diagnostic30_final_sbeam_hypotheses",
        pools,
        pairs,
        backend,
        raw_counts=[
            int(record["raw_final_hypothesis_count"]) for record in records
        ],
        unique_before_cap_counts=[
            int(record["unique_normalized_text_count_before_cap"])
            for record in records
        ],
        overlap_tokens=int(geometry["overlap_tokens"]),
        minimum_alignment=float(diagnostic_config["stitch_minimum_alignment"]),
        bootstrap_seed=int(diagnostic_config["bootstrap_seed"]),
        bootstrap_iterations=int(stage1["bootstrap_iterations"]),
    )
    analysis_telemetry = _cuda_telemetry_end(torch, analysis_started, len(pairs))
    elapsed_values = [
        float(record["telemetry"]["elapsed_seconds"]) for record in records
    ]
    peak_allocated_values = [
        int(record["telemetry"]["peak_allocated_bytes"])
        for record in records
        if record["telemetry"].get("peak_allocated_bytes") is not None
    ]
    peak_reserved_values = [
        int(record["telemetry"]["peak_reserved_bytes"])
        for record in records
        if record["telemetry"].get("peak_reserved_bytes") is not None
    ]
    profile["pool_semantics"] = (
        "Up to eight unique normalized strings from each installed Vec2Text "
        "correction50+sbeam8 final 64-way expansion, frozen before any truth-based "
        "diagnostic and ranked only by target-embedding cosine F."
    )
    profile["unique_before_cap"] = {
        "mean": mean(
            int(record["unique_normalized_text_count_before_cap"])
            for record in records
        ),
        "min": min(
            int(record["unique_normalized_text_count_before_cap"])
            for record in records
        ),
        "max": max(
            int(record["unique_normalized_text_count_before_cap"])
            for record in records
        ),
    }
    profile["independent_document_exact_fraction"] = mean(
        float(row["p_ind"] == 1.0) for row in profile["diagnostic_rows"]
    )
    profile["pair_oracle_document_exact_fraction"] = mean(
        float(row["p_oracle"] == 1.0) for row in profile["diagnostic_rows"]
    )
    profile["runtime"] = {
        "elapsed_seconds": sum(elapsed_values),
        "seconds_per_view": mean(elapsed_values),
        "seconds_per_pair": sum(elapsed_values) / len(pairs),
        "peak_allocated_bytes": (
            max(peak_allocated_values) if peak_allocated_values else None
        ),
        "peak_reserved_bytes": (
            max(peak_reserved_values) if peak_reserved_values else None
        ),
        "model_load_and_target_embedding_time_excluded": True,
        "resumed_views": resumed_views,
    }
    profile["analysis_telemetry"] = analysis_telemetry

    gate_config = dict(stage1["gate"])
    gate = classify_stage1_saturation(
        profile["summary"],
        independent_lower95_threshold=float(
            gate_config["independent_lower95_threshold"]
        ),
        headroom_upper95_threshold=float(
            gate_config["headroom_upper95_threshold"]
        ),
    )
    payload = {
        "schema_version": 1,
        "status": "STAGE1_COMPLETE",
        "stage": "K2_STRONG_SATURATION_DIAGNOSTIC30",
        "config": config,
        "environment": _environment(),
        "dataset_scan": scan,
        "partition_name": partition_name,
        "diagnostic_pair_manifest": manifest,
        "geometry": geometry,
        "target_embedding_telemetry": embedding_telemetry,
        "checkpoint": {
            "path": str(checkpoint_path),
            "fingerprint": fingerprint,
            "resumed_views": resumed_views,
            "completed_views": len(pools),
        },
        "instrumentation": {
            "records": records,
            "all_public_top1_exact_id_match": all(
                bool(record["public_top1_exact_id_match"]) for record in records
            ),
            "strong_reproduction_setting": "50 correction steps + sbeam width 8",
            "original_paper_beam_width_claimed": False,
        },
        "profile": profile,
        "gate": gate,
        "stage2_implemented": False,
        "diagnostic_30_completed": True,
        "joint_selector_implemented": False,
        "lambda_tuned": False,
        "scope_exclusions": base.get("scope_exclusions", []),
    }
    _write_stage1_artifacts(output_dir, payload)
    return payload
