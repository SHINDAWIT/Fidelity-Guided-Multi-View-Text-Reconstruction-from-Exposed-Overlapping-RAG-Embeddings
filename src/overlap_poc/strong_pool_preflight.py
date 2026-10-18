"""Reserved-three-pair strong candidate-pool preflight.

This module intentionally contains no Joint selector and no consistency weight.
Ground truth is passed only to the existing diagnostic evaluator after both
candidate pools have been frozen.
"""

from __future__ import annotations

import hashlib
import json
import platform
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import yaml

from .diagnostic_experiment import (
    _cuda_telemetry_end,
    _cuda_telemetry_start,
    _deduplicate_pairs,
    _partition_pairs,
    _scan_pairs,
    load_diagnostic_config,
)
from .diagnostics import (
    evaluate_candidate_pair,
    levenshtein_distance,
    summarize_diagnostics,
)
from .data import NaturalQuestionsLoader, PassageLoader
from .reconstruction import Candidate, Vec2TextBackend


def load_strong_preflight_config(path: str | Path) -> dict[str, Any]:
    """Load a small strong-search overlay plus its frozen primary PoC config."""

    overlay_path = Path(path)
    with overlay_path.open(encoding="utf-8") as handle:
        overlay = yaml.safe_load(handle)
    if not isinstance(overlay, dict):
        raise ValueError("strong preflight config must be a YAML mapping")
    raw_base = Path(str(overlay["base_config"]))
    base_path = raw_base if raw_base.is_absolute() else overlay_path.parent / raw_base
    return {
        "base": load_diagnostic_config(base_path),
        "strong": overlay,
        "resolved_base_config": str(base_path.resolve()),
        "resolved_overlay_config": str(overlay_path.resolve()),
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def classify_strong_pool_gate(
    summary: Mapping[str, Any],
    *,
    oracle_floor: float = 0.60,
    independent_saturation: float = 0.90,
    minimum_headroom: float = 2 / 46,
    minimum_signal_fraction: float = 0.60,
) -> dict[str, Any]:
    """Classify a reserved-three-pair pool using descriptive point estimates.

    Three pairs cannot establish an inferential result.  This is strictly a
    compute/ceiling preflight used to decide whether a 30-pair run is warranted.
    """

    p_ind = float(summary["p_ind"]["mean"])
    p_oracle = float(summary["p_oracle"]["mean"])
    headroom = float(summary["h_pair"]["mean"])
    delta_c = float(summary["delta_c"]["mean"])
    signal_fraction = float(summary["delta_c_positive_fraction_when_headroom"])

    if p_ind >= independent_saturation:
        state = "STRONG_INDEPENDENT_SATURATED"
        reason = "Strong Independent mean reaches the exploratory saturation threshold."
    elif p_oracle < oracle_floor or headroom < minimum_headroom:
        state = "STRONG_POOL_STILL_LIMITED"
        reason = (
            "The strong frozen pool does not clear both the absolute oracle "
            "floor and the two-token mean headroom threshold."
        )
    elif delta_c <= 0.0 or signal_fraction < minimum_signal_fraction:
        state = "HEADROOM_BUT_NO_OVERLAP_SIGNAL"
        reason = (
            "The strong pool has oracle headroom, but candidate-only overlap "
            "consistency does not distinguish the improving pairs."
        )
    else:
        state = "STRONG_POOL_JOINT_SIGNAL_GO"
        reason = (
            "The strong pool clears the exploratory ceiling/headroom thresholds "
            "and its oracle-improving pairs have positive overlap signal."
        )
    return {
        "state": state,
        "reason": reason,
        "basis": "reserved-3 descriptive point estimates; not an inferential claim",
        "thresholds": {
            "oracle_floor": oracle_floor,
            "independent_saturation": independent_saturation,
            "minimum_headroom": minimum_headroom,
            "minimum_signal_fraction": minimum_signal_fraction,
        },
        "observed": {
            "p_ind_mean": p_ind,
            "p_oracle_mean": p_oracle,
            "h_pair_mean": headroom,
            "delta_c_mean": delta_c,
            "delta_c_positive_fraction_when_headroom": signal_fraction,
        },
    }


def _normalized_key(candidate: Candidate) -> str:
    return " ".join(candidate.text.casefold().split())


def merge_multi_strength_pool(
    endpoint_pool: Sequence[Candidate], final_beam_pool: Sequence[Candidate]
) -> tuple[Candidate, ...]:
    """Merge base/20/50 endpoints with the public 50+sbeam8 Top-1."""

    if not final_beam_pool:
        raise ValueError("final beam pool must have a public Top-1")
    combined = list(endpoint_pool) + [final_beam_pool[0]]
    unique: dict[str, tuple[int, Candidate]] = {}
    for index, candidate in enumerate(combined):
        key = _normalized_key(candidate)
        incumbent = unique.get(key)
        if incumbent is None or candidate.local_score > incumbent[1].local_score:
            unique[key] = (index, candidate)
    ordered = sorted(
        unique.values(),
        key=lambda item: (-item[1].local_score, item[0]),
    )
    return tuple(candidate for _, candidate in ordered)


def _candidate_pairwise_values(
    pool: Sequence[Candidate], candidate_embeddings
) -> dict[str, list[float]]:
    import torch

    if len(pool) != len(candidate_embeddings):
        raise ValueError("candidate and embedding counts differ")
    token_similarities: list[float] = []
    embedding_similarities: list[float] = []
    for left_index, right_index in combinations(range(len(pool)), 2):
        left_ids = pool[left_index].content_ids
        right_ids = pool[right_index].content_ids
        denominator = max(len(left_ids), len(right_ids), 1)
        token_similarities.append(
            max(
                0.0,
                1.0 - levenshtein_distance(left_ids, right_ids) / denominator,
            )
        )
        embedding_similarities.append(
            float(
                torch.nn.functional.cosine_similarity(
                    candidate_embeddings[left_index].unsqueeze(0),
                    candidate_embeddings[right_index].unsqueeze(0),
                    dim=1,
                ).item()
            )
        )
    return {
        "token_edit_similarity": token_similarities,
        "token_edit_diversity": [1.0 - value for value in token_similarities],
        "embedding_cosine_similarity": embedding_similarities,
        "embedding_cosine_diversity": [
            1.0 - value for value in embedding_similarities
        ],
    }


def candidate_pool_diversity(
    pool: Sequence[Candidate], candidate_embeddings
) -> dict[str, Any]:
    """Within-view pairwise token and embedding diversity."""

    values = _candidate_pairwise_values(pool, candidate_embeddings)
    return {
        "pair_count": len(values["token_edit_similarity"]),
        **{name: _distribution(items) for name, items in values.items()},
    }


def _pool_profile(
    name: str,
    pools: Sequence[Sequence[Candidate]],
    pairs,
    backend: Vec2TextBackend,
    *,
    raw_counts: Sequence[int],
    unique_before_cap_counts: Sequence[int] | None = None,
    overlap_tokens: int,
    minimum_alignment: float,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    if len(pools) != len(raw_counts):
        raise ValueError("raw count must be present for every view")
    if unique_before_cap_counts is None:
        unique_before_cap_counts = [len(pool) for pool in pools]
    if len(pools) != len(unique_before_cap_counts):
        raise ValueError("pre-cap unique count must be present for every view")

    view_rows: list[dict[str, Any]] = []
    all_token_similarities: list[float] = []
    all_token_diversities: list[float] = []
    all_embedding_similarities: list[float] = []
    all_embedding_diversities: list[float] = []
    for view_index, (pool, raw_count, unique_before_cap) in enumerate(
        zip(pools, raw_counts, unique_before_cap_counts)
    ):
        embeddings = backend.embed_candidate_texts(
            [candidate.text for candidate in pool], batch_size=8
        )
        pairwise_values = _candidate_pairwise_values(pool, embeddings)
        diversity = candidate_pool_diversity(pool, embeddings)
        all_token_similarities.extend(pairwise_values["token_edit_similarity"])
        all_token_diversities.extend(pairwise_values["token_edit_diversity"])
        all_embedding_similarities.extend(
            pairwise_values["embedding_cosine_similarity"]
        )
        all_embedding_diversities.extend(
            pairwise_values["embedding_cosine_diversity"]
        )
        view_rows.append(
            {
                "view_index": view_index,
                "raw_candidate_count": raw_count,
                "unique_candidate_count_before_cap": unique_before_cap,
                "unique_candidate_count": len(pool),
                "duplicate_rate": 1.0 - unique_before_cap / raw_count,
                "cap_retention_rate": len(pool) / unique_before_cap,
                "diversity": diversity,
                "candidates": [candidate.to_dict() for candidate in pool],
            }
        )

    diagnostic_rows: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        pair_pools = pools[pair_index * 2 : pair_index * 2 + 2]
        row = evaluate_candidate_pair(
            pair_pools,
            reference_ids=pair.reference_ids,
            shared_ids=pair.shared_ids,
            overlap_tokens=overlap_tokens,
            minimum_alignment=minimum_alignment,
        )
        row.update(
            {
                "pair_id": pair.passage_id,
                "source_row_index": pair.source_row_index,
                "selected_start": pair.selected_start,
            }
        )
        diagnostic_rows.append(row)

    summary = summarize_diagnostics(
        diagnostic_rows,
        bootstrap_seed=bootstrap_seed,
        bootstrap_iterations=bootstrap_iterations,
    )
    return {
        "name": name,
        "pool_semantics": None,
        "candidate_pool_sha256": backend.candidate_pool_hash(pools),
        "view_count": len(pools),
        "unique_candidate_count_per_view": [len(pool) for pool in pools],
        "unique_candidate_count": _distribution([float(len(pool)) for pool in pools]),
        "duplicate_rate": _distribution(
            [
                1.0 - unique_count / raw
                for unique_count, raw in zip(unique_before_cap_counts, raw_counts)
            ]
        ),
        "pairwise_diversity": {
            "token_edit_similarity": _distribution(all_token_similarities),
            "token_edit_diversity": _distribution(all_token_diversities),
            "embedding_cosine_similarity": _distribution(
                all_embedding_similarities
            ),
            "embedding_cosine_diversity": _distribution(
                all_embedding_diversities
            ),
        },
        "views": view_rows,
        "diagnostic_rows": diagnostic_rows,
        "summary": summary,
    }


def _environment() -> dict[str, Any]:
    import datasets
    import torch
    import transformers
    import vec2text

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "vec2text": getattr(vec2text, "__version__", "0.0.13-installed"),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_runtime": torch.version.cuda,
            }
        )
    return report


def _peak(telemetry: Mapping[str, Any], key: str) -> int | None:
    value = telemetry.get(key)
    return int(value) if value is not None else None


def _combined_telemetry(
    first: Mapping[str, Any], second: Mapping[str, Any], observations: int
) -> dict[str, Any]:
    peaks_allocated = [
        value
        for value in (_peak(first, "peak_allocated_bytes"), _peak(second, "peak_allocated_bytes"))
        if value is not None
    ]
    peaks_reserved = [
        value
        for value in (_peak(first, "peak_reserved_bytes"), _peak(second, "peak_reserved_bytes"))
        if value is not None
    ]
    elapsed = float(first["elapsed_seconds"]) + float(second["elapsed_seconds"])
    return {
        "elapsed_seconds": elapsed,
        "seconds_per_view": elapsed / observations,
        "peak_allocated_bytes": max(peaks_allocated) if peaks_allocated else None,
        "peak_reserved_bytes": max(peaks_reserved) if peaks_reserved else None,
        "component_stages": [first.get("stage"), second.get("stage")],
        "shared_target_embedding_time_excluded": True,
    }


def _write_artifacts(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strong_pool_preflight.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Reserved-3 strong candidate-pool preflight",
        "",
        f"- Status: `{payload['status']}`",
        f"- Overall gate: `{payload['overall_gate']['state']}`",
        f"- Gate reason: {payload['overall_gate']['reason']}",
        "- Statistical scope: reserved 3 pairs; descriptive preflight only",
        "- Joint selector implemented: `false`",
        "- Lambda tuned: `false`",
        "",
        "| Pool | Unique/view mean | Duplicate rate | P_ind | P_oracle | H_pair | C_ind | C_oracle | Delta C | sec/view | Peak reserved GiB | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("multi_strength_endpoints", "final_sbeam_hypotheses"):
        profile = payload["profiles"][key]
        summary = profile["summary"]
        runtime = profile["runtime"]
        peak = runtime.get("peak_reserved_bytes")
        peak_gib = peak / 2**30 if peak is not None else float("nan")
        lines.append(
            f"| {key} | {profile['unique_candidate_count']['mean']:.3f} | "
            f"{profile['duplicate_rate']['mean']:.3f} | "
            f"{summary['p_ind']['mean']:.4f} | {summary['p_oracle']['mean']:.4f} | "
            f"{summary['h_pair']['mean']:.4f} | {summary['c_ind']['mean']:.4f} | "
            f"{summary['c_oracle']['mean']:.4f} | {summary['delta_c']['mean']:.4f} | "
            f"{runtime['seconds_per_view']:.3f} | {peak_gib:.3f} | "
            f"{profile['gate']['state']} |"
        )
    lines.extend(
        [
            "",
            "## Diversity and shared-span recovery",
            "",
            "| Pool | Token diversity | Embedding cosine | Embedding diversity | Top-1 shared | Per-view oracle shared | H>0 | Delta C>0 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in ("multi_strength_endpoints", "final_sbeam_hypotheses"):
        profile = payload["profiles"][key]
        summary = profile["summary"]
        diversity = profile["pairwise_diversity"]
        lines.append(
            f"| {key} | {diversity['token_edit_diversity']['mean']:.4f} | "
            f"{diversity['embedding_cosine_similarity']['mean']:.4f} | "
            f"{diversity['embedding_cosine_diversity']['mean']:.4f} | "
            f"{summary['independent_shared_recovery_mean']['mean']:.4f} | "
            f"{summary['oracle_shared_recovery_mean']['mean']:.4f} | "
            f"{summary['h_pair_positive_fraction']:.4f} | "
            f"{summary['delta_c_positive_fraction']:.4f} |"
        )
    records = payload["instrumentation"]["records"]
    lines.extend(
        [
            "",
            "## Final-beam exposure",
            "",
            "| View | Raw final | Unique before cap | Returned | Duplicate rate | Public Top-1 exact | Generate sec |",
            "| ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['observation_index']} | "
            f"{record['raw_final_hypothesis_count']} | "
            f"{record['unique_normalized_text_count_before_cap']} | "
            f"{record['returned_unique_count']} | "
            f"{record['duplicate_rate_by_normalized_text']:.4f} | "
            f"{str(record['public_top1_exact_id_match']).lower()} | "
            f"{record['elapsed_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Instrumentation invariant",
            "",
            "The final-sbeam pool observes the installed Vec2Text final 64-way "
            "expansion, reranks with the same target cosine, removes normalized-string "
            "duplicates, and exposes at most eight. Every public Top-1 must match the "
            "instrumented cosine Top-1 exactly in generated token-ID space.",
            "",
            "`sbeam8` is this repository's strong reproduction setting; it is not "
            "claimed as a beam width fixed by the original Vec2Text paper.",
        ]
    )
    (output_dir / "strong_pool_preflight.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_strong_pool_preflight(
    config: Mapping[str, Any],
    *,
    loader: PassageLoader | None = None,
    tokenizer: Any | None = None,
    backend: Vec2TextBackend | None = None,
) -> dict[str, Any]:
    """Compare strong endpoint and instrumented final-beam pools on 3 pairs."""

    import torch
    from transformers import AutoTokenizer

    base = dict(config["base"])
    strong = dict(config["strong"])
    dataset_config = dict(base["dataset"])
    geometry = dict(base["geometry"])
    model_config = dict(base["models"])
    partition_config = dict(base["partitions"])
    diagnostic_config = dict(base["diagnostic"])
    output_dir = Path(strong.get("output_dir", "results/k2_strong_pool_preflight"))

    if int(partition_config["sizes"]["preflight"]) != 3:
        raise ValueError("strong preflight is frozen to exactly three reserved pairs")
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
        partition_seed=int(partition_config["seed"]),
        sizes=partition_config["sizes"],
    )
    reserved_pairs = partitions["preflight"]

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

    texts = [window.text for pair in reserved_pairs for window in pair.windows]
    expected_ids = [
        window.model_input_ids for pair in reserved_pairs for window in pair.windows
    ]
    embedding_started = _cuda_telemetry_start(torch)
    embeddings = backend.embed_raw_windows(texts, expected_input_ids=expected_ids)
    embedding_telemetry = _cuda_telemetry_end(
        torch, embedding_started, len(reserved_pairs)
    )
    embedding_telemetry.update(
        {
            "stage": "shared_fp32_target_embeddings",
            "observations": len(texts),
            "embedding_dtype": str(embeddings.dtype),
            "embedding_shape": list(embeddings.shape),
            "embedding_sha256": hashlib.sha256(
                embeddings.contiguous().numpy().tobytes()
            ).hexdigest(),
        }
    )

    weak_config = dict(strong["weak_reference_pool"])
    weak_started = _cuda_telemetry_start(torch)
    weak_pools = backend.generate_frozen_candidate_pools(
        embeddings,
        weak_config["settings"],
        top_n=int(weak_config["top_n"]),
        batch_size=int(weak_config.get("batch_size", 1)),
    )
    weak_telemetry = _cuda_telemetry_end(torch, weak_started, len(reserved_pairs))
    weak_telemetry.update(
        {"stage": "weak_reference_generation", "seconds_per_view": weak_telemetry["elapsed_seconds"] / len(texts)}
    )

    endpoint_config = dict(strong["multi_strength_endpoints"])
    endpoint_started = _cuda_telemetry_start(torch)
    endpoint_pools = backend.generate_frozen_candidate_pools(
        embeddings,
        endpoint_config["settings"],
        top_n=int(endpoint_config["top_n"]),
        batch_size=int(endpoint_config.get("batch_size", 1)),
    )
    endpoint_telemetry = _cuda_telemetry_end(
        torch, endpoint_started, len(reserved_pairs)
    )
    endpoint_telemetry.update(
        {"stage": "base_20_50_endpoint_generation", "seconds_per_view": endpoint_telemetry["elapsed_seconds"] / len(texts)}
    )

    sbeam_config = dict(strong["final_sbeam_hypotheses"])
    sbeam_started = _cuda_telemetry_start(torch)
    sbeam_pools, instrumentation = backend.generate_final_sbeam_candidate_pools(
        embeddings,
        correction_steps=int(sbeam_config["correction_steps"]),
        sequence_beam_width=int(sbeam_config["sequence_beam_width"]),
        top_n=int(sbeam_config["top_n"]),
    )
    sbeam_telemetry = _cuda_telemetry_end(torch, sbeam_started, len(reserved_pairs))
    sbeam_telemetry.update(
        {
            "stage": "instrumented_50_sbeam8_generation",
            "seconds_per_view": sbeam_telemetry["elapsed_seconds"] / len(texts),
            "instrumented_generate_seconds_per_view": mean(
                float(row["elapsed_seconds"]) for row in instrumentation
            ),
            "all_public_top1_exact_id_match": all(
                bool(row["public_top1_exact_id_match"]) for row in instrumentation
            ),
        }
    )

    multi_strength_pools = tuple(
        merge_multi_strength_pool(endpoint_pool, sbeam_pool)
        for endpoint_pool, sbeam_pool in zip(endpoint_pools, sbeam_pools)
    )
    analysis_started = _cuda_telemetry_start(torch)
    profile_kwargs = {
        "pairs": reserved_pairs,
        "backend": backend,
        "overlap_tokens": int(geometry["overlap_tokens"]),
        "minimum_alignment": float(
            diagnostic_config["stitch_minimum_alignment"]
        ),
        "bootstrap_seed": int(diagnostic_config["bootstrap_seed"]),
        "bootstrap_iterations": int(
            strong.get("bootstrap_iterations", 2000)
        ),
    }
    weak_profile = _pool_profile(
        "weak_reference_pool",
        weak_pools,
        raw_counts=[len(weak_config["settings"])] * len(texts),
        **profile_kwargs,
    )
    endpoint_profile = _pool_profile(
        "multi_strength_endpoints",
        multi_strength_pools,
        raw_counts=[len(endpoint_config["settings"]) + 1] * len(texts),
        **profile_kwargs,
    )
    sbeam_profile = _pool_profile(
        "final_sbeam_hypotheses",
        sbeam_pools,
        raw_counts=[
            int(row["raw_final_hypothesis_count"]) for row in instrumentation
        ],
        unique_before_cap_counts=[
            int(row["unique_normalized_text_count_before_cap"])
            for row in instrumentation
        ],
        **profile_kwargs,
    )
    analysis_telemetry = _cuda_telemetry_end(
        torch, analysis_started, len(reserved_pairs)
    )
    analysis_telemetry["stage"] = "candidate_diversity_and_oracle_diagnostics"

    gate_parameters = {
        "oracle_floor": float(diagnostic_config["oracle_floor"]),
        "independent_saturation": float(
            diagnostic_config["independent_saturation"]
        ),
        "minimum_headroom": float(diagnostic_config["minimum_headroom"]),
        "minimum_signal_fraction": float(
            diagnostic_config["minimum_signal_fraction"]
        ),
    }
    endpoint_profile["gate"] = classify_strong_pool_gate(
        endpoint_profile["summary"], **gate_parameters
    )
    sbeam_profile["gate"] = classify_strong_pool_gate(
        sbeam_profile["summary"], **gate_parameters
    )
    endpoint_profile["pool_semantics"] = (
        "Unique final best endpoints from base/0, correction20, correction50, "
        "and the verified public Top-1 from correction50+sbeam8, ranked by the "
        "same frozen target-embedding cosine F."
    )
    sbeam_profile["pool_semantics"] = (
        "Up to eight unique normalized strings from the existing final 64-way "
        "correction50+sbeam8 expansion, ranked by the same target cosine F; "
        "instrumentation only."
    )
    weak_profile["pool_semantics"] = (
        "Current base0/correction1+sbeam3/correction2+sbeam3 endpoint control."
    )
    endpoint_profile["runtime"] = _combined_telemetry(
        endpoint_telemetry, sbeam_telemetry, len(texts)
    )
    sbeam_profile["runtime"] = dict(sbeam_telemetry)
    weak_profile["runtime"] = dict(weak_telemetry)

    profiles = {
        "weak_reference": weak_profile,
        "multi_strength_endpoints": endpoint_profile,
        "final_sbeam_hypotheses": sbeam_profile,
    }
    strong_profiles = (endpoint_profile, sbeam_profile)
    go_profiles = [
        profile
        for profile in strong_profiles
        if profile["gate"]["state"] == "STRONG_POOL_JOINT_SIGNAL_GO"
    ]
    if go_profiles:
        selected = max(
            go_profiles,
            key=lambda profile: (
                profile["summary"]["h_pair"]["mean"],
                profile["summary"]["delta_c"]["mean"],
                profile["summary"]["p_oracle"]["mean"],
            ),
        )
        overall_gate = dict(selected["gate"])
        overall_gate["selected_pool"] = selected["name"]
    else:
        state_order = {
            "HEADROOM_BUT_NO_OVERLAP_SIGNAL": 3,
            "STRONG_INDEPENDENT_SATURATED": 2,
            "STRONG_POOL_STILL_LIMITED": 1,
        }
        selected = max(
            strong_profiles,
            key=lambda profile: (
                state_order.get(profile["gate"]["state"], 0),
                profile["summary"]["p_oracle"]["mean"],
                profile["summary"]["h_pair"]["mean"],
            ),
        )
        overall_gate = dict(selected["gate"])
        overall_gate["selected_pool"] = selected["name"]

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "STRONG_PREFLIGHT_COMPLETE",
        "statistical_scope": (
            "Reserved 3 pairs only; descriptive compute/ceiling preflight. "
            "No diagnostic-30 inference is claimed."
        ),
        "config": config,
        "environment": _environment(),
        "dataset_scan": scan,
        "reserved_pair_manifest": [pair.passage_id for pair in reserved_pairs],
        "geometry": geometry,
        "target_embedding_telemetry": embedding_telemetry,
        "generation_telemetry": {
            "weak_reference": weak_telemetry,
            "base_20_50_endpoints": endpoint_telemetry,
            "final_50_sbeam8": sbeam_telemetry,
            "diagnostic_analysis": analysis_telemetry,
        },
        "instrumentation": {
            "semantics": (
                "Observe the final embed_generated_hypothesis call only; do not "
                "change Vec2Text generation/search/ranking inputs. Rank the observed "
                "final expansion by the existing target cosine and expose at most "
                "beam-width unique normalized strings."
            ),
            "records": instrumentation,
            "all_public_top1_exact_id_match": all(
                bool(row["public_top1_exact_id_match"]) for row in instrumentation
            ),
            "strong_reproduction_setting": "50 correction steps + sbeam width 8",
            "original_paper_beam_width_claimed": False,
        },
        "profiles": profiles,
        "comparison_to_weak_reference": {
            name: {
                "p_ind_mean_gain": profile["summary"]["p_ind"]["mean"]
                - weak_profile["summary"]["p_ind"]["mean"],
                "p_oracle_mean_gain": profile["summary"]["p_oracle"]["mean"]
                - weak_profile["summary"]["p_oracle"]["mean"],
                "h_pair_mean_gain": profile["summary"]["h_pair"]["mean"]
                - weak_profile["summary"]["h_pair"]["mean"],
                "delta_c_mean_gain": profile["summary"]["delta_c"]["mean"]
                - weak_profile["summary"]["delta_c"]["mean"],
            }
            for name, profile in (
                ("multi_strength_endpoints", endpoint_profile),
                ("final_sbeam_hypotheses", sbeam_profile),
            )
        },
        "overall_gate": overall_gate,
        "diagnostic_30_expanded": False,
        "joint_selector_implemented": False,
        "lambda_tuned": False,
        "scope_exclusions": base.get("scope_exclusions", []),
    }
    _write_artifacts(output_dir, payload)
    return payload
