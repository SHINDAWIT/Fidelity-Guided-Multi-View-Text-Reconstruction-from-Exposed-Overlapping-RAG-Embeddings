"""Stage 3A long-document round-trip-stable chain feasibility scan.

This module performs tokenizer-only dataset construction. It creates no
embeddings, runs no Vec2Text inversion, and contains no Joint selector.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import yaml

from .chunking import StableTokenChain, find_roundtrip_stable_chains
from .data import NaturalQuestionsLoader, PassageLoader
from .diagnostic_experiment import _wilson_interval, load_diagnostic_config
from .stage1_strong_saturation import _write_json_atomic


def load_stage3a_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        stage3a = yaml.safe_load(handle)
    if not isinstance(stage3a, dict):
        raise ValueError("Stage 3A config must be a YAML mapping")
    raw_base = Path(str(stage3a["base_config"]))
    base_path = raw_base if raw_base.is_absolute() else config_path.parent / raw_base
    raw_reference = Path(str(stage3a["k2_reference_artifact"]))
    reference_path = (
        raw_reference
        if raw_reference.is_absolute()
        else config_path.parent / raw_reference
    )
    return {
        "base": load_diagnostic_config(base_path),
        "stage3a": stage3a,
        "resolved_base_config": str(base_path.resolve()),
        "resolved_stage3a_config": str(config_path.resolve()),
        "resolved_k2_reference_artifact": str(reference_path.resolve()),
    }


def _quantile(values: Sequence[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return int(ordered[index])


def _start_distribution(values: Sequence[int]) -> dict[str, Any]:
    histogram = Counter(int(value) for value in values)
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "histogram": {str(key): histogram[key] for key in sorted(histogram)},
    }


def _multiplicity_counts(values: Sequence[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(key): counts[key] for key in sorted(counts)}


def _deduplicate_chains(
    chains: Sequence[StableTokenChain],
) -> tuple[list[StableTokenChain], dict[str, Any]]:
    by_passage_id: dict[str, StableTokenChain] = {}
    duplicate_ids: list[str] = []
    for chain in sorted(
        chains, key=lambda item: (item.source_row_index, item.passage_id)
    ):
        if chain.passage_id in by_passage_id:
            duplicate_ids.append(chain.passage_id)
            continue
        by_passage_id[chain.passage_id] = chain

    by_reference: dict[tuple[int, ...], StableTokenChain] = {}
    duplicate_reference_ids: list[str] = []
    for chain in sorted(
        by_passage_id.values(),
        key=lambda item: (item.passage_id, item.source_row_index),
    ):
        incumbent = by_reference.get(chain.reference_ids)
        if incumbent is None:
            by_reference[chain.reference_ids] = chain
            continue
        survivor = min(
            (incumbent, chain),
            key=lambda item: (item.passage_id, item.source_row_index),
        )
        removed = chain if survivor is incumbent else incumbent
        by_reference[chain.reference_ids] = survivor
        duplicate_reference_ids.append(removed.passage_id)

    deduplicated = sorted(
        by_reference.values(),
        key=lambda item: (item.source_row_index, item.passage_id),
    )
    return deduplicated, {
        "raw_stable_chains": len(chains),
        "duplicate_passage_id_count": len(duplicate_ids),
        "duplicate_passage_ids": duplicate_ids,
        "duplicate_reference_id_sequence_count": len(duplicate_reference_ids),
        "duplicate_reference_passage_ids": duplicate_reference_ids,
        "deduplicated_stable_documents": len(deduplicated),
        "reference_deduplication_semantics": (
            "exact tokenizer-ID equality over the selected K-window union"
        ),
    }


def choose_nested_cohort_k(
    available_documents_by_k: Mapping[int, int], target_size: int
) -> int | None:
    """Choose the largest K that supports the requested paired cohort."""

    if target_size < 1:
        raise ValueError("target_size must be positive")
    feasible = [
        int(k)
        for k, count in available_documents_by_k.items()
        if int(count) >= target_size
    ]
    return max(feasible) if feasible else None


def _cohort_digest(
    chain: StableTokenChain,
    *,
    dataset_revision: str,
    cohort_seed: int,
    max_k: int,
) -> bytes:
    return hashlib.sha256(
        f"{dataset_revision}\0{cohort_seed}\0{max_k}\0{chain.passage_id}".encode(
            "utf-8"
        )
    ).digest()


def _prefix_payload(
    chain: StableTokenChain,
    *,
    k: int,
    content_tokens: int,
    stride: int,
) -> dict[str, Any]:
    windows = chain.windows[:k]
    latent_length = content_tokens + stride * (k - 1)
    reference_ids = chain.reference_ids[:latent_length]
    multiplicity = tuple(
        sum(
            window.document_token_start
            <= chain.selected_start + relative_position
            < window.document_token_end
            for window in windows
        )
        for relative_position in range(latent_length)
    )
    for window_index, window in enumerate(windows):
        relative_start = window_index * stride
        if window.content_ids != reference_ids[
            relative_start : relative_start + content_tokens
        ]:
            raise AssertionError("nested prefix does not preserve source token IDs")
    return {
        "k": k,
        "latent_content_length": latent_length,
        "reference_ids": list(reference_ids),
        "observation_multiplicity": list(multiplicity),
        "multiplicity_counts": _multiplicity_counts(multiplicity),
        "window_indices": list(range(k)),
    }


def _chain_document_payload(
    chain: StableTokenChain,
    *,
    ks: Sequence[int],
    content_token_count: int,
    content_tokens: int,
    stride: int,
) -> dict[str, Any]:
    return {
        "passage_id": chain.passage_id,
        "source_row_index": chain.source_row_index,
        "passage_content_token_count": content_token_count,
        "selected_start": chain.selected_start,
        "windows": [
            {
                "window_index": window.window_index,
                "text": window.text,
                "content_ids": list(window.content_ids),
                "model_input_ids": list(window.model_input_ids),
                "attention_mask": list(window.attention_mask),
                "document_token_start": window.document_token_start,
                "document_token_end": window.document_token_end,
                "char_start": window.char_start,
                "char_end": window.char_end,
            }
            for window in chain.windows
        ],
        "prefixes": {
            str(k): _prefix_payload(
                chain,
                k=k,
                content_tokens=content_tokens,
                stride=stride,
            )
            for k in ks
        },
    }


def _write_stage3a_report(output_dir: Path, payload: Mapping[str, Any]) -> None:
    _write_json_atomic(output_dir / "stage3a_feasibility.json", payload)
    gate = payload["gate"]
    lines = [
        "# Stage 3A — long-document feasibility scan",
        "",
        f"- Status: `{payload['status']}`",
        f"- Gate: `{gate['state']}`",
        f"- Requested cohort size: {gate['target_cohort_size']}",
        f"- Selected max K: `{gate['selected_max_k']}`",
        f"- Full requested K=8 cohort feasible: `{gate['requested_max_k_feasible']}`",
        f"- Stage 3B eligible: `{gate['stage3b_eligible']}` (not executed)",
        "",
        "| K | L(K) | Length eligible | All-window stable | Stable / eligible | Dedup documents | Dedup / scan | Fixed-start stable | Selected start median / p95 / max | Multiplicity counts |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for k in payload["k_values"]:
        profile = payload["profiles"][str(k)]
        selected = profile["selected_start_distribution"]
        lines.append(
            f"| {k} | {profile['required_content_tokens']} | "
            f"{profile['length_eligible_passages']} | "
            f"{profile['all_windows_roundtrip_stable_passages']} | "
            f"{profile['acceptance_yield_among_length_eligible']:.4f} | "
            f"{profile['deduplication']['deduplicated_stable_documents']} | "
            f"{profile['deduplicated_yield_over_scan']:.4f} | "
            f"{profile['fixed_start_stability_rate']:.4f} | "
            f"{selected['median']} / {selected['p95']} / {selected['max']} | "
            f"{profile['multiplicity_counts_per_document']} |"
        )
    lines.extend(
        [
            "",
            "Each chain uses 31 content IDs plus EOS per standalone model input, "
            "stride 15, and exact adjacent overlap 16. Distance-two windows share "
            "one boundary content token, so K>=3 contains multiplicity-three source "
            "positions; these are explicitly stored in the cohort artifact.",
            "No embeddings, Vec2Text inversion, Joint method, lambda tuning, beam "
            "sweep, or K-scaling reconstruction evaluation is executed in Stage 3A.",
        ]
    )
    (output_dir / "stage3a_feasibility.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_stage3a_feasibility(
    config: Mapping[str, Any],
    *,
    loader: PassageLoader | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Run the tokenizer-only K-chain feasibility scan and nested cohort gate."""

    from transformers import AutoTokenizer

    base = dict(config["base"])
    stage3a = dict(config["stage3a"])
    dataset = dict(base["dataset"])
    geometry = dict(base["geometry"])
    models = dict(base["models"])
    k_values = tuple(int(k) for k in stage3a["k_values"])
    if k_values != (2, 3, 4, 6, 8):
        raise ValueError("Stage 3A K values changed from the frozen scan grid")
    content_tokens = int(geometry["content_tokens"])
    overlap_tokens = int(geometry["overlap_tokens"])
    stride = int(geometry["stride_tokens"])
    model_positions = int(geometry["model_positions"])
    if (content_tokens, overlap_tokens, stride, model_positions) != (31, 16, 15, 32):
        raise ValueError("Stage 3A Primary geometry must remain 31/16/15/32")

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        models["embedder_name"],
        revision=models.get("embedder_revision"),
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("round-trip raw-substring mode requires a fast tokenizer")
    loader = loader or NaturalQuestionsLoader(
        dataset_name=dataset["name"],
        dataset_config=dataset.get("config"),
        split=dataset["split"],
        revision=dataset["revision"],
        max_words=None,
        streaming=bool(dataset.get("streaming", True)),
    )
    passages = loader.load(int(dataset["scan_limit"]))

    chains_by_k: dict[int, list[StableTokenChain]] = {k: [] for k in k_values}
    content_counts_by_id: dict[str, int] = {}
    aggregates = {
        k: {
            "length_eligible": 0,
            "candidate_starts": 0,
            "stable_starts": 0,
            "fixed_start_stable": 0,
            "selected_starts": [],
            "all_stable_start_positions": [],
            "rejections": Counter(),
        }
        for k in k_values
    }
    for fallback_index, passage in enumerate(passages):
        source_row_index = int(passage.metadata.get("row_index", fallback_index))
        searches = find_roundtrip_stable_chains(
            passage.text,
            tokenizer,
            passage_id=passage.passage_id,
            source_row_index=source_row_index,
            ks=k_values,
            content_tokens=content_tokens,
            overlap_tokens=overlap_tokens,
            model_positions=model_positions,
        )
        content_counts_by_id[passage.passage_id] = searches[2].content_token_count
        for k, search in searches.items():
            aggregate = aggregates[k]
            aggregate["length_eligible"] += int(
                search.content_token_count >= search.required_content_tokens
            )
            aggregate["candidate_starts"] += search.candidate_starts
            aggregate["stable_starts"] += search.stable_starts
            aggregate["fixed_start_stable"] += int(search.fixed_start_stable)
            aggregate["all_stable_start_positions"].extend(
                search.stable_start_positions
            )
            if search.chain is None:
                aggregate["rejections"][search.rejection_reason or "unknown"] += 1
                continue
            chains_by_k[k].append(search.chain)
            aggregate["selected_starts"].append(search.chain.selected_start)

    profiles: dict[str, dict[str, Any]] = {}
    deduplicated_by_k: dict[int, list[StableTokenChain]] = {}
    for k in k_values:
        aggregate = aggregates[k]
        chains = chains_by_k[k]
        deduplicated, deduplication = _deduplicate_chains(chains)
        deduplicated_by_k[k] = deduplicated
        length_eligible = int(aggregate["length_eligible"])
        stable_count = len(chains)
        acceptance = stable_count / length_eligible if length_eligible else 0.0
        wilson = (
            _wilson_interval(stable_count, length_eligible)
            if length_eligible
            else (0.0, 0.0)
        )
        candidate_starts = int(aggregate["candidate_starts"])
        stable_starts = int(aggregate["stable_starts"])
        first_chain = chains[0] if chains else None
        profiles[str(k)] = {
            "k": k,
            "required_content_tokens": content_tokens + stride * (k - 1),
            "scanned_passages": len(passages),
            "length_eligible_passages": length_eligible,
            "length_eligibility_yield": (
                length_eligible / len(passages) if passages else 0.0
            ),
            "all_windows_roundtrip_stable_passages": stable_count,
            "acceptance_yield_among_length_eligible": acceptance,
            "acceptance_wilson95": list(wilson),
            "stable_yield_over_scan": (
                stable_count / len(passages) if passages else 0.0
            ),
            "deduplication": deduplication,
            "deduplicated_yield_over_scan": (
                len(deduplicated) / len(passages) if passages else 0.0
            ),
            "candidate_chain_starts": candidate_starts,
            "stable_chain_starts": stable_starts,
            "position_stability_rate": (
                stable_starts / candidate_starts if candidate_starts else 0.0
            ),
            "fixed_start_stable_passages": int(
                aggregate["fixed_start_stable"]
            ),
            "fixed_start_stability_rate": (
                int(aggregate["fixed_start_stable"]) / length_eligible
                if length_eligible
                else 0.0
            ),
            "selected_start_distribution": _start_distribution(
                aggregate["selected_starts"]
            ),
            "all_stable_start_distribution": _start_distribution(
                aggregate["all_stable_start_positions"]
            ),
            "rejection_counts": dict(sorted(aggregate["rejections"].items())),
            "multiplicity_counts_per_document": (
                _multiplicity_counts(first_chain.observation_multiplicity)
                if first_chain is not None
                else {}
            ),
            "maximum_observation_multiplicity": (
                max(first_chain.observation_multiplicity)
                if first_chain is not None
                else None
            ),
        }

    reference_path = Path(str(config["resolved_k2_reference_artifact"]))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_scan = reference["dataset_scan"]
    k2_profile = profiles["2"]
    k2_crosscheck = {
        "reference_artifact": str(reference_path),
        "reference_artifact_sha256": hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
        "length_eligible_match": (
            k2_profile["length_eligible_passages"]
            == reference_scan["length_eligible_rows"]
        ),
        "raw_stable_count_match": (
            k2_profile["all_windows_roundtrip_stable_passages"]
            == reference_scan["raw_stable_pairs"]
        ),
        "candidate_start_count_match": (
            k2_profile["candidate_chain_starts"]
            == reference_scan["candidate_starts"]
        ),
        "stable_start_count_match": (
            k2_profile["stable_chain_starts"] == reference_scan["stable_starts"]
        ),
        "deduplicated_count_match": (
            k2_profile["deduplication"]["deduplicated_stable_documents"]
            == reference_scan["deduplication"]["deduplicated_stable_pairs"]
        ),
    }
    if not all(
        value
        for key, value in k2_crosscheck.items()
        if key.endswith("_match")
    ):
        raise RuntimeError("generic K-chain scan does not reproduce the frozen K=2 scan")

    target_size = int(stage3a["target_nested_cohort_size"])
    selected_max_k = choose_nested_cohort_k(
        {
            k: len(deduplicated_by_k[k])
            for k in k_values
        },
        target_size,
    )
    requested_max_k = max(k_values)
    stage3b_eligible = selected_max_k is not None
    requested_max_k_feasible = selected_max_k == requested_max_k
    nested_ks = tuple(k for k in k_values if selected_max_k and k <= selected_max_k)
    output_dir = Path(stage3a["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_summary: dict[str, Any] | None = None
    if selected_max_k is not None:
        cohort_seed = int(stage3a["cohort_seed"])
        ordered = sorted(
            deduplicated_by_k[selected_max_k],
            key=lambda chain: (
                _cohort_digest(
                    chain,
                    dataset_revision=dataset["revision"],
                    cohort_seed=cohort_seed,
                    max_k=selected_max_k,
                ),
                chain.passage_id,
            ),
        )
        cohort = ordered[:target_size]
        documents = [
            _chain_document_payload(
                chain,
                ks=nested_ks,
                content_token_count=content_counts_by_id[chain.passage_id],
                content_tokens=content_tokens,
                stride=stride,
            )
            for chain in cohort
        ]
        cohort_payload = {
            "schema_version": 1,
            "stage": "STAGE3A_NESTED_LONG_DOCUMENT_COHORT",
            "dataset": dataset,
            "geometry": {
                **geometry,
                "k_values": list(nested_ks),
                "latent_lengths": {
                    str(k): content_tokens + stride * (k - 1) for k in nested_ks
                },
                "pairwise_window_overlap_by_index_distance": {
                    "1": overlap_tokens,
                    "2": max(0, content_tokens - 2 * stride),
                    "3_or_more": 0,
                },
            },
            "selection": {
                "seed": cohort_seed,
                "method": "sha256(dataset_revision, seed, max_k, passage_id)",
                "selected_max_k": selected_max_k,
                "cohort_size": len(cohort),
            },
            "manifest": [chain.passage_id for chain in cohort],
            "documents": documents,
            "cache_semantics": (
                "Stage 3B may generate each max-K window once and reuse the same "
                "cached Independent result for every shorter nested prefix"
            ),
        }
        cohort_path = output_dir / "stage3a_nested_cohort.json"
        _write_json_atomic(cohort_path, cohort_payload)
        cohort_summary = {
            "path": str(cohort_path),
            "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
            "size": len(cohort),
            "selected_max_k": selected_max_k,
            "nested_k_values": list(nested_ks),
            "manifest": cohort_payload["manifest"],
            "selected_start_distribution": _start_distribution(
                [chain.selected_start for chain in cohort]
            ),
            "multiplicity_counts_by_k": {
                str(k): documents[0]["prefixes"][str(k)]["multiplicity_counts"]
                for k in nested_ks
            },
        }

    gate = {
        "state": (
            "STAGE3A_NESTED_COHORT_READY"
            if stage3b_eligible
            else "STAGE3A_INSUFFICIENT_COHORT"
        ),
        "target_cohort_size": target_size,
        "requested_max_k": requested_max_k,
        "selected_max_k": selected_max_k,
        "requested_max_k_feasible": requested_max_k_feasible,
        "stage3b_eligible": stage3b_eligible,
        "stage3b_executed": False,
        "stage3c_executed": False,
        "reason": (
            f"K={selected_max_k} has at least {target_size} deduplicated stable "
            "documents for a common nested prefix cohort."
            if selected_max_k is not None
            else f"No requested K has {target_size} deduplicated stable documents."
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "STAGE3A_COMPLETE",
        "stage": "LONG_DOCUMENT_TOKENIZER_FEASIBILITY_ONLY",
        "config": config,
        "geometry": {
            **geometry,
            "latent_length_formula": "31 + 15 * (K - 1)",
            "distance_two_overlap_tokens": 1,
        },
        "scanned_passages": len(passages),
        "k_values": list(k_values),
        "profiles": profiles,
        "k2_frozen_scan_crosscheck": k2_crosscheck,
        "nested_cohort": cohort_summary,
        "gate": gate,
        "embeddings_generated": False,
        "vec2text_executed": False,
        "joint_selector_implemented": False,
        "lambda_tuned": False,
        "beam_width_sweep_implemented": False,
        "scope_exclusions": base.get("scope_exclusions", []),
    }
    _write_stage3a_report(output_dir, payload)
    return payload
