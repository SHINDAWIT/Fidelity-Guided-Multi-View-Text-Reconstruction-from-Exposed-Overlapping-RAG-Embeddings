"""Freeze a disjoint Plan-2 holdout before any Stage 5B inversion is run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .chunking import StableTokenChain, find_roundtrip_stable_chains
from .data import NaturalQuestionsLoader, PassageLoader
from .diagnostic_experiment import load_diagnostic_config
from .stage1_strong_saturation import _write_json_atomic
from .stage3_longdoc_feasibility import (
    _chain_document_payload,
    _cohort_digest,
    _deduplicate_chains,
)
from .stage5_nooverlap import _build_nooverlap_documents


_IMPLEMENTATION_VERSION = "stage5b-plan2-holdout-manifest-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def load_stage5b_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 5B config must be a YAML mapping")
    base_candidate = Path(str(raw["base_config"]))
    base_path = (
        base_candidate
        if base_candidate.is_absolute()
        else config_path.parent / base_candidate
    )
    cohort_candidate = Path(str(raw["exploratory_cohort_artifact"]))
    cohort_path = (
        cohort_candidate
        if cohort_candidate.is_absolute()
        else config_path.parent / cohort_candidate
    )
    return {
        "base": load_diagnostic_config(base_path),
        "stage5b": raw,
        "resolved_base_config": str(base_path.resolve()),
        "resolved_exploratory_cohort_artifact": str(cohort_path.resolve()),
        "resolved_stage5b_config": str(config_path.resolve()),
    }


def classify_nooverlap_boundary_stratum(document: Mapping[str, Any]) -> str:
    """Assign a preprocessing-only stratum before inversion outputs exist."""

    windows = list(document["windows"])
    if any(
        int(window["source_tokens_deferred_for_capacity"]) > 0
        for window in windows
    ):
        return "BOUNDARY_ADJUSTMENT_REQUIRED"
    if all(bool(window["source_ids_equal_standalone_ids"]) for window in windows):
        return "STRICT_FIXED_STABLE"
    return "CAPACITY_SAFE_NO_SHIFT"


def select_disjoint_holdout(
    ordered: Sequence[StableTokenChain],
    exploratory_manifest: Sequence[str],
    *,
    holdout_size: int,
) -> list[StableTokenChain]:
    """Verify the frozen exploratory prefix, then take the next ordered items."""

    exploratory_ids = [str(value) for value in exploratory_manifest]
    ordered_ids = [chain.passage_id for chain in ordered]
    if ordered_ids[: len(exploratory_ids)] != exploratory_ids:
        raise RuntimeError(
            "recomputed K=8 ordering does not reproduce the frozen exploratory cohort"
        )
    start = len(exploratory_ids)
    stop = start + holdout_size
    selected = list(ordered[start:stop])
    if len(selected) != holdout_size:
        raise RuntimeError("insufficient unused Plan-2 K=8 documents for holdout")
    if set(exploratory_ids) & {chain.passage_id for chain in selected}:
        raise AssertionError("Stage 5B holdout overlaps the exploratory cohort")
    return selected


def select_after_frozen_prefixes(
    ordered: Sequence[StableTokenChain],
    frozen_prefixes: Sequence[Sequence[str]],
    *,
    evaluation_size: int,
) -> list[StableTokenChain]:
    """Verify consecutive frozen cohorts, then take one untouched evaluation slice."""

    frozen_ids = [str(value) for prefix in frozen_prefixes for value in prefix]
    ordered_ids = [chain.passage_id for chain in ordered]
    if ordered_ids[: len(frozen_ids)] != frozen_ids:
        raise RuntimeError("recomputed K=8 ordering differs from frozen prior cohorts")
    selected = list(ordered[len(frozen_ids) : len(frozen_ids) + evaluation_size])
    if len(selected) != evaluation_size:
        raise RuntimeError("insufficient unused Plan-2 K=8 documents for evaluation")
    if set(frozen_ids) & {chain.passage_id for chain in selected}:
        raise AssertionError("fresh evaluation overlaps a prior frozen cohort")
    return selected


def run_stage5b_holdout_manifest(
    config: Mapping[str, Any],
    *,
    loader: PassageLoader | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Rescan the pinned frame and freeze a disjoint holdout; never run Vec2Text."""

    from transformers import AutoTokenizer

    base = dict(config["base"])
    stage5b = dict(config["stage5b"])
    dataset = dict(base["dataset"])
    geometry = dict(base["geometry"])
    models = dict(base["models"])
    if (
        int(geometry["content_tokens"]),
        int(geometry["overlap_tokens"]),
        int(geometry["stride_tokens"]),
        int(geometry["model_positions"]),
    ) != (31, 16, 15, 32):
        raise ValueError("Stage 5B Plan-2 geometry must remain 31/16/15/32")
    if int(stage5b["max_k"]) != 8:
        raise ValueError("Stage 5B holdout must remain K=8")

    exploratory_path = Path(str(config["resolved_exploratory_cohort_artifact"]))
    if _sha256(exploratory_path) != str(stage5b["exploratory_cohort_sha256"]):
        raise RuntimeError("frozen exploratory cohort hash changed")
    exploratory = _load_object(exploratory_path)

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        models["embedder_name"],
        revision=models.get("embedder_revision"),
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Stage 5B Plan-2 construction requires a fast tokenizer")
    loader = loader or NaturalQuestionsLoader(
        dataset_name=dataset["name"],
        dataset_config=dataset.get("config"),
        split=dataset["split"],
        revision=dataset["revision"],
        max_words=None,
        streaming=bool(dataset.get("streaming", True)),
    )
    passages = loader.load(int(dataset["scan_limit"]))

    stable_chains: list[StableTokenChain] = []
    content_counts: dict[str, int] = {}
    for fallback_index, passage in enumerate(passages):
        source_row_index = int(passage.metadata.get("row_index", fallback_index))
        search = find_roundtrip_stable_chains(
            passage.text,
            tokenizer,
            passage_id=passage.passage_id,
            source_row_index=source_row_index,
            ks=(8,),
            content_tokens=31,
            overlap_tokens=16,
            model_positions=32,
        )[8]
        content_counts[passage.passage_id] = search.content_token_count
        if search.chain is not None:
            stable_chains.append(search.chain)
    deduplicated, deduplication = _deduplicate_chains(stable_chains)
    expected_stable = int(stage5b["expected_deduplicated_k8_documents"])
    if len(deduplicated) != expected_stable:
        raise RuntimeError(
            f"pinned Plan-2 K=8 count changed: {len(deduplicated)} != {expected_stable}"
        )
    cohort_seed = int(stage5b["cohort_seed"])
    ordered = sorted(
        deduplicated,
        key=lambda chain: (
            _cohort_digest(
                chain,
                dataset_revision=dataset["revision"],
                cohort_seed=cohort_seed,
                max_k=8,
            ),
            chain.passage_id,
        ),
    )
    holdout_size = int(stage5b["holdout_documents"])
    selected = select_disjoint_holdout(
        ordered,
        exploratory["manifest"],
        holdout_size=holdout_size,
    )
    overlap_documents = [
        _chain_document_payload(
            chain,
            ks=(8,),
            content_token_count=content_counts[chain.passage_id],
            content_tokens=31,
            stride=15,
        )
        for chain in selected
    ]
    overlap_cohort = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "stage": "STAGE5B_PLAN2_OVERLAP_HOLDOUT_COHORT",
        "dataset": dataset,
        "geometry": {
            **geometry,
            "k": 8,
            "latent_content_tokens": 136,
        },
        "selection": {
            "seed": cohort_seed,
            "method": (
                "same Stage 3A sha256 ordering; exclude frozen first 30 and take "
                f"the next {holdout_size} before observing inversion outcomes"
            ),
            "exploratory_documents_excluded": len(exploratory["manifest"]),
            "available_unused_documents": len(ordered) - len(exploratory["manifest"]),
            "holdout_documents": holdout_size,
        },
        "manifest": [chain.passage_id for chain in selected],
        "documents": overlap_documents,
    }
    nooverlap_documents = _build_nooverlap_documents(
        overlap_cohort,
        passages,
        tokenizer,
        max_standalone_content_tokens=31,
        expected_vectors_per_document=5,
        model_positions=32,
    )
    strata_by_passage = {
        str(document["passage_id"]): classify_nooverlap_boundary_stratum(document)
        for document in nooverlap_documents
    }
    strata = {
        name: [
            passage_id
            for passage_id in overlap_cohort["manifest"]
            if strata_by_passage[passage_id] == name
        ]
        for name in (
            "STRICT_FIXED_STABLE",
            "CAPACITY_SAFE_NO_SHIFT",
            "BOUNDARY_ADJUSTMENT_REQUIRED",
        )
    }
    nooverlap_manifest = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "stage": "STAGE5B_PLAN2_NOOVERLAP_HOLDOUT_MANIFEST",
        "dataset": dataset,
        "geometry": {
            "source_content_tokens": 136,
            "max_standalone_content_tokens": 31,
            "model_positions": 32,
            "nominal_source_stride_tokens": 31,
            "vectors_per_document": 5,
        },
        "manifest": list(overlap_cohort["manifest"]),
        "documents": nooverlap_documents,
        "pre_inversion_boundary_stratum_by_passage": strata_by_passage,
        "pre_inversion_strata": {
            name: {"documents": len(passage_ids), "passage_ids": passage_ids}
            for name, passage_ids in strata.items()
        },
        "stratum_definitions": {
            "STRICT_FIXED_STABLE": (
                "No source boundary shift and every standalone chunk ID sequence "
                "equals its whole-document source-ID slice."
            ),
            "CAPACITY_SAFE_NO_SHIFT": (
                "No source boundary shift is required, but at least one standalone "
                "chunk ID sequence differs from its source-ID slice."
            ),
            "BOUNDARY_ADJUSTMENT_REQUIRED": (
                "At least one nominal right boundary is moved left because the raw "
                "substring would otherwise exceed 31 standalone content IDs."
            ),
        },
    }

    output_dir = Path(str(stage5b["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap_path = output_dir / "stage5b_overlap_holdout_cohort.json"
    nooverlap_path = output_dir / "stage5b_nooverlap_holdout_manifest.json"
    _write_json_atomic(overlap_path, overlap_cohort)
    _write_json_atomic(nooverlap_path, nooverlap_manifest)
    shared_first_view_match = all(
        overlap["windows"][0]["model_input_ids"]
        == nooverlap["windows"][0]["model_input_ids"]
        for overlap, nooverlap in zip(
            overlap_documents, nooverlap_documents, strict=True
        )
    )
    payload = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE5B_HOLDOUT_MANIFEST_READY",
        "stage": "PLAN2_FROZEN_HOLDOUT_PREINVERSION",
        "frozen_inputs": {
            "exploratory_cohort": {
                "path": str(exploratory_path),
                "sha256": _sha256(exploratory_path),
            },
            "dataset_revision": dataset["revision"],
            "scan_limit": int(dataset["scan_limit"]),
            "cohort_seed": cohort_seed,
        },
        "selection": overlap_cohort["selection"],
        "scan": {
            "scanned_passages": len(passages),
            "raw_stable_k8_chains": len(stable_chains),
            "deduplication": deduplication,
        },
        "artifacts": {
            "overlap_cohort": {
                "path": str(overlap_path),
                "sha256": _sha256(overlap_path),
            },
            "nooverlap_manifest": {
                "path": str(nooverlap_path),
                "sha256": _sha256(nooverlap_path),
            },
        },
        "pre_inversion_strata": nooverlap_manifest["pre_inversion_strata"],
        "invariants": {
            "exploratory_and_holdout_disjoint": not bool(
                set(exploratory["manifest"]) & set(overlap_cohort["manifest"])
            ),
            "same_60_documents_for_overlap_and_nooverlap": (
                overlap_cohort["manifest"] == nooverlap_manifest["manifest"]
            ),
            "same_136_source_ids": all(
                overlap["prefixes"]["8"]["reference_ids"]
                == nooverlap["reference_ids"]
                for overlap, nooverlap in zip(
                    overlap_documents, nooverlap_documents, strict=True
                )
            ),
            "all_overlap_windows_roundtrip_stable": True,
            "all_nooverlap_inputs_capacity_safe": all(
                len(window["content_ids"]) <= 31
                for document in nooverlap_documents
                for window in document["windows"]
            ),
            "shared_first_view_identical": shared_first_view_match,
            "strata_fixed_before_inversion": True,
        },
        "planned_unique_views": {
            "overlap": holdout_size * 8,
            "nooverlap": holdout_size * 5,
            "shared_first_views": holdout_size,
            "total_after_exact_view_reuse": holdout_size * 12,
        },
        "vec2text_executed": False,
        "embeddings_generated": False,
        "method_implemented": False,
    }
    summary_path = output_dir / "stage5b_holdout_manifest.json"
    _write_json_atomic(summary_path, payload)
    return payload
