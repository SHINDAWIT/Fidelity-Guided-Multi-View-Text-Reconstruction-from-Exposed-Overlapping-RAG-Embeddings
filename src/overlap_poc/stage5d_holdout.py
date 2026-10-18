"""Freeze a fresh Stage 5D assembly evaluation cohort before inversion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

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
from .stage5b_holdout import (
    classify_nooverlap_boundary_stratum,
    select_after_frozen_prefixes,
)


_IMPLEMENTATION_VERSION = "stage5d-fresh-holdout-v1"
_STRATA = (
    "STRICT_FIXED_STABLE",
    "CAPACITY_SAFE_NO_SHIFT",
    "BOUNDARY_ADJUSTMENT_REQUIRED",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def load_stage5d_holdout_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 5D holdout config must be a YAML mapping")

    def resolve(value: str) -> str:
        candidate = Path(value)
        return str(
            (candidate if candidate.is_absolute() else config_path.parent / candidate)
            .resolve()
        )

    resolved = dict(raw)
    resolved["base"] = load_diagnostic_config(resolve(str(raw["base_config"])))
    resolved["resolved_config"] = str(config_path.resolve())
    for name, entry in raw["frozen_prior_cohorts"].items():
        resolved[f"resolved_{name}"] = resolve(str(entry["path"]))
    resolved["resolved_stage5c_result"] = resolve(str(raw["stage5c_result"]["path"]))
    return resolved


def _ordered_stable_chains(
    config: Mapping[str, Any],
    *,
    loader: PassageLoader,
    tokenizer: Any,
) -> tuple[list[StableTokenChain], dict[str, int], list[Any], dict[str, Any]]:
    dataset = dict(config["base"]["dataset"])
    passages = loader.load(int(dataset["scan_limit"]))
    stable: list[StableTokenChain] = []
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
            stable.append(search.chain)
    deduplicated, deduplication = _deduplicate_chains(stable)
    expected = int(config["expected_deduplicated_k8_documents"])
    if len(deduplicated) != expected:
        raise RuntimeError(f"pinned K=8 count changed: {len(deduplicated)} != {expected}")
    seed = int(config["cohort_seed"])
    ordered = sorted(
        deduplicated,
        key=lambda chain: (
            _cohort_digest(
                chain,
                dataset_revision=dataset["revision"],
                cohort_seed=seed,
                max_k=8,
            ),
            chain.passage_id,
        ),
    )
    return ordered, content_counts, passages, deduplication


def run_stage5d_holdout_manifest(
    config: Mapping[str, Any],
    *,
    loader: PassageLoader | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Freeze the next 60 ordered documents; never generate embeddings."""

    from transformers import AutoTokenizer

    if config.get("frozen_before_inversion") is not True:
        raise RuntimeError("Stage 5D cohort was not declared frozen before inversion")
    geometry = dict(config["base"]["geometry"])
    if (
        int(geometry["content_tokens"]),
        int(geometry["overlap_tokens"]),
        int(geometry["stride_tokens"]),
        int(geometry["model_positions"]),
    ) != (31, 16, 15, 32):
        raise RuntimeError("Stage 5D geometry must remain 31/16/15/32")

    prior_values = []
    for name, entry in config["frozen_prior_cohorts"].items():
        path = Path(str(config[f"resolved_{name}"]))
        if _sha256(path) != str(entry["sha256"]):
            raise RuntimeError(f"frozen prior cohort changed: {name}")
        prior_values.append(_load_json(path))
    stage5c_path = Path(str(config["resolved_stage5c_result"]))
    if _sha256(stage5c_path) != str(config["stage5c_result"]["sha256"]):
        raise RuntimeError("frozen Stage 5C result changed")
    if _load_json(stage5c_path).get("gate", {}).get("state") != "PASS":
        raise RuntimeError("Stage 5D requires the frozen Stage 5C PASS")

    models = dict(config["base"]["models"])
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        models["embedder_name"],
        revision=models.get("embedder_revision"),
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Stage 5D Plan-2 construction requires a fast tokenizer")
    dataset = dict(config["base"]["dataset"])
    loader = loader or NaturalQuestionsLoader(
        dataset_name=dataset["name"],
        dataset_config=dataset.get("config"),
        split=dataset["split"],
        revision=dataset["revision"],
        max_words=None,
        streaming=bool(dataset.get("streaming", True)),
    )
    ordered, content_counts, passages, deduplication = _ordered_stable_chains(
        config, loader=loader, tokenizer=tokenizer
    )
    prior_manifests = [list(value["manifest"]) for value in prior_values]
    evaluation_size = int(config["evaluation_documents"])
    selected = select_after_frozen_prefixes(
        ordered, prior_manifests, evaluation_size=evaluation_size
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
        "stage": "STAGE5D_PLAN2_OVERLAP_FRESH_EVALUATION_COHORT",
        "dataset": dataset,
        "geometry": {**geometry, "k": 8, "latent_content_tokens": 136},
        "selection": {
            "seed": int(config["cohort_seed"]),
            "method": "same frozen SHA-256 order; take positions 90..149",
            "prior_documents_excluded": sum(map(len, prior_manifests)),
            "evaluation_documents": evaluation_size,
            "unused_reserve_after_evaluation": len(ordered)
            - sum(map(len, prior_manifests))
            - evaluation_size,
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
        for name in _STRATA
    }
    nooverlap_manifest = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "stage": "STAGE5D_PLAN2_NOOVERLAP_FRESH_EVALUATION_MANIFEST",
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
            name: {"documents": len(ids), "passage_ids": ids}
            for name, ids in strata.items()
        },
    }

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap_path = output_dir / str(config["overlap_filename"])
    nooverlap_path = output_dir / str(config["nooverlap_filename"])
    _write_json_atomic(overlap_path, overlap_cohort)
    _write_json_atomic(nooverlap_path, nooverlap_manifest)
    prior_ids = {passage_id for manifest in prior_manifests for passage_id in manifest}
    payload = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE5D_FRESH_EVALUATION_MANIFEST_READY",
        "config": {"path": str(config["resolved_config"]), "sha256": _sha256(Path(str(config["resolved_config"])))},
        "selection": overlap_cohort["selection"],
        "scan": {
            "scanned_passages": int(dataset["scan_limit"]),
            "deduplicated_stable_k8_documents": len(ordered),
            "deduplication": deduplication,
        },
        "artifacts": {
            "overlap_cohort": {"path": str(overlap_path), "sha256": _sha256(overlap_path)},
            "nooverlap_manifest": {"path": str(nooverlap_path), "sha256": _sha256(nooverlap_path)},
        },
        "pre_inversion_strata": nooverlap_manifest["pre_inversion_strata"],
        "invariants": {
            "prior_and_evaluation_disjoint": not bool(prior_ids & set(overlap_cohort["manifest"])),
            "same_documents_and_order": overlap_cohort["manifest"] == nooverlap_manifest["manifest"],
            "same_136_source_ids": all(
                overlap["prefixes"]["8"]["reference_ids"] == nooverlap["reference_ids"]
                for overlap, nooverlap in zip(overlap_documents, nooverlap_documents, strict=True)
            ),
            "all_overlap_windows_roundtrip_stable": True,
            "all_nooverlap_inputs_capacity_safe": all(
                len(window["content_ids"]) <= 31
                for document in nooverlap_documents
                for window in document["windows"]
            ),
            "shared_first_view_identical": all(
                overlap["windows"][0]["model_input_ids"] == nooverlap["windows"][0]["model_input_ids"]
                for overlap, nooverlap in zip(overlap_documents, nooverlap_documents, strict=True)
            ),
            "method_not_evaluated_before_freeze": True,
        },
        "planned_views": {
            "logical_overlap": evaluation_size * 8,
            "logical_nooverlap": evaluation_size * 5,
            "shared": evaluation_size,
            "unique_after_exact_reuse": evaluation_size * 12,
        },
        "vec2text_executed": False,
        "assembly_evaluated": False,
    }
    _write_json_atomic(output_dir / str(config["summary_filename"]), payload)
    return payload
