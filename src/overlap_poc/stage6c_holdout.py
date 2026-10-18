"""Freeze SHA-order positions 150..209 after the Stage 6B algorithm freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .data import NaturalQuestionsLoader, PassageLoader
from .diagnostic_experiment import load_diagnostic_config
from .stage1_strong_saturation import _write_json_atomic
from .stage3_longdoc_feasibility import _chain_document_payload
from .stage5_nooverlap import _build_nooverlap_documents
from .stage5b_holdout import classify_nooverlap_boundary_stratum
from .stage5d_holdout import _ordered_stable_chains


_IMPLEMENTATION_VERSION = "stage6c-final-unseen-cohort-v1"
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


def _validate_frozen_code(freeze: Mapping[str, Any]) -> None:
    for group, entries in freeze.get("code_hashes", {}).items():
        for name, entry in entries.items():
            path = Path(str(entry["path"]))
            if not path.is_file() or _sha256(path) != str(entry["sha256"]):
                raise RuntimeError(
                    f"Stage 6B frozen {group} code changed before cohort freeze: {name}"
                )


def _standalone_window_is_valid(
    window: Mapping[str, Any], tokenizer: Any, *, exact_content_tokens: int | None
) -> bool:
    content = tuple(int(value) for value in window["content_ids"])
    model_input = tuple(int(value) for value in window["model_input_ids"])
    attention = tuple(int(value) for value in window["attention_mask"])
    independent = tuple(
        int(value)
        for value in tokenizer(
            str(window["text"]), add_special_tokens=False, truncation=False
        )["input_ids"]
    )
    content_length_ok = (
        len(content) == exact_content_tokens
        if exact_content_tokens is not None
        else 0 < len(content) <= 31
    )
    nonpad = len(content) + 1
    return bool(
        content_length_ok
        and independent == content
        and len(model_input) == 32
        and len(attention) == 32
        and model_input[: len(content)] == content
        and model_input[len(content)] == int(tokenizer.eos_token_id)
        and attention[:nonpad] == (1,) * nonpad
        and attention[nonpad:] == (0,) * (32 - nonpad)
    )


def load_stage6c_holdout_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 6C cohort config must be a YAML mapping")

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
    resolved["resolved_algorithm_freeze"] = resolve(
        str(raw["algorithm_freeze"]["path"])
    )
    resolved["resolved_prior_cohorts"] = {
        name: {**entry, "path": resolve(str(entry["path"]))}
        for name, entry in raw["frozen_prior_cohorts"].items()
    }
    return resolved


def _validate_frozen_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if config.get("stage") != "stage6c_final_unseen_cohort_v1":
        raise RuntimeError("Stage 6C cohort stage identity changed")
    if config.get("frozen_after_stage6b") is not True:
        raise RuntimeError("Stage 6C cohort must be frozen after Stage 6B")
    base_path = Path(str(config["resolved_base_config"]))
    if _sha256(base_path) != str(config["base_config"]["sha256"]):
        raise RuntimeError("Stage 6C base config changed")
    freeze_path = Path(str(config["resolved_algorithm_freeze"]))
    if _sha256(freeze_path) != str(config["algorithm_freeze"]["sha256"]):
        raise RuntimeError("Stage 6B algorithm freeze changed")
    freeze = _load_json(freeze_path)
    if (
        freeze.get("status") != "STAGE6B_ALGORITHM_FROZEN"
        or freeze.get("freeze_closed") is not True
    ):
        raise RuntimeError("Stage 6B algorithm freeze is not closed")
    _validate_frozen_code(freeze)
    if freeze.get("artifact_hashes", {}).get("base_protocol", {}).get(
        "sha256"
    ) != str(config["base_config"]["sha256"]):
        raise RuntimeError("Stage 6C base config is not the Stage 6B frozen protocol")

    prior_values: list[dict[str, Any]] = []
    prior_ids: list[str] = []
    frozen_names = {
        "exploratory_0_29": "exploratory_overlap_cohort",
        "stage5bc_30_89": "stage5bc_overlap_cohort",
        "stage5d_90_149": "stage5d_overlap_cohort",
    }
    for name, entry in config["resolved_prior_cohorts"].items():
        frozen_name = frozen_names.get(name)
        if frozen_name is None or freeze.get("artifact_hashes", {}).get(
            frozen_name, {}
        ).get("sha256") != str(entry["sha256"]):
            raise RuntimeError(f"prior cohort is not bound by Stage 6B: {name}")
        path = Path(str(entry["path"]))
        if _sha256(path) != str(entry["sha256"]):
            raise RuntimeError(f"frozen prior cohort changed: {name}")
        value = _load_json(path)
        manifest = [str(item) for item in value.get("manifest", [])]
        if value.get("stage") != entry["expected_stage"]:
            raise RuntimeError(f"prior cohort stage changed: {name}")
        if len(manifest) != int(entry["expected_documents"]):
            raise RuntimeError(f"prior cohort document count changed: {name}")
        prior_values.append(value)
        prior_ids.extend(manifest)
    if len(prior_ids) != 150 or len(set(prior_ids)) != 150:
        raise RuntimeError("Stage 6C requires exactly 150 distinct prior documents")
    return prior_values


def run_stage6c_holdout_manifest(
    config: Mapping[str, Any],
    *,
    loader: PassageLoader | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Freeze the untouched reserve without running embeddings or Vec2Text."""

    from transformers import AutoTokenizer

    prior_values = _validate_frozen_inputs(config)
    geometry = dict(config["geometry"])
    if tuple(
        int(geometry[name])
        for name in (
            "content_tokens",
            "overlap_tokens",
            "stride_tokens",
            "model_positions",
            "source_tokens",
            "overlap_windows_per_document",
            "nooverlap_windows_per_document",
        )
    ) != (31, 16, 15, 32, 136, 8, 5):
        raise RuntimeError("Stage 6C geometry changed")
    if (
        int(config["prior_documents"]),
        int(config["selection_start_inclusive"]),
        int(config["selection_stop_exclusive"]),
        int(config["evaluation_documents"]),
    ) != (150, 150, 210, 60):
        raise RuntimeError("Stage 6C frozen SHA-order slice changed")

    models = dict(config["base"]["models"])
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        models["embedder_name"],
        revision=models.get("embedder_revision"),
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Stage 6C requires the pinned fast tokenizer")
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
    prior_ids = [
        str(passage_id)
        for value in prior_values
        for passage_id in value["manifest"]
    ]
    ordered_ids = [chain.passage_id for chain in ordered]
    if ordered_ids[:150] != prior_ids:
        raise RuntimeError("prior cohorts do not reproduce SHA-order positions 0..149")
    selected = list(ordered[150:210])
    if len(selected) != 60:
        raise RuntimeError("Stage 6C reserve slice does not contain 60 documents")
    selected_ids = [chain.passage_id for chain in selected]
    if set(selected_ids) & set(prior_ids):
        raise AssertionError("Stage 6C final cohort overlaps a prior cohort")

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
        "stage": "STAGE6C_PLAN2_OVERLAP_FINAL_UNSEEN_COHORT",
        "dataset": dataset,
        "geometry": {**config["base"]["geometry"], "k": 8, "latent_content_tokens": 136},
        "selection": {
            "seed": int(config["cohort_seed"]),
            "method": "same frozen SHA-256 order; exact positions 150..209",
            "start_inclusive": 150,
            "stop_exclusive": 210,
            "prior_documents_excluded": 150,
            "evaluation_documents": 60,
            "unused_reserve_after_evaluation": 0,
        },
        "manifest": selected_ids,
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
    nooverlap_manifest = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "stage": "STAGE6C_PLAN2_NOOVERLAP_FINAL_UNSEEN_MANIFEST",
        "dataset": dataset,
        "geometry": {
            "source_content_tokens": 136,
            "max_standalone_content_tokens": 31,
            "model_positions": 32,
            "nominal_source_stride_tokens": 31,
            "vectors_per_document": 5,
        },
        "manifest": selected_ids,
        "documents": nooverlap_documents,
        "pre_inversion_boundary_stratum_by_passage": strata_by_passage,
        "pre_inversion_strata": {
            name: {
                "documents": sum(value == name for value in strata_by_passage.values()),
                "passage_ids": [
                    passage_id
                    for passage_id in selected_ids
                    if strata_by_passage[passage_id] == name
                ],
            }
            for name in _STRATA
        },
    }

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap_path = output_dir / str(config["overlap_filename"])
    nooverlap_path = output_dir / str(config["nooverlap_filename"])
    _write_json_atomic(overlap_path, overlap_cohort)
    _write_json_atomic(nooverlap_path, nooverlap_manifest)
    invariants = {
        "algorithm_frozen_before_cohort_construction": True,
        "prior_prefix_is_exact_sha_positions_0_149": ordered_ids[:150] == prior_ids,
        "selection_is_exact_sha_positions_150_209": selected_ids == ordered_ids[150:210],
        "prior_and_final_disjoint": not bool(set(prior_ids) & set(selected_ids)),
        "same_documents_and_order": selected_ids == nooverlap_manifest["manifest"],
        "same_136_source_ids": all(
            len(overlap["prefixes"]["8"]["reference_ids"]) == 136
            and len(nooverlap["reference_ids"]) == 136
            and overlap["prefixes"]["8"]["reference_ids"]
            == nooverlap["reference_ids"]
            for overlap, nooverlap in zip(
                overlap_documents, nooverlap_documents, strict=True
            )
        ),
        "all_overlap_windows_roundtrip_stable": all(
            _standalone_window_is_valid(
                window, tokenizer, exact_content_tokens=31
            )
            for document in overlap_documents
            for window in document["windows"]
        ),
        "all_nooverlap_inputs_capacity_safe": all(
            _standalone_window_is_valid(
                window, tokenizer, exact_content_tokens=None
            )
            for document in nooverlap_documents
            for window in document["windows"]
        ),
        "expected_window_counts": all(
            len(overlap["windows"]) == 8 and len(nooverlap["windows"]) == 5
            for overlap, nooverlap in zip(
                overlap_documents, nooverlap_documents, strict=True
            )
        ),
        "nooverlap_windows_partition_reference": all(
            [
                int(token)
                for window in nooverlap["windows"]
                for token in window["source_ids"]
            ]
            == [int(token) for token in nooverlap["reference_ids"]]
            for nooverlap in nooverlap_documents
        ),
        "shared_first_view_identical": all(
            overlap["windows"][0]["model_input_ids"]
            == nooverlap["windows"][0]["model_input_ids"]
            for overlap, nooverlap in zip(
                overlap_documents, nooverlap_documents, strict=True
            )
        ),
        "vec2text_not_executed": True,
        "assembly_not_evaluated": True,
    }
    if not all(invariants.values()):
        raise RuntimeError("Stage 6C pre-inversion cohort invariant failed")
    payload = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE6C_FINAL_COHORT_FROZEN",
        "config": {
            "path": str(config["resolved_config"]),
            "sha256": _sha256(Path(str(config["resolved_config"]))),
        },
        "algorithm_freeze": {
            "path": str(config["resolved_algorithm_freeze"]),
            "sha256": str(config["algorithm_freeze"]["sha256"]),
        },
        "prior_cohorts": {
            name: {
                "path": str(entry["path"]),
                "sha256": str(entry["sha256"]),
                "documents": int(entry["expected_documents"]),
            }
            for name, entry in config["resolved_prior_cohorts"].items()
        },
        "selection": overlap_cohort["selection"],
        "scan": {
            "deduplicated_stable_k8_documents": len(ordered),
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
        "planned_views": {
            "logical_overlap": 480,
            "logical_nooverlap": 300,
            "shared": 60,
            "unique_after_exact_reuse": 720,
        },
        "invariants": invariants,
        "vec2text_executed": False,
        "assembly_evaluated": False,
    }
    _write_json_atomic(output_dir / str(config["summary_filename"]), payload)
    return payload
