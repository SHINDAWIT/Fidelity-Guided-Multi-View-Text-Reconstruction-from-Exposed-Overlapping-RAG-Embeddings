"""Create the Stage 6B immutable algorithm-freeze manifest.

This module reads only the declared Stage 6A and prior artifacts.  It neither
discovers nor reads any Stage 6C file, and it never invokes git.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import yaml


_STATUS = "STAGE6B_ALGORITHM_FROZEN"
_EXPECTED_ARTIFACTS = {
    "stage6a_result",
    "stage6a_trajectories",
    "stage6a_report",
    "stage6a_config",
    "exploratory_overlap_cohort",
    "exploratory_overlap_strong_cache",
    "exploratory_nooverlap_manifest",
    "exploratory_nooverlap_strong_cache",
    "base_protocol",
    "stage5bc_overlap_cohort",
    "stage5d_overlap_cohort",
}
_EXPECTED_CODE_GROUPS = {"inference", "evaluation", "orchestration"}
_EXPECTED_PROTOCOLS = {"stage6c_evaluation_protocol", "pyproject", "requirements"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping at {path}")
    return value


def load_stage6b_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path)

    def resolve(value: str) -> str:
        candidate = Path(value)
        return str(
            (candidate if candidate.is_absolute() else config_path.parent / candidate)
            .resolve()
        )

    resolved = deepcopy(raw)
    resolved["resolved_config"] = str(config_path)
    resolved["resolved_artifacts"] = {
        name: {**entry, "path": resolve(str(entry["path"]))}
        for name, entry in raw["artifacts"].items()
    }
    resolved["resolved_code"] = {
        group: {
            name: {**entry, "path": resolve(str(entry["path"]))}
            for name, entry in entries.items()
        }
        for group, entries in raw["code"].items()
    }
    resolved["resolved_protocols"] = {
        name: {**entry, "path": resolve(str(entry["path"]))}
        for name, entry in raw["protocols"].items()
    }
    resolved["resolved_output"] = {
        **raw["output"],
        "directory": resolve(str(raw["output"]["directory"])),
    }
    return resolved


def _validate_hashes(entries: Mapping[str, Mapping[str, Any]], label: str) -> None:
    for name, entry in entries.items():
        path = Path(str(entry["path"]))
        if not path.is_file():
            raise RuntimeError(f"Stage 6B {label} is missing: {name}")
        if _sha256(path) != str(entry["sha256"]):
            raise RuntimeError(f"Stage 6B {label} changed before freeze: {name}")


def _installed_versions(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_python = str(config["dependencies"]["python"])
    actual_python = platform.python_version()
    if actual_python != expected_python:
        raise RuntimeError(
            f"Stage 6B Python version changed: {actual_python} != {expected_python}"
        )
    packages: dict[str, str] = {}
    for package, expected in config["dependencies"]["packages"].items():
        try:
            actual = importlib.metadata.version(str(package))
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Stage 6B dependency is not installed: {package}") from error
        if actual != str(expected):
            raise RuntimeError(
                f"Stage 6B dependency version changed: {package}={actual}, expected {expected}"
            )
        packages[str(package)] = actual
    return {"python": actual_python, "packages": packages}


def _validate_algorithm(config: Mapping[str, Any], stage6a_config: Mapping[str, Any]) -> None:
    frozen = config["algorithm"]
    source = stage6a_config["pairwise_assembly"]
    geometry = stage6a_config["geometry"]
    exact = {
        "implementation_version": "stage6a-source-free-progressive-pairwise-assembly-v2",
        "method_name": source["name"],
        "merge_order": source["merge_order"],
        "alignment_algorithm": "constrained_semiglobal_token_levenshtein",
        "alignment_inputs": source["alignment_inputs"],
        "allowed_alignment_region": source["allowed_alignment_region"],
        "content_tokens": int(geometry["content_tokens"]),
        "stride_tokens": int(geometry["stride_tokens"]),
        "nominal_overlap_tokens": int(geometry["nominal_overlap_tokens"]),
        "allowed_overlap_width_tokens": [14, 15, 16, 17, 18],
        "max_boundary_shift": int(source["max_boundary_shift"]),
        "match_cost": 0.0,
        "substitution_cost": 1.0,
        "gap_cost": 1.0,
        "overlap_width_deviation_cost": float(source["deviation_cost"]),
        "boundary_objective": "edit_distance_plus_0.5_times_total_absolute_width_deviation_from_16",
        "boundary_tie_break_order": [
            "objective",
            "total_width_deviation",
            "edit_distance",
            "absolute_right_width_deviation",
            "absolute_left_width_deviation",
            "longer_right_width",
            "first_encountered_smaller_left_width",
        ],
        "token_alignment_traceback_priority": [
            "exact_match_diagonal",
            "substitution_diagonal",
            "left_token_to_gap",
            "gap_to_right_token",
        ],
        "progressive_left_boundary_fidelity": "immediately_preceding_original_window_local_score",
        "match_handling": "emit_once",
        "substitution_handling": "emit_token_from_fidelity_owner",
        "gap_handling": "emit_fidelity_owner_token_if_non_null_otherwise_omit",
        "fidelity_field": source["fidelity_field"],
        "fidelity_rule": source["fidelity_rule"],
        "fidelity_tie_tolerance": float(source["tie_tolerance"]),
        "fidelity_tie_break": source["tie_rule"],
        "multiplicity_3_endpoint_rule": source["multiplicity_3_endpoint_rule"],
        "third_view_handling": "multiplicity_3_endpoints_keep_lower_window_index",
        "global_method": "none",
        "learned_parameters": "none",
        "ground_truth_or_source_position_used_by_inference": source[
            "ground_truth_or_source_position_used_by_inference"
        ],
        "candidate_local_score_definition": "cosine_similarity_of_final_hypothesis_embedding_to_target_embedding",
        "candidate_top1_invariant": "public_top1_exactly_matches_instrumented_argmax_local_score",
    }
    if dict(frozen) != exact:
        changed = sorted(
            key for key in set(frozen) | set(exact) if frozen.get(key) != exact.get(key)
        )
        raise RuntimeError(f"Stage 6B exact algorithm rule changed: {changed}")


def _validate_protocol_pins(config: Mapping[str, Any], base: Mapping[str, Any]) -> None:
    for key, value in config["dataset"].items():
        if base["dataset"].get(key) != value:
            raise RuntimeError(f"Stage 6B dataset pin changed: {key}")
    for key in (
        "embedder_name",
        "embedder_revision",
        "inversion_checkpoint",
        "inversion_revision",
        "corrector_checkpoint",
        "corrector_revision",
    ):
        if base["models"].get(key) != config["models"][key]:
            raise RuntimeError(f"Stage 6B models pin changed: {key}")
    expected_strong = {
        "strong_setting": "correction50+sbeam8",
        "correction_steps": 50,
        "sequence_beam_width": 8,
        "top_n": 1,
        "maximum_model_positions": 32,
    }
    for key, expected in expected_strong.items():
        if config["models"].get(key) != expected:
            raise RuntimeError(f"Stage 6B strong inversion setting changed: {key}")
    tokenizer = config["tokenizer"]
    if tokenizer["name"] != base["models"]["embedder_name"]:
        raise RuntimeError("Stage 6B tokenizer name differs from the embedder tokenizer")
    if tokenizer["revision"] != base["models"]["embedder_revision"]:
        raise RuntimeError("Stage 6B tokenizer revision changed")
    expected_tokenizer = {
        "implementation": "transformers.AutoTokenizer",
        "reconstructed_text_retokenization": {
            "add_special_tokens": False,
            "truncation": False,
            "eos_and_pad_removed_from_content_ids": True,
        },
        "fast_required": True,
        "content_tokens": 31,
        "eos_tokens": 1,
    }
    for key, expected in expected_tokenizer.items():
        if tokenizer.get(key) != expected:
            raise RuntimeError(f"Stage 6B tokenizer rule changed: {key}")


def _validated_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("stage") != "stage6b_algorithm_freeze_v1":
        raise RuntimeError("Stage 6B config stage identity changed")
    if config.get("frozen_before_stage6c_outputs_inspected") is not True:
        raise RuntimeError("Stage 6B was not declared frozen before Stage 6C inspection")
    if set(config["resolved_artifacts"]) != _EXPECTED_ARTIFACTS:
        raise RuntimeError("Stage 6B declared artifact set changed")
    if set(config["resolved_code"]) != _EXPECTED_CODE_GROUPS:
        raise RuntimeError("Stage 6B code groups changed")
    if set(config["resolved_protocols"]) != _EXPECTED_PROTOCOLS:
        raise RuntimeError("Stage 6B protocol set changed")
    _validate_hashes(config["resolved_artifacts"], "artifact")
    for group, entries in config["resolved_code"].items():
        if not entries:
            raise RuntimeError(f"Stage 6B {group} code set is empty")
        _validate_hashes(entries, f"{group} code")
    _validate_hashes(config["resolved_protocols"], "protocol")

    artifacts = config["resolved_artifacts"]
    stage6a_result = _load_json(Path(artifacts["stage6a_result"]["path"]))
    trajectories = _load_json(Path(artifacts["stage6a_trajectories"]["path"]))
    stage6a_config = _load_yaml(Path(artifacts["stage6a_config"]["path"]))
    base = _load_yaml(Path(artifacts["base_protocol"]["path"]))
    if stage6a_result.get("status") != "STAGE6A_SOURCE_FREE_ASSEMBLY_COMPLETE":
        raise RuntimeError("Stage 6A result is incomplete")
    if stage6a_result.get("gate", {}).get("state") != "SOURCE_FREE_ASSEMBLY_READY":
        raise RuntimeError("Stage 6A did not authorize algorithm freeze")
    if stage6a_result.get("implementation_version") != config["algorithm"][
        "implementation_version"
    ]:
        raise RuntimeError("Stage 6A implementation version changed")
    if stage6a_result.get("config", {}).get("sha256") != artifacts["stage6a_config"][
        "sha256"
    ]:
        raise RuntimeError("Stage 6A result points to another config")
    inference_hash = config["resolved_code"]["inference"][
        "stage6a_source_free_assembly"
    ]["sha256"]
    if stage6a_result.get("implementation_sha256") != inference_hash:
        raise RuntimeError("Stage 6A result points to another implementation")
    for group in ("inference", "evaluation"):
        for name, entry in config["resolved_code"][group].items():
            recorded = stage6a_result.get("behavior_file_sha256", {}).get(name)
            if recorded is not None and recorded != entry["sha256"]:
                raise RuntimeError(f"Stage 6A behavior hash changed: {name}")
    if trajectories.get("config_sha256") != artifacts["stage6a_config"]["sha256"]:
        raise RuntimeError("Stage 6A trajectories point to another config")
    if trajectories.get("implementation_sha256") != inference_hash:
        raise RuntimeError("Stage 6A trajectories point to another implementation")
    if len(trajectories.get("trajectories", [])) != 30:
        raise RuntimeError("Stage 6A trajectory document count changed")
    for name in (
        "exploratory_overlap_cohort",
        "exploratory_overlap_strong_cache",
        "exploratory_nooverlap_manifest",
        "exploratory_nooverlap_strong_cache",
    ):
        recorded = stage6a_result.get("inputs", {}).get(name, {}).get("sha256")
        if recorded != artifacts[name]["sha256"]:
            raise RuntimeError(f"Stage 6A prior input linkage changed: {name}")
    repository = config["repository"]
    if (
        repository.get("commit") is not None
        or repository.get("commit_status") != "unavailable_not_a_git_worktree"
        or repository.get("git_probe_performed_by_freeze") is not False
    ):
        raise RuntimeError("Stage 6B repository-unavailable state changed")
    if stage6a_result.get("repository_commit") is not None or stage6a_result.get(
        "repository_commit_status"
    ) != repository["commit_status"]:
        raise RuntimeError("Stage 6B repository state differs from Stage 6A")
    _validate_algorithm(config, stage6a_config)
    _validate_protocol_pins(config, base)
    versions = _installed_versions(config)
    return {
        "stage6a_result": stage6a_result,
        "dependency_versions": versions,
    }


def build_stage6b_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all declared inputs and return the immutable freeze payload."""

    validated = _validated_sources(config)
    artifact_hashes = {
        name: {
            "path": str(entry["path"]),
            "sha256": str(entry["sha256"]),
        }
        for name, entry in config["resolved_artifacts"].items()
    }
    code_hashes = {
        group: {
            name: {
                "path": str(entry["path"]),
                "sha256": str(entry["sha256"]),
            }
            for name, entry in entries.items()
        }
        for group, entries in config["resolved_code"].items()
    }
    protocol_hashes = {
        name: str(entry["sha256"])
        for name, entry in config["resolved_protocols"].items()
    }
    stage6a_result = validated["stage6a_result"]
    return {
        "schema_version": 1,
        "status": _STATUS,
        "freeze_closed": True,
        "freeze_target": str(config["freeze_target"]),
        "frozen_before_stage6c_outputs_inspected": True,
        "frozen_before_stage6c_inversion": True,
        "config": {
            "path": str(config["resolved_config"]),
            "sha256": _sha256(Path(str(config["resolved_config"]))),
        },
        "artifact_hashes": artifact_hashes,
        "code_hashes": code_hashes,
        "protocol_hashes": protocol_hashes,
        "protocol_files": {
            name: {
                "path": str(entry["path"]),
                "sha256": str(entry["sha256"]),
            }
            for name, entry in config["resolved_protocols"].items()
        },
        "algorithm": deepcopy(config["algorithm"]),
        "evaluation": deepcopy(config["evaluation"]),
        "dataset": deepcopy(config["dataset"]),
        "models": deepcopy(config["models"]),
        "tokenizer": deepcopy(config["tokenizer"]),
        "dependency_versions": validated["dependency_versions"],
        "repository": deepcopy(config["repository"]),
        "stage6a_authorization": {
            "status": stage6a_result["status"],
            "gate_state": stage6a_result["gate"]["state"],
            "development_only": bool(stage6a_result["gate"]["development_only"]),
            "paper_claim_authorized": bool(
                stage6a_result["gate"]["paper_claim_authorized"]
            ),
        },
        "stage6c_artifacts_read": False,
        "git_probe_performed": False,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_stage6b_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = build_stage6b_freeze(config)
    output = config["resolved_output"]
    _write_json_atomic(
        Path(str(output["directory"])) / str(output["filename"]), payload
    )
    return payload
