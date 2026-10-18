"""Validation and snapshotting for the pre-inversion Stage 5B protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .stage1_strong_saturation import _write_json_atomic


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_stage5b_protocol(
    config_path: str | Path, *, output_path: str | Path | None = None
) -> dict[str, Any]:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 5B protocol must be a YAML mapping")
    if raw.get("frozen_before_strong_inversion") is not True:
        raise RuntimeError("Stage 5B protocol was not frozen before inversion")
    validated_artifacts = {}
    loaded_artifacts = {}
    for name, entry in raw["artifacts"].items():
        candidate = Path(str(entry["path"]))
        path = candidate if candidate.is_absolute() else config_path.parent / candidate
        path = path.resolve()
        actual = _sha256(path)
        expected = str(entry["sha256"])
        if actual != expected:
            raise RuntimeError(
                f"Stage 5B frozen artifact hash mismatch for {name}: "
                f"{actual} != {expected}"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        validated_artifacts[name] = {"path": str(path), "sha256": actual}
        loaded_artifacts[name] = value

    holdout = loaded_artifacts["holdout_summary"]
    perfect = loaded_artifacts["perfect_reassembly_control"]
    if holdout["status"] != "STAGE5B_HOLDOUT_MANIFEST_READY":
        raise RuntimeError("Stage 5B holdout is not frozen")
    if perfect["status"] != "PERFECT_RECONSTRUCTION_REASSEMBLY_CONTROL_COMPLETE":
        raise RuntimeError("Stage 5B perfect-input control is incomplete")
    expected_strata = {
        str(name): int(count) for name, count in raw["cohort"]["strata"].items()
    }
    actual_strata = {
        str(name): int(value["documents"])
        for name, value in holdout["pre_inversion_strata"].items()
    }
    if actual_strata != expected_strata:
        raise RuntimeError("Stage 5B frozen stratum sizes changed")
    if int(perfect["documents"]) != int(raw["cohort"]["documents"]):
        raise RuntimeError("Stage 5B perfect control cohort size changed")

    payload = {
        "schema_version": 1,
        "status": "STAGE5B_EVALUATION_PROTOCOL_FROZEN",
        "protocol_version": raw["protocol_version"],
        "protocol_path": str(config_path.resolve()),
        "protocol_sha256": _sha256(config_path),
        "validated_artifacts": validated_artifacts,
        "cohort": raw["cohort"],
        "strong_reproduction_setting": raw["strong_reproduction_setting"],
        "evaluation": raw["evaluation"],
        "stratified_analysis": raw["stratified_analysis"],
        "gate": raw["gate"],
        "strong_inversion_executed": False,
        "method_implementation_authorized": False,
    }
    if output_path is not None:
        _write_json_atomic(Path(output_path), payload)
    return payload
