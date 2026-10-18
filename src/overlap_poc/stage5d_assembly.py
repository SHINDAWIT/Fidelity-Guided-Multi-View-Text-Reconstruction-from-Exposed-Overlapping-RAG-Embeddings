"""Confirmatory Stage 5D source-oblivious document assembly evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import time
from typing import Any, Mapping, Sequence

import yaml

from .diagnostics import (
    align_token_ids,
    stitch_candidate_sequence_constrained,
    stitch_candidate_sequence_fidelity_constrained,
)
from .reconstruction import Candidate
from .stage1_strong_saturation import _candidate_from_dict, _write_json_atomic
from .stage3_longdoc_strong import _percentile
from .stage5b_evaluate import nooverlap_single_view_source_row
from .stage5c_fidelity_selector import _score_document


_IMPLEMENTATION_VERSION = "stage5d-source-oblivious-assembly-v1"
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


def load_stage5d_assembly_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 5D assembly config must be a YAML mapping")

    def resolve(value: str) -> str:
        candidate = Path(value)
        return str(
            (candidate if candidate.is_absolute() else config_path.parent / candidate)
            .resolve()
        )

    resolved = dict(raw)
    resolved["resolved_config"] = str(config_path.resolve())
    resolved["resolved_inputs"] = {
        name: {**entry, "path": resolve(str(entry["path"]))}
        for name, entry in raw["inputs"].items()
    }
    resolved["resolved_strong_cache"] = resolve(str(raw["strong_cache"]))
    return resolved


def _validate_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if config.get("frozen_before_candidate_outputs_evaluated") is not True:
        raise RuntimeError("Stage 5D assembly was not frozen before candidate evaluation")
    inputs = {}
    for name, entry in config["resolved_inputs"].items():
        path = Path(str(entry["path"]))
        if _sha256(path) != str(entry["sha256"]):
            raise RuntimeError(f"Stage 5D frozen input changed: {name}")
        if path.suffix == ".json":
            inputs[name] = _load_json(path)
        elif path.suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"expected YAML mapping at {path}")
            inputs[name] = value
    summary = inputs["holdout_summary"]
    if summary.get("status") != "STAGE5D_FRESH_EVALUATION_MANIFEST_READY":
        raise RuntimeError("Stage 5D holdout manifest is incomplete")
    if not all(bool(value) for value in summary["invariants"].values()):
        raise RuntimeError("Stage 5D holdout invariant failed")
    if inputs["stage5c_result"].get("gate", {}).get("state") != "PASS":
        raise RuntimeError("Stage 5D requires Stage 5C PASS")

    stage4 = inputs["stage4_frozen_stitcher"]["stronger_stitcher"]
    existing = config["existing_assembly"]
    new = config["new_assembly"]
    frozen_values = (
        int(stage4["nominal_overlap_tokens"]),
        int(stage4["max_boundary_shift"]),
        float(stage4["deviation_cost"]),
    )
    if frozen_values != (
        int(config["geometry"]["nominal_overlap_tokens"]),
        int(existing["max_boundary_shift"]),
        float(existing["deviation_cost"]),
    ) or frozen_values != (
        int(config["geometry"]["nominal_overlap_tokens"]),
        int(new["max_boundary_shift"]),
        float(new["deviation_cost"]),
    ):
        raise RuntimeError("Stage 5D changed the frozen Stage 4 alignment parameters")
    selector = config["selector"]
    expected_selector = {
        "rule": "maximum_window_embedding_fidelity",
        "fidelity_field": "Candidate.local_score",
        "tie_rule": "lower_window_index",
        "multiplicity_3_rule": "lower_window_index",
        "threshold_or_tuning": "none",
    }
    if any(selector.get(name) != value for name, value in expected_selector.items()):
        raise RuntimeError("Stage 5D changed the frozen Stage 5C selector")

    strong_path = Path(str(config["resolved_strong_cache"]))
    strong = _load_json(strong_path)
    if strong.get("status") != "STAGE5D_STRONG_TOP1_CACHE_COMPLETE":
        raise RuntimeError("Stage 5D strong cache is incomplete")
    if strong.get("strong_reproduction_setting") != "correction50+sbeam8":
        raise RuntimeError("Stage 5D candidate generation changed")
    if strong.get("freeze_sha256") != config["inputs"]["holdout_summary"]["sha256"]:
        raise RuntimeError("Stage 5D strong cache refers to another cohort freeze")
    if strong.get("config", {}).get("sha256") != config["inputs"]["strong_generation_config"]["sha256"]:
        raise RuntimeError("Stage 5D strong cache refers to another generation config")
    if len(strong.get("top1_candidates", [])) != 720:
        raise RuntimeError("Stage 5D strong candidate count changed")
    records = list(strong.get("records", []))
    if len(records) != 720 or not all(
        bool(record.get("public_top1_exact_id_match")) for record in records
    ):
        raise RuntimeError("Stage 5D public/instrumented Top-1 invariant failed")
    if any(int(record.get("embed_call_count", -1)) != 51 for record in records):
        raise RuntimeError("Stage 5D correction budget changed")
    if any(
        int(record.get("search_hypothesis_rows_embedded", -1)) != 3145
        for record in records
    ):
        raise RuntimeError("Stage 5D search-row budget changed")
    inputs["strong_cache"] = strong
    return inputs


def _candidates_for_document(
    document: Mapping[str, Any],
    *,
    policy: str,
    candidates: Sequence[Candidate],
    alias_to_index: Mapping[str, int],
) -> tuple[Candidate, ...]:
    passage_id = str(document["passage_id"])
    return tuple(
        candidates[int(alias_to_index[f"{passage_id}:{policy}:{window['window_index']}"])]
        for window in document["windows"]
    )


def fixed_coordinate_fidelity_assembly(
    candidates: Sequence[Candidate],
    *,
    content_tokens: int = 31,
    stride: int = 15,
    tie_tolerance: float = 1e-12,
) -> tuple[int, ...]:
    """Stage 5C fixed-index realization, expressed without any source input."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    output_length = content_tokens + stride * (len(candidates) - 1)
    output: list[int] = []
    for position in range(output_length):
        observations: list[tuple[int, int | None, float]] = []
        for view_index, candidate in enumerate(candidates):
            local_position = position - stride * view_index
            if 0 <= local_position < content_tokens:
                prediction = (
                    int(candidate.content_ids[local_position])
                    if local_position < len(candidate.content_ids)
                    else None
                )
                observations.append((view_index, prediction, float(candidate.local_score)))
        if len(observations) == 2:
            top_score = max(value[2] for value in observations)
            selected = min(
                (value for value in observations if top_score - value[2] <= tie_tolerance),
                key=lambda value: value[0],
            )
        else:
            selected = min(observations, key=lambda value: value[0])
        if selected[1] is not None:
            output.append(int(selected[1]))
    return tuple(output)


def _document_score(prediction: Sequence[int], reference: Sequence[int]) -> dict[str, Any]:
    alignment = align_token_ids(prediction, reference)
    exact_matches = len(alignment.exact_matches)
    return {
        "prediction_length": len(prediction),
        "source_token_count": len(reference),
        "exact_matched_token_count": exact_matches,
        "token_recovery": exact_matches / len(reference),
        "edit_distance": alignment.distance,
        "normalized_edit_similarity": max(0.0, 1.0 - alignment.distance / len(reference)),
        "document_exact": tuple(prediction) == tuple(reference),
        "substitution_count": len(alignment.substitutions),
        "prediction_insertion_count": len(alignment.prediction_insertions),
        "reference_deletion_count": len(alignment.reference_deletions),
        "exact_reference_positions": [
            reference_index for _prediction_index, reference_index in alignment.exact_matches
        ],
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = (
        "nooverlap",
        "overlap_fixed_owner",
        "overlap_fidelity_fixed_coordinate",
        "overlap_fidelity_geometry_aware",
    )
    result = {"documents": len(rows)}
    for method in methods:
        values = [row[method] for row in rows]
        total = sum(int(value["source_token_count"]) for value in values)
        correct = sum(int(value["exact_matched_token_count"]) for value in values)
        result[method] = {
            "token_recovery": correct / total,
            "normalized_edit_similarity": mean(float(value["normalized_edit_similarity"]) for value in values),
            "document_exact_fraction": mean(bool(value["document_exact"]) for value in values),
            "total_edit_distance": sum(int(value["edit_distance"]) for value in values),
            "prediction_insertions": sum(int(value["prediction_insertion_count"]) for value in values),
            "reference_deletions": sum(int(value["reference_deletion_count"]) for value in values),
            "substitutions": sum(int(value["substitution_count"]) for value in values),
            "mean_prediction_length": mean(int(value["prediction_length"]) for value in values),
        }
    source_total = sum(int(row["source_aligned"]["source_token_count"]) for row in rows)
    nooverlap_source = sum(int(row["source_aligned"]["nooverlap_correct"]) for row in rows)
    selector_source = sum(int(row["source_aligned"]["selector_correct"]) for row in rows)
    oracle_source = sum(int(row["source_aligned"]["oracle_correct"]) for row in rows)
    result["diagnostic_upper_bounds"] = {
        "nooverlap_source_aligned_recovery": nooverlap_source / source_total,
        "fidelity_selector_source_aligned_recovery": selector_source / source_total,
        "best_of_views_oracle_source_aligned_recovery": oracle_source / source_total,
    }
    new_recovery = result["overlap_fidelity_geometry_aware"]["token_recovery"]
    nooverlap_recovery = result["nooverlap"]["token_recovery"]
    source_selector = selector_source / source_total
    result["transfer"] = {
        "source_aligned_selector_gain_over_nooverlap": source_selector - nooverlap_source / source_total,
        "assembly_loss_from_source_aligned_selector": source_selector - new_recovery,
        "end_to_end_gain_over_nooverlap": new_recovery - nooverlap_recovery,
    }
    return result


def _bootstrap(
    rows: Sequence[Mapping[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    metric_names = (
        "new_minus_nooverlap",
        "new_minus_fixed_owner",
        "new_minus_fixed_coordinate",
        "source_aligned_selector_minus_nooverlap",
        "source_aligned_selector_minus_new",
    )
    rng = random.Random(seed)
    samples = {name: [] for name in metric_names}
    for _ in range(iterations):
        selected = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        for name in metric_names:
            samples[name].append(mean(float(row["deltas"][name]) for row in selected))
    return {
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "fresh frozen document",
        "mean_delta": {
            name: mean(float(row["deltas"][name]) for row in rows)
            for name in metric_names
        },
        "ci95": {
            name: [_percentile(values, 0.025), _percentile(values, 0.975)]
            for name, values in samples.items()
        },
    }


def _gate(bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    primary = bootstrap["ci95"]["new_minus_nooverlap"]
    assembly = bootstrap["ci95"]["new_minus_fixed_coordinate"]
    if float(primary[0]) > 0.0:
        state = "PASS"
    elif float(assembly[0]) > 0.0:
        state = "PARTIAL"
    else:
        state = "FAIL"
    return {
        "state": state,
        "primary_new_minus_nooverlap_ci95": primary,
        "assembly_new_minus_fixed_coordinate_ci95": assembly,
        "selector_or_assembly_tuning_authorized": False,
        "interpretation": {
            "PASS": "The frozen source-oblivious fidelity assembly improves no-overlap on untouched documents.",
            "PARTIAL": "The assembly loss is reduced, but overlap does not reliably exceed no-overlap.",
            "FAIL": "The source-aligned selector gain does not transfer through the frozen source-oblivious assembly.",
        }[state],
    }


def _write_report(output_dir: Path, result: Mapping[str, Any], filename: str) -> None:
    main = result["main_result"]
    transfer = main["transfer"]
    bootstrap = result["bootstrap"]
    lines = [
        "# Stage 5D — source-oblivious fidelity assembly",
        "",
        f"- Gate: `{result['gate']['state']}`",
        "- Evaluation cohort: untouched SHA-256 positions 90..149; Stage 3 and Stage 5B/5C documents excluded.",
        "- Assembly inputs: prediction IDs, frozen local_score, order, and 31/15/16 geometry only.",
        "",
        "| Method | Token recovery | Normalized edit similarity | Document exact |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "nooverlap": "No-overlap single-view",
        "overlap_fixed_owner": "Overlap fixed-owner existing assembly",
        "overlap_fidelity_fixed_coordinate": "Overlap fidelity + fixed-coordinate assembly",
        "overlap_fidelity_geometry_aware": "Overlap fidelity + geometry-aware assembly",
    }
    for name, label in labels.items():
        value = main[name]
        lines.append(
            f"| {label} | {value['token_recovery']:.6f} | "
            f"{value['normalized_edit_similarity']:.6f} | {value['document_exact_fraction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Gain transfer",
            "",
            f"- Source-aligned selector gain: {transfer['source_aligned_selector_gain_over_nooverlap']:+.6f}",
            f"- Loss during assembly: {transfer['assembly_loss_from_source_aligned_selector']:+.6f}",
            f"- Final end-to-end gain: {transfer['end_to_end_gain_over_nooverlap']:+.6f}",
            f"- Final gain CI95: {bootstrap['ci95']['new_minus_nooverlap']}",
            f"- New minus fixed-coordinate CI95: {bootstrap['ci95']['new_minus_fixed_coordinate']}",
            "",
            "## Diagnostic upper bounds",
            "",
        ]
    )
    for name, value in main["diagnostic_upper_bounds"].items():
        lines.append(f"- {name}: {value:.6f}")
    lines.extend(["", "## Residual loss", ""])
    residual = result["failure_analysis"]
    lines.append(
        f"- New assembly edits: substitutions={residual['substitutions']}, "
        f"insertions={residual['prediction_insertions']}, deletions={residual['reference_deletions']}."
    )
    lines.append("- Worst documents:")
    for row in residual["worst_documents"]:
        lines.append(
            f"  - `{row['passage_id']}` ({row['pre_inversion_stratum']}): "
            f"recovery={row['token_recovery']:.4f}, edits={row['edit_distance']}"
        )
    (output_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage5d_assembly(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = _validate_inputs(config)
    overlap = inputs["overlap_cohort"]
    nooverlap = inputs["nooverlap_manifest"]
    strong = inputs["strong_cache"]
    candidates = tuple(_candidate_from_dict(value) for value in strong["top1_candidates"])
    aliases = {
        str(alias): int(index)
        for alias, index in strong["alias_to_canonical_index"].items()
    }
    nooverlap_by_id = {
        str(document["passage_id"]): document for document in nooverlap["documents"]
    }
    strata = {
        str(key): str(value)
        for key, value in nooverlap["pre_inversion_boundary_stratum_by_passage"].items()
    }
    geometry = config["geometry"]
    overlap_tokens = int(geometry["nominal_overlap_tokens"])
    stride = int(geometry["stride_tokens"])
    max_shift = int(config["new_assembly"]["max_boundary_shift"])
    deviation_cost = float(config["new_assembly"]["deviation_cost"])
    tolerance = float(config["selector"]["tie_tolerance"])
    rows = []
    assembly_runtime = 0.0
    for document in overlap["documents"]:
        passage_id = str(document["passage_id"])
        nooverlap_document = nooverlap_by_id[passage_id]
        overlap_candidates = _candidates_for_document(
            document, policy="overlap", candidates=candidates, alias_to_index=aliases
        )
        nooverlap_candidates = _candidates_for_document(
            nooverlap_document,
            policy="nooverlap",
            candidates=candidates,
            alias_to_index=aliases,
        )
        reference = tuple(int(value) for value in document["prefixes"]["8"]["reference_ids"])
        nooverlap_prediction = tuple(
            token for candidate in nooverlap_candidates for token in candidate.content_ids
        )
        fixed = stitch_candidate_sequence_constrained(
            overlap_candidates,
            overlap_tokens=overlap_tokens,
            max_boundary_shift=max_shift,
            deviation_cost=deviation_cost,
        )
        fixed_coordinate = fixed_coordinate_fidelity_assembly(
            overlap_candidates,
            content_tokens=int(geometry["content_tokens"]),
            stride=stride,
            tie_tolerance=tolerance,
        )
        started = time.perf_counter()
        assembled = stitch_candidate_sequence_fidelity_constrained(
            overlap_candidates,
            overlap_tokens=overlap_tokens,
            max_boundary_shift=max_shift,
            deviation_cost=deviation_cost,
            tie_tolerance=tolerance,
        )
        assembly_runtime += time.perf_counter() - started
        scores = {index: float(candidate.local_score) for index, candidate in enumerate(overlap_candidates)}
        source_selector, _failures = _score_document(
            document,
            overlap_candidates,
            decision_scores=scores,
            tie_tolerance=tolerance,
        )
        nooverlap_source = nooverlap_single_view_source_row(
            nooverlap_document, nooverlap_candidates
        )
        method_scores = {
            "nooverlap": _document_score(nooverlap_prediction, reference),
            "overlap_fixed_owner": _document_score(fixed.content_ids, reference),
            "overlap_fidelity_fixed_coordinate": _document_score(fixed_coordinate, reference),
            "overlap_fidelity_geometry_aware": _document_score(assembled.content_ids, reference),
        }
        row = {
            "passage_id": passage_id,
            "pre_inversion_stratum": strata[passage_id],
            **method_scores,
            "source_aligned": {
                "source_token_count": int(source_selector["source_token_count"]),
                "nooverlap_correct": int(nooverlap_source["correct_source_token_count"]),
                "selector_correct": int(source_selector["selector_correct"]),
                "oracle_correct": int(source_selector["oracle_correct"]),
            },
            "assembly_diagnostic": {
                "boundary_left_widths": list(assembled.boundary_left_widths),
                "boundary_right_widths": list(assembled.boundary_right_widths),
                "boundary_owner_views": list(assembled.boundary_owner_views),
                "inter_boundary_gap_widths": list(assembled.inter_boundary_gap_widths),
                "inter_boundary_duplicate_widths": list(assembled.inter_boundary_duplicate_widths),
            },
        }
        row["deltas"] = {
            "new_minus_nooverlap": method_scores["overlap_fidelity_geometry_aware"]["token_recovery"] - method_scores["nooverlap"]["token_recovery"],
            "new_minus_fixed_owner": method_scores["overlap_fidelity_geometry_aware"]["token_recovery"] - method_scores["overlap_fixed_owner"]["token_recovery"],
            "new_minus_fixed_coordinate": method_scores["overlap_fidelity_geometry_aware"]["token_recovery"] - method_scores["overlap_fidelity_fixed_coordinate"]["token_recovery"],
            "source_aligned_selector_minus_nooverlap": source_selector["selector_recovery"] - nooverlap_source["source_token_recovery"],
            "source_aligned_selector_minus_new": source_selector["selector_recovery"] - method_scores["overlap_fidelity_geometry_aware"]["token_recovery"],
        }
        rows.append(row)

    evaluation = config["evaluation"]
    bootstrap = _bootstrap(
        rows,
        iterations=int(evaluation["bootstrap_iterations"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    main = _summary(rows)
    robustness = {}
    for offset, name in enumerate((*_STRATA, "POOLED_NONADJUSTED"), start=1):
        selected = [
            row
            for row in rows
            if (
                row["pre_inversion_stratum"] != "BOUNDARY_ADJUSTMENT_REQUIRED"
                if name == "POOLED_NONADJUSTED"
                else row["pre_inversion_stratum"] == name
            )
        ]
        robustness[name] = {
            "summary": _summary(selected),
            "bootstrap": _bootstrap(
                selected,
                iterations=int(evaluation["bootstrap_iterations"]),
                seed=int(evaluation["bootstrap_seed"]) + offset,
            ),
        }
    new_rows = [row["overlap_fidelity_geometry_aware"] for row in rows]
    worst = sorted(
        rows,
        key=lambda row: (
            float(row["overlap_fidelity_geometry_aware"]["token_recovery"]),
            -int(row["overlap_fidelity_geometry_aware"]["edit_distance"]),
        ),
    )[:5]
    failure = {
        "substitutions": sum(int(row["substitution_count"]) for row in new_rows),
        "prediction_insertions": sum(int(row["prediction_insertion_count"]) for row in new_rows),
        "reference_deletions": sum(int(row["reference_deletion_count"]) for row in new_rows),
        "by_boundary_stratum": {
            name: {
                "documents": sum(row["pre_inversion_stratum"] == name for row in rows),
                "edit_distance": sum(
                    int(row["overlap_fidelity_geometry_aware"]["edit_distance"])
                    for row in rows
                    if row["pre_inversion_stratum"] == name
                ),
            }
            for name in _STRATA
        },
        "worst_documents": [
            {
                "passage_id": row["passage_id"],
                "pre_inversion_stratum": row["pre_inversion_stratum"],
                "token_recovery": row["overlap_fidelity_geometry_aware"]["token_recovery"],
                "edit_distance": row["overlap_fidelity_geometry_aware"]["edit_distance"],
            }
            for row in worst
        ],
    }
    result = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE5D_ASSEMBLY_EVALUATION_COMPLETE",
        "estimand": "Untouched 60-document Plan-2 K=8 cohort at frozen SHA-256 positions 90..149.",
        "config": {"path": str(config["resolved_config"]), "sha256": _sha256(Path(str(config["resolved_config"])))},
        "strong_cache": {"path": str(config["resolved_strong_cache"]), "sha256": _sha256(Path(str(config["resolved_strong_cache"]))), "telemetry": strong["telemetry"]},
        "main_result": main,
        "bootstrap": bootstrap,
        "robustness": robustness,
        "failure_analysis": failure,
        "assembly_runtime": {
            "total_seconds": assembly_runtime,
            "seconds_per_document": assembly_runtime / len(rows),
        },
        "gate": _gate(bootstrap),
        "ground_truth_used_by_assembly": False,
        "source_alignment_used_by_assembly": False,
        "selector_changed_or_tuned": False,
        "candidate_generation_changed": False,
    }
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / str(config["result_filename"]), result)
    _write_json_atomic(
        output_dir / str(config["trajectory_filename"]),
        {
            "schema_version": 1,
            "config_sha256": result["config"]["sha256"],
            "strong_cache_sha256": result["strong_cache"]["sha256"],
            "trajectories": rows,
        },
    )
    _write_report(output_dir, result, str(config["report_filename"]))
    return result
