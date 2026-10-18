"""Frozen Stage 5B source-level holdout evaluation.

H1 compares GT-aligned source positions, never reassembled text. H2 evaluates
the already-frozen candidate-visible boundary-fidelity diagnostic. Ground truth
is used only for evaluation/oracle labels; no correction method is implemented.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml

from .diagnostics import align_token_ids
from .reconstruction import Candidate
from .stage1_strong_saturation import _candidate_from_dict, _write_json_atomic
from .stage3_longdoc_strong import _aggregate_prefix, _percentile
from .stage4_document_correction import (
    _aggregate_fidelity,
    _agreement_summary,
    _evaluate_document,
    _observable_boundary_summary,
    _observations,
    _sum_nested_counts,
)
from .stage5_nooverlap import _aggregate_nooverlap, _evaluate_nooverlap_document
from .stage5b_protocol import validate_stage5b_protocol


_IMPLEMENTATION_VERSION = "stage5b-source-holdout-evaluation-v1"
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


def load_stage5b_evaluation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 5B evaluation config must be a YAML mapping")

    def resolve(value: str) -> str:
        candidate = Path(value)
        return str(
            (candidate if candidate.is_absolute() else config_path.parent / candidate)
            .resolve()
        )

    resolved = dict(raw)
    for key in (
        "base_config",
        "evaluation_protocol",
        "evaluation_snapshot",
        "strong_generation_config",
        "strong_cache",
    ):
        resolved[f"resolved_{key}"] = resolve(str(raw[key]))
    resolved["resolved_config"] = str(config_path.resolve())
    return resolved


def _validate_inputs(config: Mapping[str, Any]) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    protocol_path = Path(str(config["resolved_evaluation_protocol"]))
    snapshot_path = Path(str(config["resolved_evaluation_snapshot"]))
    if _sha256(protocol_path) != str(config["evaluation_protocol_sha256"]):
        raise RuntimeError("Stage 5B evaluation protocol changed")
    if _sha256(snapshot_path) != str(config["evaluation_snapshot_sha256"]):
        raise RuntimeError("Stage 5B evaluation snapshot changed")
    protocol = validate_stage5b_protocol(protocol_path)
    snapshot = _load_json(snapshot_path)
    if snapshot["protocol_sha256"] != protocol["protocol_sha256"]:
        raise RuntimeError("Stage 5B protocol snapshot refers to another protocol")

    artifacts = {
        name: _load_json(Path(str(entry["path"])))
        for name, entry in protocol["validated_artifacts"].items()
    }
    overlap = artifacts["overlap_cohort"]
    nooverlap = artifacts["nooverlap_manifest"]
    strong_cache_path = Path(str(config["resolved_strong_cache"]))
    strong = _load_json(strong_cache_path)
    if strong.get("status") != "STAGE5B_STRONG_TOP1_CACHE_COMPLETE":
        raise RuntimeError("Stage 5B strong Top-1 cache is incomplete")
    if strong.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("Stage 5B strong cache refers to another protocol")
    if strong.get("strong_reproduction_setting") != "correction50+sbeam8":
        raise RuntimeError("Stage 5B cache is not the frozen strong setting")
    if len(strong.get("top1_candidates", [])) != 720:
        raise RuntimeError("Stage 5B strong cache must contain 720 unique views")
    if len(strong.get("alias_to_canonical_index", {})) != 780:
        raise RuntimeError("Stage 5B logical alias count changed")
    records = list(strong.get("records", []))
    if len(records) != 720 or not all(
        bool(record.get("public_top1_exact_id_match")) for record in records
    ):
        raise RuntimeError("Stage 5B strong Top-1 instrumentation invariant failed")
    if any(int(record.get("embed_call_count", -1)) != 51 for record in records):
        raise RuntimeError("Stage 5B per-view embedding-query budget changed")
    if any(
        int(record.get("search_hypothesis_rows_embedded", -1)) != 3145
        for record in records
    ):
        raise RuntimeError("Stage 5B hypothesis-row budget changed")
    if [document["passage_id"] for document in overlap["documents"]] != [
        document["passage_id"] for document in nooverlap["documents"]
    ]:
        raise RuntimeError("Stage 5B paired document order changed")
    return protocol, overlap, nooverlap, strong, artifacts["holdout_summary"]


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


def overlap_best_of_views_source_row(
    document: Mapping[str, Any], candidates: Sequence[Candidate]
) -> dict[str, Any]:
    observations, _aligned_views = _observations(document, candidates, k=8)
    correct = [any(bool(value["correct"]) for value in values) for values in observations]
    return {
        "source_token_count": len(correct),
        "correct_source_token_count": sum(correct),
        "source_token_recovery": mean(correct),
        "source_position_document_exact": all(correct),
        "correct_source_positions": [
            index for index, value in enumerate(correct) if value
        ],
        "definition": (
            "For each frozen source position, correct iff at least one GT-aligned "
            "overlapping Independent Top-1 view reconstructs the token exactly."
        ),
    }


def nooverlap_single_view_source_row(
    document: Mapping[str, Any], candidates: Sequence[Candidate]
) -> dict[str, Any]:
    reference = tuple(int(value) for value in document["reference_ids"])
    if len(candidates) != len(document["windows"]):
        raise ValueError("no-overlap candidate count differs from window count")
    observed = [0] * len(reference)
    correct = [False] * len(reference)
    window_rows = []
    for window, candidate in zip(document["windows"], candidates):
        source_ids = tuple(int(value) for value in window["source_ids"])
        alignment = align_token_ids(candidate.content_ids, source_ids)
        start = int(window["relative_token_start"])
        for relative_position in range(len(source_ids)):
            observed[start + relative_position] += 1
        for _prediction_index, reference_index in alignment.exact_matches:
            correct[start + reference_index] = True
        window_rows.append(
            {
                "window_index": int(window["window_index"]),
                "source_token_count": len(source_ids),
                "correct_source_token_count": len(alignment.exact_matches),
                "source_token_recovery": len(alignment.exact_matches) / len(source_ids),
                "source_position_exact": len(alignment.exact_matches) == len(source_ids),
                "prediction_insertion_count": len(alignment.prediction_insertions),
            }
        )
    if observed != [1] * len(reference):
        raise AssertionError("no-overlap source windows no longer partition 136 positions")
    return {
        "source_token_count": len(reference),
        "correct_source_token_count": sum(correct),
        "source_token_recovery": mean(correct),
        "source_position_document_exact": all(correct),
        "correct_source_positions": [
            index for index, value in enumerate(correct) if value
        ],
        "window_rows": window_rows,
        "definition": (
            "Each disjoint reconstruction is independently GT-aligned to its "
            "frozen source-ID slice; no text or candidate-ID reassembly is used."
        ),
    }


def _h2_components(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    boundaries = [
        boundary
        for diagnostic in diagnostics
        for boundary in diagnostic["candidate_aligned_boundaries"]
        if bool(boundary["observable_boundary_disagreement"])
    ]
    event_count = len(boundaries)
    fidelity_correct = sum(
        float(boundary["fidelity_selected_correct_shared_tokens"])
        for boundary in boundaries
    )
    random_correct = sum(
        float(boundary["random_expected_correct_shared_tokens"])
        for boundary in boundaries
    )
    denominator = event_count * 16
    return {
        "event_count": event_count,
        "fidelity_correct_shared_tokens": fidelity_correct,
        "random_expected_correct_shared_tokens": random_correct,
        "denominator_shared_tokens": denominator,
        "higher_fidelity_shared_token_accuracy": (
            fidelity_correct / denominator if denominator else None
        ),
        "random_side_expected_shared_token_accuracy": (
            random_correct / denominator if denominator else None
        ),
        "uplift_over_random": (
            (fidelity_correct - random_correct) / denominator
            if denominator
            else None
        ),
    }


def _bootstrap(
    rows: Sequence[Mapping[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    h1_samples: list[float] = []
    h2_samples: list[float] = []
    for _ in range(iterations):
        selected = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        h1_samples.append(mean(float(row["h1_delta"]) for row in selected))
        h2 = _h2_components([row["overlap_diagnostic"] for row in selected])
        if h2["uplift_over_random"] is not None:
            h2_samples.append(float(h2["uplift_over_random"]))
    return {
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "frozen document",
        "h1_mean_delta": mean(float(row["h1_delta"]) for row in rows),
        "h1_ci95": [
            _percentile(h1_samples, 0.025),
            _percentile(h1_samples, 0.975),
        ],
        "h2_ci95": (
            [_percentile(h2_samples, 0.025), _percentile(h2_samples, 0.975)]
            if h2_samples
            else [None, None]
        ),
        "h2_bootstrap_replicates_with_events": len(h2_samples),
    }


def _source_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tokens = sum(int(row["overlap_oracle"]["source_token_count"]) for row in rows)
    overlap_correct = sum(
        int(row["overlap_oracle"]["correct_source_token_count"]) for row in rows
    )
    nooverlap_correct = sum(
        int(row["nooverlap_single"]["correct_source_token_count"]) for row in rows
    )
    return {
        "documents": len(rows),
        "source_token_count": tokens,
        "overlap_best_of_views_source_recovery": overlap_correct / tokens,
        "nooverlap_single_view_source_recovery": nooverlap_correct / tokens,
        "overlap_minus_nooverlap_source_recovery": (
            overlap_correct - nooverlap_correct
        )
        / tokens,
        "overlap_source_position_document_exact_fraction": mean(
            bool(row["overlap_oracle"]["source_position_document_exact"])
            for row in rows
        ),
        "nooverlap_source_position_document_exact_fraction": mean(
            bool(row["nooverlap_single"]["source_position_document_exact"])
            for row in rows
        ),
    }


def _classify_gate(
    *,
    overall: Mapping[str, Any],
    nonadjusted: Mapping[str, Any],
    adjusted: Mapping[str, Any],
    h2: Mapping[str, Any],
) -> dict[str, Any]:
    all_positive = float(overall["bootstrap"]["h1_ci95"][0]) > 0.0
    nonadjusted_positive = float(nonadjusted["bootstrap"]["h1_ci95"][0]) > 0.0
    adjusted_positive = float(adjusted["bootstrap"]["h1_ci95"][0]) > 0.0
    h2_lower = overall["bootstrap"]["h2_ci95"][0]
    h2_positive = int(h2["event_count"]) > 0 and h2_lower is not None and float(h2_lower) > 0.0
    if not all_positive:
        state = "NO_OVERLAP_INFORMATION_GAIN"
    elif adjusted_positive and not nonadjusted_positive:
        state = "BOUNDARY_CONDITIONAL_SIGNAL"
    elif nonadjusted_positive and not h2_positive:
        state = "OVERLAP_REDUNDANCY_NOT_EXPLOITABLE"
    elif nonadjusted_positive and h2_positive:
        state = "GENERAL_OVERLAP_SIGNAL"
    else:
        state = "STRATA_PATTERN_INCONCLUSIVE"
    return {
        "state": state,
        "h1_all_document_ci95_lower_above_zero": all_positive,
        "h1_pooled_nonadjusted_ci95_lower_above_zero": nonadjusted_positive,
        "h1_boundary_adjusted_ci95_lower_above_zero": adjusted_positive,
        "h2_nonzero_event_count_and_ci95_lower_above_zero": h2_positive,
        "method_implementation_authorized": False,
        "next_step": (
            "Stage 5C Plan-1 controlled robustness only"
            if state in {"BOUNDARY_CONDITIONAL_SIGNAL", "GENERAL_OVERLAP_SIGNAL"}
            else "Stop or reassess without implementing a selector"
        ),
    }


def _write_report(output_dir: Path, filename: str, result: Mapping[str, Any]) -> None:
    source = result["source_level"]
    h2 = result["h2_fidelity"]
    lines = [
        "# Stage 5B — frozen Plan-2 holdout evaluation",
        "",
        f"- Gate: `{result['gate']['state']}`",
        "- Strong setting: repository reproduction `correction50+sbeam8`; not an original-paper beam-width claim.",
        "- H1 uses source-position alignment only; raw-text document assembly is excluded.",
        "- Ground truth is diagnostic/evaluation-only; no selector or correction method was implemented.",
        "",
        "| Group | Docs | Overlap Best-of-Views | No-overlap single-view | Delta | H1 CI95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in source.items():
        summary = values["summary"]
        lines.append(
            f"| {name} | {summary['documents']} | "
            f"{summary['overlap_best_of_views_source_recovery']:.4f} | "
            f"{summary['nooverlap_single_view_source_recovery']:.4f} | "
            f"{summary['overlap_minus_nooverlap_source_recovery']:+.4f} | "
            f"{values['bootstrap']['h1_ci95']} |"
        )
    lines.extend(
        [
            "",
            f"- H2 observable imperfect-boundary events: {h2['event_count']}",
            f"- Higher-fidelity shared-token accuracy: {h2['higher_fidelity_shared_token_accuracy']}",
            f"- Random-side expected accuracy: {h2['random_side_expected_shared_token_accuracy']}",
            f"- Uplift: {h2['uplift_over_random']}",
            f"- H2 bootstrap CI95: {source['ALL']['bootstrap']['h2_ci95']}",
        ]
    )
    (output_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage5b_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    protocol, overlap, nooverlap, strong, holdout = _validate_inputs(config)
    base = yaml.safe_load(Path(str(config["resolved_base_config"])).read_text(encoding="utf-8"))
    candidates = tuple(
        _candidate_from_dict(raw) for raw in strong["top1_candidates"]
    )
    alias_to_index = {
        str(alias): int(index)
        for alias, index in strong["alias_to_canonical_index"].items()
    }
    tokenizer = AutoTokenizer.from_pretrained(
        base["models"]["embedder_name"],
        revision=base["models"].get("embedder_revision"),
    )
    nooverlap_by_id = {
        str(document["passage_id"]): document
        for document in nooverlap["documents"]
    }
    strata = {
        str(key): str(value)
        for key, value in nooverlap["pre_inversion_boundary_stratum_by_passage"].items()
    }
    rows = []
    overlap_rows = []
    nooverlap_rows = []
    diagnostics = []
    stitch = dict(config["stronger_stitcher"])
    for overlap_document in overlap["documents"]:
        passage_id = str(overlap_document["passage_id"])
        nooverlap_document = nooverlap_by_id[passage_id]
        overlap_candidates = _candidates_for_document(
            overlap_document,
            policy="overlap",
            candidates=candidates,
            alias_to_index=alias_to_index,
        )
        nooverlap_candidates = _candidates_for_document(
            nooverlap_document,
            policy="nooverlap",
            candidates=candidates,
            alias_to_index=alias_to_index,
        )
        overlap_oracle = overlap_best_of_views_source_row(
            overlap_document, overlap_candidates
        )
        nooverlap_single = nooverlap_single_view_source_row(
            nooverlap_document, nooverlap_candidates
        )
        overlap_row, _stitched, diagnostic = _evaluate_document(
            overlap_document,
            overlap_candidates,
            k=8,
            overlap_tokens=int(stitch["nominal_overlap_tokens"]),
            stride=int(config["geometry"]["stride_tokens"]),
            max_boundary_shift=int(stitch["max_boundary_shift"]),
            deviation_cost=float(stitch["deviation_cost"]),
        )
        nooverlap_row = _evaluate_nooverlap_document(
            nooverlap_document, nooverlap_candidates, tokenizer
        )
        rows.append(
            {
                "passage_id": passage_id,
                "pre_inversion_stratum": strata[passage_id],
                "overlap_oracle": overlap_oracle,
                "nooverlap_single": nooverlap_single,
                "h1_delta": overlap_oracle["source_token_recovery"]
                - nooverlap_single["source_token_recovery"],
                "overlap_improved": overlap_row,
                "nooverlap_end_to_end": nooverlap_row,
                "overlap_diagnostic": diagnostic,
            }
        )
        overlap_rows.append(overlap_row)
        nooverlap_rows.append(nooverlap_row)
        diagnostics.append(diagnostic)

    bootstrap_config = protocol["evaluation"]["bootstrap"]
    iterations = int(bootstrap_config["iterations"])
    seed = int(bootstrap_config["seed"])

    def group_payload(selected: Sequence[Mapping[str, Any]], offset: int) -> dict[str, Any]:
        return {
            "summary": _source_summary(selected),
            "bootstrap": _bootstrap(
                selected, iterations=iterations, seed=seed + offset
            ),
            "passage_ids": [str(row["passage_id"]) for row in selected],
        }

    grouped: dict[str, dict[str, Any]] = {"ALL": group_payload(rows, 0)}
    for offset, stratum in enumerate(_STRATA, start=1):
        selected = [row for row in rows if row["pre_inversion_stratum"] == stratum]
        grouped[stratum] = group_payload(selected, offset)
    nonadjusted = [
        row
        for row in rows
        if row["pre_inversion_stratum"]
        in {"STRICT_FIXED_STABLE", "CAPACITY_SAFE_NO_SHIFT"}
    ]
    grouped["POOLED_NONADJUSTED"] = group_payload(nonadjusted, 4)

    h2 = _h2_components(diagnostics)
    agreement = {
        multiplicity: _agreement_summary(
            _sum_nested_counts(diagnostics, multiplicity)
        )
        for multiplicity in ("2", "3")
    }
    fidelity = {
        multiplicity: _aggregate_fidelity(diagnostics, multiplicity)
        for multiplicity in ("all", "2", "3")
    }
    overlap_summary = _aggregate_prefix(overlap_rows)
    nooverlap_summary = _aggregate_nooverlap(nooverlap_rows)
    gate = _classify_gate(
        overall=grouped["ALL"],
        nonadjusted=grouped["POOLED_NONADJUSTED"],
        adjusted=grouped["BOUNDARY_ADJUSTMENT_REQUIRED"],
        h2=h2,
    )
    result = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE5B_HOLDOUT_EVALUATION_COMPLETE",
        "estimand": (
            "Frozen 60-document Plan-2 holdout drawn from K=8 round-trip-stable "
            "NQ chains; not the full pinned NQ scan frame."
        ),
        "protocol": {
            "path": str(config["resolved_evaluation_protocol"]),
            "sha256": protocol["protocol_sha256"],
            "snapshot_sha256": str(config["evaluation_snapshot_sha256"]),
        },
        "strong_cache": {
            "path": str(config["resolved_strong_cache"]),
            "sha256": _sha256(Path(str(config["resolved_strong_cache"]))),
            "telemetry": strong["telemetry"],
        },
        "cohort": {
            "documents": len(rows),
            "pre_inversion_strata": holdout["pre_inversion_strata"],
            "paired_design": True,
        },
        "source_level": grouped,
        "h2_fidelity": h2,
        "agreement": agreement,
        "gt_aligned_fidelity_diagnostic": fidelity,
        "candidate_aligned_observable": _observable_boundary_summary(
            diagnostics, overlap_tokens=16
        ),
        "secondary_end_to_end": {
            "overlap_improved_stitch": overlap_summary,
            "nooverlap": nooverlap_summary,
            "raw_text_assembly_not_used_in_h1": True,
        },
        "gate": gate,
        "ground_truth_passed_to_selector": False,
        "selector_or_correction_method_implemented": False,
        "stage5c_or_method_executed": False,
    }
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / str(config["result_filename"]), result)
    _write_json_atomic(
        output_dir / str(config["trajectory_filename"]),
        {
            "schema_version": 1,
            "protocol_sha256": protocol["protocol_sha256"],
            "strong_cache_sha256": result["strong_cache"]["sha256"],
            "trajectories": rows,
        },
    )
    _write_report(output_dir, str(config["report_filename"]), result)
    return result

