"""Stage 6A development-only source-free fidelity-guided assembly."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import time
from typing import Any, Mapping, Sequence

import yaml

from .diagnostics import (
    FidelityStitchedSequenceIDs,
    _aligned_pairs,
    align_token_ids,
    stitch_candidate_ids_constrained,
    stitch_candidate_sequence,
    stitch_candidate_sequence_constrained,
    stitch_candidate_sequence_fidelity_constrained,
)
from .reconstruction import Candidate
from .stage1_strong_saturation import _candidate_from_dict, _write_json_atomic
from .stage3_longdoc_strong import _percentile
from .stage4_document_correction import _observations
from .stage5c_fidelity_selector import _score_document
from .stage5d_assembly import fixed_coordinate_fidelity_assembly


_IMPLEMENTATION_VERSION = "stage6a-source-free-progressive-pairwise-assembly-v2"
_INPUT_NAMES = (
    "exploratory_overlap_cohort",
    "exploratory_overlap_strong_cache",
    "exploratory_nooverlap_manifest",
    "exploratory_nooverlap_strong_cache",
)
_TAXONOMY = (
    "VIEW_ALL_WRONG",
    "ALIGNMENT_FAILURE",
    "FIDELITY_WRONG_DIRECTION",
    "GAP_HANDLING_FAILURE",
    "GLOBAL_ASSEMBLY_FAILURE",
    "EVALUATION_AMBIGUITY",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def load_stage6a_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 6A config must be a YAML mapping")
    resolved = dict(raw)
    resolved["resolved_config"] = str(config_path.resolve())
    resolved["resolved_inputs"] = {
        name: {
            **entry,
            "path": str(
                (
                    Path(str(entry["path"]))
                    if Path(str(entry["path"])).is_absolute()
                    else config_path.parent / str(entry["path"])
                ).resolve()
            ),
        }
        for name, entry in raw["inputs"].items()
    }
    return resolved


def _validate_candidate_cache(
    cache: Mapping[str, Any], *, expected_count: int, expected_keys: Sequence[str]
) -> None:
    if cache.get("strong_reproduction_setting") != "correction50+sbeam8":
        raise RuntimeError("Stage 6A candidate generation setting changed")
    if cache.get("public_top1_exact_id_match") is not True:
        raise RuntimeError("Stage 6A public/instrumented Top-1 invariant failed")
    candidates = list(cache.get("top1_candidates", []))
    view_keys = list(cache.get("view_keys", []))
    if len(candidates) != expected_count or len(view_keys) != expected_count:
        raise RuntimeError("Stage 6A strong candidate count changed")
    if view_keys != list(expected_keys) or len(set(view_keys)) != expected_count:
        raise RuntimeError("Stage 6A strong candidate order changed")
    if any(
        int(candidate.get("correction_steps", -1)) != 50
        or int(candidate.get("sequence_beam_width", -1)) != 8
        for candidate in candidates
    ):
        raise RuntimeError("Stage 6A candidate generation budget changed")


def _validate_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if config.get("stage") != "stage6a_source_free_progressive_pairwise_assembly_v2":
        raise RuntimeError("Stage 6A config stage identity changed")
    if config.get("development_only") is not True:
        raise RuntimeError("Stage 6A must remain development-only")
    if set(config["resolved_inputs"]) != set(_INPUT_NAMES):
        raise RuntimeError("Stage 6A input set changed")
    values: dict[str, dict[str, Any]] = {}
    for name in _INPUT_NAMES:
        entry = config["resolved_inputs"][name]
        path = Path(str(entry["path"]))
        lowered_path = str(path).lower()
        if (
            "stage5b_plan2_holdout" in lowered_path
            or "stage5c_fidelity_selector" in lowered_path
        ):
            raise RuntimeError(
                "Stage 6A development cannot consume the Stage 5B/C holdout"
            )
        if _sha256(path) != str(entry["sha256"]):
            raise RuntimeError(f"Stage 6A frozen development input changed: {name}")
        values[name] = _load_json(path)

    overlap = values["exploratory_overlap_cohort"]
    overlap_cache = values["exploratory_overlap_strong_cache"]
    nooverlap = values["exploratory_nooverlap_manifest"]
    nooverlap_cache = values["exploratory_nooverlap_strong_cache"]
    expected_docs = int(config["cohort"]["documents"])
    overlap_docs = list(overlap.get("documents", []))
    nooverlap_docs = list(nooverlap.get("documents", []))
    overlap_ids = [str(row["passage_id"]) for row in overlap_docs]
    nooverlap_ids = [str(row["passage_id"]) for row in nooverlap_docs]
    if overlap.get("stage") != "STAGE3A_NESTED_LONG_DOCUMENT_COHORT":
        raise RuntimeError("Stage 6A overlap cohort is not the Stage 3 exploratory cohort")
    if nooverlap.get("stage") != "STAGE5A_NO_OVERLAP_FROZEN_SOURCE_MANIFEST":
        raise RuntimeError("Stage 6A no-overlap manifest is not frozen Stage 5A input")
    if len(overlap_docs) != expected_docs or len(nooverlap_docs) != expected_docs:
        raise RuntimeError(
            f"Stage 6A requires exactly {expected_docs} exploratory documents"
        )
    if overlap_ids != nooverlap_ids:
        raise RuntimeError("Stage 6A paired exploratory document order changed")
    if overlap_ids != list(overlap.get("manifest", [])) or overlap_ids != list(
        nooverlap.get("manifest", [])
    ):
        raise RuntimeError("Stage 6A manifest order changed")
    overlap_hash = str(config["resolved_inputs"]["exploratory_overlap_cohort"]["sha256"])
    nooverlap_hash = str(
        config["resolved_inputs"]["exploratory_nooverlap_manifest"]["sha256"]
    )
    if overlap_cache.get("cohort_sha256") != overlap_hash:
        raise RuntimeError("Stage 6A overlap cache belongs to another cohort")
    if nooverlap.get("source_cohort_sha256") != overlap_hash:
        raise RuntimeError("Stage 6A no-overlap manifest belongs to another cohort")
    if nooverlap_cache.get("manifest_sha256") != nooverlap_hash:
        raise RuntimeError("Stage 6A no-overlap cache belongs to another manifest")
    if any(len(row.get("windows", [])) != 8 for row in overlap_docs):
        raise RuntimeError("Stage 6A overlap window count changed")
    if any(len(row.get("windows", [])) != 5 for row in nooverlap_docs):
        raise RuntimeError("Stage 6A no-overlap window count changed")
    for left, right in zip(overlap_docs, nooverlap_docs):
        reference = list(left["prefixes"]["8"]["reference_ids"])
        if len(reference) != 136 or reference != list(right["reference_ids"]):
            raise RuntimeError("Stage 6A paired source span changed")
    overlap_keys = [f"{pid}:{index}" for pid in overlap_ids for index in range(8)]
    nooverlap_keys = [
        f"{pid}:nooverlap:{index}" for pid in nooverlap_ids for index in range(5)
    ]
    _validate_candidate_cache(overlap_cache, expected_count=240, expected_keys=overlap_keys)
    _validate_candidate_cache(
        nooverlap_cache, expected_count=150, expected_keys=nooverlap_keys
    )
    records = list(nooverlap_cache.get("records", []))
    if len(records) != 150 or any(
        not bool(row.get("public_top1_exact_id_match"))
        or int(row.get("embed_call_count", -1)) != 51
        or int(row.get("search_hypothesis_rows_embedded", -1)) != 3145
        for row in records
    ):
        raise RuntimeError("Stage 6A no-overlap cache telemetry changed")
    geometry = config["geometry"]
    if (
        int(geometry["content_tokens"]),
        int(geometry["stride_tokens"]),
        int(geometry["nominal_overlap_tokens"]),
        int(geometry["windows_per_document"]),
        int(geometry["source_tokens_per_document"]),
    ) != (31, 15, 16, 8, 136):
        raise RuntimeError("Stage 6A public overlap geometry changed")
    assembly = config["pairwise_assembly"]
    if (
        str(assembly["name"]),
        str(assembly["merge_order"]),
        int(assembly["max_boundary_shift"]),
        float(assembly["deviation_cost"]),
        str(assembly["fidelity_field"]),
        str(assembly["tie_rule"]),
        bool(assembly["boundary_independent_pairwise_chain_reported_as_diagnostic"]),
    ) != (
        "accumulated_progressive_fidelity_owned_constrained_semiglobal_v1",
        "merge_accumulated_output_with_next_window",
        2,
        0.5,
        "Candidate.local_score",
        "lower_window_index",
        True,
    ):
        raise RuntimeError("Stage 6A frozen pairwise rule changed")
    return values


def assemble_source_free_boundary_independent(
    candidates: Sequence[Candidate],
    *,
    overlap_tokens: int = 16,
    max_boundary_shift: int = 2,
    deviation_cost: float = 0.5,
    tie_tolerance: float = 1e-12,
) -> FidelityStitchedSequenceIDs:
    """Assemble candidates without accepting source text, positions, or GT."""

    return stitch_candidate_sequence_fidelity_constrained(
        candidates,
        overlap_tokens=overlap_tokens,
        max_boundary_shift=max_boundary_shift,
        deviation_cost=deviation_cost,
        tie_tolerance=tie_tolerance,
    )


def assemble_source_free_progressive(
    candidates: Sequence[Candidate],
    *,
    overlap_tokens: int = 16,
    max_boundary_shift: int = 2,
    deviation_cost: float = 0.5,
    tie_tolerance: float = 1e-12,
) -> FidelityStitchedSequenceIDs:
    """Literally apply ``merge(merge(W0, W1), W2)`` without source input.

    The accumulated sequence supplies the left suffix at every step.  Its
    boundary confidence remains that of the immediately preceding window,
    because public geometry says that window owns the suffix being compared.
    This function is retained as the requested early-error-propagation
    baseline; the boundary-independent pairwise chain is evaluated separately.
    """

    if len(candidates) < 2:
        raise ValueError("progressive assembly requires at least two candidates")
    assembled = list(candidates[0].content_ids)
    boundaries = []
    owners: list[int] = []
    for boundary_index, right in enumerate(candidates[1:]):
        left_window = candidates[boundary_index]
        accumulated = Candidate(
            text="",
            local_score=float(left_window.local_score),
            correction_steps=int(left_window.correction_steps),
            sequence_beam_width=int(left_window.sequence_beam_width),
            generation_id=f"stage6a-progressive:{boundary_index}",
            content_ids=tuple(assembled),
        )
        boundary = stitch_candidate_ids_constrained(
            accumulated,
            right,
            overlap_tokens=overlap_tokens,
            max_boundary_shift=max_boundary_shift,
            deviation_cost=deviation_cost,
        )
        boundaries.append(boundary)
        left_segment = tuple(assembled[-boundary.aligned_left_tokens :])
        right_segment = tuple(right.content_ids[: boundary.consumed_right_tokens])
        aligned = _aligned_pairs(left_segment, right_segment)
        right_wins = (
            float(right.local_score)
            > float(left_window.local_score) + tie_tolerance
        )
        owners.append(boundary_index + 1 if right_wins else boundary_index)
        merged: list[int] = []
        for column_index, (left_token, right_token) in enumerate(aligned):
            multiplicity_three_endpoint = (
                column_index == 0 and boundary_index > 0
            ) or (
                column_index == len(aligned) - 1
                and boundary_index < len(candidates) - 2
            )
            chosen = (
                left_token
                if multiplicity_three_endpoint or not right_wins
                else right_token
            )
            if chosen is not None:
                merged.append(int(chosen))
        assembled = assembled[: -boundary.aligned_left_tokens]
        assembled.extend(merged)
        assembled.extend(right.content_ids[boundary.consumed_right_tokens :])

    gaps: list[int] = []
    duplicates: list[int] = []
    for view_index in range(1, len(candidates) - 1):
        delta = (
            len(candidates[view_index].content_ids)
            - boundaries[view_index - 1].consumed_right_tokens
            - boundaries[view_index].aligned_left_tokens
        )
        gaps.append(max(0, delta))
        duplicates.append(max(0, -delta))
    return FidelityStitchedSequenceIDs(
        content_ids=tuple(assembled),
        boundary_left_widths=tuple(
            boundary.aligned_left_tokens for boundary in boundaries
        ),
        boundary_right_widths=tuple(
            boundary.consumed_right_tokens for boundary in boundaries
        ),
        boundary_alignment_scores=tuple(
            boundary.alignment_score for boundary in boundaries
        ),
        boundary_owner_views=tuple(owners),
        inter_boundary_gap_widths=tuple(gaps),
        inter_boundary_duplicate_widths=tuple(duplicates),
    )


def assemble_source_free_pairwise(
    candidates: Sequence[Candidate],
    *,
    overlap_tokens: int = 16,
    max_boundary_shift: int = 2,
    deviation_cost: float = 0.5,
    tie_tolerance: float = 1e-12,
) -> FidelityStitchedSequenceIDs:
    """Stage 6A proposed method: the requested accumulated pairwise merge."""

    return assemble_source_free_progressive(
        candidates,
        overlap_tokens=overlap_tokens,
        max_boundary_shift=max_boundary_shift,
        deviation_cost=deviation_cost,
        tie_tolerance=tie_tolerance,
    )


def _document_score(prediction: Sequence[int], reference: Sequence[int]) -> dict[str, Any]:
    alignment = align_token_ids(prediction, reference)
    exact = len(alignment.exact_matches)
    return {
        "prediction_length": len(prediction),
        "source_token_count": len(reference),
        "exact_matched_token_count": exact,
        "token_recovery": exact / len(reference),
        "normalized_edit_similarity": max(0.0, 1.0 - alignment.distance / len(reference)),
        "edit_distance": alignment.distance,
        "substitution_count": len(alignment.substitutions),
        "prediction_insertion_count": len(alignment.prediction_insertions),
        "reference_deletion_count": len(alignment.reference_deletions),
        "document_exact": tuple(prediction) == tuple(reference),
        "exact_reference_positions": [right for _left, right in alignment.exact_matches],
    }


def _transfer_metrics(
    method_recovery: float,
    fixed_owner_recovery: float,
    source_aligned_recovery: float,
) -> dict[str, float | None]:
    denominator = source_aligned_recovery - fixed_owner_recovery
    return {
        "realization_gap": source_aligned_recovery - method_recovery,
        "source_aligned_gain_over_fixed_owner": denominator,
        "source_free_gain_over_fixed_owner": method_recovery - fixed_owner_recovery,
        "gain_recovery_ratio": (
            (method_recovery - fixed_owner_recovery) / denominator
            if abs(denominator) > 1e-12
            else None
        ),
    }


def _classify_source_error(
    *,
    all_views_wrong: bool,
    source_selector_correct: bool,
    oracle_correct: bool,
    has_gap_observation: bool,
    boundary_has_gap_or_duplicate: bool,
    global_method_used: bool = False,
    evaluation_ambiguous: bool = False,
) -> str:
    """Apply the frozen conservative post-hoc attribution priority."""

    if evaluation_ambiguous:
        return "EVALUATION_AMBIGUITY"
    if all_views_wrong:
        return "VIEW_ALL_WRONG"
    if not source_selector_correct and oracle_correct:
        return "FIDELITY_WRONG_DIRECTION"
    if has_gap_observation or boundary_has_gap_or_duplicate:
        return "GAP_HANDLING_FAILURE"
    if global_method_used:
        return "GLOBAL_ASSEMBLY_FAILURE"
    return "ALIGNMENT_FAILURE"


def _bootstrap(
    rows: Sequence[Mapping[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    names = (
        "method_minus_nooverlap",
        "method_minus_fixed_owner",
        "method_minus_constrained_stitch",
        "method_minus_fixed_coordinate",
        "method_minus_boundary_independent",
        "source_aligned_minus_method",
    )
    rng = random.Random(seed)
    samples = {name: [] for name in names}
    for _ in range(iterations):
        selected = [rows[rng.randrange(len(rows))] for _ in rows]
        for name in names:
            samples[name].append(mean(float(row["deltas"][name]) for row in selected))
    return {
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "exploratory development document",
        "mean_delta": {
            name: mean(float(row["deltas"][name]) for row in rows) for name in names
        },
        "ci95": {
            name: [_percentile(values, 0.025), _percentile(values, 0.975)]
            for name, values in samples.items()
        },
    }


def _gate(summary: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    method = float(summary["source_free_pairwise"]["token_recovery"])
    source_aligned = summary.get("source_aligned_diagnostics")
    fixed = float(
        source_aligned["fixed_owner_recovery"]
        if source_aligned is not None
        else summary["overlap_fixed_owner"]["token_recovery"]
    )
    nooverlap = float(summary["nooverlap"]["token_recovery"])
    ratio = summary["transfer"]["gain_recovery_ratio"]
    ci_lower = float(bootstrap["ci95"]["method_minus_fixed_owner"][0])
    checks = {
        "method_point_estimate_gt_fixed_owner": method > fixed,
        "method_point_estimate_gte_nooverlap": method >= nooverlap,
        "method_minus_fixed_owner_ci95_lower_gt_zero": ci_lower > 0.0,
        "gain_recovery_ratio_gte_0_5": ratio is not None and float(ratio) >= 0.5,
    }
    ready = all(checks.values())
    return {
        "state": "SOURCE_FREE_ASSEMBLY_READY" if ready else "ASSEMBLY_MECHANISM_FAIL",
        "checks": checks,
        "development_only": True,
        "paper_claim_authorized": False,
        "global_method_implemented": False,
        "interpretation": (
            "Pairwise source-free assembly warrants a frozen unseen validation."
            if ready
            else "The simple pairwise mechanism does not justify validation; inspect whether global consistency is the bottleneck without broad tuning."
        ),
    }


def _candidate_map(cache: Mapping[str, Any]) -> dict[str, Candidate]:
    return {
        str(key): _candidate_from_dict(candidate)
        for key, candidate in zip(cache["view_keys"], cache["top1_candidates"])
    }


def _select_source_view(
    observations: Sequence[Mapping[str, Any]], *, tie_tolerance: float
) -> Mapping[str, Any]:
    ordered = sorted(observations, key=lambda row: int(row["view_index"]))
    if len(ordered) != 2:
        return ordered[0]
    top = max(float(row["fidelity"]) for row in ordered)
    return min(
        (row for row in ordered if top - float(row["fidelity"]) <= tie_tolerance),
        key=lambda row: int(row["view_index"]),
    )


def _residual_positions_matchable_in_an_alternative_optimal_trace(
    prediction: Sequence[int], reference: Sequence[int]
) -> set[int]:
    """Find residual reference positions exact-matchable on another optimum.

    A deterministic Levenshtein trace can choose one of several equal-cost
    paths.  This forward/backward check marks a residual reference position as
    evaluation-ambiguous when an exact diagonal edge for that position lies on
    at least one globally optimal path.
    """

    left = tuple(prediction)
    right = tuple(reference)
    rows = len(left) + 1
    columns = len(right) + 1
    forward = [[0] * columns for _ in range(rows)]
    backward = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        forward[row][0] = row
        backward[row][columns - 1] = len(left) - row
    for column in range(columns):
        forward[0][column] = column
        backward[rows - 1][column] = len(right) - column
    for row in range(1, rows):
        for column in range(1, columns):
            forward[row][column] = min(
                forward[row - 1][column] + 1,
                forward[row][column - 1] + 1,
                forward[row - 1][column - 1]
                + int(left[row - 1] != right[column - 1]),
            )
    for row in range(len(left) - 1, -1, -1):
        for column in range(len(right) - 1, -1, -1):
            backward[row][column] = min(
                backward[row + 1][column] + 1,
                backward[row][column + 1] + 1,
                backward[row + 1][column + 1]
                + int(left[row] != right[column]),
            )
    optimum = forward[-1][-1]
    deterministic_exact = {
        reference_index
        for _prediction_index, reference_index in align_token_ids(
            left, right
        ).exact_matches
    }
    ambiguous: set[int] = set()
    for reference_index, reference_token in enumerate(right):
        if reference_index in deterministic_exact:
            continue
        if any(
            prediction_token == reference_token
            and forward[prediction_index][reference_index]
            + backward[prediction_index + 1][reference_index + 1]
            == optimum
            for prediction_index, prediction_token in enumerate(left)
        ):
            ambiguous.add(reference_index)
    return ambiguous


def _error_diagnostics(
    document: Mapping[str, Any],
    candidates: Sequence[Candidate],
    prediction: Sequence[int],
    assembled: FidelityStitchedSequenceIDs,
    *,
    tie_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference = tuple(int(value) for value in document["prefixes"]["8"]["reference_ids"])
    observations, _ = _observations(document, candidates, k=8)
    alignment = align_token_ids(prediction, reference)
    exact_positions = {right for _left, right in alignment.exact_matches}
    ambiguous_positions = (
        _residual_positions_matchable_in_an_alternative_optimal_trace(
            prediction, reference
        )
    )
    substitutions = {right: left for left, right in alignment.substitutions}
    deletion_positions = set(alignment.reference_deletions)
    counts: Counter[str] = Counter()
    realization_cases: list[dict[str, Any]] = []
    for position in range(len(reference)):
        if position in exact_positions:
            continue
        values = list(observations[position])
        selected = _select_source_view(values, tie_tolerance=tie_tolerance)
        correct_views = [int(row["view_index"]) for row in values if bool(row["correct"])]
        selector_correct = bool(selected["correct"])
        oracle_correct = bool(correct_views)
        nearest_boundary = min(range(7), key=lambda b: abs(position - (15 * (b + 1) + 7.5)))
        internal = min(nearest_boundary, len(assembled.inter_boundary_gap_widths) - 1)
        gap_width = int(assembled.inter_boundary_gap_widths[internal])
        duplicate_width = int(assembled.inter_boundary_duplicate_widths[internal])
        has_gap_observation = any(not isinstance(row["prediction"], int) for row in values)
        category = _classify_source_error(
            all_views_wrong=not oracle_correct,
            source_selector_correct=selector_correct,
            oracle_correct=oracle_correct,
            has_gap_observation=has_gap_observation,
            boundary_has_gap_or_duplicate=bool(gap_width or duplicate_width),
            evaluation_ambiguous=position in ambiguous_positions,
        )
        counts[category] += 1
        if selector_correct:
            selected_score = float(selected["fidelity"])
            competing = [
                float(row["fidelity"])
                for row in values
                if int(row["view_index"]) != int(selected["view_index"])
            ]
            prediction_index = substitutions.get(position)
            operation = "substitution" if position in substitutions else "deletion"
            if position not in substitutions and position not in deletion_positions:
                operation = "evaluation_ambiguity"
            realization_cases.append(
                {
                    "passage_id": str(document["passage_id"]),
                    "source_position": position,
                    "error_operation": operation,
                    "multiplicity": len(values),
                    "covering_views": [int(row["view_index"]) for row in values],
                    "selected_views": [int(selected["view_index"])],
                    "correct_views": correct_views,
                    "fidelity_margin": (
                        selected_score - max(competing) if competing else None
                    ),
                    "nearest_boundary_index": nearest_boundary,
                    "boundary_offset_from_nominal_overlap_start": position
                    - 15 * (nearest_boundary + 1),
                    "pairwise_alignment_score": float(
                        assembled.boundary_alignment_scores[nearest_boundary]
                    ),
                    "boundary_left_width": int(
                        assembled.boundary_left_widths[nearest_boundary]
                    ),
                    "boundary_right_width": int(
                        assembled.boundary_right_widths[nearest_boundary]
                    ),
                    "boundary_gap_width": gap_width,
                    "boundary_duplicate_width": duplicate_width,
                    "third_view_support": {
                        "present": len(values) == 3,
                        "reference_support_count": len(correct_views),
                        "selected_token_support_count": sum(
                            row["prediction"] == selected["prediction"] for row in values
                        ),
                    },
                    "reference_token_id": reference[position],
                    "source_aligned_selected_token_id": (
                        selected["prediction"] if isinstance(selected["prediction"], int) else None
                    ),
                    "assembled_token_id": (
                        int(prediction[prediction_index])
                        if prediction_index is not None
                        else None
                    ),
                    "taxonomy": category,
                    "evaluation_ambiguous": position in ambiguous_positions,
                }
            )
    reconciled = sum(counts.values())
    expected = len(reference) - len(exact_positions)
    if reconciled != expected:
        raise AssertionError("Stage 6A source-position taxonomy did not reconcile")
    return (
        {
            "counts": {name: int(counts[name]) for name in _TAXONOMY},
            "total_residual_source_position_errors": expected,
            "prediction_insertions_stored_separately": len(alignment.prediction_insertions),
            "alternative_optimal_trace_ambiguity_positions": len(
                ambiguous_positions
            ),
            "reconciled": True,
        },
        realization_cases,
    )


def _error_taxonomy(
    document: Mapping[str, Any],
    candidates: Sequence[Candidate],
    assembled: FidelityStitchedSequenceIDs,
    reference: Sequence[int],
    *,
    stride: int,
    overlap_tokens: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    """Return the frozen post-hoc taxonomy for one development document.

    ``reference``, ``stride``, and ``overlap_tokens`` are accepted only by this
    evaluator-facing helper.  The source-free assembler API accepts none of
    them, which keeps ground truth and source coordinates out of inference.
    """

    frozen_reference = tuple(
        int(value) for value in document["prefixes"]["8"]["reference_ids"]
    )
    if tuple(int(value) for value in reference) != frozen_reference:
        raise RuntimeError(
            "Stage 6A taxonomy reference does not match the frozen document"
        )
    if (stride, overlap_tokens) != (15, 16):
        raise RuntimeError("Stage 6A taxonomy geometry changed")
    taxonomy, _cases = _error_diagnostics(
        document,
        candidates,
        assembled.content_ids,
        assembled,
        tie_tolerance=tie_tolerance,
    )
    return {
        **taxonomy,
        "source_position_error_count": taxonomy[
            "total_residual_source_position_errors"
        ],
        "counts_reconcile_source_position_errors": taxonomy["reconciled"],
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = (
        "nooverlap",
        "overlap_legacy_stitch",
        "overlap_fixed_owner",
        "fixed_coordinate_fidelity",
        "source_free_boundary_independent",
        "source_free_pairwise",
    )
    result: dict[str, Any] = {"documents": len(rows)}
    for method in methods:
        scores = [row[method] for row in rows]
        total = sum(int(row["source_token_count"]) for row in scores)
        result[method] = {
            "token_recovery": sum(int(row["exact_matched_token_count"]) for row in scores) / total,
            "normalized_edit_similarity": mean(float(row["normalized_edit_similarity"]) for row in scores),
            "total_edit_distance": sum(int(row["edit_distance"]) for row in scores),
            "substitutions": sum(int(row["substitution_count"]) for row in scores),
            "prediction_insertions": sum(int(row["prediction_insertion_count"]) for row in scores),
            "reference_deletions": sum(int(row["reference_deletion_count"]) for row in scores),
            "document_exact_fraction": mean(bool(row["document_exact"]) for row in scores),
        }
    source_total = sum(int(row["source_aligned"]["source_token_count"]) for row in rows)
    result["source_aligned_diagnostics"] = {
        "fixed_owner_recovery": sum(int(row["source_aligned"]["fixed_owner_correct"]) for row in rows) / source_total,
        "fidelity_selector_recovery": sum(int(row["source_aligned"]["selector_correct"]) for row in rows) / source_total,
        "best_of_views_oracle_recovery": sum(int(row["source_aligned"]["oracle_correct"]) for row in rows) / source_total,
    }
    result["transfer"] = _transfer_metrics(
        float(result["source_free_pairwise"]["token_recovery"]),
        float(result["source_aligned_diagnostics"]["fixed_owner_recovery"]),
        float(result["source_aligned_diagnostics"]["fidelity_selector_recovery"]),
    )
    result["transfer_reference"] = "paired_source_aligned_fixed_owner_diagnostic"
    result["source_free_gain_over_constrained_stitch"] = (
        float(result["source_free_pairwise"]["token_recovery"])
        - float(result["overlap_fixed_owner"]["token_recovery"])
    )
    result["progressive_propagation"] = {
        "documents_with_output_difference": sum(
            not bool(row["progressive_matches_boundary_independent_output"])
            for row in rows
        ),
        "documents_with_boundary_decision_difference": sum(
            int(row["progressive_boundary_decision_divergence_count"]) > 0
            for row in rows
        ),
        "total_boundary_decision_divergences": sum(
            int(row["progressive_boundary_decision_divergence_count"])
            for row in rows
        ),
        "token_recovery_delta_progressive_minus_boundary_independent": (
            float(result["source_free_pairwise"]["token_recovery"])
            - float(result["source_free_boundary_independent"]["token_recovery"])
        ),
    }
    return result


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# Stage 6A — Source-Free Fidelity-Guided Pairwise Assembly",
        "",
        f"- Development gate: `{result['gate']['state']}`",
        "- Cohort: existing Stage 3/4 exploratory 30 documents only.",
        "- This is development evidence only; it does not authorize a paper claim.",
        "- No Vec2Text rerun; GT/source alignment was used only for post-hoc evaluation.",
        "- No global DP/path method was implemented.",
        "- Both the requested accumulated progressive baseline and a boundary-independent adjacent pairwise chain are reported; neither searches arbitrary substrings.",
        "",
        "| Method | Token recovery | Normalized edit similarity | Document exact |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "nooverlap": "No-overlap strong",
        "overlap_legacy_stitch": "Existing legacy stitcher",
        "overlap_fixed_owner": "Overlap constrained fixed-owner stitcher",
        "fixed_coordinate_fidelity": "Existing fixed-index fidelity realization",
        "source_free_boundary_independent": "Source-Free Fidelity Assembly (boundary-independent diagnostic)",
        "source_free_pairwise": "Source-Free Fidelity Assembly (accumulated progressive; proposed)",
    }
    for name, label in labels.items():
        value = summary[name]
        lines.append(
            f"| {label} | {value['token_recovery']:.6f} | {value['normalized_edit_similarity']:.6f} | {value['document_exact_fraction']:.6f} |"
        )
    diagnostic = summary["source_aligned_diagnostics"]
    lines.extend(
        [
            "",
            "## Ceiling diagnostics (not attack methods)",
            "",
            f"- Source-Aligned Fixed Owner (development cohort): {diagnostic['fixed_owner_recovery']:.6f}",
            f"- Source-Aligned Fidelity Selector (development cohort): {diagnostic['fidelity_selector_recovery']:.6f}",
            f"- Best-of-Views Oracle (development cohort): {diagnostic['best_of_views_oracle_recovery']:.6f}",
            f"- Stage 5C frozen-holdout Source-Aligned Fidelity reference: {result['stage5c_reference_label']:.6f} (descriptive, unpaired, and not used by the gate)",
            "",
            "## Realization transfer",
            "",
            f"- Paired development realization gap: {summary['transfer']['realization_gap']:+.6f}",
            f"- Unpaired Stage 5C-reference minus development method: {result['stage5c_reference_minus_development_method']:+.6f}",
            f"- Gain recovery ratio (paired Source-Aligned Fixed Owner denominator): {summary['transfer']['gain_recovery_ratio']}",
            f"- Pairwise minus no-overlap CI95: {result['bootstrap']['ci95']['method_minus_nooverlap']}",
            f"- Pairwise minus Source-Aligned Fixed Owner CI95: {result['bootstrap']['ci95']['method_minus_fixed_owner']}",
            f"- Pairwise minus constrained fixed-owner stitch CI95: {result['bootstrap']['ci95']['method_minus_constrained_stitch']}",
            f"- Pairwise minus fixed-coordinate fidelity CI95: {result['bootstrap']['ci95']['method_minus_fixed_coordinate']}",
            f"- Progressive minus boundary-independent chain CI95: {result['bootstrap']['ci95']['method_minus_boundary_independent']}",
            f"- Progressive outputs differing from boundary-independent chain: {summary['progressive_propagation']['documents_with_output_difference']} / {summary['documents']}",
            f"- Progressive boundary-decision divergences: {summary['progressive_propagation']['total_boundary_decision_divergences']}",
            "",
            "## Error taxonomy",
            "",
            "These are hierarchical post-hoc diagnostic attributions, not causal provenance; nearest-boundary fields are observational.",
            "",
        ]
    )
    for name, count in result["error_taxonomy"]["counts"].items():
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            f"- Prediction insertions (stored separately): {result['error_taxonomy']['prediction_insertions_stored_separately']}",
            f"- Source-aligned-correct but source-free-wrong cases: {len(result['source_aligned_correct_source_free_wrong_cases'])}",
            "",
            "The simple pairwise gate passed, so the preregistered condition for implementing a global DP/path assembler was not met. Stage 6B freeze and a still-unseen Stage 6C cohort are required before a confirmatory attack claim.",
        ]
    )
    _write_text_atomic(path, "\n".join(lines) + "\n")


def run_stage6a(config: Mapping[str, Any]) -> dict[str, Any]:
    values = _validate_inputs(config)
    overlap = values["exploratory_overlap_cohort"]
    nooverlap = values["exploratory_nooverlap_manifest"]
    overlap_map = _candidate_map(values["exploratory_overlap_strong_cache"])
    nooverlap_map = _candidate_map(values["exploratory_nooverlap_strong_cache"])
    nooverlap_by_id = {str(row["passage_id"]): row for row in nooverlap["documents"]}
    assembly = config["pairwise_assembly"]
    overlap_tokens = int(config["geometry"]["nominal_overlap_tokens"])
    max_shift = int(assembly["max_boundary_shift"])
    deviation_cost = float(assembly["deviation_cost"])
    tie_tolerance = float(assembly["tie_tolerance"])
    rows: list[dict[str, Any]] = []
    taxonomy_counts: Counter[str] = Counter()
    realization_cases: list[dict[str, Any]] = []
    runtime = 0.0
    boundary_independent_runtime = 0.0
    for document in overlap["documents"]:
        passage_id = str(document["passage_id"])
        nooverlap_document = nooverlap_by_id[passage_id]
        overlap_candidates = tuple(overlap_map[f"{passage_id}:{index}"] for index in range(8))
        nooverlap_candidates = tuple(
            nooverlap_map[f"{passage_id}:nooverlap:{index}"] for index in range(5)
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
        legacy = stitch_candidate_sequence(
            overlap_candidates,
            overlap_tokens=overlap_tokens,
        )
        fixed_coordinate = fixed_coordinate_fidelity_assembly(
            overlap_candidates,
            content_tokens=31,
            stride=15,
            tie_tolerance=tie_tolerance,
        )
        boundary_independent_started = time.perf_counter()
        boundary_independent = assemble_source_free_boundary_independent(
            overlap_candidates,
            overlap_tokens=overlap_tokens,
            max_boundary_shift=max_shift,
            deviation_cost=deviation_cost,
            tie_tolerance=tie_tolerance,
        )
        boundary_independent_elapsed = (
            time.perf_counter() - boundary_independent_started
        )
        boundary_independent_runtime += boundary_independent_elapsed
        started = time.perf_counter()
        proposed = assemble_source_free_pairwise(
            overlap_candidates,
            overlap_tokens=overlap_tokens,
            max_boundary_shift=max_shift,
            deviation_cost=deviation_cost,
            tie_tolerance=tie_tolerance,
        )
        elapsed = time.perf_counter() - started
        runtime += elapsed
        source_aligned, _ = _score_document(
            document,
            overlap_candidates,
            decision_scores={index: float(candidate.local_score) for index, candidate in enumerate(overlap_candidates)},
            tie_tolerance=tie_tolerance,
        )
        method_scores = {
            "nooverlap": _document_score(nooverlap_prediction, reference),
            "overlap_legacy_stitch": _document_score(legacy.content_ids, reference),
            "overlap_fixed_owner": _document_score(fixed.content_ids, reference),
            "fixed_coordinate_fidelity": _document_score(fixed_coordinate, reference),
            "source_free_boundary_independent": _document_score(
                boundary_independent.content_ids, reference
            ),
            "source_free_pairwise": _document_score(proposed.content_ids, reference),
        }
        taxonomy, cases = _error_diagnostics(
            document,
            overlap_candidates,
            proposed.content_ids,
            proposed,
            tie_tolerance=tie_tolerance,
        )
        taxonomy_counts.update(taxonomy["counts"])
        realization_cases.extend(cases)
        row: dict[str, Any] = {
            "passage_id": passage_id,
            **method_scores,
            "source_aligned": {
                "source_token_count": int(source_aligned["source_token_count"]),
                "fixed_owner_correct": int(source_aligned["fixed_owner_correct"]),
                "selector_correct": int(source_aligned["selector_correct"]),
                "oracle_correct": int(source_aligned["oracle_correct"]),
            },
            "assembly_runtime_seconds": elapsed,
            "boundary_independent_assembly_runtime_seconds": (
                boundary_independent_elapsed
            ),
            "progressive_matches_boundary_independent_output": (
                proposed.content_ids == boundary_independent.content_ids
            ),
            "progressive_boundary_decision_divergence_count": sum(
                left != right
                for left, right in zip(
                    zip(
                        proposed.boundary_left_widths,
                        proposed.boundary_right_widths,
                        proposed.boundary_owner_views,
                    ),
                    zip(
                        boundary_independent.boundary_left_widths,
                        boundary_independent.boundary_right_widths,
                        boundary_independent.boundary_owner_views,
                    ),
                )
            ),
            "assembly_diagnostic": {
                "boundary_left_widths": list(proposed.boundary_left_widths),
                "boundary_right_widths": list(proposed.boundary_right_widths),
                "boundary_alignment_scores": list(proposed.boundary_alignment_scores),
                "boundary_owner_views": list(proposed.boundary_owner_views),
                "inter_boundary_gap_widths": list(proposed.inter_boundary_gap_widths),
                "inter_boundary_duplicate_widths": list(proposed.inter_boundary_duplicate_widths),
            },
            "error_taxonomy": taxonomy,
        }
        row["deltas"] = {
            "method_minus_nooverlap": method_scores["source_free_pairwise"]["token_recovery"] - method_scores["nooverlap"]["token_recovery"],
            "method_minus_fixed_owner": method_scores["source_free_pairwise"]["token_recovery"] - source_aligned["fixed_owner_recovery"],
            "method_minus_constrained_stitch": method_scores["source_free_pairwise"]["token_recovery"] - method_scores["overlap_fixed_owner"]["token_recovery"],
            "method_minus_fixed_coordinate": method_scores["source_free_pairwise"]["token_recovery"] - method_scores["fixed_coordinate_fidelity"]["token_recovery"],
            "method_minus_boundary_independent": method_scores["source_free_pairwise"]["token_recovery"] - method_scores["source_free_boundary_independent"]["token_recovery"],
            "source_aligned_minus_method": source_aligned["selector_recovery"] - method_scores["source_free_pairwise"]["token_recovery"],
        }
        rows.append(row)

    summary = _summary(rows)
    evaluation = config["evaluation"]
    bootstrap = _bootstrap(
        rows,
        iterations=int(evaluation["bootstrap_iterations"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    source_path = Path(__file__)
    behavior_paths = {
        "stage6a_source_free_assembly": source_path,
        "diagnostics": source_path.with_name("diagnostics.py"),
        "reconstruction": source_path.with_name("reconstruction.py"),
        "stage1_strong_saturation": source_path.with_name(
            "stage1_strong_saturation.py"
        ),
        "stage3_longdoc_strong": source_path.with_name(
            "stage3_longdoc_strong.py"
        ),
        "stage4_document_correction": source_path.with_name(
            "stage4_document_correction.py"
        ),
        "stage5c_fidelity_selector": source_path.with_name(
            "stage5c_fidelity_selector.py"
        ),
        "stage5d_assembly": source_path.with_name("stage5d_assembly.py"),
    }
    behavior_hashes = {
        name: _sha256(path) for name, path in behavior_paths.items()
    }
    result = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE6A_SOURCE_FREE_ASSEMBLY_COMPLETE",
        "estimand": "Existing exploratory Stage 3/4 K=8 30-document development cohort only.",
        "config": {"path": str(config["resolved_config"]), "sha256": _sha256(Path(str(config["resolved_config"])))},
        "implementation_sha256": _sha256(source_path),
        "behavior_file_sha256": behavior_hashes,
        "repository_commit": None,
        "repository_commit_status": "unavailable_not_a_git_worktree",
        "inputs": {
            name: {"path": str(config["resolved_inputs"][name]["path"]), "sha256": str(config["resolved_inputs"][name]["sha256"])}
            for name in _INPUT_NAMES
        },
        "summary": summary,
        "bootstrap": bootstrap,
        "stage5c_reference_label": float(evaluation["stage5c_reference_source_aligned_fidelity"]),
        "stage5c_reference_minus_development_method": float(
            evaluation["stage5c_reference_source_aligned_fidelity"]
        )
        - float(summary["source_free_pairwise"]["token_recovery"]),
        "error_taxonomy": {
            "counts": {name: int(taxonomy_counts[name]) for name in _TAXONOMY},
            "total_residual_source_position_errors": sum(taxonomy_counts.values()),
            "operational_priority": [
                "EVALUATION_AMBIGUITY",
                "VIEW_ALL_WRONG",
                "FIDELITY_WRONG_DIRECTION",
                "GAP_HANDLING_FAILURE",
                "GLOBAL_ASSEMBLY_FAILURE",
                "ALIGNMENT_FAILURE",
            ],
            "definitions": {
                "VIEW_ALL_WRONG": "No covering reconstructed view is correct at the GT position.",
                "ALIGNMENT_FAILURE": "A correct source-aligned choice exists but candidate-only assembly does not realize it, without an observable gap-width trigger.",
                "FIDELITY_WRONG_DIRECTION": "A view is correct but the frozen source-aligned fidelity choice points elsewhere.",
                "GAP_HANDLING_FAILURE": "The residual is associated with a deletion observation or candidate-coordinate gap/duplicate width; this post-hoc label is not causal proof.",
                "GLOBAL_ASSEMBLY_FAILURE": "Reserved for a global method; zero because Stage 6A implements no global method.",
                "EVALUATION_AMBIGUITY": "The deterministic trace marks the position wrong, but an exact match for it lies on another globally optimal Levenshtein path.",
            },
            "attribution_scope": "Post-hoc hierarchical diagnostic, not causal provenance. Boundary fields use the nearest public-geometry boundary and are observational.",
            "evaluation_ambiguity_assessed": True,
            "prediction_insertions_stored_separately": sum(
                int(row["source_free_pairwise"]["prediction_insertion_count"]) for row in rows
            ),
            "reconciled": sum(taxonomy_counts.values())
            == sum(136 - int(row["source_free_pairwise"]["exact_matched_token_count"]) for row in rows),
        },
        "source_aligned_correct_source_free_wrong_cases": realization_cases,
        "runtime": {
            "accumulated_progressive_total_seconds": runtime,
            "accumulated_progressive_seconds_per_document": runtime / len(rows),
            "boundary_independent_pairwise_total_seconds": (
                boundary_independent_runtime
            ),
            "boundary_independent_pairwise_seconds_per_document": (
                boundary_independent_runtime / len(rows)
            ),
        },
        "gate": _gate(summary, bootstrap),
        "vec2text_reexecuted": False,
        "ground_truth_used_by_inference": False,
        "source_alignment_used_by_inference": False,
        "global_dp_or_path_method_implemented": False,
        "stage5bc_holdout_used_for_tuning_or_inference": False,
    }
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / str(config["result_filename"]), result)
    _write_json_atomic(
        output_dir / str(config["trajectory_filename"]),
        {
            "schema_version": 1,
            "config_sha256": result["config"]["sha256"],
            "implementation_sha256": result["implementation_sha256"],
            "behavior_file_sha256": result["behavior_file_sha256"],
            "trajectories": rows,
        },
    )
    _write_report(output_dir / str(config["report_filename"]), result)
    return result
