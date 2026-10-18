"""Stage 4 pre-method diagnostics for long-document overlap correction.

Stage 4A strengthens only the deterministic stitching baseline. Stage 4B uses
ground truth after that baseline is frozen to measure a source-position
Best-of-Views ceiling and observable confidence signals. No reconstruction,
candidate selection, voting, or path-decoding method is implemented here.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median
import time
from typing import Any, Mapping, Sequence

import yaml

from .diagnostics import (
    StitchedSequenceIDs,
    align_token_ids,
    stitch_candidate_sequence_constrained,
)
from .reconstruction import Candidate
from .stage1_strong_saturation import _candidate_from_dict, _write_json_atomic
from .stage3_longdoc_strong import (
    _aggregate_prefix,
    _distribution,
    _evaluate_document_prefix,
    _percentile,
    _safe_ratio,
)


_IMPLEMENTATION_VERSION = "stage4-document-correction-gate-v1"
_FIDELITY_TOLERANCE = 1e-12
_DELETION = "<DEL>"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stage4_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Stage 4 config must be a YAML mapping")

    resolved = dict(raw)
    for key in ("cohort_artifact", "strong_top1_cache", "stage3_result"):
        candidate = Path(str(raw[key]))
        resolved[f"resolved_{key}"] = str(
            (candidate if candidate.is_absolute() else config_path.parent / candidate)
            .resolve()
        )
    resolved["resolved_config"] = str(config_path.resolve())
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _validate_inputs(
    config: Mapping[str, Any],
    cohort: Mapping[str, Any],
    cache: Mapping[str, Any],
    stage3: Mapping[str, Any],
) -> dict[str, Any]:
    cohort_path = Path(str(config["resolved_cohort_artifact"]))
    cache_path = Path(str(config["resolved_strong_top1_cache"]))
    stage3_path = Path(str(config["resolved_stage3_result"]))
    expected_hashes = {
        "cohort": str(config["cohort_sha256"]),
        "strong_top1_cache": str(config["strong_top1_cache_sha256"]),
        "stage3_result": str(config["stage3_result_sha256"]),
    }
    actual_hashes = {
        "cohort": _sha256(cohort_path),
        "strong_top1_cache": _sha256(cache_path),
        "stage3_result": _sha256(stage3_path),
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            f"Stage 4 frozen artifact hash mismatch: expected={expected_hashes} "
            f"actual={actual_hashes}"
        )

    k_values = tuple(int(value) for value in config["k_values"])
    if k_values != (2, 3, 4, 6, 8):
        raise RuntimeError("Stage 4 K grid changed")
    documents = list(cohort["documents"])
    expected_documents = int(config["expected_documents"])
    if len(documents) != expected_documents:
        raise RuntimeError("Stage 4 cohort size changed")
    manifest = list(cohort["manifest"])
    if [document["passage_id"] for document in documents] != manifest:
        raise RuntimeError("Stage 4 cohort order changed")
    expected_view_keys = [
        f"{document['passage_id']}:{window_index}"
        for document in documents
        for window_index in range(8)
    ]
    if list(cache["view_keys"]) != expected_view_keys:
        raise RuntimeError("Stage 4 strong-cache view order changed")
    candidates = list(cache["top1_candidates"])
    if len(candidates) != expected_documents * 8:
        raise RuntimeError("Stage 4 strong-cache candidate count changed")
    if str(cache["cohort_sha256"]) != actual_hashes["cohort"]:
        raise RuntimeError("Strong cache refers to another Stage 3A cohort")
    if not bool(cache["public_top1_exact_id_match"]):
        raise RuntimeError("Strong cache did not preserve public Top-1")
    if str(cache["strong_reproduction_setting"]) != "correction50+sbeam8":
        raise RuntimeError("Stage 4 cache is not correction50+sbeam8")
    if stage3["gate"]["state"] != "LONGDOC_RECOVERABLE_REDUNDANCY_GO":
        raise RuntimeError("Stage 4 requires the completed Stage 3 GO artifact")
    if stage3["cohort"]["manifest"] != manifest:
        raise RuntimeError("Stage 3 result manifest differs from Stage 4 cohort")
    if stage3["strong_top1_cache"]["sha256"] != actual_hashes["strong_top1_cache"]:
        raise RuntimeError("Stage 3 result refers to another strong cache")
    return {
        "hashes": actual_hashes,
        "manifest_match": True,
        "candidate_count": len(candidates),
        "public_instrumented_top1_match": True,
        "vec2text_reexecuted": False,
    }


def _aligned_view(
    candidate: Candidate,
    window: Mapping[str, Any],
    *,
    selected_start: int,
) -> dict[str, Any]:
    reference = tuple(int(value) for value in window["content_ids"])
    alignment = align_token_ids(candidate.content_ids, reference)
    labels: list[int | str] = [_DELETION] * len(reference)
    prediction_to_global: dict[int, int] = {}
    relative_start = int(window["document_token_start"]) - selected_start
    for prediction_index, reference_index in (
        alignment.exact_matches + alignment.substitutions
    ):
        labels[reference_index] = int(candidate.content_ids[prediction_index])
        prediction_to_global[prediction_index] = relative_start + reference_index
    return {
        "alignment": alignment,
        "labels": tuple(labels),
        "prediction_to_global": prediction_to_global,
        "relative_start": relative_start,
    }


def _observations(
    document: Mapping[str, Any],
    candidates: Sequence[Candidate],
    *,
    k: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    reference = tuple(int(value) for value in document["prefixes"][str(k)]["reference_ids"])
    selected_start = int(document["selected_start"])
    by_position: list[list[dict[str, Any]]] = [[] for _ in reference]
    aligned_views: list[dict[str, Any]] = []
    for view_index, (window, candidate) in enumerate(
        zip(document["windows"][:k], candidates[:k])
    ):
        aligned = _aligned_view(candidate, window, selected_start=selected_start)
        aligned_views.append(aligned)
        for local_position, prediction in enumerate(aligned["labels"]):
            global_position = aligned["relative_start"] + local_position
            by_position[global_position].append(
                {
                    "view_index": view_index,
                    "prediction": prediction,
                    "correct": prediction == reference[global_position],
                    "fidelity": float(candidate.local_score),
                }
            )
    expected_multiplicity = tuple(
        int(value)
        for value in document["prefixes"][str(k)]["observation_multiplicity"]
    )
    if tuple(len(values) for values in by_position) != expected_multiplicity:
        raise AssertionError("Stage 4 observation multiplicity changed")
    return tuple(tuple(values) for values in by_position), tuple(aligned_views)


def _agreement_counts(
    observations: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter[str]] = {"2": Counter(), "3": Counter()}
    for values in observations:
        multiplicity = len(values)
        if multiplicity < 2:
            continue
        group = groups[str(multiplicity)]
        predictions = [value["prediction"] for value in values]
        correct_count = sum(bool(value["correct"]) for value in values)
        unanimous = len(set(predictions)) == 1
        has_gap = _DELETION in predictions
        group["source_token_count"] += 1
        group["gap_containing_count"] += int(has_gap)
        if unanimous:
            group["UNANIMOUS_count"] += 1
            group["UNANIMOUS_correct_count"] += int(correct_count == multiplicity)
        else:
            group["DISAGREEMENT_count"] += 1
            if correct_count == 0:
                group["DISAGREEMENT_all_wrong_count"] += 1
            elif correct_count == 1:
                group["DISAGREEMENT_one_correct_count"] += 1
            else:
                group["DISAGREEMENT_two_or_more_correct_count"] += 1
        if not has_gap:
            group["GAP_FREE_count"] += 1
            if unanimous:
                group["GAP_FREE_UNANIMOUS_count"] += 1
                group["GAP_FREE_UNANIMOUS_correct_count"] += int(
                    correct_count == multiplicity
                )
            else:
                group["GAP_FREE_DISAGREEMENT_count"] += 1
                group["GAP_FREE_DISAGREEMENT_any_correct_count"] += int(
                    correct_count > 0
                )
    return {group: dict(counts) for group, counts in groups.items()}


def _fidelity_counts(
    observations: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    def empty_group() -> dict[str, Any]:
        return {
            "event_count": 0,
            "recoverable_event_count": 0,
            "all_wrong_event_count": 0,
            "positive_margin_count": 0,
            "tie_margin_count": 0,
            "negative_margin_count": 0,
            "fractional_credit_sum": 0.0,
            "chance_credit_sum": 0.0,
            "unique_top_count": 0,
            "unique_top_correct_count": 0,
            "margins": [],
        }

    groups: dict[str, dict[str, Any]] = {
        "all": empty_group(),
        "2": empty_group(),
        "3": empty_group(),
    }
    for values in observations:
        multiplicity = len(values)
        correct_count = sum(bool(value["correct"]) for value in values)
        predictions = [value["prediction"] for value in values]
        if multiplicity < 2 or len(set(predictions)) == 1:
            continue
        top_score = max(float(value["fidelity"]) for value in values)
        top = [
            value
            for value in values
            if abs(float(value["fidelity"]) - top_score) <= _FIDELITY_TOLERANCE
        ]
        fractional_credit = sum(bool(value["correct"]) for value in top) / len(top)
        chance_credit = correct_count / multiplicity
        for group_name in ("all", str(multiplicity)):
            group = groups[group_name]
            group["event_count"] = int(group.get("event_count", 0)) + 1
            group["fractional_credit_sum"] = float(
                group.get("fractional_credit_sum", 0.0)
            ) + fractional_credit
            group["chance_credit_sum"] = float(
                group.get("chance_credit_sum", 0.0)
            ) + chance_credit
            group["unique_top_count"] = int(group.get("unique_top_count", 0)) + int(
                len(top) == 1
            )
            group["unique_top_correct_count"] = int(
                group.get("unique_top_correct_count", 0)
            ) + int(len(top) == 1 and bool(top[0]["correct"]))
            if correct_count == 0:
                group["all_wrong_event_count"] = int(
                    group.get("all_wrong_event_count", 0)
                ) + 1
                continue

            correct_scores = [
                float(value["fidelity"]) for value in values if value["correct"]
            ]
            wrong_scores = [
                float(value["fidelity"])
                for value in values
                if not value["correct"]
            ]
            margin = max(correct_scores) - max(wrong_scores)
            group["recoverable_event_count"] = int(
                group.get("recoverable_event_count", 0)
            ) + 1
            group["positive_margin_count"] = int(
                group.get("positive_margin_count", 0)
            ) + int(margin > _FIDELITY_TOLERANCE)
            group["tie_margin_count"] = int(
                group.get("tie_margin_count", 0)
            ) + int(abs(margin) <= _FIDELITY_TOLERANCE)
            group["negative_margin_count"] = int(
                group.get("negative_margin_count", 0)
            ) + int(margin < -_FIDELITY_TOLERANCE)
            group["margins"].append(margin)
    return groups


def _candidate_aligned_boundaries(
    document: Mapping[str, Any],
    candidates: Sequence[Candidate],
    observations: Sequence[Sequence[Mapping[str, Any]]],
    aligned_views: Sequence[Mapping[str, Any]],
    stitched: StitchedSequenceIDs,
    *,
    k: int,
    overlap_tokens: int,
    stride: int,
) -> list[dict[str, Any]]:
    reference = tuple(int(value) for value in document["prefixes"][str(k)]["reference_ids"])
    rows: list[dict[str, Any]] = []
    for boundary_index in range(k - 1):
        left_width = int(stitched.aligned_left_tokens[boundary_index])
        right_width = int(stitched.consumed_right_tokens[boundary_index])
        left = candidates[boundary_index]
        right = candidates[boundary_index + 1]
        left_slice = tuple(left.content_ids[-left_width:])
        right_slice = tuple(right.content_ids[:right_width])
        alignment = align_token_ids(left_slice, right_slice)
        left_offset = len(left.content_ids) - left_width
        exact_agreements = 0
        same_source = 0
        correct_agreements = 0
        for left_local, right_local in alignment.exact_matches:
            exact_agreements += 1
            left_prediction_index = left_offset + left_local
            right_prediction_index = right_local
            left_global = aligned_views[boundary_index]["prediction_to_global"].get(
                left_prediction_index
            )
            right_global = aligned_views[boundary_index + 1][
                "prediction_to_global"
            ].get(right_prediction_index)
            corresponds = left_global is not None and left_global == right_global
            same_source += int(corresponds)
            correct_agreements += int(
                corresponds and left_slice[left_local] == reference[left_global]
            )

        substitution_same_source = 0
        directional_event_count = 0
        directional_recoverable_count = 0
        directional_all_wrong_count = 0
        directional_fractional_credit = 0.0
        directional_chance_credit = 0.0
        for left_local, right_local in alignment.substitutions:
            left_prediction_index = left_offset + left_local
            right_prediction_index = right_local
            left_global = aligned_views[boundary_index]["prediction_to_global"].get(
                left_prediction_index
            )
            right_global = aligned_views[boundary_index + 1][
                "prediction_to_global"
            ].get(right_prediction_index)
            corresponds = left_global is not None and left_global == right_global
            if not corresponds:
                continue
            substitution_same_source += 1
            left_correct = left_slice[left_local] == reference[left_global]
            right_correct = right_slice[right_local] == reference[right_global]
            correct_count = int(left_correct) + int(right_correct)
            directional_event_count += 1
            directional_chance_credit += correct_count / 2.0
            directional_recoverable_count += int(correct_count > 0)
            directional_all_wrong_count += int(correct_count == 0)
            score_delta = float(left.local_score) - float(right.local_score)
            if abs(score_delta) <= _FIDELITY_TOLERANCE:
                directional_fractional_credit += correct_count / 2.0
            elif score_delta > 0:
                directional_fractional_credit += int(left_correct)
            else:
                directional_fractional_credit += int(right_correct)

        shared_start = (boundary_index + 1) * stride
        shared_positions = range(shared_start, shared_start + overlap_tokens)
        both_correct = 0
        mixed = 0
        both_wrong = 0
        correct_observations = 0
        left_correct_tokens = 0
        right_correct_tokens = 0
        for position in shared_positions:
            by_view = {value["view_index"]: value for value in observations[position]}
            left_correct = bool(by_view[boundary_index]["correct"])
            right_correct = bool(by_view[boundary_index + 1]["correct"])
            left_correct_tokens += int(left_correct)
            right_correct_tokens += int(right_correct)
            correct_observations += int(left_correct) + int(right_correct)
            if left_correct and right_correct:
                both_correct += 1
            elif left_correct or right_correct:
                mixed += 1
            else:
                both_wrong += 1
        observable_disagreement = (
            float(stitched.alignment_scores[boundary_index])
            < 1.0 - _FIDELITY_TOLERANCE
        )
        if abs(float(left.local_score) - float(right.local_score)) <= _FIDELITY_TOLERANCE:
            fidelity_selected_correct_tokens = (
                left_correct_tokens + right_correct_tokens
            ) / 2.0
        elif float(left.local_score) > float(right.local_score):
            fidelity_selected_correct_tokens = float(left_correct_tokens)
        else:
            fidelity_selected_correct_tokens = float(right_correct_tokens)
        random_expected_correct_tokens = (
            left_correct_tokens + right_correct_tokens
        ) / 2.0
        rows.append(
            {
                "boundary_index": boundary_index,
                "alignment_score": float(stitched.alignment_scores[boundary_index]),
                "aligned_left_tokens": left_width,
                "consumed_right_tokens": right_width,
                "candidate_exact_agreement_count": exact_agreements,
                "candidate_exact_agreement_same_source_count": same_source,
                "candidate_exact_agreement_correct_count": correct_agreements,
                "candidate_alignment_substitution_count": len(alignment.substitutions),
                "candidate_alignment_substitution_same_source_count": (
                    substitution_same_source
                ),
                "candidate_aligned_directional_event_count": directional_event_count,
                "candidate_aligned_directional_recoverable_count": (
                    directional_recoverable_count
                ),
                "candidate_aligned_directional_all_wrong_count": (
                    directional_all_wrong_count
                ),
                "candidate_aligned_fidelity_fractional_credit_sum": (
                    directional_fractional_credit
                ),
                "candidate_aligned_random_credit_sum": directional_chance_credit,
                "candidate_alignment_left_insertion_count": len(
                    alignment.prediction_insertions
                ),
                "candidate_alignment_right_deletion_count": len(
                    alignment.reference_deletions
                ),
                "gt_both_correct_source_tokens": both_correct,
                "gt_mixed_source_tokens": mixed,
                "gt_both_wrong_source_tokens": both_wrong,
                "gt_shared_observation_accuracy": correct_observations
                / (2 * overlap_tokens),
                "observable_boundary_disagreement": observable_disagreement,
                "gt_left_correct_shared_tokens": left_correct_tokens,
                "gt_right_correct_shared_tokens": right_correct_tokens,
                "fidelity_selected_correct_shared_tokens": (
                    fidelity_selected_correct_tokens
                ),
                "random_expected_correct_shared_tokens": (
                    random_expected_correct_tokens
                ),
                "fidelity_selected_better_side": int(
                    fidelity_selected_correct_tokens
                    > random_expected_correct_tokens + _FIDELITY_TOLERANCE
                ),
                "fidelity_selected_worse_side": int(
                    fidelity_selected_correct_tokens
                    < random_expected_correct_tokens - _FIDELITY_TOLERANCE
                ),
                "views_equal_shared_correctness": int(
                    left_correct_tokens == right_correct_tokens
                ),
            }
        )
    return rows


def _oracle_row(
    reference: Sequence[int],
    multiplicity: Sequence[int],
    observations: Sequence[Sequence[Mapping[str, Any]]],
    improved_row: Mapping[str, Any],
) -> dict[str, Any]:
    final_alignment = align_token_ids(
        improved_row["stitched_prediction_ids"], reference
    )
    baseline_correct = {
        reference_index for _, reference_index in final_alignment.exact_matches
    }
    available = {
        position
        for position, values in enumerate(observations)
        if any(bool(value["correct"]) for value in values)
    }
    baseline_errors = set(range(len(reference))) - baseline_correct
    recoverable = baseline_errors & available
    residual = baseline_errors - available
    recoverable_multiply = {
        position for position in recoverable if int(multiplicity[position]) >= 2
    }
    recoverable_single = recoverable - recoverable_multiply
    residual_after_multiview_only = baseline_errors - recoverable_multiply
    cause_by_position = {
        int(value["source_position"]): str(value["cause"])
        for value in improved_row["final_error_attribution_by_source_position"]
    }
    gain_attribution = Counter(
        cause_by_position[position] for position in recoverable
    )
    return {
        "total_source_tokens": len(reference),
        "improved_baseline_source_error_count": len(baseline_errors),
        "improved_baseline_prediction_insertion_count": int(
            improved_row["final_prediction_insertion_count"]
        ),
        "oracle_recoverable_source_position_count": len(recoverable),
        "oracle_recoverable_single_observed_count": len(recoverable_single),
        "oracle_recoverable_multiply_observed_count": len(recoverable_multiply),
        "oracle_residual_source_error_count": len(residual),
        "oracle_normalized_source_token_error": len(residual) / len(reference),
        "oracle_source_token_recovery": 1.0 - len(residual) / len(reference),
        "oracle_source_position_document_exact": not residual,
        "oracle_absolute_token_improvement": len(recoverable),
        "multiview_oracle_absolute_token_improvement": len(recoverable_multiply),
        "multiview_only_oracle_residual_source_error_count": len(
            residual_after_multiview_only
        ),
        "multiview_only_oracle_source_token_recovery": (
            1.0 - len(residual_after_multiview_only) / len(reference)
        ),
        "multiview_only_oracle_source_position_document_exact": not (
            residual_after_multiview_only
        ),
        "multiview_oracle_relative_source_error_reduction": _safe_ratio(
            len(recoverable_multiply), len(baseline_errors)
        ),
        "oracle_relative_source_error_reduction": _safe_ratio(
            len(recoverable), len(baseline_errors)
        ),
        "recoverable_gain_attribution": {
            cause: gain_attribution[cause]
            for cause in (
                "SOURCE_VIEW_WRONG",
                "STITCH_ONLY_ERROR",
                "ALIGNMENT_AMBIGUITY",
            )
        },
        "definition": (
            "Non-degrading GT-aligned source-position Best-of-Views ceiling: a "
            "baseline-correct source position remains correct; otherwise it is "
            "corrected iff any observing independent view is locally exact. "
            "Prediction insertions are reported separately and are not silently "
            "counted as token-level oracle corrections."
        ),
    }


def _evaluate_document(
    document: Mapping[str, Any],
    candidates: Sequence[Candidate],
    *,
    k: int,
    overlap_tokens: int,
    stride: int,
    max_boundary_shift: int,
    deviation_cost: float,
) -> tuple[dict[str, Any], StitchedSequenceIDs, dict[str, Any]]:
    stitched = stitch_candidate_sequence_constrained(
        candidates[:k],
        overlap_tokens=overlap_tokens,
        max_boundary_shift=max_boundary_shift,
        deviation_cost=deviation_cost,
    )
    records = tuple({"telemetry": {"elapsed_seconds": 0.0}} for _ in range(k))
    row = _evaluate_document_prefix(
        document,
        candidates[:k],
        records,
        k=k,
        content_tokens=31,
        overlap_tokens=overlap_tokens,
        stride=stride,
        minimum_alignment=0.5,
        stitch_sequence=lambda _candidates: stitched,
    )
    observations, aligned_views = _observations(document, candidates, k=k)
    reference = tuple(int(value) for value in document["prefixes"][str(k)]["reference_ids"])
    multiplicity = tuple(
        int(value)
        for value in document["prefixes"][str(k)]["observation_multiplicity"]
    )
    diagnostic = {
        "oracle": _oracle_row(reference, multiplicity, observations, row),
        "agreement": {"by_multiplicity": _agreement_counts(observations)},
        "fidelity": {"by_multiplicity": _fidelity_counts(observations)},
        "candidate_aligned_boundaries": _candidate_aligned_boundaries(
            document,
            candidates,
            observations,
            aligned_views,
            stitched,
            k=k,
            overlap_tokens=overlap_tokens,
            stride=stride,
        ),
    }
    return row, stitched, diagnostic


def _sum_nested_counts(
    diagnostics: Sequence[Mapping[str, Any]], group: str
) -> Counter[str]:
    total: Counter[str] = Counter()
    for diagnostic in diagnostics:
        total.update(diagnostic["agreement"]["by_multiplicity"][group])
    return total


def _agreement_summary(counts: Mapping[str, int]) -> dict[str, Any]:
    unanimous = int(counts.get("UNANIMOUS_count", 0))
    disagreement = int(counts.get("DISAGREEMENT_count", 0))
    disagreement_any_correct = int(counts.get("DISAGREEMENT_one_correct_count", 0)) + int(
        counts.get("DISAGREEMENT_two_or_more_correct_count", 0)
    )
    gap_free_unanimous = int(counts.get("GAP_FREE_UNANIMOUS_count", 0))
    gap_free_disagreement = int(counts.get("GAP_FREE_DISAGREEMENT_count", 0))
    return {
        **{key: int(value) for key, value in counts.items()},
        "P_correct_given_UNANIMOUS": _safe_ratio(
            int(counts.get("UNANIMOUS_correct_count", 0)), unanimous
        ),
        "P_any_correct_given_DISAGREEMENT": _safe_ratio(
            disagreement_any_correct, disagreement
        ),
        "gap_free_sensitivity": {
            "P_correct_given_UNANIMOUS": _safe_ratio(
                int(counts.get("GAP_FREE_UNANIMOUS_correct_count", 0)),
                gap_free_unanimous,
            ),
            "P_any_correct_given_DISAGREEMENT": _safe_ratio(
                int(counts.get("GAP_FREE_DISAGREEMENT_any_correct_count", 0)),
                gap_free_disagreement,
            ),
        },
    }


def _aggregate_fidelity(
    diagnostics: Sequence[Mapping[str, Any]], group: str
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    fractional = 0.0
    chance = 0.0
    margins: list[float] = []
    for diagnostic in diagnostics:
        values = diagnostic["fidelity"]["by_multiplicity"][group]
        for key in (
            "event_count",
            "recoverable_event_count",
            "all_wrong_event_count",
            "positive_margin_count",
            "tie_margin_count",
            "negative_margin_count",
            "unique_top_count",
            "unique_top_correct_count",
        ):
            counts[key] += int(values.get(key, 0))
        fractional += float(values.get("fractional_credit_sum", 0.0))
        chance += float(values.get("chance_credit_sum", 0.0))
        margins.extend(float(value) for value in values.get("margins", []))
    events = counts["event_count"]
    fractional_accuracy = _safe_ratio(fractional, events)
    chance_accuracy = _safe_ratio(chance, events)
    count_keys = (
        "event_count",
        "recoverable_event_count",
        "all_wrong_event_count",
        "positive_margin_count",
        "tie_margin_count",
        "negative_margin_count",
        "unique_top_count",
        "unique_top_correct_count",
    )
    return {
        **{key: counts[key] for key in count_keys},
        "correct_max_fidelity_beats_wrong_fraction": _safe_ratio(
            counts["positive_margin_count"], counts["recoverable_event_count"]
        ),
        "fidelity_margin": _distribution(margins),
        "fractional_top_fidelity_accuracy": fractional_accuracy,
        "random_view_expected_accuracy": chance_accuracy,
        "fractional_accuracy_uplift_over_random": (
            None
            if fractional_accuracy is None or chance_accuracy is None
            else fractional_accuracy - chance_accuracy
        ),
        "unique_top_coverage": _safe_ratio(counts["unique_top_count"], events),
        "unique_top_accuracy": _safe_ratio(
            counts["unique_top_correct_count"], counts["unique_top_count"]
        ),
        "tie_policy": (
            "Scores within 1e-12 share fractional credit; no index-based "
            "positional tie-break is counted as selector evidence."
        ),
        "denominator_definition": (
            "Selector accuracy uses every GT-aligned disagreement, including "
            "ALL_WRONG events with zero credit. Correct-vs-wrong fidelity margins "
            "are defined only on recoverable disagreement events."
        ),
    }


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            result[order[position]] = average_rank
        start = end
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (lvalue - left_mean) * (rvalue - right_mean)
        for lvalue, rvalue in zip(left, right)
    )
    left_sum = sum((value - left_mean) ** 2 for value in left)
    right_sum = sum((value - right_mean) ** 2 for value in right)
    denominator = (left_sum * right_sum) ** 0.5
    return numerator / denominator if denominator else None


def _observable_boundary_summary(
    diagnostics: Sequence[Mapping[str, Any]], *, overlap_tokens: int
) -> dict[str, Any]:
    rows = [
        row
        for diagnostic in diagnostics
        for row in diagnostic["candidate_aligned_boundaries"]
    ]
    exact = sum(int(row["candidate_exact_agreement_count"]) for row in rows)
    substitutions = sum(
        int(row["candidate_alignment_substitution_count"]) for row in rows
    )
    substitution_same_source = sum(
        int(row["candidate_alignment_substitution_same_source_count"])
        for row in rows
    )
    directional_events = sum(
        int(row["candidate_aligned_directional_event_count"]) for row in rows
    )
    directional_fractional = sum(
        float(row["candidate_aligned_fidelity_fractional_credit_sum"])
        for row in rows
    )
    directional_chance = sum(
        float(row["candidate_aligned_random_credit_sum"]) for row in rows
    )
    directional_accuracy = _safe_ratio(directional_fractional, directional_events)
    directional_random = _safe_ratio(directional_chance, directional_events)
    disagreement_boundaries = [
        row for row in rows if bool(row["observable_boundary_disagreement"])
    ]
    disagreement_boundary_count = len(disagreement_boundaries)
    fidelity_boundary_correct = sum(
        float(row["fidelity_selected_correct_shared_tokens"])
        for row in disagreement_boundaries
    )
    random_boundary_correct = sum(
        float(row["random_expected_correct_shared_tokens"])
        for row in disagreement_boundaries
    )
    fidelity_boundary_accuracy = _safe_ratio(
        fidelity_boundary_correct, disagreement_boundary_count * overlap_tokens
    )
    random_boundary_accuracy = _safe_ratio(
        random_boundary_correct, disagreement_boundary_count * overlap_tokens
    )
    same_source = sum(
        int(row["candidate_exact_agreement_same_source_count"]) for row in rows
    )
    correct = sum(
        int(row["candidate_exact_agreement_correct_count"]) for row in rows
    )
    scores = [float(row["alignment_score"]) for row in rows]
    accuracies = [float(row["gt_shared_observation_accuracy"]) for row in rows]
    with_mixed = [
        float(row["alignment_score"])
        for row in rows
        if int(row["gt_mixed_source_tokens"]) > 0
    ]
    without_mixed = [
        float(row["alignment_score"])
        for row in rows
        if int(row["gt_mixed_source_tokens"]) == 0
    ]
    return {
        "attack_visible_features": {
            "boundary_count": len(rows),
            "candidate_exact_agreement_count": exact,
            "candidate_aligned_substitution_count": substitutions,
            "candidate_exact_agreement_coverage_of_nominal_overlap": _safe_ratio(
                exact, len(rows) * overlap_tokens
            ),
            "alignment_score": _distribution(scores),
            "observable_disagreement_boundary_count": disagreement_boundary_count,
            "directional_selector_capable_for_multiplicity_2": False,
            "directional_limit": (
                "Pairwise overlap consistency is symmetric: it can flag an "
                "unreliable boundary but cannot identify which of two views is "
                "correct."
            ),
        },
        "gt_calibration": {
            "candidate_exact_agreement_same_source_precision": _safe_ratio(
                same_source, exact
            ),
            "candidate_aligned_substitution_same_source_precision": _safe_ratio(
                substitution_same_source, substitutions
            ),
            "P_correct_given_candidate_aligned_exact_agreement": _safe_ratio(
                correct, exact
            ),
            "alignment_score_with_gt_mixed_token": _distribution(with_mixed),
            "alignment_score_without_gt_mixed_token": _distribution(without_mixed),
            "spearman_alignment_score_vs_gt_shared_observation_accuracy": _pearson(
                _rank(scores), _rank(accuracies)
            ),
            "candidate_aligned_fidelity_selection": {
                "event_count": directional_events,
                "recoverable_event_count": sum(
                    int(row["candidate_aligned_directional_recoverable_count"])
                    for row in rows
                ),
                "all_wrong_event_count": sum(
                    int(row["candidate_aligned_directional_all_wrong_count"])
                    for row in rows
                ),
                "fractional_accuracy": directional_accuracy,
                "random_view_expected_accuracy": directional_random,
                "uplift_over_random": (
                    None
                    if directional_accuracy is None or directional_random is None
                    else directional_accuracy - directional_random
                ),
                "definition": (
                    "The boundary correspondence and score choice use candidate "
                    "tokens, policy, and per-window fidelity only. Ground truth is "
                    "used solely to score whether the chosen side was correct and "
                    "whether the paired candidates mapped to the same source token."
                ),
            },
            "boundary_fidelity_selection": {
                "observable_disagreement_boundary_count": (
                    disagreement_boundary_count
                ),
                "directionally_informative_boundary_count": sum(
                    not bool(row["views_equal_shared_correctness"])
                    for row in disagreement_boundaries
                ),
                "higher_fidelity_better_side_count": sum(
                    int(row["fidelity_selected_better_side"])
                    for row in disagreement_boundaries
                ),
                "higher_fidelity_worse_side_count": sum(
                    int(row["fidelity_selected_worse_side"])
                    for row in disagreement_boundaries
                ),
                "equal_shared_correctness_count": sum(
                    int(row["views_equal_shared_correctness"])
                    for row in disagreement_boundaries
                ),
                "higher_fidelity_shared_token_accuracy": fidelity_boundary_accuracy,
                "random_side_expected_shared_token_accuracy": (
                    random_boundary_accuracy
                ),
                "shared_token_accuracy_uplift_over_random": (
                    None
                    if fidelity_boundary_accuracy is None
                    or random_boundary_accuracy is None
                    else fidelity_boundary_accuracy - random_boundary_accuracy
                ),
                "definition": (
                    "Every candidate-visible imperfect boundary is retained. The "
                    "higher-fidelity window is chosen without GT, then GT scores "
                    "how many of the policy-defined 16 shared source tokens that "
                    "window reconstructed exactly."
                ),
            },
            "attack_visible": False,
        },
    }


def _aggregate_oracle(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [diagnostic["oracle"] for diagnostic in diagnostics]
    source_tokens = sum(int(row["total_source_tokens"]) for row in rows)
    baseline_errors = sum(
        int(row["improved_baseline_source_error_count"]) for row in rows
    )
    residual = sum(int(row["oracle_residual_source_error_count"]) for row in rows)
    gain = sum(int(row["oracle_absolute_token_improvement"]) for row in rows)
    multiview_gain = sum(
        int(row["multiview_oracle_absolute_token_improvement"]) for row in rows
    )
    multiview_residual = sum(
        int(row["multiview_only_oracle_residual_source_error_count"])
        for row in rows
    )
    gain_attribution = {
        cause: sum(int(row["recoverable_gain_attribution"][cause]) for row in rows)
        for cause in (
            "SOURCE_VIEW_WRONG",
            "STITCH_ONLY_ERROR",
            "ALIGNMENT_AMBIGUITY",
        )
    }
    return {
        "source_token_count": source_tokens,
        "improved_baseline_source_error_count": baseline_errors,
        "improved_baseline_prediction_insertion_count": sum(
            int(row["improved_baseline_prediction_insertion_count"]) for row in rows
        ),
        "oracle_residual_source_error_count": residual,
        "oracle_normalized_source_token_error": residual / source_tokens,
        "oracle_source_token_recovery": 1.0 - residual / source_tokens,
        "oracle_source_position_document_exact_fraction": mean(
            bool(row["oracle_source_position_document_exact"]) for row in rows
        ),
        "oracle_absolute_token_improvement": gain,
        "multiview_oracle_absolute_token_improvement": multiview_gain,
        "multiview_only_oracle_residual_source_error_count": multiview_residual,
        "multiview_only_oracle_source_token_recovery": (
            1.0 - multiview_residual / source_tokens
        ),
        "multiview_only_oracle_source_position_document_exact_fraction": mean(
            bool(row["multiview_only_oracle_source_position_document_exact"])
            for row in rows
        ),
        "oracle_error_tokens_per_document": _distribution(
            [float(row["oracle_residual_source_error_count"]) for row in rows]
        ),
        "absolute_token_improvement_per_document": _distribution(
            [float(row["oracle_absolute_token_improvement"]) for row in rows]
        ),
        "oracle_relative_source_error_reduction": _safe_ratio(gain, baseline_errors),
        "multiview_oracle_relative_source_error_reduction": _safe_ratio(
            multiview_gain, baseline_errors
        ),
        "oracle_recoverable_single_observed_count": sum(
            int(row["oracle_recoverable_single_observed_count"]) for row in rows
        ),
        "oracle_recoverable_multiply_observed_count": sum(
            int(row["oracle_recoverable_multiply_observed_count"]) for row in rows
        ),
        "recoverable_gain_attribution": gain_attribution,
        "comparison_warning": (
            "This source-position oracle is compared with the improved baseline's "
            "unmatched source positions. It is not subtracted directly from total "
            "Levenshtein edits, which also contain prediction insertions. The gate "
            "uses only multiplicity>=2 gains so residual single-view stitching "
            "headroom is not misreported as multi-view redundancy."
        ),
    }


def _attribution_breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    causes = (
        "SOURCE_VIEW_WRONG",
        "STITCH_ONLY_ERROR",
        "ALIGNMENT_AMBIGUITY",
    )
    source = Counter(
        str(value["cause"])
        for row in rows
        for value in row["final_error_attribution_by_source_position"]
    )
    insertions = Counter(
        str(value["cause"])
        for row in rows
        for value in row["final_prediction_insertion_attribution"]
    )
    consumed = Counter(
        int(value) for row in rows for value in row["stitch_consumed_right_tokens"]
    )
    aligned_left = Counter(
        int(value) for row in rows for value in row["stitch_aligned_left_tokens"]
    )
    return {
        "source_position_errors_by_cause": {
            cause: source[cause] for cause in causes
        },
        "prediction_insertions_by_cause": {
            cause: insertions[cause] for cause in causes
        },
        "consumed_right_width_counts": {
            str(width): count for width, count in sorted(consumed.items())
        },
        "aligned_left_width_counts": {
            str(width): count for width, count in sorted(aligned_left.items())
        },
    }


def _bootstrap(
    rows_by_k: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    document_count = len(rows_by_k[min(rows_by_k)])
    rng = random.Random(seed)
    samples = {
        k: {
            "legacy_minus_improved_edit_distance": [],
            "oracle_source_error_gain": [],
            "multiview_oracle_source_error_gain": [],
            "fidelity_fractional_uplift_over_random": [],
            "candidate_aligned_fidelity_uplift_over_random": [],
            "boundary_fidelity_shared_token_uplift_over_random": [],
        }
        for k in rows_by_k
    }
    for _ in range(iterations):
        indices = [rng.randrange(document_count) for _ in range(document_count)]
        for k, rows in rows_by_k.items():
            selected = [rows[index] for index in indices]
            samples[k]["legacy_minus_improved_edit_distance"].append(
                mean(
                    float(row["legacy"]["document_edit_distance"])
                    - float(row["improved"]["document_edit_distance"])
                    for row in selected
                )
            )
            samples[k]["oracle_source_error_gain"].append(
                mean(
                    float(row["diagnostic"]["oracle"]["oracle_absolute_token_improvement"])
                    for row in selected
                )
            )
            samples[k]["multiview_oracle_source_error_gain"].append(
                mean(
                    float(
                        row["diagnostic"]["oracle"][
                            "multiview_oracle_absolute_token_improvement"
                        ]
                    )
                    for row in selected
                )
            )
            fidelity = [
                row["diagnostic"]["fidelity"]["by_multiplicity"]["all"]
                for row in selected
            ]
            events = sum(int(value.get("event_count", 0)) for value in fidelity)
            if events:
                fractional = sum(
                    float(value.get("fractional_credit_sum", 0.0))
                    for value in fidelity
                ) / events
                chance = sum(
                    float(value.get("chance_credit_sum", 0.0)) for value in fidelity
                ) / events
                samples[k]["fidelity_fractional_uplift_over_random"].append(
                    fractional - chance
                )
            boundaries = [
                boundary
                for row in selected
                for boundary in row["diagnostic"]["candidate_aligned_boundaries"]
            ]
            directional_events = sum(
                int(value["candidate_aligned_directional_event_count"])
                for value in boundaries
            )
            if directional_events:
                directional_fractional = sum(
                    float(value["candidate_aligned_fidelity_fractional_credit_sum"])
                    for value in boundaries
                ) / directional_events
                directional_chance = sum(
                    float(value["candidate_aligned_random_credit_sum"])
                    for value in boundaries
                ) / directional_events
                samples[k][
                    "candidate_aligned_fidelity_uplift_over_random"
                ].append(directional_fractional - directional_chance)
            disagreement_boundaries = [
                value
                for value in boundaries
                if bool(value["observable_boundary_disagreement"])
            ]
            if disagreement_boundaries:
                fidelity_correct = sum(
                    float(value["fidelity_selected_correct_shared_tokens"])
                    for value in disagreement_boundaries
                )
                random_correct = sum(
                    float(value["random_expected_correct_shared_tokens"])
                    for value in disagreement_boundaries
                )
                denominator = len(disagreement_boundaries) * 16
                samples[k][
                    "boundary_fidelity_shared_token_uplift_over_random"
                ].append((fidelity_correct - random_correct) / denominator)
    return {
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "frozen document manifest index",
        "by_k": {
            str(k): {
                metric: [
                    _percentile(values, 0.025),
                    _percentile(values, 0.975),
                ]
                for metric, values in metrics.items()
            }
            for k, metrics in samples.items()
        },
    }


def _classify_gate(
    summaries: Mapping[int, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    *,
    primary_k: int,
) -> dict[str, Any]:
    summary = summaries[primary_k]
    intervals = bootstrap["by_k"][str(primary_k)]
    oracle_ci = intervals["multiview_oracle_source_error_gain"]
    fidelity_ci = intervals[
        "boundary_fidelity_shared_token_uplift_over_random"
    ]
    oracle_established = oracle_ci[0] is not None and float(oracle_ci[0]) > 0.0
    selector_established = (
        fidelity_ci[0] is not None and float(fidelity_ci[0]) > 0.0
    )
    if not oracle_established:
        state = "STITCHER_DOMINATED_STOP"
        reason = (
            "After freezing the stronger stitcher, paired document bootstrap "
            "does not establish additional multiplicity>=2 Best-of-Views "
            "source-error headroom."
        )
    elif not selector_established:
        state = "ORACLE_REDUNDANCY_BUT_NO_SELECTOR"
        reason = (
            "Multiplicity>=2 Best-of-Views source-error headroom remains, but the "
            "candidate-visible imperfect-boundary fidelity signal pre-specified "
            "in the frozen Stage 4 config does not beat its random-side expectation "
            "with a positive document-bootstrap lower bound."
        )
    else:
        state = "DOCUMENT_CORRECTION_GO"
        reason = (
            "Multiplicity>=2 Best-of-Views source-error headroom remains and "
            "candidate-visible imperfect-boundary fidelity has non-random "
            "directional association with shared-span correctness."
        )
    return {
        "state": state,
        "reason": reason,
        "primary_k": primary_k,
        "oracle_headroom_established": oracle_established,
        "observable_directional_selector_signal_established": selector_established,
        "oracle_source_error_gain_mean_ci95": oracle_ci,
        "boundary_fidelity_shared_token_uplift_over_random_ci95": fidelity_ci,
        "observed": {
            "legacy_total_edits": summary["legacy"]["document_level"][
                "absolute_error_token_count"
            ],
            "improved_total_edits": summary["improved"]["document_level"][
                "absolute_error_token_count"
            ],
            "oracle_absolute_source_token_improvement": summary["oracle"][
                "oracle_absolute_token_improvement"
            ],
            "multiview_oracle_absolute_source_token_improvement": summary["oracle"][
                "multiview_oracle_absolute_token_improvement"
            ],
            "fidelity_fractional_uplift_over_random": summary["fidelity"]["all"][
                "fractional_accuracy_uplift_over_random"
            ],
            "candidate_aligned_fidelity_uplift_over_random": summary[
                "candidate_aligned_observable"
            ]["gt_calibration"]["candidate_aligned_fidelity_selection"][
                "uplift_over_random"
            ],
            "boundary_fidelity_shared_token_uplift_over_random": summary[
                "candidate_aligned_observable"
            ]["gt_calibration"]["boundary_fidelity_selection"][
                "shared_token_accuracy_uplift_over_random"
            ],
        },
        "actual_document_correction_implemented": False,
        "classification_scope": (
            "Exploratory non-random feasibility gate. No practical effect-size "
            "threshold was preregistered, so confirmatory and practical claims "
            "require a new frozen cohort and an explicit effect-size criterion."
        ),
    }


def _assert_legacy_reproduction(
    recomputed: Mapping[str, Any], stored: Mapping[str, Any], *, k: int
) -> None:
    checks = (
        ("absolute_error_token_count", "document_level"),
        ("document_exact_fraction", "document_level"),
    )
    for metric, section in checks:
        if recomputed[section][metric] != stored[section][metric]:
            raise AssertionError(f"Stage 4 legacy K={k} {metric} did not reproduce")
    for cause in ("SOURCE_VIEW_WRONG", "STITCH_ONLY_ERROR", "ALIGNMENT_AMBIGUITY"):
        if recomputed["final_error_attribution"][cause] != stored[
            "final_error_attribution"
        ][cause]:
            raise AssertionError(f"Stage 4 legacy K={k} {cause} did not reproduce")


def _write_report(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "stage4_document_correction_gate.json", payload)
    lines = [
        "# Stage 4 — stitching and document-correction ceiling diagnostics",
        "",
        f"- Status: `{payload['status']}`",
        f"- Gate: `{payload['gate']['state']}`",
        f"- Reason: {payload['gate']['reason']}",
        "- Vec2Text rerun: `false`; frozen correction50+sbeam8 Top-1 cache reused.",
        "- Actual correction/voting/path selector implemented: `false`.",
        "",
        "## Stage 4A — Independent stitching",
        "",
        "| K | Legacy sim | Improved sim | Legacy token recovery | Improved token recovery | Legacy edits | Improved edits | Improved source errors | Improved insertions | Legacy stitch-only | Improved stitch-only |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in payload["k_values"]:
        summary = payload["summaries"][str(k)]
        legacy = summary["legacy"]
        improved = summary["improved"]
        lines.append(
            f"| {k} | {legacy['document_level']['normalized_edit_similarity']['mean']:.4f} | "
            f"{improved['document_level']['normalized_edit_similarity']['mean']:.4f} | "
            f"{legacy['document_level']['pooled_exact_matched_token_recovery']:.4f} | "
            f"{improved['document_level']['pooled_exact_matched_token_recovery']:.4f} | "
            f"{legacy['document_level']['absolute_error_token_count']} | "
            f"{improved['document_level']['absolute_error_token_count']} | "
            f"{improved['document_level']['final_source_token_error_count']} | "
            f"{improved['document_level']['final_prediction_insertion_count']} | "
            f"{legacy['final_error_attribution']['STITCH_ONLY_ERROR']} | "
            f"{improved['final_error_attribution']['STITCH_ONLY_ERROR']} |"
        )
    lines.extend(
        [
            "",
            "## Stage 4B — GT-aligned Best-of-Views source-position ceiling",
            "",
            "| K | Improved source errors | Full oracle residual | Full gain | Multi-view gain | Multi-view relative reduction | Multi-view-only recovery | Oracle source-position doc exact |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for k in payload["k_values"]:
        oracle = payload["summaries"][str(k)]["oracle"]
        relative = oracle["multiview_oracle_relative_source_error_reduction"]
        lines.append(
            f"| {k} | {oracle['improved_baseline_source_error_count']} | "
            f"{oracle['oracle_residual_source_error_count']} | "
            f"{oracle['oracle_absolute_token_improvement']} | "
            f"{oracle['multiview_oracle_absolute_token_improvement']} | "
            f"{('N/A' if relative is None else f'{relative:.4f}')} | "
            f"{oracle['multiview_only_oracle_source_token_recovery']:.4f} | "
            f"{oracle['multiview_only_oracle_source_position_document_exact_fraction']:.4f} |"
        )
    k8 = payload["summaries"]["8"]
    lines.extend(
        [
            "",
            "## K=8 agreement and observable signals",
            "",
            "| Multiplicity | Unanimous | P(correct | unanimous) | Disagreement | One correct | 2+ correct | All wrong | P(any correct | disagreement) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in ("2", "3"):
        agreement = k8["agreement"][group]
        lines.append(
            f"| {group} | {agreement.get('UNANIMOUS_count', 0)} | "
            f"{agreement['P_correct_given_UNANIMOUS']:.4f} | "
            f"{agreement.get('DISAGREEMENT_count', 0)} | "
            f"{agreement.get('DISAGREEMENT_one_correct_count', 0)} | "
            f"{agreement.get('DISAGREEMENT_two_or_more_correct_count', 0)} | "
            f"{agreement.get('DISAGREEMENT_all_wrong_count', 0)} | "
            f"{agreement['P_any_correct_given_DISAGREEMENT']:.4f} |"
        )
    fidelity = k8["fidelity"]["all"]
    observable = k8["candidate_aligned_observable"]
    observable_features = observable["attack_visible_features"]
    observable_calibration = observable["gt_calibration"]
    candidate_fidelity = observable_calibration[
        "candidate_aligned_fidelity_selection"
    ]
    boundary_fidelity = observable_calibration["boundary_fidelity_selection"]
    lines.extend(
        [
            "",
            f"- Fidelity disagreement events: {fidelity['event_count']} (recoverable={fidelity['recoverable_event_count']}, all-wrong={fidelity['all_wrong_event_count']})",
            f"- Fractional top-fidelity accuracy over all disagreements: {fidelity['fractional_top_fidelity_accuracy']:.4f}",
            f"- Random-view expected accuracy: {fidelity['random_view_expected_accuracy']:.4f}",
            f"- Fidelity uplift: {fidelity['fractional_accuracy_uplift_over_random']:.4f}",
            f"- Candidate-aligned fidelity events: {candidate_fidelity['event_count']}",
            f"- Candidate-aligned fidelity accuracy: {candidate_fidelity['fractional_accuracy']:.4f}",
            f"- Candidate-aligned fidelity random expectation: {candidate_fidelity['random_view_expected_accuracy']:.4f}",
            f"- Candidate-aligned fidelity uplift: {candidate_fidelity['uplift_over_random']:.4f}",
            f"- Observable imperfect boundaries: {boundary_fidelity['observable_disagreement_boundary_count']}",
            f"- Higher-fidelity boundary shared-token accuracy: {boundary_fidelity['higher_fidelity_shared_token_accuracy']:.4f}",
            f"- Random-side expected shared-token accuracy: {boundary_fidelity['random_side_expected_shared_token_accuracy']:.4f}",
            f"- Boundary fidelity shared-token uplift: {boundary_fidelity['shared_token_accuracy_uplift_over_random']:.4f}",
            f"- Candidate-aligned exact-agreement correctness (GT calibration only): {observable_calibration['P_correct_given_candidate_aligned_exact_agreement']:.4f}",
            f"- Symmetric overlap consistency can direction-select the correct m=2 view: `{str(observable_features['directional_selector_capable_for_multiplicity_2']).lower()}`",
            f"- K=8 multi-view oracle-gain bootstrap CI95 (tokens/document): {payload['gate']['oracle_source_error_gain_mean_ci95']}",
            f"- K=8 boundary-fidelity shared-token-uplift bootstrap CI95: {payload['gate']['boundary_fidelity_shared_token_uplift_over_random_ci95']}",
            "",
            "The stronger stitcher reduces sequence edit distance mainly by eliminating duplicated prediction insertions. It does not dominate the legacy baseline on exact source-position token recovery, so both quantities are reported separately.",
            "",
            "The oracle is a source-position ceiling. Prediction insertions remain separate; it is not compared by subtracting its residual directly from total Levenshtein edits.",
        ]
    )
    (output_dir / "stage4_document_correction_gate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_stage4(config: Mapping[str, Any]) -> dict[str, Any]:
    cohort_path = Path(str(config["resolved_cohort_artifact"]))
    cache_path = Path(str(config["resolved_strong_top1_cache"]))
    stage3_path = Path(str(config["resolved_stage3_result"]))
    cohort = _load_json(cohort_path)
    cache = _load_json(cache_path)
    stage3 = _load_json(stage3_path)
    validation = _validate_inputs(config, cohort, cache, stage3)

    documents = list(cohort["documents"])
    candidates = tuple(
        _candidate_from_dict(raw) for raw in cache["top1_candidates"]
    )
    candidates_by_document = [
        candidates[index * 8 : (index + 1) * 8]
        for index in range(len(documents))
    ]
    stitch_config = dict(config["stronger_stitcher"])
    overlap_tokens = int(stitch_config["nominal_overlap_tokens"])
    max_boundary_shift = int(stitch_config["max_boundary_shift"])
    deviation_cost = float(stitch_config["deviation_cost"])
    stride = 15

    summaries: dict[int, dict[str, Any]] = {}
    bootstrap_rows: dict[int, list[dict[str, Any]]] = {}
    trajectories = [
        {"passage_id": document["passage_id"], "k_prefixes": {}}
        for document in documents
    ]
    total_started = time.perf_counter()
    for k in (int(value) for value in config["k_values"]):
        legacy_started = time.perf_counter()
        legacy_rows = []
        for document, document_candidates in zip(documents, candidates_by_document):
            zero_records = tuple(
                {"telemetry": {"elapsed_seconds": 0.0}} for _ in range(k)
            )
            legacy_rows.append(
                _evaluate_document_prefix(
                    document,
                    document_candidates[:k],
                    zero_records,
                    k=k,
                    content_tokens=31,
                    overlap_tokens=overlap_tokens,
                    stride=stride,
                    minimum_alignment=0.5,
                )
            )
        legacy_seconds = time.perf_counter() - legacy_started
        legacy_summary = _aggregate_prefix(legacy_rows)
        _assert_legacy_reproduction(
            legacy_summary, stage3["summaries"][str(k)], k=k
        )

        improved_started = time.perf_counter()
        improved_rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for document, document_candidates in zip(documents, candidates_by_document):
            row, _stitched, diagnostic = _evaluate_document(
                document,
                document_candidates,
                k=k,
                overlap_tokens=overlap_tokens,
                stride=stride,
                max_boundary_shift=max_boundary_shift,
                deviation_cost=deviation_cost,
            )
            improved_rows.append(row)
            diagnostics.append(diagnostic)
        improved_seconds = time.perf_counter() - improved_started
        improved_summary = _aggregate_prefix(improved_rows)

        oracle = _aggregate_oracle(diagnostics)
        agreement = {
            group: _agreement_summary(_sum_nested_counts(diagnostics, group))
            for group in ("2", "3")
        }
        fidelity = {
            group: _aggregate_fidelity(diagnostics, group)
            for group in ("all", "2", "3")
        }
        observable = _observable_boundary_summary(
            diagnostics, overlap_tokens=overlap_tokens
        )
        summaries[k] = {
            "legacy": legacy_summary,
            "improved": improved_summary,
            "legacy_attribution_breakdown": _attribution_breakdown(legacy_rows),
            "improved_attribution_breakdown": _attribution_breakdown(improved_rows),
            "stitching_delta": {
                "total_edit_reduction": legacy_summary["document_level"][
                    "absolute_error_token_count"
                ]
                - improved_summary["document_level"]["absolute_error_token_count"],
                "relative_total_edit_reduction": _safe_ratio(
                    legacy_summary["document_level"]["absolute_error_token_count"]
                    - improved_summary["document_level"][
                        "absolute_error_token_count"
                    ],
                    legacy_summary["document_level"]["absolute_error_token_count"],
                ),
                "stitch_only_error_reduction": legacy_summary[
                    "final_error_attribution"
                ]["STITCH_ONLY_ERROR"]
                - improved_summary["final_error_attribution"]["STITCH_ONLY_ERROR"],
                "relative_stitch_only_error_reduction": _safe_ratio(
                    legacy_summary["final_error_attribution"]["STITCH_ONLY_ERROR"]
                    - improved_summary["final_error_attribution"][
                        "STITCH_ONLY_ERROR"
                    ],
                    legacy_summary["final_error_attribution"]["STITCH_ONLY_ERROR"],
                ),
            },
            "oracle": oracle,
            "agreement": agreement,
            "fidelity": fidelity,
            "candidate_aligned_observable": observable,
            "runtime": {
                "legacy_evaluation_seconds": legacy_seconds,
                "improved_stitch_and_diagnostic_seconds": improved_seconds,
                "improved_seconds_per_document": improved_seconds / len(documents),
            },
        }
        combined_rows = []
        for index, (legacy, improved, diagnostic) in enumerate(
            zip(legacy_rows, improved_rows, diagnostics)
        ):
            compact = {
                "passage_id": documents[index]["passage_id"],
                "legacy": legacy,
                "improved": improved,
                "diagnostic": diagnostic,
            }
            combined_rows.append(compact)
            trajectories[index]["k_prefixes"][str(k)] = compact
        bootstrap_rows[k] = combined_rows

    bootstrap = _bootstrap(
        bootstrap_rows,
        iterations=int(config["bootstrap_iterations"]),
        seed=int(config["bootstrap_seed"]),
    )
    gate = _classify_gate(
        summaries,
        bootstrap,
        primary_k=int(config["gate"]["primary_k"]),
    )
    output_dir = Path(str(config["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    trajectories_path = output_dir / "stage4_paired_trajectories.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        trajectories_path,
        {
            "schema_version": 1,
            "implementation_version": _IMPLEMENTATION_VERSION,
            "cohort_sha256": validation["hashes"]["cohort"],
            "strong_top1_cache_sha256": validation["hashes"]["strong_top1_cache"],
            "trajectories": trajectories,
        },
    )
    payload = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE4_COMPLETE",
        "stage": "Stage 4A stronger stitching plus Stage 4B oracle/signal diagnostics",
        "config": dict(config),
        "validation": validation,
        "estimand": stage3["estimand"],
        "selection_limitation": stage3["selection_limitation"],
        "k_values": [int(value) for value in config["k_values"]],
        "stronger_stitcher": stitch_config,
        "summaries": {str(k): value for k, value in summaries.items()},
        "paired_document_bootstrap_95": bootstrap,
        "paired_trajectories": {
            "path": str(trajectories_path),
            "sha256": _sha256(trajectories_path),
        },
        "gate": gate,
        "telemetry": {
            "stage4_total_wall_seconds": time.perf_counter() - total_started,
            "vec2text_reexecuted": False,
            "gpu_required": False,
            "strong_cache_views_reused": len(candidates),
            "frozen_strong_inversion": {
                "views": int(stage3["telemetry"]["views"]),
                "original_total_search_runtime_seconds": float(
                    stage3["telemetry"]["total_search_runtime_seconds"]
                ),
                "original_runtime_seconds_per_view": float(
                    stage3["telemetry"]["runtime_seconds_per_view"]
                ),
                "original_search_embedding_queries_total": int(
                    stage3["telemetry"]["search_embedding_queries_total"]
                ),
                "original_peak_allocated_bytes": int(
                    stage3["telemetry"]["peak_allocated_bytes"]
                ),
            },
        },
        "oracle_uses_ground_truth": True,
        "stitcher_uses_ground_truth": False,
        "gt_aligned_agreement_is_attack_visible": False,
        "candidate_aligned_boundary_features_are_attack_visible": True,
        "candidate_aligned_boundary_gt_calibration_is_attack_visible": False,
        "candidate_level_joint_reranker_implemented": False,
        "lambda_tuned": False,
        "search_time_joint_vec2text_implemented": False,
        "majority_or_confidence_voting_implemented": False,
        "path_decoder_implemented": False,
        "llm_correction_implemented": False,
        "new_inversion_decoder_implemented": False,
    }
    _write_report(output_dir, payload)
    return payload
