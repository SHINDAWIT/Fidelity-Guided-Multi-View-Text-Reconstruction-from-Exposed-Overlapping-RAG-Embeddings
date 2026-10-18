"""Controlled Stage 5C evaluation of one frozen fidelity selector.

The selector uses only window order, overlap multiplicity, and each frozen
Independent Top-1 reconstruction's target-embedding fidelity. Ground truth is
consulted only after a view has been selected, for evaluation and the oracle.
"""

from __future__ import annotations

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
from .stage3_longdoc_strong import _percentile
from .stage4_document_correction import _observations
from .stage5b_evaluate import nooverlap_single_view_source_row


_IMPLEMENTATION_VERSION = "stage5c-fidelity-selector-v1"
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


def load_stage5c_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 5C config must be a YAML mapping")
    if raw.get("frozen_before_evaluation") is not True:
        raise RuntimeError("Stage 5C selector was not frozen before evaluation")
    resolved = dict(raw)
    resolved_inputs = {}
    for name, entry in raw["inputs"].items():
        candidate = Path(str(entry["path"]))
        resolved_inputs[name] = {
            **entry,
            "path": str(
                (
                    candidate
                    if candidate.is_absolute()
                    else config_path.parent / candidate
                ).resolve()
            ),
        }
    resolved["inputs"] = resolved_inputs
    resolved["resolved_config"] = str(config_path.resolve())
    return resolved


def _validate_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = {}
    for name, entry in config["inputs"].items():
        path = Path(str(entry["path"]))
        actual = _sha256(path)
        if actual != str(entry["sha256"]):
            raise RuntimeError(f"Stage 5C frozen input changed: {name}")
        values[name] = _load_json(path)
    stage5b = values["stage5b_result"]
    strong = values["strong_cache"]
    if stage5b.get("status") != "STAGE5B_HOLDOUT_EVALUATION_COMPLETE":
        raise RuntimeError("Stage 5B evaluation is incomplete")
    if stage5b.get("gate", {}).get("state") != "GENERAL_OVERLAP_SIGNAL":
        raise RuntimeError("Stage 5C requires the frozen Stage 5B general signal")
    if strong.get("status") != "STAGE5B_STRONG_TOP1_CACHE_COMPLETE":
        raise RuntimeError("Stage 5B strong cache is incomplete")
    if strong.get("strong_reproduction_setting") != "correction50+sbeam8":
        raise RuntimeError("Stage 5C must reuse correction50+sbeam8")
    if len(strong.get("top1_candidates", [])) != 720:
        raise RuntimeError("Stage 5C strong candidate count changed")
    overlap = values["overlap_cohort"]
    nooverlap = values["nooverlap_manifest"]
    expected_documents = int(config["cohort"]["documents"])
    if len(overlap["documents"]) != expected_documents:
        raise RuntimeError("Stage 5C overlap document count changed")
    if [document["passage_id"] for document in overlap["documents"]] != [
        document["passage_id"] for document in nooverlap["documents"]
    ]:
        raise RuntimeError("Stage 5C paired document order changed")
    selector = config["selector"]
    expected_selector = {
        "decision_unit": "source_position_with_exactly_two_policy_observations",
        "rule": "maximum_window_embedding_fidelity",
        "fidelity_field": "Candidate.local_score",
        "tie_rule": "lower_window_index",
        "multiplicity_3_rule": "lower_window_index_no_stage5c_selection",
        "ground_truth_used_for_decision": False,
        "prediction_tokens_used_for_decision": False,
        "threshold_or_tuning": "none",
    }
    for key, expected in expected_selector.items():
        if selector.get(key) != expected:
            raise RuntimeError(f"Stage 5C selector rule changed: {key}")
    return values


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


def _choose_view_index(
    values: Sequence[Mapping[str, Any]],
    scores: Mapping[int, float],
    *,
    tie_tolerance: float,
) -> int:
    """Choose using only view identity and externally supplied fidelity."""

    top_score = max(float(scores[int(value["view_index"])]) for value in values)
    tied = [
        int(value["view_index"])
        for value in values
        if abs(float(scores[int(value["view_index"])]) - top_score)
        <= tie_tolerance
    ]
    return min(tied)


def _score_document(
    document: Mapping[str, Any],
    candidates: Sequence[Candidate],
    *,
    decision_scores: Mapping[int, float],
    tie_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observations, _aligned_views = _observations(document, candidates, k=8)
    reference = tuple(int(value) for value in document["prefixes"]["8"]["reference_ids"])
    fixed_correct = 0
    selector_correct = 0
    oracle_correct = 0
    selected_positions = 0
    ties = 0
    failures = []
    for position, values in enumerate(observations):
        ordered = sorted(values, key=lambda value: int(value["view_index"]))
        fixed = ordered[0]
        fixed_correct += int(bool(fixed["correct"]))
        selected = fixed
        oracle_position_correct = bool(fixed["correct"])
        if len(ordered) == 2:
            selected_positions += 1
            scores = [decision_scores[int(value["view_index"])] for value in ordered]
            ties += int(abs(float(scores[0]) - float(scores[1])) <= tie_tolerance)
            selected_index = _choose_view_index(
                ordered, decision_scores, tie_tolerance=tie_tolerance
            )
            selected = next(
                value for value in ordered if int(value["view_index"]) == selected_index
            )
            oracle_position_correct = any(bool(value["correct"]) for value in ordered)
            if not bool(selected["correct"]) and oracle_position_correct:
                correct_values = [value for value in ordered if bool(value["correct"])]
                failures.append(
                    {
                        "source_position": position,
                        "reference_token_id": reference[position],
                        "selected_view_index": selected_index,
                        "selected_prediction": selected["prediction"],
                        "selected_fidelity": float(decision_scores[selected_index]),
                        "correct_alternative_view_index": int(
                            correct_values[0]["view_index"]
                        ),
                        "correct_alternative_fidelity": float(
                            decision_scores[int(correct_values[0]["view_index"])]
                        ),
                        "fidelity_margin_wrong_minus_correct": float(
                            decision_scores[selected_index]
                            - decision_scores[int(correct_values[0]["view_index"])]
                        ),
                        "tie": abs(
                            float(decision_scores[selected_index])
                            - float(
                                decision_scores[
                                    int(correct_values[0]["view_index"])
                                ]
                            )
                        )
                        <= tie_tolerance,
                    }
                )
        selector_correct += int(bool(selected["correct"]))
        oracle_correct += int(oracle_position_correct)
    total = len(observations)
    return (
        {
            "source_token_count": total,
            "fixed_owner_correct": fixed_correct,
            "selector_correct": selector_correct,
            "oracle_correct": oracle_correct,
            "fixed_owner_recovery": fixed_correct / total,
            "selector_recovery": selector_correct / total,
            "oracle_recovery": oracle_correct / total,
            "selected_multiplicity_2_positions": selected_positions,
            "fidelity_tie_positions": ties,
            "selector_minus_fixed_owner": (selector_correct - fixed_correct) / total,
            "oracle_minus_fixed_owner": (oracle_correct - fixed_correct) / total,
        },
        failures,
    )


def _nominal_position_score(
    document: Mapping[str, Any],
    candidates: Sequence[Candidate],
    *,
    decision_scores: Mapping[int, float],
    tie_tolerance: float,
) -> dict[str, Any]:
    """GT-free realizability sensitivity using policy-local candidate indices."""

    reference = tuple(int(value) for value in document["prefixes"]["8"]["reference_ids"])
    selected_start = int(document["selected_start"])
    views_by_position: list[list[tuple[int, int | None]]] = [
        [] for _ in reference
    ]
    for view_index, (window, candidate) in enumerate(zip(document["windows"], candidates)):
        relative_start = int(window["document_token_start"]) - selected_start
        for local_position in range(31):
            prediction = (
                int(candidate.content_ids[local_position])
                if local_position < len(candidate.content_ids)
                else None
            )
            views_by_position[relative_start + local_position].append(
                (view_index, prediction)
            )
    fixed_correct = 0
    selector_correct = 0
    oracle_correct = 0
    for position, values in enumerate(views_by_position):
        ordered = sorted(values)
        fixed_prediction = ordered[0][1]
        selected_prediction = fixed_prediction
        oracle_position = fixed_prediction == reference[position]
        if len(ordered) == 2:
            pseudo = [{"view_index": view_index} for view_index, _ in ordered]
            selected_index = _choose_view_index(
                pseudo, decision_scores, tie_tolerance=tie_tolerance
            )
            selected_prediction = next(
                prediction
                for view_index, prediction in ordered
                if view_index == selected_index
            )
            oracle_position = any(
                prediction == reference[position] for _view_index, prediction in ordered
            )
        fixed_correct += int(fixed_prediction == reference[position])
        selector_correct += int(selected_prediction == reference[position])
        oracle_correct += int(oracle_position)
    total = len(reference)
    return {
        "fixed_owner_recovery": fixed_correct / total,
        "selector_recovery": selector_correct / total,
        "oracle_recovery": oracle_correct / total,
        "selector_minus_fixed_owner": (selector_correct - fixed_correct) / total,
    }


def _bootstrap(
    rows: Sequence[Mapping[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    metrics = (
        "selector_minus_nooverlap",
        "selector_minus_fixed_owner",
        "oracle_minus_nooverlap",
        "shuffled_minus_fixed_owner",
        "nonadjacent_minus_fixed_owner",
        "nominal_selector_minus_nominal_fixed_owner",
    )
    rng = random.Random(seed)
    samples = {metric: [] for metric in metrics}
    ratios = []
    for _ in range(iterations):
        selected = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        estimates = {
            metric: mean(float(row[metric]) for row in selected) for metric in metrics
        }
        for metric, estimate in estimates.items():
            samples[metric].append(estimate)
        denominator = estimates["oracle_minus_nooverlap"]
        if abs(denominator) > 1e-12:
            ratios.append(estimates["selector_minus_nooverlap"] / denominator)
    estimates = {metric: mean(float(row[metric]) for row in rows) for metric in metrics}
    denominator = estimates["oracle_minus_nooverlap"]
    capture = (
        estimates["selector_minus_nooverlap"] / denominator
        if abs(denominator) > 1e-12
        else None
    )
    return {
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "frozen document",
        "mean_delta": estimates,
        "ci95": {
            metric: [_percentile(values, 0.025), _percentile(values, 0.975)]
            for metric, values in samples.items()
        },
        "headroom_capture_ratio": capture,
        "headroom_capture_ratio_ci95": (
            [_percentile(ratios, 0.025), _percentile(ratios, 0.975)]
            if ratios
            else [None, None]
        ),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = sum(int(row["selector"]["source_token_count"]) for row in rows)
    fixed = sum(int(row["selector"]["fixed_owner_correct"]) for row in rows)
    selected = sum(int(row["selector"]["selector_correct"]) for row in rows)
    oracle = sum(int(row["selector"]["oracle_correct"]) for row in rows)
    nooverlap = sum(
        int(row["nooverlap"]["correct_source_token_count"]) for row in rows
    )
    return {
        "documents": len(rows),
        "source_token_count": total,
        "nooverlap_single_view_recovery": nooverlap / total,
        "overlap_fixed_owner_recovery": fixed / total,
        "overlap_fidelity_selector_recovery": selected / total,
        "overlap_best_of_two_oracle_recovery": oracle / total,
        "selector_gain_over_nooverlap": (selected - nooverlap) / total,
        "selector_gain_over_fixed_owner": (selected - fixed) / total,
        "oracle_headroom_over_nooverlap": (oracle - nooverlap) / total,
        "selected_multiplicity_2_positions": sum(
            int(row["selector"]["selected_multiplicity_2_positions"])
            for row in rows
        ),
        "fidelity_tie_positions": sum(
            int(row["selector"]["fidelity_tie_positions"]) for row in rows
        ),
        "nominal_index_realization": {
            "fixed_owner_recovery": mean(
                float(row["nominal"]["fixed_owner_recovery"]) for row in rows
            ),
            "selector_recovery": mean(
                float(row["nominal"]["selector_recovery"]) for row in rows
            ),
            "oracle_recovery": mean(
                float(row["nominal"]["oracle_recovery"]) for row in rows
            ),
        },
    }


def _gate(
    summary: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    stage5b_h2: Mapping[str, Any],
) -> dict[str, Any]:
    estimates = bootstrap["mean_delta"]
    intervals = bootstrap["ci95"]
    primary_positive = float(intervals["selector_minus_nooverlap"][0]) > 0.0
    isolated_positive = float(intervals["selector_minus_fixed_owner"][0]) > 0.0
    shuffled_positive = float(intervals["shuffled_minus_fixed_owner"][0]) > 0.0
    nonadjacent_positive = float(intervals["nonadjacent_minus_fixed_owner"][0]) > 0.0
    selector_nonpositive = float(estimates["selector_minus_nooverlap"]) <= 0.0
    h2_positive = float(stage5b_h2["uplift_over_random"]) > 0.0
    if selector_nonpositive or shuffled_positive or nonadjacent_positive:
        state = "FAIL"
    elif primary_positive and isolated_positive:
        state = "PASS"
    elif h2_positive:
        state = "PARTIAL"
    else:
        state = "FAIL"
    return {
        "state": state,
        "selector_minus_nooverlap_ci95_lower_above_zero": primary_positive,
        "selector_minus_fixed_owner_ci95_lower_above_zero": isolated_positive,
        "shuffled_control_ci95_lower_above_zero": shuffled_positive,
        "nonadjacent_control_ci95_lower_above_zero": nonadjacent_positive,
        "stage5b_disagreement_signal_positive": h2_positive,
        "new_attack_structure_authorized": False,
        "interpretation": {
            "PASS": "Observable fidelity recovers part of overlap headroom beyond both single-view baselines without surviving score-decoupling controls.",
            "PARTIAL": "Fidelity remains a local disagreement signal, but whole-holdout coverage or uncertainty prevents a practical selector claim.",
            "FAIL": "The frozen selector does not improve the external single-view baseline, or comparable gain survives a score-decoupling control.",
        }[state],
    }


def _token_text(tokenizer: Any, value: int | str) -> str:
    if isinstance(value, int):
        return str(tokenizer.convert_ids_to_tokens(value))
    return str(value)


def _write_report(output_dir: Path, filename: str, result: Mapping[str, Any]) -> None:
    main = result["main_result"]
    bootstrap = result["bootstrap"]
    disagreement = result["disagreement_analysis"]
    lines = [
        "# Stage 5C — frozen fidelity selector",
        "",
        f"- Gate: `{result['gate']['state']}`",
        "- Selector: maximum observable window embedding fidelity; no GT, token correctness, threshold, lexical score, or tuning is used for decisions.",
        "- Existing correction50+sbeam8 cache reused; no Vec2Text execution.",
        "",
        "## Main result",
        "",
        f"- No-overlap single-view: {main['nooverlap_single_view_recovery']:.6f}",
        f"- Overlap fixed-owner single-view: {main['overlap_fixed_owner_recovery']:.6f}",
        f"- Overlap fidelity selector: {main['overlap_fidelity_selector_recovery']:.6f}",
        f"- Overlap Best-of-Two oracle: {main['overlap_best_of_two_oracle_recovery']:.6f}",
        f"- Selector gain over no-overlap: {main['selector_gain_over_nooverlap']:+.6f} (CI95 {bootstrap['ci95']['selector_minus_nooverlap']})",
        f"- Selector gain over fixed owner: {main['selector_gain_over_fixed_owner']:+.6f} (CI95 {bootstrap['ci95']['selector_minus_fixed_owner']})",
        f"- Headroom capture ratio: {bootstrap['headroom_capture_ratio']}",
        "",
        "## Disagreement-boundary analysis",
        "",
        f"- Events: {disagreement['event_count']}",
        f"- Higher-fidelity accuracy: {disagreement['higher_fidelity_shared_token_accuracy']:.6f}",
        f"- Random expectation: {disagreement['random_side_expected_shared_token_accuracy']:.6f}",
        f"- Uplift: {disagreement['uplift_over_random']:+.6f}",
        f"- CI95: {disagreement['ci95']}",
        "",
        "## Robustness and controls",
        "",
        "| Group | Fidelity selector | No-overlap | Gain | CI95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in result["robustness"].items():
        summary = values["summary"]
        lines.append(
            f"| {name} | {summary['overlap_fidelity_selector_recovery']:.6f} | "
            f"{summary['nooverlap_single_view_recovery']:.6f} | "
            f"{summary['selector_gain_over_nooverlap']:+.6f} | "
            f"{values['bootstrap']['ci95']['selector_minus_nooverlap']} |"
        )
    lines.extend(
        [
            "",
            f"- Shuffled-score gain over fixed owner: {bootstrap['mean_delta']['shuffled_minus_fixed_owner']:+.6f} (CI95 {bootstrap['ci95']['shuffled_minus_fixed_owner']})",
            f"- Non-adjacent-score gain over fixed owner: {bootstrap['mean_delta']['nonadjacent_minus_fixed_owner']:+.6f} (CI95 {bootstrap['ci95']['nonadjacent_minus_fixed_owner']})",
            "- Direct non-adjacent reconstruction substitution is intentionally not reported: the views do not observe the same source token.",
            "",
            "## Failure analysis",
            "",
        ]
    )
    for example in result["failure_analysis"]["representative_examples"]:
        lines.append(
            f"- `{example['passage_id']}` position {example['source_position']}: "
            f"GT={example['reference_token']} selected={example['selected_prediction_token']} "
            f"margin={example['fidelity_margin_wrong_minus_correct']:+.8f}"
        )
    (output_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage5c(config: Mapping[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    inputs = _validate_inputs(config)
    overlap = inputs["overlap_cohort"]
    nooverlap = inputs["nooverlap_manifest"]
    strong = inputs["strong_cache"]
    stage5b = inputs["stage5b_result"]
    candidates = tuple(
        _candidate_from_dict(raw) for raw in strong["top1_candidates"]
    )
    alias_to_index = {
        str(alias): int(index)
        for alias, index in strong["alias_to_canonical_index"].items()
    }
    nooverlap_by_id = {
        str(document["passage_id"]): document
        for document in nooverlap["documents"]
    }
    stratum_by_id = {
        str(key): str(value)
        for key, value in nooverlap["pre_inversion_boundary_stratum_by_passage"].items()
    }
    overlap_documents = list(overlap["documents"])
    scores_by_document = []
    candidates_by_document = []
    for document in overlap_documents:
        values = _candidates_for_document(
            document,
            policy="overlap",
            candidates=candidates,
            alias_to_index=alias_to_index,
        )
        candidates_by_document.append(values)
        scores_by_document.append(
            {index: float(candidate.local_score) for index, candidate in enumerate(values)}
        )

    tolerance = float(config["selector"]["tie_tolerance"])
    rows = []
    all_failures = []
    for document_index, (document, overlap_candidates) in enumerate(
        zip(overlap_documents, candidates_by_document)
    ):
        passage_id = str(document["passage_id"])
        nooverlap_document = nooverlap_by_id[passage_id]
        nooverlap_candidates = _candidates_for_document(
            nooverlap_document,
            policy="nooverlap",
            candidates=candidates,
            alias_to_index=alias_to_index,
        )
        true_scores = scores_by_document[document_index]
        shuffled_scores = scores_by_document[(document_index + 1) % len(scores_by_document)]
        nonadjacent_scores = {
            index: true_scores[(index + 2) % 8] for index in range(8)
        }
        selector, failures = _score_document(
            document,
            overlap_candidates,
            decision_scores=true_scores,
            tie_tolerance=tolerance,
        )
        shuffled, _ = _score_document(
            document,
            overlap_candidates,
            decision_scores=shuffled_scores,
            tie_tolerance=tolerance,
        )
        nonadjacent, _ = _score_document(
            document,
            overlap_candidates,
            decision_scores=nonadjacent_scores,
            tie_tolerance=tolerance,
        )
        nominal = _nominal_position_score(
            document,
            overlap_candidates,
            decision_scores=true_scores,
            tie_tolerance=tolerance,
        )
        nooverlap_row = nooverlap_single_view_source_row(
            nooverlap_document, nooverlap_candidates
        )
        row = {
            "passage_id": passage_id,
            "pre_inversion_stratum": stratum_by_id[passage_id],
            "selector": selector,
            "shuffled_score_control": shuffled,
            "nonadjacent_score_control": nonadjacent,
            "nominal": nominal,
            "nooverlap": nooverlap_row,
            "selector_minus_nooverlap": selector["selector_recovery"]
            - nooverlap_row["source_token_recovery"],
            "selector_minus_fixed_owner": selector["selector_minus_fixed_owner"],
            "oracle_minus_nooverlap": selector["oracle_recovery"]
            - nooverlap_row["source_token_recovery"],
            "shuffled_minus_fixed_owner": shuffled["selector_minus_fixed_owner"],
            "nonadjacent_minus_fixed_owner": nonadjacent["selector_minus_fixed_owner"],
            "nominal_selector_minus_nominal_fixed_owner": nominal[
                "selector_minus_fixed_owner"
            ],
        }
        rows.append(row)
        for failure in failures:
            all_failures.append(
                {
                    "passage_id": passage_id,
                    "pre_inversion_stratum": stratum_by_id[passage_id],
                    **failure,
                }
            )

    bootstrap_config = config["bootstrap"]
    iterations = int(bootstrap_config["iterations"])
    seed = int(bootstrap_config["seed"])
    bootstrap = _bootstrap(rows, iterations=iterations, seed=seed)
    main = _summary(rows)

    def grouped(selected: Sequence[Mapping[str, Any]], offset: int) -> dict[str, Any]:
        return {
            "summary": _summary(selected),
            "bootstrap": _bootstrap(
                selected, iterations=iterations, seed=seed + offset
            ),
        }

    robustness = {}
    adjusted = [
        row
        for row in rows
        if row["pre_inversion_stratum"] == "BOUNDARY_ADJUSTMENT_REQUIRED"
    ]
    nonadjusted = [
        row
        for row in rows
        if row["pre_inversion_stratum"] != "BOUNDARY_ADJUSTMENT_REQUIRED"
    ]
    robustness["BOUNDARY_ADJUSTMENT_REQUIRED"] = grouped(adjusted, 1)
    robustness["POOLED_NONADJUSTED"] = grouped(nonadjusted, 2)
    for offset, stratum in enumerate(_STRATA, start=3):
        robustness[stratum] = grouped(
            [row for row in rows if row["pre_inversion_stratum"] == stratum],
            offset,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/gtr-t5-base",
        revision="9801579ce813fb37541e6098dffd17959fffcd6e",
    )
    representative = sorted(
        all_failures,
        key=lambda value: float(value["fidelity_margin_wrong_minus_correct"]),
        reverse=True,
    )[:5]
    for value in representative:
        value["reference_token"] = _token_text(
            tokenizer, int(value["reference_token_id"])
        )
        value["selected_prediction_token"] = _token_text(
            tokenizer, value["selected_prediction"]
        )
    failure_patterns = {
        "selector_wrong_oracle_correct_positions": len(all_failures),
        "affected_document_count": len(
            {str(value["passage_id"]) for value in all_failures}
        ),
        "wrong_higher_fidelity_over_correct_lower_fidelity": sum(
            float(value["fidelity_margin_wrong_minus_correct"]) > tolerance
            for value in all_failures
        ),
        "fidelity_tie_resolved_to_wrong_lower_index": sum(
            bool(value["tie"]) for value in all_failures
        ),
        "selected_deletion_count": sum(
            value["selected_prediction"] == "<DEL>" for value in all_failures
        ),
        "selected_substitution_count": sum(
            value["selected_prediction"] != "<DEL>" for value in all_failures
        ),
        "wrong_minus_correct_fidelity_margin": {
            "minimum": min(
                (
                    float(value["fidelity_margin_wrong_minus_correct"])
                    for value in all_failures
                ),
                default=None,
            ),
            "mean": (
                mean(
                    float(value["fidelity_margin_wrong_minus_correct"])
                    for value in all_failures
                )
                if all_failures
                else None
            ),
            "maximum": max(
                (
                    float(value["fidelity_margin_wrong_minus_correct"])
                    for value in all_failures
                ),
                default=None,
            ),
            "at_most_0_005_count": sum(
                float(value["fidelity_margin_wrong_minus_correct"]) <= 0.005
                for value in all_failures
            ),
        },
    }
    disagreement = {
        **stage5b["h2_fidelity"],
        "ci95": stage5b["source_level"]["ALL"]["bootstrap"]["h2_ci95"],
        "reused_without_redefinition": True,
    }
    gate = _gate(main, bootstrap, disagreement)
    result = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE5C_FIDELITY_SELECTOR_COMPLETE",
        "config": {
            "path": str(config["resolved_config"]),
            "sha256": _sha256(Path(str(config["resolved_config"]))),
            "frozen_before_evaluation": True,
        },
        "input_hashes": {
            name: str(entry["sha256"]) for name, entry in config["inputs"].items()
        },
        "selector": config["selector"],
        "main_result": main,
        "bootstrap": bootstrap,
        "disagreement_analysis": disagreement,
        "robustness": robustness,
        "negative_controls": {
            "shuffled_scores": config["negative_controls"]["shuffled_scores"],
            "nonadjacent_scores": config["negative_controls"]["nonadjacent_scores"],
            "direct_nonadjacent_reconstruction_gain_defined": False,
            "reason": config["negative_controls"]["note"],
        },
        "failure_analysis": {
            **failure_patterns,
            "representative_examples": representative,
        },
        "gate": gate,
        "vec2text_reexecuted": False,
        "ground_truth_used_for_selector_decision": False,
        "new_correction_or_joint_method_implemented": False,
    }
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / str(config["result_filename"]), result)
    _write_json_atomic(
        output_dir / str(config["trajectory_filename"]),
        {
            "schema_version": 1,
            "config_sha256": result["config"]["sha256"],
            "trajectories": rows,
        },
    )
    _write_report(output_dir, str(config["report_filename"]), result)
    return result
