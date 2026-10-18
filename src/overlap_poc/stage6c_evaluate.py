"""Single frozen evaluation of the Stage 6C final unseen cohort."""

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

from .diagnostics import stitch_candidate_sequence_constrained
from .reconstruction import Candidate
from .stage1_strong_saturation import _candidate_from_dict, _write_json_atomic
from .stage3_longdoc_strong import _percentile
from .stage5c_fidelity_selector import _score_document
from .stage5d_assembly import fixed_coordinate_fidelity_assembly
from .stage6a_source_free_assembly import (
    _TAXONOMY,
    _document_score,
    _error_diagnostics,
    _sha256,
    _transfer_metrics,
    assemble_source_free_pairwise,
)


_IMPLEMENTATION_VERSION = "stage6c-final-evaluation-v1"
_SEQUENCE_METHODS = (
    "nooverlap_strong",
    "constrained_fixed_owner",
    "fixed_coordinate_fidelity",
    "source_free_progressive",
)
_BOOTSTRAP_METRICS = (
    "proposed_minus_nooverlap",
    "proposed_minus_source_aligned_fixed_owner",
    "proposed_minus_constrained_fixed_owner",
    "proposed_minus_fixed_coordinate",
    "source_aligned_minus_proposed",
    "proposed_sequence_similarity_minus_nooverlap",
    "proposed_sequence_similarity_minus_fixed_coordinate",
    "local_fidelity_minus_source_aligned_fixed_owner",
    "shuffled_fidelity_minus_source_aligned_fixed_owner",
    "nonadjacent_fidelity_minus_source_aligned_fixed_owner",
)
_EXPECTED_METHODS = {
    "nooverlap_strong": "concatenate_ordered_nooverlap_top1_candidates",
    "overlap_fixed_owner": "source_aligned_lower_window_index_diagnostic",
    "constrained_fixed_owner": "constrained_suffix_prefix_left_owner_sequence",
    "fixed_coordinate_fidelity": "nominal_31_15_coordinates_with_higher_local_score",
    "source_free_progressive_proposed": "accumulated_progressive_stage6a_v2",
    "source_aligned_fidelity": "diagnostic_ceiling_only",
    "best_of_views_oracle": "oracle_ceiling_only",
}
_EXPECTED_NEGATIVE_CONTROLS = {
    "scope": "source_aligned_selector_diagnostic_only",
    "shuffled_fidelity": "next_document_cyclic_same_window_index",
    "nonadjacent_fidelity": "same_document_window_index_plus_2_mod_8",
    "document_order_is_part_of_definition": True,
}
_EXPECTED_PASS_RULES = [
    "proposed_minus_nooverlap_point_gt_zero",
    "proposed_minus_nooverlap_ci95_lower_gt_zero",
    "proposed_minus_source_aligned_fixed_owner_ci95_lower_gt_zero",
    "gain_recovery_ratio_gte_0.5",
    "sequence_guard_pass",
    "local_fidelity_selector_ci95_lower_gt_zero",
    "shuffled_selector_ci95_lower_lte_zero",
    "nonadjacent_selector_ci95_lower_lte_zero",
]
_EXPECTED_PARTIAL_RULES = [
    "proposed_minus_nooverlap_point_gte_zero",
    "proposed_minus_nooverlap_ci95_includes_zero",
    "proposed_minus_source_aligned_fixed_owner_ci95_lower_gt_zero",
    "gain_recovery_ratio_gte_0.5",
    "sequence_guard_pass",
    "local_fidelity_selector_ci95_lower_gt_zero",
    "shuffled_selector_ci95_lower_lte_zero",
    "nonadjacent_selector_ci95_lower_lte_zero",
]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def load_stage6c_evaluation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 6C evaluation config must be a YAML mapping")

    def resolve(value: str) -> str:
        candidate = Path(value)
        return str(
            (candidate if candidate.is_absolute() else config_path.parent / candidate)
            .resolve()
        )

    resolved = dict(raw)
    resolved["resolved_config"] = str(config_path.resolve())
    resolved["resolved_protocol"] = resolve(str(raw["protocol"]["path"]))
    resolved["resolved_inputs"] = {
        name: {**entry, "path": resolve(str(entry["path"]))}
        for name, entry in raw["inputs"].items()
    }
    return resolved


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("stage") != "stage6c_final_evaluation_protocol_v1":
        raise RuntimeError("Stage 6C evaluation protocol identity changed")
    if protocol.get("frozen_before_stage6c_inversion") is not True:
        raise RuntimeError("Stage 6C protocol was not frozen before inversion")
    cohort = protocol["cohort"]
    if (
        int(cohort["deterministic_sha_order_start_inclusive"]),
        int(cohort["deterministic_sha_order_stop_exclusive"]),
        int(cohort["documents"]),
        int(cohort["source_tokens_per_document"]),
    ) != (150, 210, 60, 136):
        raise RuntimeError("Stage 6C protocol cohort changed")
    bootstrap = protocol["bootstrap"]
    if (
        int(bootstrap["iterations"]),
        int(bootstrap["seed"]),
        str(bootstrap["unit"]),
    ) != (10000, 20260910, "final_validation_document"):
        raise RuntimeError("Stage 6C bootstrap protocol changed")
    if float(protocol["realization"]["substantial_gain_recovery_threshold"]) != 0.5:
        raise RuntimeError("Stage 6C gain recovery threshold changed")
    if protocol.get("methods") != _EXPECTED_METHODS:
        raise RuntimeError("Stage 6C method definitions changed")
    if protocol.get("negative_controls") != _EXPECTED_NEGATIVE_CONTROLS:
        raise RuntimeError("Stage 6C negative-control definitions changed")
    if protocol.get("sequence_guard") != {
        "proposed_normalized_similarity_gte_nooverlap": True,
        "proposed_total_edit_distance_lte_nooverlap": True,
    }:
        raise RuntimeError("Stage 6C sequence guard changed")
    gate = protocol.get("final_gate", {})
    if (
        gate.get("DOCUMENT_ATTACK_PASS") != _EXPECTED_PASS_RULES
        or gate.get("OVERLAP_ASSEMBLY_ONLY_PASS") != _EXPECTED_PARTIAL_RULES
        or gate.get("DOCUMENT_ATTACK_FAIL") != "otherwise"
    ):
        raise RuntimeError("Stage 6C final gate changed")


def _validate_frozen_code(freeze: Mapping[str, Any]) -> None:
    for group, entries in freeze.get("code_hashes", {}).items():
        for name, entry in entries.items():
            path = Path(str(entry["path"]))
            if not path.is_file() or _sha256(path) != str(entry["sha256"]):
                raise RuntimeError(
                    f"Stage 6B frozen {group} code changed before evaluation: {name}"
                )


def _validate_frozen_protocol_files(freeze: Mapping[str, Any]) -> None:
    for name, entry in freeze.get("protocol_files", {}).items():
        path = Path(str(entry["path"]))
        if not path.is_file() or _sha256(path) != str(entry["sha256"]):
            raise RuntimeError(
                f"Stage 6B frozen protocol changed before evaluation: {name}"
            )


def _validate_cohort_links(
    cohort_freeze: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
) -> None:
    if cohort_freeze.get("algorithm_freeze", {}).get("sha256") != str(
        inputs["algorithm_freeze"]["sha256"]
    ):
        raise RuntimeError("Stage 6C cohort points to another algorithm freeze")
    for name in ("overlap_cohort", "nooverlap_manifest"):
        if cohort_freeze.get("artifacts", {}).get(name, {}).get("sha256") != str(
            inputs[name]["sha256"]
        ):
            raise RuntimeError(f"Stage 6C cohort artifact linkage changed: {name}")


def _validate_inputs(
    config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("stage") != "stage6c_final_evaluation_v1":
        raise RuntimeError("Stage 6C evaluation stage identity changed")
    if config.get("single_final_evaluation") is not True:
        raise RuntimeError("Stage 6C must remain a single final evaluation")
    protocol_path = Path(str(config["resolved_protocol"]))
    if _sha256(protocol_path) != str(config["protocol"]["sha256"]):
        raise RuntimeError("Stage 6C frozen evaluation protocol changed")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise RuntimeError("Stage 6C protocol is not a mapping")
    _validate_protocol(protocol)
    loaded: dict[str, dict[str, Any]] = {}
    for name, entry in config["resolved_inputs"].items():
        path = Path(str(entry["path"]))
        if name != "strong_generation_config" and "stage5d_assembly" in str(path):
            raise RuntimeError("Stage 6C evaluation cannot consume Stage 5D outputs")
        if _sha256(path) != str(entry["sha256"]):
            raise RuntimeError(f"Stage 6C frozen evaluation input changed: {name}")
        if path.suffix == ".json":
            loaded[name] = _load_json(path)
    freeze = loaded["algorithm_freeze"]
    cohort_freeze = loaded["cohort_freeze"]
    overlap = loaded["overlap_cohort"]
    nooverlap = loaded["nooverlap_manifest"]
    strong = loaded["strong_cache"]
    if (
        freeze.get("status") != "STAGE6B_ALGORITHM_FROZEN"
        or freeze.get("freeze_closed") is not True
    ):
        raise RuntimeError("Stage 6B algorithm freeze is not closed")
    _validate_frozen_code(freeze)
    _validate_frozen_protocol_files(freeze)
    frozen_protocol = freeze.get("protocol_hashes", {}).get(
        "stage6c_evaluation_protocol"
    )
    if frozen_protocol != str(config["protocol"]["sha256"]):
        raise RuntimeError("Stage 6B does not bind the Stage 6C evaluation protocol")
    if cohort_freeze.get("status") != "STAGE6C_FINAL_COHORT_FROZEN":
        raise RuntimeError("Stage 6C cohort freeze is incomplete")
    _validate_cohort_links(cohort_freeze, config["resolved_inputs"])
    if overlap["manifest"] != nooverlap["manifest"] or len(overlap["manifest"]) != 60:
        raise RuntimeError("Stage 6C paired cohort changed")
    if strong.get("status") != "STAGE6C_STRONG_TOP1_CACHE_COMPLETE":
        raise RuntimeError("Stage 6C strong cache is incomplete")
    if strong.get("algorithm_freeze_sha256") != str(
        config["resolved_inputs"]["algorithm_freeze"]["sha256"]
    ):
        raise RuntimeError("Stage 6C strong cache refers to another algorithm freeze")
    if strong.get("cohort_freeze_sha256") != str(
        config["resolved_inputs"]["cohort_freeze"]["sha256"]
    ):
        raise RuntimeError("Stage 6C strong cache refers to another cohort freeze")
    if strong.get("config", {}).get("sha256") != str(
        config["resolved_inputs"]["strong_generation_config"]["sha256"]
    ):
        raise RuntimeError("Stage 6C strong cache refers to another generation config")
    if strong.get("strong_reproduction_setting") != "correction50+sbeam8":
        raise RuntimeError("Stage 6C strong setting changed")
    candidates = list(strong.get("top1_candidates", []))
    records = list(strong.get("records", []))
    if len(candidates) != 720 or len(records) != 720:
        raise RuntimeError("Stage 6C strong candidate count changed")
    if any(
        not bool(record.get("public_top1_exact_id_match"))
        or int(record.get("embed_call_count", -1)) != 51
        or int(record.get("search_hypothesis_rows_embedded", -1)) != 3145
        for record in records
    ):
        raise RuntimeError("Stage 6C candidate-generation invariant changed")
    if strong.get("evaluation_executed") is not False:
        raise RuntimeError("Stage 6C strong cache was not sealed before evaluation")
    return loaded, protocol


def _candidate_map(strong: Mapping[str, Any]) -> tuple[tuple[Candidate, ...], dict[str, int]]:
    candidates = tuple(
        _candidate_from_dict(candidate) for candidate in strong["top1_candidates"]
    )
    aliases = {
        str(alias): int(index)
        for alias, index in strong["alias_to_canonical_index"].items()
    }
    return candidates, aliases


def _candidates_for_document(
    document: Mapping[str, Any],
    *,
    policy: str,
    candidates: Sequence[Candidate],
    aliases: Mapping[str, int],
) -> tuple[Candidate, ...]:
    passage_id = str(document["passage_id"])
    return tuple(
        candidates[aliases[f"{passage_id}:{policy}:{window['window_index']}"]]
        for window in document["windows"]
    )


def _length_summary(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [int(score["prediction_length"]) for score in scores]
    biases = [length - 136 for length in lengths]
    return {
        "mean_prediction_length": mean(lengths),
        "min_prediction_length": min(lengths),
        "p25_prediction_length": _percentile(lengths, 0.25),
        "median_prediction_length": _percentile(lengths, 0.5),
        "p75_prediction_length": _percentile(lengths, 0.75),
        "max_prediction_length": max(lengths),
        "mean_signed_length_bias_tokens": mean(biases),
        "mean_relative_length_bias": mean(bias / 136 for bias in biases),
        "mean_absolute_length_error_tokens": mean(abs(bias) for bias in biases),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"documents": len(rows), "source_tokens": 136 * len(rows)}
    for method in _SEQUENCE_METHODS:
        scores = [row[method] for row in rows]
        total = sum(int(score["source_token_count"]) for score in scores)
        result[method] = {
            "token_recovery": sum(
                int(score["exact_matched_token_count"]) for score in scores
            )
            / total,
            "normalized_sequence_similarity": mean(
                float(score["normalized_edit_similarity"]) for score in scores
            ),
            "total_edit_distance": sum(int(score["edit_distance"]) for score in scores),
            "substitutions": sum(int(score["substitution_count"]) for score in scores),
            "insertions": sum(
                int(score["prediction_insertion_count"]) for score in scores
            ),
            "deletions": sum(
                int(score["reference_deletion_count"]) for score in scores
            ),
            "document_exact_fraction": mean(
                bool(score["document_exact"]) for score in scores
            ),
            "output_length": _length_summary(scores),
        }
    total = 136 * len(rows)
    fixed = sum(int(row["source_aligned"]["fixed_owner_correct"]) for row in rows)
    selected = sum(int(row["source_aligned"]["selector_correct"]) for row in rows)
    oracle = sum(int(row["source_aligned"]["oracle_correct"]) for row in rows)
    shuffled = sum(int(row["negative_controls"]["shuffled_correct"]) for row in rows)
    nonadjacent = sum(
        int(row["negative_controls"]["nonadjacent_correct"]) for row in rows
    )
    result["source_aligned_diagnostics"] = {
        "fixed_owner_recovery": fixed / total,
        "fidelity_selector_recovery": selected / total,
        "best_of_views_oracle_recovery": oracle / total,
    }
    result["negative_controls"] = {
        "local_fidelity_recovery": selected / total,
        "shuffled_fidelity_recovery": shuffled / total,
        "nonadjacent_fidelity_recovery": nonadjacent / total,
        "fixed_owner_recovery": fixed / total,
        "local_gain_over_fixed_owner": (selected - fixed) / total,
        "shuffled_gain_over_fixed_owner": (shuffled - fixed) / total,
        "nonadjacent_gain_over_fixed_owner": (nonadjacent - fixed) / total,
    }
    result["transfer"] = _transfer_metrics(
        float(result["source_free_progressive"]["token_recovery"]),
        fixed / total,
        selected / total,
    )
    return result


def _bootstrap(
    rows: Sequence[Mapping[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    samples = {name: [] for name in _BOOTSTRAP_METRICS}
    ratios: list[float] = []
    for _ in range(iterations):
        selected_rows = [rows[rng.randrange(len(rows))] for _ in rows]
        estimates = {
            name: mean(float(row["deltas"][name]) for row in selected_rows)
            for name in _BOOTSTRAP_METRICS
        }
        for name, estimate in estimates.items():
            samples[name].append(estimate)
        numerator = estimates["proposed_minus_source_aligned_fixed_owner"]
        denominator = numerator + estimates["source_aligned_minus_proposed"]
        if abs(denominator) > 1e-12:
            ratios.append(numerator / denominator)
    estimates = {
        name: mean(float(row["deltas"][name]) for row in rows)
        for name in _BOOTSTRAP_METRICS
    }
    return {
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "final validation document",
        "mean_delta": estimates,
        "ci95": {
            name: [_percentile(values, 0.025), _percentile(values, 0.975)]
            for name, values in samples.items()
        },
        "gain_recovery_ratio_ci95": [
            _percentile(ratios, 0.025),
            _percentile(ratios, 0.975),
        ]
        if ratios
        else [None, None],
    }


def _gate(
    summary: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    del protocol
    ci = bootstrap["ci95"]
    proposed = summary["source_free_progressive"]
    nooverlap = summary["nooverlap_strong"]
    primary_delta = float(bootstrap["mean_delta"]["proposed_minus_nooverlap"])
    primary_ci = ci["proposed_minus_nooverlap"]
    fixed_ci = ci["proposed_minus_source_aligned_fixed_owner"]
    controls = summary["negative_controls"]
    checks = {
        "proposed_minus_nooverlap_point_gt_zero": primary_delta > 0.0,
        "proposed_minus_nooverlap_ci95_lower_gt_zero": float(primary_ci[0]) > 0.0,
        "proposed_minus_source_aligned_fixed_owner_ci95_lower_gt_zero": float(
            fixed_ci[0]
        )
        > 0.0,
        "gain_recovery_ratio_gte_0_5": summary["transfer"]["gain_recovery_ratio"]
        is not None
        and float(summary["transfer"]["gain_recovery_ratio"]) >= 0.5,
        "proposed_normalized_similarity_gte_nooverlap": float(
            proposed["normalized_sequence_similarity"]
        )
        >= float(nooverlap["normalized_sequence_similarity"]),
        "proposed_total_edit_distance_lte_nooverlap": int(
            proposed["total_edit_distance"]
        )
        <= int(nooverlap["total_edit_distance"]),
        "local_fidelity_selector_ci95_lower_gt_zero": float(
            ci["local_fidelity_minus_source_aligned_fixed_owner"][0]
        )
        > 0.0,
        "shuffled_selector_ci95_lower_lte_zero": float(
            ci["shuffled_fidelity_minus_source_aligned_fixed_owner"][0]
        )
        <= 0.0,
        "nonadjacent_selector_ci95_lower_lte_zero": float(
            ci["nonadjacent_fidelity_minus_source_aligned_fixed_owner"][0]
        )
        <= 0.0,
    }
    sequence_guard = (
        checks["proposed_normalized_similarity_gte_nooverlap"]
        and checks["proposed_total_edit_distance_lte_nooverlap"]
    )
    controls_guard = (
        checks["local_fidelity_selector_ci95_lower_gt_zero"]
        and checks["shuffled_selector_ci95_lower_lte_zero"]
        and checks["nonadjacent_selector_ci95_lower_lte_zero"]
    )
    shared_guards = (
        checks["proposed_minus_source_aligned_fixed_owner_ci95_lower_gt_zero"]
        and checks["gain_recovery_ratio_gte_0_5"]
        and sequence_guard
        and controls_guard
    )
    if (
        checks["proposed_minus_nooverlap_point_gt_zero"]
        and checks["proposed_minus_nooverlap_ci95_lower_gt_zero"]
        and shared_guards
    ):
        state = "DOCUMENT_ATTACK_PASS"
    elif (
        primary_delta >= 0.0
        and float(primary_ci[0]) <= 0.0 <= float(primary_ci[1])
        and shared_guards
    ):
        state = "OVERLAP_ASSEMBLY_ONLY_PASS"
    else:
        state = "DOCUMENT_ATTACK_FAIL"
    return {
        "state": state,
        "checks": checks,
        "sequence_guard_pass": sequence_guard,
        "negative_control_guard_pass": controls_guard,
        "post_final_tuning_authorized": False,
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    summary = result["primary_table"]
    bootstrap = result["paired_bootstrap"]
    lines = [
        "# Stage 6C — Final Unseen Document Attack Validation",
        "",
        f"- Final gate: `{result['final_gate']['state']}`",
        "- Cohort: deterministic SHA-order positions 150..209 (60 documents).",
        "- Inference used only Top-1 reconstructed token IDs, window order, public geometry, and Candidate.local_score.",
        "- Source-Aligned Fidelity and Best-of-Views Oracle are diagnostics, not attack methods.",
        "",
        "| Method | Recovery | Sequence similarity | Edits | Substitutions | Insertions | Deletions | Document exact | Mean length bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "nooverlap_strong": "No-overlap strong",
        "constrained_fixed_owner": "Constrained fixed-owner stitcher",
        "fixed_coordinate_fidelity": "Fixed-coordinate fidelity",
        "source_free_progressive": "Source-Free Progressive Proposed",
    }
    for name, label in labels.items():
        value = summary[name]
        lines.append(
            f"| {label} | {value['token_recovery']:.6f} | {value['normalized_sequence_similarity']:.6f} | "
            f"{value['total_edit_distance']} | {value['substitutions']} | {value['insertions']} | "
            f"{value['deletions']} | {value['document_exact_fraction']:.6f} | "
            f"{value['output_length']['mean_signed_length_bias_tokens']:+.3f} |"
        )
    diagnostic = summary["source_aligned_diagnostics"]
    lines.extend(
        [
            f"| Overlap Fixed Owner (source-aligned diagnostic) | {diagnostic['fixed_owner_recovery']:.6f} | — | — | — | — | — | — | — |",
            f"| Source-Aligned Fidelity (diagnostic ceiling) | {diagnostic['fidelity_selector_recovery']:.6f} | — | — | — | — | — | — | — |",
            f"| Best-of-Views Oracle (oracle ceiling) | {diagnostic['best_of_views_oracle_recovery']:.6f} | — | — | — | — | — | — | — |",
            "",
            "## Paired primary comparisons",
            "",
            f"- Proposed minus no-overlap: {bootstrap['mean_delta']['proposed_minus_nooverlap']:+.6f}; CI95 {bootstrap['ci95']['proposed_minus_nooverlap']}",
            f"- Proposed minus Source-Aligned Fixed Owner: {bootstrap['mean_delta']['proposed_minus_source_aligned_fixed_owner']:+.6f}; CI95 {bootstrap['ci95']['proposed_minus_source_aligned_fixed_owner']}",
            f"- Proposed minus constrained fixed-owner: {bootstrap['mean_delta']['proposed_minus_constrained_fixed_owner']:+.6f}; CI95 {bootstrap['ci95']['proposed_minus_constrained_fixed_owner']}",
            "",
            "## Realization",
            "",
            f"- Realization gap: {summary['transfer']['realization_gap']:+.6f}",
            f"- Gain recovery ratio: {summary['transfer']['gain_recovery_ratio']}",
            f"- Gain recovery ratio CI95: {bootstrap['gain_recovery_ratio_ci95']}",
            "",
            "## Frozen negative controls",
            "",
            f"- Local fidelity gain over fixed owner: {summary['negative_controls']['local_gain_over_fixed_owner']:+.6f}; CI95 {bootstrap['ci95']['local_fidelity_minus_source_aligned_fixed_owner']}",
            f"- Shuffled fidelity gain: {summary['negative_controls']['shuffled_gain_over_fixed_owner']:+.6f}; CI95 {bootstrap['ci95']['shuffled_fidelity_minus_source_aligned_fixed_owner']}",
            f"- Non-adjacent fidelity gain: {summary['negative_controls']['nonadjacent_gain_over_fixed_owner']:+.6f}; CI95 {bootstrap['ci95']['nonadjacent_fidelity_minus_source_aligned_fixed_owner']}",
            "",
            "## Residual error taxonomy",
            "",
        ]
    )
    for name, count in result["residual_error_taxonomy"]["counts"].items():
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            f"- Source-Aligned-correct / Source-Free-wrong cases: {len(result['source_aligned_correct_source_free_wrong_cases'])}",
            "",
            "No inference or gate rule may be changed after this result.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_stage6c_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    loaded, protocol = _validate_inputs(config)
    overlap = loaded["overlap_cohort"]
    nooverlap = loaded["nooverlap_manifest"]
    strong = loaded["strong_cache"]
    candidates, aliases = _candidate_map(strong)
    nooverlap_by_id = {
        str(document["passage_id"]): document
        for document in nooverlap["documents"]
    }
    overlap_documents = list(overlap["documents"])
    candidates_by_document = [
        _candidates_for_document(
            document,
            policy="overlap",
            candidates=candidates,
            aliases=aliases,
        )
        for document in overlap_documents
    ]
    scores_by_document = [
        {
            index: float(candidate.local_score)
            for index, candidate in enumerate(document_candidates)
        }
        for document_candidates in candidates_by_document
    ]
    rows: list[dict[str, Any]] = []
    taxonomy_counts: Counter[str] = Counter()
    realization_cases: list[dict[str, Any]] = []
    total_runtime = 0.0
    for document_index, (document, overlap_candidates) in enumerate(
        zip(overlap_documents, candidates_by_document, strict=True)
    ):
        passage_id = str(document["passage_id"])
        nooverlap_document = nooverlap_by_id[passage_id]
        nooverlap_candidates = _candidates_for_document(
            nooverlap_document,
            policy="nooverlap",
            candidates=candidates,
            aliases=aliases,
        )
        reference = tuple(
            int(value) for value in document["prefixes"]["8"]["reference_ids"]
        )
        nooverlap_prediction = tuple(
            token
            for candidate in nooverlap_candidates
            for token in candidate.content_ids
        )
        constrained = stitch_candidate_sequence_constrained(
            overlap_candidates,
            overlap_tokens=16,
            max_boundary_shift=2,
            deviation_cost=0.5,
        )
        fixed_coordinate = fixed_coordinate_fidelity_assembly(
            overlap_candidates,
            content_tokens=31,
            stride=15,
            tie_tolerance=1e-12,
        )
        started = time.perf_counter()
        proposed = assemble_source_free_pairwise(
            overlap_candidates,
            overlap_tokens=16,
            max_boundary_shift=2,
            deviation_cost=0.5,
            tie_tolerance=1e-12,
        )
        elapsed = time.perf_counter() - started
        total_runtime += elapsed
        true_scores = scores_by_document[document_index]
        shuffled_scores = scores_by_document[
            (document_index + 1) % len(scores_by_document)
        ]
        nonadjacent_scores = {
            index: true_scores[(index + 2) % 8] for index in range(8)
        }
        source_aligned, _ = _score_document(
            document,
            overlap_candidates,
            decision_scores=true_scores,
            tie_tolerance=1e-12,
        )
        shuffled, _ = _score_document(
            document,
            overlap_candidates,
            decision_scores=shuffled_scores,
            tie_tolerance=1e-12,
        )
        nonadjacent, _ = _score_document(
            document,
            overlap_candidates,
            decision_scores=nonadjacent_scores,
            tie_tolerance=1e-12,
        )
        method_scores = {
            "nooverlap_strong": _document_score(nooverlap_prediction, reference),
            "constrained_fixed_owner": _document_score(
                constrained.content_ids, reference
            ),
            "fixed_coordinate_fidelity": _document_score(
                fixed_coordinate, reference
            ),
            "source_free_progressive": _document_score(
                proposed.content_ids, reference
            ),
        }
        taxonomy, cases = _error_diagnostics(
            document,
            overlap_candidates,
            proposed.content_ids,
            proposed,
            tie_tolerance=1e-12,
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
            "negative_controls": {
                "shuffled_correct": int(shuffled["selector_correct"]),
                "nonadjacent_correct": int(nonadjacent["selector_correct"]),
            },
            "assembly_runtime_seconds": elapsed,
            "assembly_diagnostic": {
                "boundary_left_widths": list(proposed.boundary_left_widths),
                "boundary_right_widths": list(proposed.boundary_right_widths),
                "boundary_alignment_scores": list(
                    proposed.boundary_alignment_scores
                ),
                "boundary_owner_views": list(proposed.boundary_owner_views),
                "inter_boundary_gap_widths": list(
                    proposed.inter_boundary_gap_widths
                ),
                "inter_boundary_duplicate_widths": list(
                    proposed.inter_boundary_duplicate_widths
                ),
            },
            "error_taxonomy": taxonomy,
        }
        row["deltas"] = {
            "proposed_minus_nooverlap": method_scores["source_free_progressive"]["token_recovery"]
            - method_scores["nooverlap_strong"]["token_recovery"],
            "proposed_minus_source_aligned_fixed_owner": method_scores["source_free_progressive"]["token_recovery"]
            - source_aligned["fixed_owner_recovery"],
            "proposed_minus_constrained_fixed_owner": method_scores["source_free_progressive"]["token_recovery"]
            - method_scores["constrained_fixed_owner"]["token_recovery"],
            "proposed_minus_fixed_coordinate": method_scores["source_free_progressive"]["token_recovery"]
            - method_scores["fixed_coordinate_fidelity"]["token_recovery"],
            "source_aligned_minus_proposed": source_aligned["selector_recovery"]
            - method_scores["source_free_progressive"]["token_recovery"],
            "proposed_sequence_similarity_minus_nooverlap": method_scores["source_free_progressive"]["normalized_edit_similarity"]
            - method_scores["nooverlap_strong"]["normalized_edit_similarity"],
            "proposed_sequence_similarity_minus_fixed_coordinate": method_scores["source_free_progressive"]["normalized_edit_similarity"]
            - method_scores["fixed_coordinate_fidelity"]["normalized_edit_similarity"],
            "local_fidelity_minus_source_aligned_fixed_owner": source_aligned["selector_minus_fixed_owner"],
            "shuffled_fidelity_minus_source_aligned_fixed_owner": shuffled["selector_minus_fixed_owner"],
            "nonadjacent_fidelity_minus_source_aligned_fixed_owner": nonadjacent["selector_minus_fixed_owner"],
        }
        rows.append(row)

    summary = _summary(rows)
    bootstrap = _bootstrap(
        rows,
        iterations=int(protocol["bootstrap"]["iterations"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    gate = _gate(summary, bootstrap, protocol)
    result = {
        "schema_version": 1,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "status": "STAGE6C_FINAL_EVALUATION_COMPLETE",
        "estimand": "Deterministic SHA-order positions 150..209; 60 final unseen documents.",
        "config": {
            "path": str(config["resolved_config"]),
            "sha256": _sha256(Path(str(config["resolved_config"]))),
        },
        "protocol": {
            "path": str(config["resolved_protocol"]),
            "sha256": str(config["protocol"]["sha256"]),
        },
        "input_hashes": {
            name: {"path": str(entry["path"]), "sha256": str(entry["sha256"])}
            for name, entry in config["resolved_inputs"].items()
        },
        "primary_table": summary,
        "paired_bootstrap": bootstrap,
        "sequence_level_tradeoff": {
            "proposed_minus_nooverlap_similarity": bootstrap["mean_delta"][
                "proposed_sequence_similarity_minus_nooverlap"
            ],
            "proposed_minus_fixed_coordinate_similarity": bootstrap["mean_delta"][
                "proposed_sequence_similarity_minus_fixed_coordinate"
            ],
            "sequence_guard_pass": gate["sequence_guard_pass"],
        },
        "frozen_negative_controls": summary["negative_controls"],
        "residual_error_taxonomy": {
            "counts": {name: int(taxonomy_counts[name]) for name in _TAXONOMY},
            "total_residual_source_position_errors": sum(taxonomy_counts.values()),
            "prediction_insertions_stored_separately": int(
                summary["source_free_progressive"]["insertions"]
            ),
            "reconciled": sum(taxonomy_counts.values())
            == 136 * len(rows)
            - sum(
                int(row["source_free_progressive"]["exact_matched_token_count"])
                for row in rows
            ),
            "attribution_scope": "Frozen Stage 6A post-hoc hierarchical diagnostic; not causal provenance.",
        },
        "source_aligned_correct_source_free_wrong_cases": realization_cases,
        "assembly_runtime": {
            "total_seconds": total_runtime,
            "seconds_per_document": total_runtime / len(rows),
        },
        "final_gate": gate,
        "ground_truth_used_by_inference": False,
        "source_alignment_used_by_inference": False,
        "oracle_used_by_inference": False,
        "post_final_rule_change_authorized": False,
    }
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / str(config["result_filename"]), result)
    _write_json_atomic(
        output_dir / str(config["trajectory_filename"]),
        {
            "schema_version": 1,
            "config_sha256": result["config"]["sha256"],
            "protocol_sha256": result["protocol"]["sha256"],
            "trajectories": rows,
        },
    )
    _write_report(output_dir / str(config["report_filename"]), result)
    return result
