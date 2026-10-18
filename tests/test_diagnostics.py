from __future__ import annotations

import pytest

from overlap_poc.diagnostics import (
    align_token_ids,
    candidate_consistency,
    classify_diagnostic_gate,
    evaluate_candidate_pair,
    levenshtein_distance,
    ordered_reconstruction_score,
    stitch_candidate_ids,
    stitch_candidate_ids_constrained,
    stitch_candidate_sequence,
    stitch_candidate_sequence_constrained,
    stitch_candidate_sequence_fidelity_constrained,
    summarize_diagnostics,
    window_reconstruction_score,
)
from overlap_poc.reconstruction import Candidate


def _candidate(ids, score, name):
    return Candidate(
        text=name,
        local_score=score,
        correction_steps=0,
        content_ids=tuple(ids),
    )


def test_fixed_consistency_penalizes_short_exact_fragments() -> None:
    left = _candidate(range(5), 1.0, "left")
    right = _candidate(range(5), 1.0, "right")
    assert candidate_consistency(left, right) == 5 / 16


def test_independent_oracle_and_delta_c_are_computed_from_one_pool() -> None:
    reference = tuple(range(1, 47))
    left_good = tuple(range(1, 32))
    right_good = tuple(range(16, 47))
    left_bad = left_good[:-3] + (901, 902, 903)
    right_bad = (801, 802, 803) + right_good[3:]
    pools = [
        [_candidate(left_bad, 0.99, "left-bad"), _candidate(left_good, 0.95, "left-good")],
        [_candidate(right_bad, 0.99, "right-bad"), _candidate(right_good, 0.95, "right-good")],
    ]
    row = evaluate_candidate_pair(
        pools,
        reference_ids=reference,
        shared_ids=tuple(range(16, 32)),
    )
    assert row["independent_indices"] == [0, 0]
    # The right candidate's corrupted prefix is removed by the shared stitcher,
    # so the final-outcome oracle keeps its higher-fidelity endpoint on ties.
    assert row["oracle_indices"] == [1, 0]
    assert row["p_oracle"] == 1.0
    assert row["h_pair"] > 0
    assert row["c_oracle"] == 13 / 16
    assert row["delta_c"] > 0


def test_stitch_and_m46_use_ordered_content_ids() -> None:
    left = _candidate(range(1, 32), 1.0, "left")
    right = _candidate(range(16, 47), 1.0, "right")
    stitched = stitch_candidate_ids(left, right)
    assert stitched.consumed_right_tokens == 16
    assert stitched.content_ids == tuple(range(1, 47))
    assert ordered_reconstruction_score(tuple(range(1, 47)), stitched.content_ids) == 1.0
    assert levenshtein_distance((1, 2, 3), (1, 9, 3)) == 1


def test_multi_window_stitch_preserves_pairwise_policy_and_provenance() -> None:
    candidates = [
        _candidate(range(1, 32), 1.0, "w0"),
        _candidate(range(16, 47), 1.0, "w1"),
        _candidate(range(31, 62), 1.0, "w2"),
    ]
    stitched = stitch_candidate_sequence(candidates)

    assert stitched.content_ids == tuple(range(1, 62))
    assert stitched.consumed_right_tokens == (16, 16)
    assert stitched.origins[0] == (0, 0)
    assert stitched.origins[31] == (1, 16)
    assert stitched.origins[-1] == (2, 30)


def test_constrained_stitch_handles_overlap_deletion_without_ground_truth() -> None:
    left = _candidate(range(1, 32), 1.0, "left")
    right_ids = list(range(16, 47))
    del right_ids[5]
    right = _candidate(right_ids, 1.0, "right-deletion")

    stitched = stitch_candidate_ids_constrained(left, right)

    assert stitched.aligned_left_tokens == 16
    assert stitched.consumed_right_tokens == 15
    assert stitched.content_ids == tuple(range(1, 47))


def test_constrained_stitch_handles_overlap_insertion_without_ground_truth() -> None:
    left = _candidate(range(1, 32), 1.0, "left")
    right_ids = list(range(16, 47))
    right_ids.insert(5, 999)
    right = _candidate(right_ids, 1.0, "right-insertion")

    stitched = stitch_candidate_ids_constrained(left, right)

    assert stitched.aligned_left_tokens == 16
    assert stitched.consumed_right_tokens == 17
    assert stitched.content_ids == tuple(range(1, 47))


def test_constrained_stitch_uses_nominal_policy_when_boundaries_disagree() -> None:
    left = _candidate(range(1, 32), 1.0, "left")
    right = _candidate(range(101, 132), 1.0, "unrelated-right")

    legacy = stitch_candidate_ids(left, right)
    constrained = stitch_candidate_ids_constrained(left, right)

    assert legacy.consumed_right_tokens == 0
    assert constrained.aligned_left_tokens == 16
    assert constrained.consumed_right_tokens == 16
    assert len(constrained.content_ids) == 46


def test_constrained_sequence_preserves_provenance() -> None:
    candidates = [
        _candidate(range(1, 32), 1.0, "w0"),
        _candidate(range(16, 47), 1.0, "w1"),
        _candidate(range(31, 62), 1.0, "w2"),
    ]

    stitched = stitch_candidate_sequence_constrained(candidates)

    assert stitched.content_ids == tuple(range(1, 62))
    assert stitched.consumed_right_tokens == (16, 16)
    assert stitched.aligned_left_tokens == (16, 16)
    assert stitched.origins[-1] == (2, 30)


def test_fidelity_constrained_assembly_uses_right_overlap_without_truth() -> None:
    candidates = [
        _candidate(range(1, 32), 0.8, "w0"),
        _candidate(range(16, 47), 0.9, "w1"),
        _candidate(range(31, 62), 0.7, "w2"),
    ]
    left = list(candidates[0].content_ids)
    left[20] = 999
    candidates[0] = _candidate(left, 0.8, "w0-error")

    stitched = stitch_candidate_sequence_fidelity_constrained(candidates)

    assert stitched.content_ids == tuple(range(1, 62))
    assert stitched.boundary_owner_views == (1, 1)
    assert stitched.inter_boundary_duplicate_widths == (1,)
    assert stitched.inter_boundary_gap_widths == (0,)


def test_fidelity_constrained_assembly_preserves_lower_index_on_tie() -> None:
    left = _candidate(range(1, 32), 0.9, "left")
    right_ids = list(range(16, 47))
    right_ids[5] = 999
    right = _candidate(right_ids, 0.9, "right")

    stitched = stitch_candidate_sequence_fidelity_constrained([left, right])

    assert stitched.boundary_owner_views == (0,)
    assert 999 not in stitched.content_ids


def test_fidelity_constrained_assembly_keeps_only_real_triple_endpoint_fixed() -> None:
    left_ids = list(range(1, 32))
    right_ids = list(range(16, 47))
    left_ids[15] = 998
    left_ids[30] = 997
    stitched = stitch_candidate_sequence_fidelity_constrained(
        [_candidate(left_ids, 0.8, "left"), _candidate(right_ids, 0.9, "right")]
    )

    # K=2 has no multiplicity-three endpoint, so the higher-fidelity right view
    # owns both ends of the overlap.
    assert stitched.content_ids == tuple(range(1, 47))


def test_alignment_trace_separates_truth_errors_from_prediction_insertions() -> None:
    alignment = align_token_ids((1, 9, 2, 4), (1, 2, 3))

    assert alignment.distance == 2
    assert alignment.exact_matches == ((0, 0), (2, 1))
    assert alignment.prediction_insertions == (1,)
    assert alignment.substitutions == ((3, 2),)
    assert alignment.reference_deletions == ()


def test_per_view_recovery_and_oracle_stay_inside_frozen_pool() -> None:
    reference = tuple(range(1, 47))
    left_truth = tuple(range(1, 32))
    right_truth = tuple(range(16, 47))
    left_one_error = left_truth[:-1] + (999,)
    right_one_error = (998,) + right_truth[1:]
    row = evaluate_candidate_pair(
        [
            [
                _candidate(left_one_error, 0.99, "left-top"),
                _candidate(left_truth, 0.98, "left-oracle"),
            ],
            [
                _candidate(right_one_error, 0.99, "right-top"),
                _candidate(right_truth, 0.98, "right-oracle"),
            ],
        ],
        reference_ids=reference,
        shared_ids=tuple(range(16, 32)),
    )
    assert row["independent_window_token_recovery"] == [30 / 31, 30 / 31]
    assert row["independent_window_exact"] == [False, False]
    assert row["per_view_oracle_window_token_recovery"] == [1.0, 1.0]
    assert row["per_view_oracle_window_exact"] == [True, True]
    assert window_reconstruction_score(left_truth, left_one_error) == 30 / 31
    assert row["overlap_both_correct_count"] == 14
    assert row["overlap_both_wrong_count"] == 0
    assert row["overlap_left_only_correct_count"] == 1
    assert row["overlap_right_only_correct_count"] == 1
    assert row["recoverable_overlap_error"] == 1.0

    summary = summarize_diagnostics([row], bootstrap_iterations=10)
    assert summary["independent_window_exact_fraction"] == 0.0
    assert summary["per_view_oracle_window_exact_fraction"] == 1.0
    assert summary["recoverable_overlap_error"]["pooled_fraction"] == 1.0


@pytest.mark.parametrize(
    ("intervals", "signal_fraction", "expected"),
    [
        ({"p_ind": [0.91, 0.95], "p_oracle": [0.95, 1.0], "h_pair": [0.0, 0.04], "delta_c": [0.0, 0.1]}, 1.0, "INDEPENDENT_SATURATED"),
        ({"p_ind": [0.30, 0.50], "p_oracle": [0.40, 0.59], "h_pair": [0.01, 0.03], "delta_c": [0.0, 0.1]}, 1.0, "POOL_LIMITED_STOP"),
        ({"p_ind": [0.60, 0.70], "p_oracle": [0.75, 0.85], "h_pair": [0.08, 0.15], "delta_c": [-0.02, 0.05]}, 0.5, "HEADROOM_BUT_NO_SIGNAL"),
        ({"p_ind": [0.60, 0.70], "p_oracle": [0.75, 0.85], "h_pair": [0.08, 0.15], "delta_c": [0.01, 0.08]}, 0.8, "JOINT_SIGNAL_GO"),
    ],
)
def test_four_gate_states(intervals, signal_fraction, expected) -> None:
    summary = {
        "bootstrap_95": intervals,
        "delta_c_positive_fraction_when_headroom": signal_fraction,
    }
    assert classify_diagnostic_gate(summary)["state"] == expected
