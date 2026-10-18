from __future__ import annotations

from overlap_poc.stage5c_fidelity_selector import _choose_view_index, _gate


def test_selector_uses_only_fidelity_and_lower_index_tie_break() -> None:
    values = [
        {"view_index": 2, "correct": True, "prediction": 9},
        {"view_index": 1, "correct": False, "prediction": 8},
    ]
    assert _choose_view_index(values, {1: 0.8, 2: 0.9}, tie_tolerance=1e-12) == 2
    assert _choose_view_index(values, {1: 0.9, 2: 0.9}, tie_tolerance=1e-12) == 1


def test_gate_pass_requires_positive_primary_and_isolated_but_not_controls() -> None:
    bootstrap = {
        "mean_delta": {"selector_minus_nooverlap": 0.01},
        "ci95": {
            "selector_minus_nooverlap": [0.001, 0.02],
            "selector_minus_fixed_owner": [0.001, 0.02],
            "shuffled_minus_fixed_owner": [-0.01, 0.01],
            "nonadjacent_minus_fixed_owner": [-0.01, 0.01],
        },
    }
    result = _gate({}, bootstrap, {"uplift_over_random": 0.1})
    assert result["state"] == "PASS"
    assert result["new_attack_structure_authorized"] is False


def test_gate_fails_when_shuffled_control_is_positive() -> None:
    bootstrap = {
        "mean_delta": {"selector_minus_nooverlap": 0.01},
        "ci95": {
            "selector_minus_nooverlap": [0.001, 0.02],
            "selector_minus_fixed_owner": [0.001, 0.02],
            "shuffled_minus_fixed_owner": [0.001, 0.01],
            "nonadjacent_minus_fixed_owner": [-0.01, 0.01],
        },
    }
    assert _gate({}, bootstrap, {"uplift_over_random": 0.1})["state"] == "FAIL"

