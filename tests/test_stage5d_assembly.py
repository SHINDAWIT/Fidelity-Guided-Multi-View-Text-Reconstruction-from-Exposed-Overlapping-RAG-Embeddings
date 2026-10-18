from __future__ import annotations

from overlap_poc.reconstruction import Candidate
from overlap_poc.stage5d_assembly import fixed_coordinate_fidelity_assembly, _gate


def _candidate(ids, score):
    return Candidate(
        text="synthetic",
        local_score=score,
        correction_steps=50,
        sequence_beam_width=8,
        content_ids=tuple(ids),
    )


def test_fixed_coordinate_rule_selects_fidelity_only_for_multiplicity_two() -> None:
    w0 = list(range(1, 32))
    w1 = list(range(16, 47))
    w2 = list(range(31, 62))
    w0[20] = 900
    w0[30] = 777

    assembled = fixed_coordinate_fidelity_assembly(
        [_candidate(w0, 0.8), _candidate(w1, 0.9), _candidate(w2, 0.95)]
    )

    assert assembled[20] == 21  # multiplicity two: higher-fidelity W1
    assert assembled[30] == 777  # multiplicity three: frozen lower-index W0
    assert len(assembled) == 61


def test_stage5d_gate_is_frozen_to_primary_then_assembly_transfer() -> None:
    assert _gate(
        {
            "ci95": {
                "new_minus_nooverlap": [0.001, 0.02],
                "new_minus_fixed_coordinate": [0.01, 0.03],
            }
        }
    )["state"] == "PASS"
    assert _gate(
        {
            "ci95": {
                "new_minus_nooverlap": [-0.01, 0.01],
                "new_minus_fixed_coordinate": [0.001, 0.02],
            }
        }
    )["state"] == "PARTIAL"
    assert _gate(
        {
            "ci95": {
                "new_minus_nooverlap": [-0.02, 0.01],
                "new_minus_fixed_coordinate": [-0.01, 0.02],
            }
        }
    )["state"] == "FAIL"

