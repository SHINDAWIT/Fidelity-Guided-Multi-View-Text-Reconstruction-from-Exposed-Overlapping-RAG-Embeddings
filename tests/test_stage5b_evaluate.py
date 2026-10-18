from __future__ import annotations

from overlap_poc.reconstruction import Candidate
from overlap_poc.stage5b_evaluate import (
    _bootstrap,
    _classify_gate,
    nooverlap_single_view_source_row,
)


def _candidate(ids: list[int]) -> Candidate:
    return Candidate(
        text="x",
        local_score=1.0,
        correction_steps=50,
        sequence_beam_width=8,
        generation_id="test",
        content_ids=tuple(ids),
    )


def test_nooverlap_source_metric_aligns_each_window_without_assembly() -> None:
    document = {
        "reference_ids": [1, 2, 3, 4],
        "windows": [
            {"source_ids": [1, 2], "relative_token_start": 0, "window_index": 0},
            {"source_ids": [3, 4], "relative_token_start": 2, "window_index": 1},
        ],
    }
    row = nooverlap_single_view_source_row(
        document, [_candidate([1, 2]), _candidate([9, 4])]
    )
    assert row["correct_source_token_count"] == 3
    assert row["source_token_recovery"] == 0.75
    assert row["source_position_document_exact"] is False


def test_bootstrap_resamples_frozen_documents() -> None:
    rows = [
        {"h1_delta": 0.1, "overlap_diagnostic": {"candidate_aligned_boundaries": []}},
        {"h1_delta": 0.3, "overlap_diagnostic": {"candidate_aligned_boundaries": []}},
    ]
    result = _bootstrap(rows, iterations=100, seed=9)
    assert result["resampling_unit"] == "frozen document"
    assert result["h1_mean_delta"] == 0.2
    assert result["h2_ci95"] == [None, None]


def test_gate_requires_both_general_h1_and_h2() -> None:
    def group(lower: float) -> dict:
        return {"bootstrap": {"h1_ci95": [lower, 0.1], "h2_ci95": [0.01, 0.02]}}

    result = _classify_gate(
        overall=group(0.01),
        nonadjusted=group(0.01),
        adjusted=group(0.01),
        h2={"event_count": 2},
    )
    assert result["state"] == "GENERAL_OVERLAP_SIGNAL"
    assert result["method_implementation_authorized"] is False
