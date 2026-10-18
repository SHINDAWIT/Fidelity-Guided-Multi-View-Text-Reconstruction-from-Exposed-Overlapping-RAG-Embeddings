from overlap_poc.stage1_strong_saturation import classify_stage1_saturation


def _summary(p_ind_interval, headroom_interval):
    return {
        "bootstrap_95": {
            "p_ind": p_ind_interval,
            "p_oracle": [0.0, 1.0],
            "h_pair": headroom_interval,
            "delta_c": [0.0, 0.0],
        }
    }


def test_stage1_stops_posthoc_when_headroom_is_small_and_reports_saturation() -> None:
    stopped = classify_stage1_saturation(
        _summary([0.92, 0.98], [0.0, 0.03])
    )
    assert stopped["state"] == "K2_STRONG_POSTHOC_STOP"
    assert stopped["next_step"] == "PIVOT_TO_STAGE2_COMPUTE_BUDGET_DIAGNOSTIC"

    low_quality_but_no_posthoc_headroom = classify_stage1_saturation(
        _summary([0.85, 0.98], [0.0, 0.03])
    )
    assert low_quality_but_no_posthoc_headroom["state"] == "K2_STRONG_POSTHOC_STOP"
    assert not low_quality_but_no_posthoc_headroom["observed"][
        "absolute_saturation_established"
    ]

    meaningful_headroom = classify_stage1_saturation(
        _summary([0.92, 0.98], [0.01, 0.08])
    )
    assert meaningful_headroom["state"] == "K2_STRONG_AMBIGUITY_REMAINS"
