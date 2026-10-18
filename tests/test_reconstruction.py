import pytest

from overlap_poc.reconstruction import (
    Candidate,
    Vec2TextBackend,
    merge_reconstructions,
    overlap_consistency,
    select_baseline,
    select_joint,
)


def candidate(text: str, score: float) -> Candidate:
    return Candidate(text=text, local_score=score, correction_steps=0)


def test_joint_can_trade_small_local_loss_for_overlap_consistency() -> None:
    candidate_sets = [
        [candidate("alpha shared phrase", 0.90), candidate("alpha private words", 0.91)],
        [candidate("shared phrase omega", 0.90), candidate("other tokens omega", 0.91)],
    ]
    assert [item.text for item in select_baseline(candidate_sets)] == [
        "alpha private words",
        "other tokens omega",
    ]
    assert [item.text for item in select_joint(candidate_sets, [2 / 3], 1.0)] == [
        "alpha shared phrase",
        "shared phrase omega",
    ]


def test_zero_overlap_joint_equals_independent_baseline() -> None:
    candidate_sets = [
        [candidate("a", 0.1), candidate("b", 0.2)],
        [candidate("c", 0.4), candidate("d", 0.3)],
    ]
    assert select_joint(candidate_sets, [0.0], 10.0) == select_baseline(candidate_sets)


def test_overlap_score_and_merge_use_expected_overlap() -> None:
    selected = [candidate("one two shared words", 1.0), candidate("shared words five", 1.0)]
    assert overlap_consistency(selected[0].text, selected[1].text, 0.5) == 1.0
    assert merge_reconstructions(selected, [0.5]) == "one two shared words five"


def test_alignment_scales_with_different_candidate_lengths() -> None:
    short = "intro shared phrase"
    long = "shared phrase followed by several reconstructed words"
    assert overlap_consistency(short, long, 0.5) > 0.5
    assert merge_reconstructions(
        [candidate(short, 1.0), candidate(long, 1.0)], [0.5]
    ) == "intro shared phrase followed by several reconstructed words"


def test_empty_joint_input_and_invalid_candidate_sets() -> None:
    assert select_joint([], [], 1.0) == []
    try:
        select_baseline([[]])
    except ValueError as exc:
        assert "at least one candidate" in str(exc)
    else:
        raise AssertionError("an empty per-chunk candidate set must fail")


def test_ties_are_deterministic() -> None:
    first = candidate("first", 0.5)
    second = candidate("second", 0.5)
    assert select_baseline([[first, second]]) == [first]
    assert select_joint([[first, second]], [], 2.0) == [first]


def test_explicit_cuda_request_never_falls_back_to_cpu(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        Vec2TextBackend(device="cuda")
