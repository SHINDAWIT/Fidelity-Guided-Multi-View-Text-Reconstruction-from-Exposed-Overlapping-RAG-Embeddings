"""Oracle-only diagnostics for deciding whether a Joint selector is warranted.

This module deliberately contains no Joint selector and no lambda. Ground truth
enters only :func:`evaluate_candidate_pair`; the candidate consistency helper,
Independent selector, and stitcher depend only on frozen candidates and public
window policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .reconstruction import Candidate


_EPSILON = 1e-12


@dataclass(frozen=True)
class StitchedIDs:
    content_ids: tuple[int, ...]
    consumed_right_tokens: int
    alignment_score: float
    aligned_left_tokens: int = 0


@dataclass(frozen=True)
class StitchedSequenceIDs:
    """A deterministic multi-window stitch with output-token provenance."""

    content_ids: tuple[int, ...]
    origins: tuple[tuple[int, int], ...]
    consumed_right_tokens: tuple[int, ...]
    alignment_scores: tuple[float, ...]
    aligned_left_tokens: tuple[int, ...] = ()


@dataclass(frozen=True)
class FidelityStitchedSequenceIDs:
    """Candidate-only assembly whose overlap owner is frozen by fidelity."""

    content_ids: tuple[int, ...]
    boundary_left_widths: tuple[int, ...]
    boundary_right_widths: tuple[int, ...]
    boundary_alignment_scores: tuple[float, ...]
    boundary_owner_views: tuple[int, ...]
    inter_boundary_gap_widths: tuple[int, ...]
    inter_boundary_duplicate_widths: tuple[int, ...]


@dataclass(frozen=True)
class TokenAlignment:
    """Deterministic Levenshtein trace from prediction to reference IDs."""

    distance: int
    exact_matches: tuple[tuple[int, int], ...]
    substitutions: tuple[tuple[int, int], ...]
    prediction_insertions: tuple[int, ...]
    reference_deletions: tuple[int, ...]


def levenshtein_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Unit-cost insertion/deletion/substitution edit distance."""

    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def fixed_length_similarity(
    left: Sequence[int], right: Sequence[int], *, expected_tokens: int = 16
) -> float:
    """Edit similarity with an explicit penalty for two equally short inputs."""

    if expected_tokens < 1:
        raise ValueError("expected_tokens must be positive")
    left_boundary = tuple(left[-expected_tokens:])
    right_boundary = tuple(right[:expected_tokens])
    shortfall = expected_tokens - max(len(left_boundary), len(right_boundary))
    distance = levenshtein_distance(left_boundary, right_boundary) + shortfall
    return max(0.0, 1.0 - distance / expected_tokens)


def candidate_consistency(
    left: Candidate, right: Candidate, *, overlap_tokens: int = 16
) -> float:
    """Candidate-only fixed suffix/prefix consistency; no truth is accepted."""

    return fixed_length_similarity(
        left.content_ids, right.content_ids, expected_tokens=overlap_tokens
    )


def select_independent_indices(
    candidate_pools: Sequence[Sequence[Candidate]],
) -> tuple[int, ...]:
    """Select each view by the same frozen embedding-fidelity score."""

    selected: list[int] = []
    for candidates in candidate_pools:
        if not candidates:
            raise ValueError("every view must have a non-empty candidate pool")
        selected.append(
            max(
                range(len(candidates)),
                key=lambda index: (candidates[index].local_score, -index),
            )
        )
    return tuple(selected)


def stitch_candidate_ids(
    left: Candidate,
    right: Candidate,
    *,
    overlap_tokens: int = 16,
    minimum_alignment: float = 0.5,
) -> StitchedIDs:
    """Deterministically stitch candidates using candidate IDs only."""

    left_ids = tuple(left.content_ids)
    right_ids = tuple(right.content_ids)
    limit = min(overlap_tokens, len(left_ids), len(right_ids))
    best_k = 0
    best_score = 0.0
    for width in range(1, limit + 1):
        score = (
            width
            - levenshtein_distance(left_ids[-width:], right_ids[:width])
        ) / overlap_tokens
        if score > best_score + _EPSILON or (
            abs(score - best_score) <= _EPSILON and width > best_k
        ):
            best_score = score
            best_k = width
    consumed = best_k if best_score + _EPSILON >= minimum_alignment else 0
    return StitchedIDs(
        content_ids=left_ids + right_ids[consumed:],
        consumed_right_tokens=consumed,
        alignment_score=best_score,
        aligned_left_tokens=best_k,
    )


def stitch_candidate_sequence(
    candidates: Sequence[Candidate],
    *,
    overlap_tokens: int = 16,
    minimum_alignment: float = 0.5,
) -> StitchedSequenceIDs:
    """Apply the existing adjacent deterministic stitcher to an ordered chain."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    content_ids = list(candidates[0].content_ids)
    origins = [(0, index) for index in range(len(candidates[0].content_ids))]
    consumed: list[int] = []
    scores: list[float] = []
    aligned_left: list[int] = []
    for view_index, (left, right) in enumerate(
        zip(candidates, candidates[1:]), start=1
    ):
        boundary = stitch_candidate_ids(
            left,
            right,
            overlap_tokens=overlap_tokens,
            minimum_alignment=minimum_alignment,
        )
        drop = boundary.consumed_right_tokens
        content_ids.extend(right.content_ids[drop:])
        origins.extend(
            (view_index, prediction_index)
            for prediction_index in range(drop, len(right.content_ids))
        )
        consumed.append(drop)
        scores.append(boundary.alignment_score)
        aligned_left.append(boundary.aligned_left_tokens)
    return StitchedSequenceIDs(
        content_ids=tuple(content_ids),
        origins=tuple(origins),
        consumed_right_tokens=tuple(consumed),
        alignment_scores=tuple(scores),
        aligned_left_tokens=tuple(aligned_left),
    )


def stitch_candidate_ids_constrained(
    left: Candidate,
    right: Candidate,
    *,
    overlap_tokens: int = 16,
    max_boundary_shift: int = 2,
    deviation_cost: float = 0.5,
) -> StitchedIDs:
    """Stitch one boundary with candidate-only constrained edit alignment.

    Suffix and prefix widths are searched only in a narrow band around the
    public nominal overlap.  A fixed width-deviation prior prevents shortening
    the boundary merely to avoid substitutions.  Unlike the legacy stitcher,
    this policy always consumes the selected right prefix instead of duplicating
    a whole window when two imperfect reconstructions disagree.
    """

    if overlap_tokens < 1:
        raise ValueError("overlap_tokens must be positive")
    if max_boundary_shift < 0 or max_boundary_shift >= overlap_tokens:
        raise ValueError("max_boundary_shift must be in [0, overlap_tokens)")
    if deviation_cost < 0:
        raise ValueError("deviation_cost must be non-negative")
    left_ids = tuple(left.content_ids)
    right_ids = tuple(right.content_ids)
    if not left_ids or not right_ids:
        raise ValueError("constrained stitching requires non-empty candidates")

    minimum_width = max(1, overlap_tokens - max_boundary_shift)
    maximum_width = overlap_tokens + max_boundary_shift
    left_widths = range(minimum_width, min(len(left_ids), maximum_width) + 1)
    right_widths = range(minimum_width, min(len(right_ids), maximum_width) + 1)
    if not left_widths or not right_widths:
        raise ValueError("candidate is too short for the constrained overlap band")

    best: tuple[tuple[float, int, int, int, int, int], int, int, int] | None = None
    for left_width in left_widths:
        for right_width in right_widths:
            distance = levenshtein_distance(
                left_ids[-left_width:], right_ids[:right_width]
            )
            total_deviation = abs(left_width - overlap_tokens) + abs(
                right_width - overlap_tokens
            )
            objective = distance + deviation_cost * total_deviation
            key = (
                objective,
                total_deviation,
                distance,
                abs(right_width - overlap_tokens),
                abs(left_width - overlap_tokens),
                -right_width,
            )
            if best is None or key < best[0]:
                best = (key, left_width, right_width, distance)
    if best is None:  # pragma: no cover - validated non-empty search ranges
        raise AssertionError("constrained boundary search produced no alignment")

    _, left_width, right_width, distance = best
    denominator = max(overlap_tokens, left_width, right_width)
    score = max(0.0, 1.0 - distance / denominator)
    return StitchedIDs(
        content_ids=left_ids + right_ids[right_width:],
        consumed_right_tokens=right_width,
        alignment_score=score,
        aligned_left_tokens=left_width,
    )


def _aligned_pairs(
    left: Sequence[int], right: Sequence[int]
) -> tuple[tuple[int | None, int | None], ...]:
    """Return a deterministic Levenshtein alignment without a reference string."""

    rows = len(left) + 1
    columns = len(right) + 1
    table = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        table[row][0] = row
    for column in range(columns):
        table[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            table[row][column] = min(
                table[row - 1][column] + 1,
                table[row][column - 1] + 1,
                table[row - 1][column - 1]
                + int(left[row - 1] != right[column - 1]),
            )

    row = len(left)
    column = len(right)
    aligned: list[tuple[int | None, int | None]] = []
    while row or column:
        if (
            row
            and column
            and left[row - 1] == right[column - 1]
            and table[row][column] == table[row - 1][column - 1]
        ):
            aligned.append((left[row - 1], right[column - 1]))
            row -= 1
            column -= 1
        elif (
            row
            and column
            and table[row][column] == table[row - 1][column - 1] + 1
        ):
            aligned.append((left[row - 1], right[column - 1]))
            row -= 1
            column -= 1
        elif row and table[row][column] == table[row - 1][column] + 1:
            aligned.append((left[row - 1], None))
            row -= 1
        elif column and table[row][column] == table[row][column - 1] + 1:
            aligned.append((None, right[column - 1]))
            column -= 1
        else:  # pragma: no cover - dynamic-programming invariant
            raise AssertionError("no deterministic candidate alignment exists")
    return tuple(reversed(aligned))


def stitch_candidate_sequence_fidelity_constrained(
    candidates: Sequence[Candidate],
    *,
    overlap_tokens: int = 16,
    max_boundary_shift: int = 2,
    deviation_cost: float = 0.5,
    tie_tolerance: float = 1e-12,
) -> FidelityStitchedSequenceIDs:
    """Assemble ordered reconstructions using geometry and frozen fidelity only.

    Every adjacent reconstructed suffix/prefix is aligned in the same narrow
    band as :func:`stitch_candidate_ids_constrained`.  The higher-fidelity
    reconstruction owns the aligned overlap.  The two overlap endpoints retain
    the lower-index (left) view so the Stage 5C multiplicity-three rule is not
    changed.  Interior-window gaps or duplicate coverage are resolved solely in
    candidate coordinates; no source tokens, source positions, or GT alignment
    are accepted by this function.
    """

    if len(candidates) < 2:
        raise ValueError("fidelity assembly requires at least two candidates")

    boundaries: list[StitchedIDs] = []
    segments: list[list[int]] = []
    owners: list[int] = []
    for boundary_index, (left, right) in enumerate(zip(candidates, candidates[1:])):
        boundary = stitch_candidate_ids_constrained(
            left,
            right,
            overlap_tokens=overlap_tokens,
            max_boundary_shift=max_boundary_shift,
            deviation_cost=deviation_cost,
        )
        boundaries.append(boundary)
        left_segment = tuple(left.content_ids[-boundary.aligned_left_tokens :])
        right_segment = tuple(right.content_ids[: boundary.consumed_right_tokens])
        aligned = _aligned_pairs(left_segment, right_segment)
        right_wins = float(right.local_score) > float(left.local_score) + tie_tolerance
        owner = boundary_index + 1 if right_wins else boundary_index
        owners.append(owner)

        merged: list[int] = []
        for column_index, (left_token, right_token) in enumerate(aligned):
            # With 31/15 geometry, an internal boundary's first column is also
            # observed by the previous view and its last column by the next.
            # Only those actual multiplicity-three endpoints retain the frozen
            # lower-index rule; the two document-edge endpoints remain eligible
            # for the ordinary multiplicity-two fidelity rule.
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
        segments.append(merged)

    first = candidates[0]
    assembled = list(first.content_ids[: -boundaries[0].aligned_left_tokens])
    assembled.extend(segments[0])
    gaps: list[int] = []
    duplicates: list[int] = []
    for view_index in range(1, len(candidates) - 1):
        candidate_ids = tuple(candidates[view_index].content_ids)
        prefix_width = boundaries[view_index - 1].consumed_right_tokens
        suffix_width = boundaries[view_index].aligned_left_tokens
        delta = len(candidate_ids) - prefix_width - suffix_width
        gap_width = max(0, delta)
        duplicate_width = max(0, -delta)
        gaps.append(gap_width)
        duplicates.append(duplicate_width)
        if gap_width:
            assembled.extend(candidate_ids[prefix_width : prefix_width + gap_width])
        next_segment = segments[view_index]
        assembled.extend(next_segment[min(duplicate_width, len(next_segment)) :])

    final = candidates[-1]
    assembled.extend(final.content_ids[boundaries[-1].consumed_right_tokens :])
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


def stitch_candidate_sequence_constrained(
    candidates: Sequence[Candidate],
    *,
    overlap_tokens: int = 16,
    max_boundary_shift: int = 2,
    deviation_cost: float = 0.5,
) -> StitchedSequenceIDs:
    """Apply the constrained candidate-only stitcher to an ordered chain."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    content_ids = list(candidates[0].content_ids)
    origins = [(0, index) for index in range(len(candidates[0].content_ids))]
    consumed: list[int] = []
    scores: list[float] = []
    aligned_left: list[int] = []
    for view_index, (left, right) in enumerate(
        zip(candidates, candidates[1:]), start=1
    ):
        boundary = stitch_candidate_ids_constrained(
            left,
            right,
            overlap_tokens=overlap_tokens,
            max_boundary_shift=max_boundary_shift,
            deviation_cost=deviation_cost,
        )
        drop = boundary.consumed_right_tokens
        content_ids.extend(right.content_ids[drop:])
        origins.extend(
            (view_index, prediction_index)
            for prediction_index in range(drop, len(right.content_ids))
        )
        consumed.append(drop)
        scores.append(boundary.alignment_score)
        aligned_left.append(boundary.aligned_left_tokens)
    return StitchedSequenceIDs(
        content_ids=tuple(content_ids),
        origins=tuple(origins),
        consumed_right_tokens=tuple(consumed),
        alignment_scores=tuple(scores),
        aligned_left_tokens=tuple(aligned_left),
    )


def ordered_reconstruction_score(
    reference_ids: Sequence[int], prediction_ids: Sequence[int]
) -> float:
    """M46: clipped ordered token-ID edit similarity with fixed denominator."""

    if len(reference_ids) != 46:
        raise ValueError("the K=2 reference must contain exactly 46 content IDs")
    distance = levenshtein_distance(reference_ids, prediction_ids)
    return max(0.0, 1.0 - distance / 46)


def window_reconstruction_score(
    reference_ids: Sequence[int], prediction_ids: Sequence[int]
) -> float:
    """Ordered token-ID edit recovery for one 31-content-ID observation."""

    if len(reference_ids) != 31:
        raise ValueError("a primary window reference must contain 31 content IDs")
    distance = levenshtein_distance(reference_ids, prediction_ids)
    return max(0.0, 1.0 - distance / 31)


def _shared_recovery(candidate_boundary: Sequence[int], truth: Sequence[int]) -> float:
    if len(truth) != 16:
        raise ValueError("the shared truth must contain exactly 16 content IDs")
    distance = levenshtein_distance(candidate_boundary, truth)
    return max(0.0, 1.0 - distance / 16)


def align_token_ids(
    prediction: Sequence[int], reference: Sequence[int]
) -> TokenAlignment:
    """Return a deterministic prediction/reference Levenshtein alignment.

    Backtrace precedence is exact diagonal, substitution diagonal, deletion from
    the prediction, then insertion of a missing reference token.
    """

    rows = len(prediction) + 1
    columns = len(reference) + 1
    table = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        table[row][0] = row
    for column in range(columns):
        table[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            table[row][column] = min(
                table[row - 1][column] + 1,
                table[row][column - 1] + 1,
                table[row - 1][column - 1]
                + int(prediction[row - 1] != reference[column - 1]),
            )

    row = len(prediction)
    column = len(reference)
    exact_matches: list[tuple[int, int]] = []
    substitutions: list[tuple[int, int]] = []
    prediction_insertions: list[int] = []
    reference_deletions: list[int] = []
    while row or column:
        if (
            row
            and column
            and prediction[row - 1] == reference[column - 1]
            and table[row][column] == table[row - 1][column - 1]
        ):
            exact_matches.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif (
            row
            and column
            and table[row][column] == table[row - 1][column - 1] + 1
        ):
            substitutions.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif row and table[row][column] == table[row - 1][column] + 1:
            prediction_insertions.append(row - 1)
            row -= 1
        elif column and table[row][column] == table[row][column - 1] + 1:
            reference_deletions.append(column - 1)
            column -= 1
        else:  # pragma: no cover - dynamic-programming invariant
            raise AssertionError("no deterministic Levenshtein backtrace exists")
    return TokenAlignment(
        distance=table[-1][-1],
        exact_matches=tuple(reversed(exact_matches)),
        substitutions=tuple(reversed(substitutions)),
        prediction_insertions=tuple(reversed(prediction_insertions)),
        reference_deletions=tuple(reversed(reference_deletions)),
    )


def _matched_truth_positions(
    candidate: Sequence[int], truth: Sequence[int]
) -> frozenset[int]:
    """Return exact-match truth positions under the shared deterministic trace."""

    return frozenset(
        reference_index
        for _, reference_index in align_token_ids(candidate, truth).exact_matches
    )


def _phi(errors_left: Sequence[bool], errors_right: Sequence[bool]) -> float | None:
    n11 = sum(left and right for left, right in zip(errors_left, errors_right))
    n10 = sum(left and not right for left, right in zip(errors_left, errors_right))
    n01 = sum(not left and right for left, right in zip(errors_left, errors_right))
    n00 = sum(not left and not right for left, right in zip(errors_left, errors_right))
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / denominator if denominator else None


def evaluate_candidate_pair(
    candidate_pools: Sequence[Sequence[Candidate]],
    *,
    reference_ids: Sequence[int],
    shared_ids: Sequence[int],
    overlap_tokens: int = 16,
    minimum_alignment: float = 0.5,
) -> dict[str, Any]:
    """Compute Independent, Cartesian oracle, and overlap-signal diagnostics."""

    if len(candidate_pools) != 2:
        raise ValueError("the primary diagnostic requires exactly two views")
    left_pool, right_pool = candidate_pools
    independent_indices = select_independent_indices(candidate_pools)

    def result(left_index: int, right_index: int) -> dict[str, Any]:
        left = left_pool[left_index]
        right = right_pool[right_index]
        stitched = stitch_candidate_ids(
            left,
            right,
            overlap_tokens=overlap_tokens,
            minimum_alignment=minimum_alignment,
        )
        return {
            "indices": (left_index, right_index),
            "score": ordered_reconstruction_score(reference_ids, stitched.content_ids),
            "fidelity_sum": left.local_score + right.local_score,
            "consistency": candidate_consistency(
                left, right, overlap_tokens=overlap_tokens
            ),
            "stitched_ids": stitched.content_ids,
            "consumed_right_tokens": stitched.consumed_right_tokens,
            "stitch_alignment_score": stitched.alignment_score,
        }

    independent = result(*independent_indices)
    oracle: dict[str, Any] | None = None
    for left_index, right_index in product(range(len(left_pool)), range(len(right_pool))):
        current = result(left_index, right_index)
        if oracle is None:
            oracle = current
            continue
        if current["score"] > oracle["score"] + _EPSILON:
            oracle = current
        elif abs(current["score"] - oracle["score"]) <= _EPSILON:
            if current["fidelity_sum"] > oracle["fidelity_sum"] + _EPSILON:
                oracle = current
            elif (
                abs(current["fidelity_sum"] - oracle["fidelity_sum"])
                <= _EPSILON
                and current["indices"] < oracle["indices"]
            ):
                oracle = current
    if oracle is None:  # pragma: no cover - guarded by independent selector
        raise AssertionError("oracle pool unexpectedly empty")

    truth = tuple(shared_ids)
    independent_left = left_pool[independent_indices[0]]
    independent_right = right_pool[independent_indices[1]]
    window_tokens = (len(reference_ids) + overlap_tokens) // 2
    if window_tokens != 31 or 2 * window_tokens - overlap_tokens != len(
        reference_ids
    ):
        raise ValueError("reference/overlap geometry does not describe two 31-ID windows")
    window_references = (
        tuple(reference_ids[:window_tokens]),
        tuple(reference_ids[-window_tokens:]),
    )
    independent_candidates = (independent_left, independent_right)
    independent_window_token_recovery = [
        window_reconstruction_score(reference, candidate.content_ids)
        for reference, candidate in zip(window_references, independent_candidates)
    ]
    independent_window_exact = [
        tuple(candidate.content_ids) == reference
        for reference, candidate in zip(window_references, independent_candidates)
    ]
    per_view_oracle_window_token_recovery = [
        max(
            window_reconstruction_score(reference, candidate.content_ids)
            for candidate in pool
        )
        for reference, pool in zip(window_references, candidate_pools)
    ]
    per_view_oracle_window_exact = [
        any(tuple(candidate.content_ids) == reference for candidate in pool)
        for reference, pool in zip(window_references, candidate_pools)
    ]
    left_boundary = independent_left.content_ids[-overlap_tokens:]
    right_boundary = independent_right.content_ids[:overlap_tokens]
    left_recovery = _shared_recovery(left_boundary, truth)
    right_recovery = _shared_recovery(right_boundary, truth)

    left_oracle_recovery = max(
        _shared_recovery(candidate.content_ids[-overlap_tokens:], truth)
        for candidate in left_pool
    )
    right_oracle_recovery = max(
        _shared_recovery(candidate.content_ids[:overlap_tokens], truth)
        for candidate in right_pool
    )

    left_matches = _matched_truth_positions(left_boundary, truth)
    right_matches = _matched_truth_positions(right_boundary, truth)
    left_errors = [index not in left_matches for index in range(overlap_tokens)]
    right_errors = [index not in right_matches for index in range(overlap_tokens)]
    both_correct = sum(
        not left_error and not right_error
        for left_error, right_error in zip(left_errors, right_errors)
    )
    both_wrong = sum(
        left_error and right_error
        for left_error, right_error in zip(left_errors, right_errors)
    )
    left_only_correct = sum(
        not left_error and right_error
        for left_error, right_error in zip(left_errors, right_errors)
    )
    right_only_correct = sum(
        left_error and not right_error
        for left_error, right_error in zip(left_errors, right_errors)
    )
    directed_error_count = (
        2 * both_wrong + left_only_correct + right_only_correct
    )
    recoverable_error_count = left_only_correct + right_only_correct
    error_union = {index for index in range(overlap_tokens) if left_errors[index] or right_errors[index]}
    error_intersection = {
        index for index in range(overlap_tokens) if left_errors[index] and right_errors[index]
    }
    correct_union = left_matches | right_matches
    best_single_correct = max(len(left_matches), len(right_matches))
    complementarity = (
        0.0
        if not error_union
        else 1.0 - len(error_intersection) / len(error_union)
    )

    p_ind = float(independent["score"])
    p_oracle = float(oracle["score"])
    c_ind = float(independent["consistency"])
    c_oracle = float(oracle["consistency"])
    return {
        "candidate_counts": [len(left_pool), len(right_pool)],
        "cartesian_pairs": len(left_pool) * len(right_pool),
        "independent_indices": list(independent["indices"]),
        "oracle_indices": list(oracle["indices"]),
        "p_ind": p_ind,
        "p_oracle": p_oracle,
        "h_pair": p_oracle - p_ind,
        "c_ind": c_ind,
        "c_oracle": c_oracle,
        "delta_c": c_oracle - c_ind,
        "independent_stitched_ids": list(independent["stitched_ids"]),
        "oracle_stitched_ids": list(oracle["stitched_ids"]),
        "independent_stitch_consumed": independent["consumed_right_tokens"],
        "oracle_stitch_consumed": oracle["consumed_right_tokens"],
        "independent_window_token_recovery": independent_window_token_recovery,
        "independent_window_exact": independent_window_exact,
        "per_view_oracle_window_token_recovery": (
            per_view_oracle_window_token_recovery
        ),
        "per_view_oracle_window_exact": per_view_oracle_window_exact,
        "independent_shared_recovery_left": left_recovery,
        "independent_shared_recovery_right": right_recovery,
        "independent_shared_recovery_mean": (left_recovery + right_recovery) / 2,
        "oracle_shared_recovery_left": left_oracle_recovery,
        "oracle_shared_recovery_right": right_oracle_recovery,
        "oracle_shared_recovery_mean": (
            left_oracle_recovery + right_oracle_recovery
        )
        / 2,
        "independent_predicted_overlap_agreement": c_ind,
        "error_complementarity": complementarity,
        "error_union_gain": (
            len(correct_union) - best_single_correct
        )
        / overlap_tokens,
        "error_phi": _phi(left_errors, right_errors),
        "error_saturated": not error_union,
        "overlap_both_correct_count": both_correct,
        "overlap_both_wrong_count": both_wrong,
        "overlap_left_only_correct_count": left_only_correct,
        "overlap_right_only_correct_count": right_only_correct,
        "overlap_directed_error_count": directed_error_count,
        "overlap_recoverable_error_count": recoverable_error_count,
        "recoverable_overlap_error": (
            recoverable_error_count / directed_error_count
            if directed_error_count
            else None
        ),
        "left_error_bits": [int(value) for value in left_errors],
        "right_error_bits": [int(value) for value in right_errors],
    }


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + end - 1) / 2
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    return {"mean": mean(values), "median": median(values)}


def paired_bootstrap_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260904,
    iterations: int = 10_000,
) -> dict[str, list[float]]:
    """Fixed paired percentile CIs used by the diagnostic gate."""

    if not rows or iterations < 1:
        raise ValueError("bootstrap requires rows and positive iterations")
    metric_names = ("p_ind", "p_oracle", "h_pair", "delta_c")
    matrix = np.asarray(
        [[float(row[name]) for name in metric_names] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("bootstrap metrics must be finite")
    indices = np.random.default_rng(seed).integers(
        0, len(rows), size=(iterations, len(rows))
    )
    replicate_means = matrix[indices].mean(axis=1)
    bounds = np.quantile(
        replicate_means, [0.025, 0.975], axis=0, method="linear"
    )
    if not np.isfinite(bounds).all():
        raise ValueError("bootstrap produced non-finite bounds")
    return {
        name: [float(bounds[0, index]), float(bounds[1, index])]
        for index, name in enumerate(metric_names)
    }


def summarize_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20260904,
    bootstrap_iterations: int = 10_000,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("diagnostic summary requires at least one row")
    metric_names = (
        "p_ind",
        "p_oracle",
        "h_pair",
        "c_ind",
        "c_oracle",
        "delta_c",
        "independent_shared_recovery_left",
        "independent_shared_recovery_right",
        "independent_shared_recovery_mean",
        "oracle_shared_recovery_left",
        "oracle_shared_recovery_right",
        "oracle_shared_recovery_mean",
        "error_complementarity",
        "error_union_gain",
    )
    summary = {
        name: _metric_summary([float(row[name]) for row in rows])
        for name in metric_names
    }
    h_values = [float(row["h_pair"]) for row in rows]
    delta_values = [float(row["delta_c"]) for row in rows]
    positive_headroom_delta = [
        delta
        for headroom, delta in zip(h_values, delta_values)
        if headroom > _EPSILON
    ]
    finite_phi = [
        float(row["error_phi"])
        for row in rows
        if row["error_phi"] is not None
    ]
    pooled_left_errors = [
        bool(value) for row in rows for value in row["left_error_bits"]
    ]
    pooled_right_errors = [
        bool(value) for row in rows for value in row["right_error_bits"]
    ]
    overlap_category_totals = {
        name: sum(int(row[name]) for row in rows)
        for name in (
            "overlap_both_correct_count",
            "overlap_both_wrong_count",
            "overlap_left_only_correct_count",
            "overlap_right_only_correct_count",
        )
    }
    overlap_positions = len(rows) * 16
    directed_errors = sum(int(row["overlap_directed_error_count"]) for row in rows)
    recoverable_errors = sum(
        int(row["overlap_recoverable_error_count"]) for row in rows
    )
    finite_recoverable = [
        float(row["recoverable_overlap_error"])
        for row in rows
        if row["recoverable_overlap_error"] is not None
    ]
    summary.update(
        {
            "sample_count": len(rows),
            "h_pair_positive_fraction": sum(
                value > _EPSILON for value in h_values
            )
            / len(rows),
            "delta_c_positive_fraction": sum(
                value > _EPSILON for value in delta_values
            )
            / len(rows),
            "delta_c_positive_fraction_when_headroom": (
                sum(value > _EPSILON for value in positive_headroom_delta)
                / len(positive_headroom_delta)
                if positive_headroom_delta
                else 0.0
            ),
            "h_pair_delta_c_pearson": _pearson(h_values, delta_values),
            "h_pair_delta_c_spearman": _pearson(
                _rank(h_values), _rank(delta_values)
            ),
            "mean_sample_error_phi": mean(finite_phi) if finite_phi else None,
            "finite_sample_error_phi_count": len(finite_phi),
            "pooled_error_phi": _phi(pooled_left_errors, pooled_right_errors),
            "overlap_error_categories": {
                **overlap_category_totals,
                **{
                    name.replace("_count", "_fraction"): value
                    / overlap_positions
                    for name, value in overlap_category_totals.items()
                },
                "total_positions": overlap_positions,
            },
            "recoverable_overlap_error": {
                "definition": (
                    "directed shared-span view errors whose same aligned truth "
                    "position is correct in the other view / all directed "
                    "shared-span view errors"
                ),
                "recoverable_directed_errors": recoverable_errors,
                "directed_errors": directed_errors,
                "pooled_fraction": (
                    recoverable_errors / directed_errors
                    if directed_errors
                    else None
                ),
                "mean_over_pairs_with_errors": (
                    mean(finite_recoverable) if finite_recoverable else None
                ),
                "pairs_with_errors": len(finite_recoverable),
            },
            "independent_window_token_recovery": _metric_summary(
                [
                    float(value)
                    for row in rows
                    for value in row["independent_window_token_recovery"]
                ]
            ),
            "independent_window_exact_fraction": mean(
                float(value)
                for row in rows
                for value in row["independent_window_exact"]
            ),
            "per_view_oracle_window_token_recovery": _metric_summary(
                [
                    float(value)
                    for row in rows
                    for value in row["per_view_oracle_window_token_recovery"]
                ]
            ),
            "per_view_oracle_window_exact_fraction": mean(
                float(value)
                for row in rows
                for value in row["per_view_oracle_window_exact"]
            ),
            "bootstrap_95": paired_bootstrap_intervals(
                rows,
                seed=bootstrap_seed,
                iterations=bootstrap_iterations,
            ),
        }
    )
    return summary


def classify_diagnostic_gate(
    summary: Mapping[str, Any],
    *,
    oracle_floor: float = 0.60,
    independent_saturation: float = 0.90,
    minimum_headroom: float = 2 / 46,
    minimum_signal_fraction: float = 0.60,
) -> dict[str, Any]:
    """Return one of the four pre-Joint gate states requested by the PoC."""

    intervals = summary["bootstrap_95"]
    p_ind_low, p_ind_high = intervals["p_ind"]
    p_oracle_low, p_oracle_high = intervals["p_oracle"]
    h_pair_low, h_pair_high = intervals["h_pair"]
    delta_c_low, _ = intervals["delta_c"]

    if p_ind_low >= independent_saturation:
        state = "INDEPENDENT_SATURATED"
        reason = "Independent lower 95% bound reaches the saturation threshold."
    elif p_oracle_high < oracle_floor:
        state = "POOL_LIMITED_STOP"
        reason = (
            "Even the oracle mean's upper 95% bound is below the frozen-pool floor."
        )
    elif h_pair_high < minimum_headroom:
        state = "POOL_LIMITED_STOP"
        reason = "The frozen pool's upper 95% headroom bound is below two tokens."
    elif p_oracle_low >= oracle_floor and h_pair_low >= minimum_headroom:
        signal_fraction = float(
            summary["delta_c_positive_fraction_when_headroom"]
        )
        if delta_c_low > 0.0 and signal_fraction >= minimum_signal_fraction:
            state = "JOINT_SIGNAL_GO"
            reason = (
                "Candidate headroom is established and oracle-improving pairs "
                "have a positive candidate-only consistency signal."
            )
        else:
            state = "HEADROOM_BUT_NO_SIGNAL"
            reason = (
                "Candidate headroom is established, but fixed overlap "
                "consistency does not meet the preregistered signal gate."
            )
    else:
        state = "POOL_LIMITED_STOP"
        reason = (
            "A usable frozen-pool ceiling/headroom was not established under "
            "the preregistered four-state gate."
        )
    return {
        "state": state,
        "reason": reason,
        "thresholds": {
            "oracle_floor": oracle_floor,
            "independent_saturation": independent_saturation,
            "minimum_headroom": minimum_headroom,
            "minimum_signal_fraction": minimum_signal_fraction,
        },
        "observed_bounds": {
            "p_ind_lower95": p_ind_low,
            "p_ind_upper95": p_ind_high,
            "p_oracle_lower95": p_oracle_low,
            "p_oracle_upper95": p_oracle_high,
            "h_pair_lower95": h_pair_low,
            "h_pair_upper95": h_pair_high,
            "delta_c_lower95": delta_c_low,
        },
    }


def mean_finite(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return mean(finite) if finite else None
