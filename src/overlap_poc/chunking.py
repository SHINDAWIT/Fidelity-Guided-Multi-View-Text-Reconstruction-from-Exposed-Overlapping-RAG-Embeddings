"""Chunk builders for legacy whitespace and tokenizer-native PoC inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from .pii import PIIEntity, PIIType, PositionLabel


@dataclass(frozen=True)
class ChunkPII:
    entity_id: str
    pii_type: PIIType
    value: str
    position_label: PositionLabel
    global_char_start: int
    global_char_end: int
    local_char_start: int
    local_char_end: int
    visible_text: str
    is_complete: bool


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    text: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    pii_entities: tuple[ChunkPII, ...] = ()


@dataclass(frozen=True)
class TokenWindow:
    """One independently tokenized raw window for the GTR encoder."""

    window_index: int
    text: str
    content_ids: tuple[int, ...]
    model_input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    document_token_start: int
    document_token_end: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class StableTokenPair:
    """K=2 observation with one exact 16-content-token shared span."""

    passage_id: str
    source_row_index: int
    selected_start: int
    windows: tuple[TokenWindow, TokenWindow]
    reference_ids: tuple[int, ...]
    shared_ids: tuple[int, ...]


@dataclass(frozen=True)
class StablePairSearch:
    """Round-trip search result and position-level feasibility counts."""

    pair: StableTokenPair | None
    content_token_count: int
    candidate_starts: int
    stable_starts: int
    fixed_start_stable: bool
    rejection_reason: str | None = None


@dataclass(frozen=True)
class StableTokenChain:
    """K-window round-trip-stable observation chain for one source passage."""

    passage_id: str
    source_row_index: int
    selected_start: int
    windows: tuple[TokenWindow, ...]
    reference_ids: tuple[int, ...]
    observation_multiplicity: tuple[int, ...]


@dataclass(frozen=True)
class StableChainSearch:
    """Round-trip feasibility result for one requested chain length K."""

    k: int
    chain: StableTokenChain | None
    content_token_count: int
    required_content_tokens: int
    candidate_starts: int
    stable_starts: int
    stable_start_positions: tuple[int, ...]
    fixed_start_stable: bool
    rejection_reason: str | None = None


def find_roundtrip_stable_pair(
    text: str,
    tokenizer: Any,
    *,
    passage_id: str,
    source_row_index: int,
    content_tokens: int = 31,
    overlap_tokens: int = 16,
    model_positions: int = 32,
) -> StablePairSearch:
    """Find the first K=2 raw-substring pair whose token IDs round-trip.

    The complete passage is tokenized once without special tokens. Every
    possible two-window start is checked, but only the first stable pair is
    selected. The returned raw strings are independently tokenized exactly as
    normal GTR inputs; direct token-ID embedding is deliberately not used here.
    """

    if content_tokens < 1:
        raise ValueError("content_tokens must be positive")
    if not 0 < overlap_tokens < content_tokens:
        raise ValueError("overlap_tokens must be in (0, content_tokens)")
    stride = content_tokens - overlap_tokens
    required_content_tokens = content_tokens + stride
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    document_ids = tuple(int(value) for value in encoded["input_ids"])
    offsets = tuple((int(start), int(end)) for start, end in encoded["offset_mapping"])
    if len(document_ids) != len(offsets):
        raise ValueError("token IDs and offset mapping lengths differ")
    if len(document_ids) < required_content_tokens:
        return StablePairSearch(
            pair=None,
            content_token_count=len(document_ids),
            candidate_starts=0,
            stable_starts=0,
            fixed_start_stable=False,
            rejection_reason="fewer_than_required_content_tokens",
        )

    first_pair: StableTokenPair | None = None
    stable_starts = 0
    fixed_start_stable = False
    candidate_starts = len(document_ids) - required_content_tokens + 1
    for start in range(candidate_starts):
        window_starts = (start, start + stride)
        built: list[TokenWindow] = []
        is_stable = True
        for window_index, window_start in enumerate(window_starts):
            content_ids = document_ids[
                window_start : window_start + content_tokens
            ]
            raw_span = _raw_token_span(text, offsets, window_start, content_tokens)
            if raw_span is None:
                is_stable = False
                break
            char_start, char_end, raw_text = raw_span
            independent_ids = tuple(
                int(value)
                for value in tokenizer(
                    raw_text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )
            if independent_ids != content_ids:
                is_stable = False
                break
            model_input_ids, attention_mask = _prepare_standalone_window(
                raw_text,
                content_ids,
                tokenizer,
                model_positions=model_positions,
            )
            built.append(
                TokenWindow(
                    window_index=window_index,
                    text=raw_text,
                    content_ids=content_ids,
                    model_input_ids=model_input_ids,
                    attention_mask=attention_mask,
                    document_token_start=window_start,
                    document_token_end=window_start + content_tokens,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        if not is_stable:
            continue
        stable_starts += 1
        if start == 0:
            fixed_start_stable = True
        if first_pair is None:
            left, right = built
            shared_ids = left.content_ids[-overlap_tokens:]
            if shared_ids != right.content_ids[:overlap_tokens]:
                raise AssertionError("constructed pair does not have exact overlap")
            reference_ids = left.content_ids + right.content_ids[overlap_tokens:]
            if len(reference_ids) != required_content_tokens:
                raise AssertionError("constructed union has unexpected length")
            first_pair = StableTokenPair(
                passage_id=passage_id,
                source_row_index=source_row_index,
                selected_start=start,
                windows=(left, right),
                reference_ids=reference_ids,
                shared_ids=shared_ids,
            )

    return StablePairSearch(
        pair=first_pair,
        content_token_count=len(document_ids),
        candidate_starts=candidate_starts,
        stable_starts=stable_starts,
        fixed_start_stable=fixed_start_stable,
        rejection_reason=None if first_pair is not None else "no_roundtrip_stable_pair",
    )


def find_roundtrip_stable_chains(
    text: str,
    tokenizer: Any,
    *,
    passage_id: str,
    source_row_index: int,
    ks: Sequence[int],
    content_tokens: int = 31,
    overlap_tokens: int = 16,
    model_positions: int = 32,
) -> dict[int, StableChainSearch]:
    """Find the first stable chain for each K while tokenizing each window once.

    A full passage is tokenized once without special tokens. Every possible
    31-content-token raw substring is independently tokenized and validated.
    A K-chain start is stable exactly when all starts separated by the fixed
    stride have stable standalone windows. This preserves the Primary raw-text
    path and makes the K sweep a nested geometry check rather than direct-ID
    inference.
    """

    normalized_ks = tuple(sorted(set(int(k) for k in ks)))
    if not normalized_ks or any(k < 2 for k in normalized_ks):
        raise ValueError("ks must contain one or more integers >= 2")
    if content_tokens < 1:
        raise ValueError("content_tokens must be positive")
    if not 0 < overlap_tokens < content_tokens:
        raise ValueError("overlap_tokens must be in (0, content_tokens)")
    stride = content_tokens - overlap_tokens
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    document_ids = tuple(int(value) for value in encoded["input_ids"])
    offsets = tuple(
        (int(start), int(end)) for start, end in encoded["offset_mapping"]
    )
    if len(document_ids) != len(offsets):
        raise ValueError("token IDs and offset mapping lengths differ")

    stable_windows: dict[int, TokenWindow] = {}
    window_candidate_starts = max(0, len(document_ids) - content_tokens + 1)
    for window_start in range(window_candidate_starts):
        content_ids = document_ids[window_start : window_start + content_tokens]
        raw_span = _raw_token_span(text, offsets, window_start, content_tokens)
        if raw_span is None:
            continue
        char_start, char_end, raw_text = raw_span
        independent_ids = tuple(
            int(value)
            for value in tokenizer(
                raw_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )
        if independent_ids != content_ids:
            continue
        model_input_ids, attention_mask = _prepare_standalone_window(
            raw_text,
            content_ids,
            tokenizer,
            model_positions=model_positions,
        )
        stable_windows[window_start] = TokenWindow(
            window_index=0,
            text=raw_text,
            content_ids=content_ids,
            model_input_ids=model_input_ids,
            attention_mask=attention_mask,
            document_token_start=window_start,
            document_token_end=window_start + content_tokens,
            char_start=char_start,
            char_end=char_end,
        )

    searches: dict[int, StableChainSearch] = {}
    for k in normalized_ks:
        required_content_tokens = content_tokens + stride * (k - 1)
        candidate_starts = max(
            0, len(document_ids) - required_content_tokens + 1
        )
        stable_chain_starts = [
            start
            for start in range(candidate_starts)
            if all(
                start + window_index * stride in stable_windows
                for window_index in range(k)
            )
        ]
        if not stable_chain_starts:
            reason = (
                "fewer_than_required_content_tokens"
                if len(document_ids) < required_content_tokens
                else "no_roundtrip_stable_chain"
            )
            searches[k] = StableChainSearch(
                k=k,
                chain=None,
                content_token_count=len(document_ids),
                required_content_tokens=required_content_tokens,
                candidate_starts=candidate_starts,
                stable_starts=0,
                stable_start_positions=(),
                fixed_start_stable=False,
                rejection_reason=reason,
            )
            continue

        selected_start = stable_chain_starts[0]
        windows = tuple(
            replace(
                stable_windows[selected_start + window_index * stride],
                window_index=window_index,
            )
            for window_index in range(k)
        )
        reference_ids = document_ids[
            selected_start : selected_start + required_content_tokens
        ]
        multiplicity = tuple(
            sum(
                window.document_token_start <= selected_start + relative_position
                < window.document_token_end
                for window in windows
            )
            for relative_position in range(required_content_tokens)
        )
        for window_index, window in enumerate(windows):
            relative_start = window_index * stride
            if window.content_ids != reference_ids[
                relative_start : relative_start + content_tokens
            ]:
                raise AssertionError("stable chain window differs from union reference")
            if window_index and (
                windows[window_index - 1].content_ids[-overlap_tokens:]
                != window.content_ids[:overlap_tokens]
            ):
                raise AssertionError("stable chain lost its adjacent exact overlap")
        if not multiplicity or min(multiplicity) < 1 or max(multiplicity) > 3:
            raise AssertionError("unexpected source-token observation multiplicity")
        searches[k] = StableChainSearch(
            k=k,
            chain=StableTokenChain(
                passage_id=passage_id,
                source_row_index=source_row_index,
                selected_start=selected_start,
                windows=windows,
                reference_ids=reference_ids,
                observation_multiplicity=multiplicity,
            ),
            content_token_count=len(document_ids),
            required_content_tokens=required_content_tokens,
            candidate_starts=candidate_starts,
            stable_starts=len(stable_chain_starts),
            stable_start_positions=tuple(stable_chain_starts),
            fixed_start_stable=stable_chain_starts[0] == 0,
            rejection_reason=None,
        )
    return searches


def _raw_token_span(
    text: str,
    offsets: Sequence[tuple[int, int]],
    token_start: int,
    token_count: int,
) -> tuple[int, int, str] | None:
    selected = offsets[token_start : token_start + token_count]
    non_empty = [(start, end) for start, end in selected if end > start]
    if not non_empty:
        return None
    char_start = non_empty[0][0]
    char_end = non_empty[-1][1]
    if not 0 <= char_start < char_end <= len(text):
        return None
    return char_start, char_end, text[char_start:char_end]


def _prepare_standalone_window(
    raw_text: str,
    content_ids: tuple[int, ...],
    tokenizer: Any,
    *,
    model_positions: int,
    require_full_content: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    expected = tuple(
        int(value)
        for value in tokenizer.build_inputs_with_special_tokens(list(content_ids))
    )
    prepared = tokenizer(
        raw_text,
        add_special_tokens=True,
        truncation=True,
        max_length=model_positions,
        padding="max_length",
    )
    input_ids = tuple(int(value) for value in prepared["input_ids"])
    attention_mask = tuple(int(value) for value in prepared["attention_mask"])
    if len(expected) > model_positions:
        raise ValueError("content and special tokens exceed model_positions")
    if require_full_content and len(expected) != model_positions:
        raise ValueError(
            "content/special-token contract does not fill model_positions"
        )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None and len(expected) < model_positions:
        raise ValueError("partial standalone windows require a PAD token")
    padded_expected = expected + (int(pad_token_id),) * (
        model_positions - len(expected)
    )
    expected_attention_mask = (1,) * len(expected) + (0,) * (
        model_positions - len(expected)
    )
    if input_ids != padded_expected:
        raise ValueError("standalone tokenization changed or truncated the window")
    if len(input_ids) != model_positions or len(attention_mask) != model_positions:
        raise ValueError("prepared standalone input has an unexpected length")
    if attention_mask != expected_attention_mask:
        raise ValueError("standalone attention mask differs from EOS/PAD contract")
    if require_full_content and any(value != 1 for value in attention_mask):
        raise ValueError("a full 31-content-plus-EOS window must not be padded")
    return input_ids, attention_mask


def chunk_document(
    text: str,
    *,
    chunk_size: int,
    overlap_ratio: float,
    pii_entities: Iterable[PIIEntity] = (),
) -> list[Chunk]:
    """Split ``text`` using whitespace tokens and an exact overlap fraction."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    overlap_tokens = _overlap_tokens(chunk_size, overlap_ratio)
    stride = chunk_size - overlap_tokens
    matches = list(re.finditer(r"\S+", text))
    entities = tuple(pii_entities)
    _validate_entities(text, entities)

    chunks: list[Chunk] = []
    for chunk_id, token_start in enumerate(range(0, len(matches), stride)):
        token_end = min(token_start + chunk_size, len(matches))
        if token_start >= token_end:
            break
        char_start = matches[token_start].start()
        char_end = matches[token_end - 1].end()
        chunk_text = text[char_start:char_end]
        chunk_pii = tuple(
            _project_entity(text, entity, char_start, char_end)
            for entity in entities
            if entity.char_start < char_end and entity.char_end > char_start
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                token_start=token_start,
                token_end=token_end,
                char_start=char_start,
                char_end=char_end,
                pii_entities=chunk_pii,
            )
        )
        if token_end == len(matches):
            break
    return chunks


def _overlap_tokens(chunk_size: int, overlap_ratio: float) -> int:
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1)")
    # Token counts are discrete; round half up and then use that exact count for
    # every adjacent pair. This also supports the intended 32-token PoC chunks.
    return int(chunk_size * overlap_ratio + 0.5)


def _validate_entities(text: str, entities: tuple[PIIEntity, ...]) -> None:
    for entity in entities:
        if not (0 <= entity.char_start < entity.char_end <= len(text)):
            raise ValueError(f"invalid character span for {entity.entity_id}")
        if text[entity.char_start : entity.char_end] != entity.value:
            raise ValueError(
                f"character span does not match value for {entity.entity_id}"
            )


def _project_entity(
    text: str, entity: PIIEntity, chunk_start: int, chunk_end: int
) -> ChunkPII:
    visible_start = max(entity.char_start, chunk_start)
    visible_end = min(entity.char_end, chunk_end)
    return ChunkPII(
        entity_id=entity.entity_id,
        pii_type=entity.pii_type,
        value=entity.value,
        position_label=entity.position_label,
        global_char_start=entity.char_start,
        global_char_end=entity.char_end,
        local_char_start=visible_start - chunk_start,
        local_char_end=visible_end - chunk_start,
        visible_text=text[visible_start:visible_end],
        is_complete=(
            visible_start == entity.char_start and visible_end == entity.char_end
        ),
    )
