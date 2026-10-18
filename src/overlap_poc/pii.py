"""Deterministic synthetic PII generation and controlled placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from faker import Faker

PIIType = Literal["name", "email", "phone"]
PositionLabel = Literal[
    "overlap_region", "non_overlap_region", "chunk_boundary"
]


@dataclass(frozen=True)
class PIIEntity:
    """Synthetic PII and its exact span in the injected document."""

    entity_id: str
    pii_type: PIIType
    value: str
    position_label: PositionLabel
    char_start: int
    char_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class InjectedDocument:
    text: str
    entities: tuple[PIIEntity, ...]
    seed: int
    placement_overlap_ratio: float


def inject_synthetic_pii(
    text: str,
    *,
    seed: int = 0,
    chunk_size: int = 32,
    overlap_ratio: float = 0.2,
    locale: str = "en_US",
    position_rotation: int = 0,
) -> InjectedDocument:
    """Insert a name, email, and phone at controlled whitespace-token positions.

    ``position_rotation`` rotates PII types over non-overlap, overlap, and boundary
    positions. Rotations 0/1/2 across three documents form a Latin square so type
    difficulty is not confounded with position. For a 0%-overlap experiment,
    inject once with a positive placement overlap and reuse the same document;
    labels describe controlled source positions, not every evaluated overlap rate.
    """

    if chunk_size < 8:
        raise ValueError("chunk_size must be at least 8 for controlled placement")
    overlap_tokens = _overlap_tokens(chunk_size, overlap_ratio)
    if overlap_tokens < 2:
        raise ValueError("placement overlap must contain at least two tokens")

    base_tokens = text.split()
    if len(base_tokens) < chunk_size * 2:
        raise ValueError(
            "source passage must contain at least two chunks before PII injection"
        )

    fake = Faker(locale)
    fake.seed_instance(seed)
    name = f"{fake.first_name().split()[0]} {fake.last_name().split()[-1]}"
    email = f"u{fake.random_int(min=1000, max=9999)}@mail.test"
    phone = f"555{fake.random_int(min=1000, max=9999):04d}"

    stride = chunk_size - overlap_tokens
    values: list[tuple[PIIType, str]] = [
        ("phone", phone),
        ("email", email),
        ("name", name),
    ]
    rotation = position_rotation % len(values)
    values = values[rotation:] + values[:rotation]
    position_targets: list[tuple[int, PositionLabel]] = [
        (max(1, stride // 3), "non_overlap_region"),
        (stride, "overlap_region"),
        (chunk_size - 1, "chunk_boundary"),
    ]
    placements: list[tuple[int, PIIType, str, PositionLabel]] = [
        (target, pii_type, value, label)
        for (target, label), (pii_type, value) in zip(position_targets, values)
    ]
    placements.sort(key=lambda item: item[0])

    output_tokens: list[str] = []
    inserted: list[tuple[str, PIIType, str, PositionLabel, int, int]] = []
    base_index = 0
    for entity_index, (target, pii_type, value, label) in enumerate(placements):
        if target < len(output_tokens):
            raise ValueError("chunk_size is too small for non-overlapping placements")
        take = target - len(output_tokens)
        output_tokens.extend(base_tokens[base_index : base_index + take])
        base_index += take
        value_tokens = value.split()
        token_start = len(output_tokens)
        output_tokens.extend(value_tokens)
        token_end = len(output_tokens)
        inserted.append(
            (
                f"synthetic-{entity_index}",
                pii_type,
                value,
                label,
                token_start,
                token_end,
            )
        )
    output_tokens.extend(base_tokens[base_index:])

    offsets = _token_character_offsets(output_tokens)
    entities = tuple(
        PIIEntity(
            entity_id=entity_id,
            pii_type=pii_type,
            value=value,
            position_label=label,
            char_start=offsets[token_start][0],
            char_end=offsets[token_end - 1][1],
            token_start=token_start,
            token_end=token_end,
        )
        for entity_id, pii_type, value, label, token_start, token_end in inserted
    )
    return InjectedDocument(
        text=" ".join(output_tokens),
        entities=entities,
        seed=seed,
        placement_overlap_ratio=overlap_ratio,
    )


def _overlap_tokens(chunk_size: int, overlap_ratio: float) -> int:
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1)")
    # A token overlap must be integral. Round half up so common settings such as
    # 32 tokens / 20% have one documented, deterministic realization (6 tokens).
    return int(chunk_size * overlap_ratio + 0.5)


def _token_character_offsets(tokens: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token) + 1
    return offsets
