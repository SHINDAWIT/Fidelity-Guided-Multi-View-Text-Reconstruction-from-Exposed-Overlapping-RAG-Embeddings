import pytest

from overlap_poc.chunking import chunk_document
from overlap_poc.pii import inject_synthetic_pii


BASE_TEXT = " ".join(f"token{i}" for i in range(120))


@pytest.mark.parametrize(
    ("overlap_ratio", "expected_stride"),
    [(0.0, 10), (0.2, 8), (0.4, 6)],
)
def test_chunking_applies_exact_overlap(
    overlap_ratio: float, expected_stride: int
) -> None:
    chunks = chunk_document(
        BASE_TEXT, chunk_size=10, overlap_ratio=overlap_ratio
    )
    assert chunks[0].token_start == 0
    assert chunks[0].token_end == 10
    assert chunks[1].token_start == expected_stride
    assert chunks[1].token_end == expected_stride + 10
    assert chunks[0].text == BASE_TEXT[chunks[0].char_start : chunks[0].char_end]


def test_chunk_metadata_preserves_complete_and_boundary_pii() -> None:
    document = inject_synthetic_pii(
        BASE_TEXT, seed=9, chunk_size=20, overlap_ratio=0.2
    )
    chunks = chunk_document(
        document.text,
        chunk_size=20,
        overlap_ratio=0.2,
        pii_entities=document.entities,
    )

    boundary_id = next(
        entity.entity_id
        for entity in document.entities
        if entity.position_label == "chunk_boundary"
    )
    first_piece = next(
        item for item in chunks[0].pii_entities if item.entity_id == boundary_id
    )
    second_piece = next(
        item for item in chunks[1].pii_entities if item.entity_id == boundary_id
    )
    assert not first_piece.is_complete
    assert second_piece.is_complete
    assert chunks[0].text[
        first_piece.local_char_start : first_piece.local_char_end
    ] == first_piece.visible_text
    assert first_piece.visible_text != second_piece.visible_text


def test_chunking_rounds_fractional_overlap_to_a_stable_token_count() -> None:
    chunks = chunk_document(BASE_TEXT, chunk_size=32, overlap_ratio=0.2)
    assert chunks[1].token_start == 26
