from __future__ import annotations

import re

from overlap_poc.chunking import (
    _prepare_standalone_window,
    find_roundtrip_stable_chains,
    find_roundtrip_stable_pair,
)


class WhitespaceTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    vocab_size = 10_000
    is_fast = True

    def __call__(
        self,
        value,
        *,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=False,
        max_length=None,
        padding=False,
    ):
        matches = list(re.finditer(r"\S+", value))
        ids = [100 + int(match.group()[1:]) for match in matches]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        attention = [1] * len(ids)
        if padding == "max_length" and max_length is not None:
            attention.extend([0] * (max_length - len(ids)))
            ids.extend([self.pad_token_id] * (max_length - len(ids)))
        result = {"input_ids": ids, "attention_mask": attention}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (match.start(), match.end()) for match in matches
            ]
        return result

    def build_inputs_with_special_tokens(self, ids):
        return list(ids) + [self.eos_token_id]


def test_roundtrip_stable_k2_geometry_is_exact() -> None:
    text = " ".join(f"t{index}" for index in range(70))
    search = find_roundtrip_stable_pair(
        text,
        WhitespaceTokenizer(),
        passage_id="p0",
        source_row_index=0,
    )
    assert search.pair is not None
    pair = search.pair
    left, right = pair.windows
    assert pair.selected_start == 0
    assert len(left.content_ids) == len(right.content_ids) == 31
    assert len(left.model_input_ids) == len(right.model_input_ids) == 32
    assert left.model_input_ids[-1] == right.model_input_ids[-1] == 1
    assert left.attention_mask == right.attention_mask == (1,) * 32
    assert left.content_ids[-16:] == right.content_ids[:16] == pair.shared_ids
    assert right.document_token_start - left.document_token_start == 15
    assert len(pair.reference_ids) == 46
    assert pair.reference_ids == left.content_ids + right.content_ids[16:]


def test_short_passage_has_no_partial_tail_pair() -> None:
    text = " ".join(f"t{index}" for index in range(45))
    search = find_roundtrip_stable_pair(
        text,
        WhitespaceTokenizer(),
        passage_id="short",
        source_row_index=0,
    )
    assert search.pair is None
    assert search.rejection_reason == "fewer_than_required_content_tokens"


def test_partial_standalone_window_has_eos_then_padding() -> None:
    tokenizer = WhitespaceTokenizer()
    raw_text = " ".join(f"t{index}" for index in range(12))
    content_ids = tuple(100 + index for index in range(12))

    input_ids, attention_mask = _prepare_standalone_window(
        raw_text,
        content_ids,
        tokenizer,
        model_positions=32,
        require_full_content=False,
    )

    assert input_ids[:13] == content_ids + (tokenizer.eos_token_id,)
    assert input_ids[13:] == (tokenizer.pad_token_id,) * 19
    assert attention_mask == (1,) * 13 + (0,) * 19


def test_roundtrip_stable_k8_chain_preserves_nested_geometry_and_multiplicity() -> None:
    text = " ".join(f"t{index}" for index in range(160))
    searches = find_roundtrip_stable_chains(
        text,
        WhitespaceTokenizer(),
        passage_id="long",
        source_row_index=0,
        ks=(2, 3, 4, 6, 8),
    )

    expected_lengths = {2: 46, 3: 61, 4: 76, 6: 106, 8: 136}
    for k, expected_length in expected_lengths.items():
        chain = searches[k].chain
        assert chain is not None
        assert chain.selected_start == 0
        assert len(chain.windows) == k
        assert len(chain.reference_ids) == expected_length
        assert all(len(window.content_ids) == 31 for window in chain.windows)
        assert all(len(window.model_input_ids) == 32 for window in chain.windows)
        assert all(
            right.document_token_start - left.document_token_start == 15
            for left, right in zip(chain.windows, chain.windows[1:])
        )

    chain = searches[8].chain
    assert chain is not None
    assert chain.windows[0].content_ids[-1:] == chain.windows[2].content_ids[:1]
    assert set(chain.observation_multiplicity) == {1, 2, 3}
    assert chain.observation_multiplicity.count(1) == 30
    assert chain.observation_multiplicity.count(2) == 100
    assert chain.observation_multiplicity.count(3) == 6
    assert searches[3].chain is not None
    assert searches[3].chain.reference_ids == chain.reference_ids[:61]
