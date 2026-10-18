from overlap_poc.data import (
    AI4PrivacyLoader,
    InMemoryPassageLoader,
    NaturalQuestionsLoader,
    Passage,
)


def test_nq_loader_preserves_beir_id(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    rows = [{"_id": "doc7", "title": "Title", "text": "one two three"}]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: rows),
    )
    passage = NaturalQuestionsLoader(max_words=3).load(1)[0]
    assert passage.passage_id == "doc7"


def test_nq_loader_can_disable_whitespace_truncation(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    rows = [{"_id": "doc8", "text": "one two three four"}]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: rows),
    )
    passage = NaturalQuestionsLoader(max_words=None).load(1)[0]
    assert passage.text == "one two three four"


def test_in_memory_loader_is_network_free_and_respects_limit() -> None:
    passages = [Passage(str(i), f"passage {i}") for i in range(3)]
    assert InMemoryPassageLoader(passages).load(2) == passages[:2]


def test_nq_document_extraction_removes_html_tokens() -> None:
    row = {
        "document": {
            "title": "Example",
            "tokens": {
                "token": ["<P>", "Public", "Wikipedia", "text", "</P>"],
                "is_html": [True, False, False, False, True],
            },
        }
    }
    assert NaturalQuestionsLoader._extract_document(row) == (
        "Public Wikipedia text",
        "Example",
    )


def test_nq_document_extraction_accepts_beir_passages() -> None:
    assert NaturalQuestionsLoader._extract_document(
        {"_id": "doc7", "title": "Public title", "text": "Wikipedia passage"}
    ) == ("Wikipedia passage", "Public title")


def test_ai4privacy_boundary_fails_loudly_until_mapping_is_defined() -> None:
    try:
        AI4PrivacyLoader().load(1)
    except NotImplementedError as exc:
        assert "external validation" in str(exc)
    else:
        raise AssertionError("stub should not silently load incompatible data")
