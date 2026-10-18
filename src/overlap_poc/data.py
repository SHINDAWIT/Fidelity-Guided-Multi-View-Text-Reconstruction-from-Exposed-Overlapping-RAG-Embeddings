"""Passage-loader boundaries for the PoC data sources.

The heavy Hugging Face dependency is imported only when a concrete remote loader
is used. Unit tests can therefore use an in-memory ``PassageLoader`` without
network access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Passage:
    """A public source passage before synthetic PII injection."""

    passage_id: str
    text: str
    title: str | None = None
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PassageLoader(ABC):
    """Source-agnostic boundary used by the experiment pipeline."""

    @abstractmethod
    def load(self, limit: int) -> list[Passage]:
        """Return at most ``limit`` passages in deterministic source order."""


@dataclass
class NaturalQuestionsLoader(PassageLoader):
    """Load Wikipedia passages from the public BEIR Natural Questions corpus.

    The default ``BeIR/nq`` corpus exposes compact ``title``/``text`` records and
    avoids pulling the much larger raw Natural Questions release. The extractor
    also accepts raw Natural Questions token arrays for an explicit override.
    ``streaming=True`` avoids downloading the full passage corpus.
    """

    dataset_name: str = "BeIR/nq"
    dataset_config: str | None = "corpus"
    split: str = "corpus"
    revision: str | None = None
    max_words: int | None = 256
    streaming: bool = True

    def load(self, limit: int) -> list[Passage]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if self.max_words is not None and self.max_words < 1:
            raise ValueError("max_words must be at least 1")

        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "NaturalQuestionsLoader requires the optional 'datasets' package"
            ) from exc

        kwargs: dict[str, Any] = {
            "split": self.split,
            "streaming": self.streaming,
        }
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.dataset_config is not None:
            kwargs["name"] = self.dataset_config
        rows = load_dataset(self.dataset_name, **kwargs)

        passages: list[Passage] = []
        for row_index, row in enumerate(rows):
            text, title = self._extract_document(row)
            if not text:
                continue
            passage_text = (
                text
                if self.max_words is None
                else " ".join(text.split()[: self.max_words])
            )
            passages.append(
                Passage(
                    passage_id=str(row.get("_id", row.get("id", row_index))),
                    text=passage_text,
                    title=title,
                    source="natural_questions",
                    metadata={"split": self.split, "row_index": row_index},
                )
            )
            if len(passages) == limit:
                break
        return passages

    @staticmethod
    def _extract_document(row: Mapping[str, Any]) -> tuple[str, str | None]:
        if isinstance(row.get("text"), str):
            title_value = row.get("title")
            title = str(title_value) if title_value is not None else None
            return str(row["text"]), title

        document = row.get("document")
        if not isinstance(document, Mapping):
            return "", None

        title_value = document.get("title")
        title = str(title_value) if title_value is not None else None
        tokens = document.get("tokens")
        if not isinstance(tokens, Mapping):
            return "", title

        words = tokens.get("token", ())
        html_flags = tokens.get("is_html", ())
        if not isinstance(words, Iterable):
            return "", title

        word_list = list(words)
        flag_list = list(html_flags) if isinstance(html_flags, Iterable) else []
        clean_words = [
            str(word)
            for index, word in enumerate(word_list)
            if not (index < len(flag_list) and bool(flag_list[index]))
        ]
        return " ".join(clean_words), title


@dataclass
class AI4PrivacyLoader(PassageLoader):
    """Explicit extension boundary for a later AI4Privacy validation experiment.

    AI4Privacy records are PII-annotated examples rather than Natural Questions
    Wikipedia passages. Their annotation-to-``Passage`` policy must be chosen as
    part of the external-validation experiment, so this initial PoC fails loudly
    instead of silently mixing incompatible source semantics.
    """

    dataset_name: str = "ai4privacy/pii-masking-200k"
    split: str = "train"

    def load(self, limit: int) -> list[Passage]:
        raise NotImplementedError(
            "AI4Privacy loading is reserved for external validation; implement "
            "the dataset-specific text and annotation mapping in this loader."
        )


class InMemoryPassageLoader(PassageLoader):
    """Network-free loader useful for tests and local smoke experiments."""

    def __init__(self, passages: Iterable[Passage]) -> None:
        self._passages = tuple(passages)

    def load(self, limit: int) -> list[Passage]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        return list(self._passages[:limit])
