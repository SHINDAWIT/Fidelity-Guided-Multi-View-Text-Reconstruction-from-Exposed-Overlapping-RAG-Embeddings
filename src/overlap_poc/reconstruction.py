from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol, Sequence


_TOKEN_RE = re.compile(r"[\w@.+-]+", re.UNICODE)


@dataclass(frozen=True)
class Candidate:
    text: str
    local_score: float
    correction_steps: int
    sequence_beam_width: int = 0
    generation_id: str = "legacy"
    content_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OverlapAlignment:
    score: float
    left_suffix_tokens: int
    right_prefix_tokens: int


class InversionBackend(Protocol):
    def generate_candidates(
        self, texts: Sequence[str], correction_steps: Sequence[int]
    ) -> list[list[Candidate]]: ...


def normalized_tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _boundary_tokens(text: str) -> list[str]:
    """Normalize while preserving a one-to-one mapping to whitespace tokens."""
    return [
        ("".join(_TOKEN_RE.findall(token)) or token).casefold()
        for token in text.split()
    ]


def align_overlap(left: str, right: str, overlap_ratio: float) -> OverlapAlignment:
    """Align variable-length candidate suffix/prefix regions without source text."""
    if overlap_ratio <= 0:
        return OverlapAlignment(0.0, 0, 0)
    if overlap_ratio >= 1:
        raise ValueError("overlap_ratio must be below 1")
    left_tokens = _boundary_tokens(left)
    right_tokens = _boundary_tokens(right)
    if not left_tokens or not right_tokens:
        return OverlapAlignment(0.0, 0, 0)

    expected_left = max(1, round(len(left_tokens) * overlap_ratio))
    expected_right = max(1, round(len(right_tokens) * overlap_ratio))
    left_widths = range(
        max(1, math.floor(expected_left * 0.5)),
        min(len(left_tokens), math.ceil(expected_left * 1.5)) + 1,
    )
    right_widths = range(
        max(1, math.floor(expected_right * 0.5)),
        min(len(right_tokens), math.ceil(expected_right * 1.5)) + 1,
    )
    best = OverlapAlignment(0.0, expected_left, expected_right)
    for left_width in left_widths:
        for right_width in right_widths:
            similarity = SequenceMatcher(
                None,
                left_tokens[-left_width:],
                right_tokens[:right_width],
                autojunk=False,
            ).ratio()
            deviation = 0.05 * (
                abs(left_width - expected_left) / expected_left
                + abs(right_width - expected_right) / expected_right
            )
            score = max(0.0, similarity - deviation)
            if score > best.score:
                best = OverlapAlignment(score, left_width, right_width)
    return best


def overlap_consistency(left: str, right: str, overlap_ratio: float) -> float:
    return align_overlap(left, right, overlap_ratio).score


def select_baseline(candidate_sets: Sequence[Sequence[Candidate]]) -> list[Candidate]:
    """Choose each chunk independently by target-embedding cosine similarity."""
    selected: list[Candidate] = []
    for candidates in candidate_sets:
        if not candidates:
            raise ValueError("every chunk must have at least one candidate")
        selected.append(max(candidates, key=lambda item: item.local_score))
    return selected


def select_joint(
    candidate_sets: Sequence[Sequence[Candidate]],
    overlap_ratios: Sequence[float],
    consistency_weight: float,
) -> list[Candidate]:
    """Viterbi reranking over local embedding scores and adjacent overlap agreement."""
    if not candidate_sets:
        return []
    if len(overlap_ratios) != len(candidate_sets) - 1:
        raise ValueError("overlap_ratios must contain one value per adjacent chunk pair")
    if any(not candidates for candidates in candidate_sets):
        raise ValueError("every chunk must have at least one candidate")

    scores = [candidate.local_score for candidate in candidate_sets[0]]
    backpointers: list[list[int]] = []
    for chunk_index in range(1, len(candidate_sets)):
        current_scores: list[float] = []
        current_backpointers: list[int] = []
        overlap_ratio = overlap_ratios[chunk_index - 1]
        for current in candidate_sets[chunk_index]:
            transitions = [
                previous_score
                + current.local_score
                + consistency_weight
                * overlap_consistency(previous.text, current.text, overlap_ratio)
                for previous_score, previous in zip(
                    scores, candidate_sets[chunk_index - 1]
                )
            ]
            best_previous = max(range(len(transitions)), key=transitions.__getitem__)
            current_scores.append(transitions[best_previous])
            current_backpointers.append(best_previous)
        scores = current_scores
        backpointers.append(current_backpointers)

    selected_indices = [max(range(len(scores)), key=scores.__getitem__)]
    for pointers in reversed(backpointers):
        selected_indices.append(pointers[selected_indices[-1]])
    selected_indices.reverse()
    return [
        candidate_sets[index][candidate_index]
        for index, candidate_index in enumerate(selected_indices)
    ]


def merge_reconstructions(
    selected: Sequence[Candidate],
    overlap_ratios: Sequence[float],
    minimum_alignment: float = 0.5,
) -> str:
    """Merge on an aligned candidate boundary, never a source-token index."""
    if not selected:
        return ""
    if len(overlap_ratios) != len(selected) - 1:
        raise ValueError("overlap_ratios must contain one value per adjacent chunk pair")
    merged = selected[0].text.strip()
    previous_text = selected[0].text
    for candidate, overlap_ratio in zip(selected[1:], overlap_ratios):
        tokens = candidate.text.strip().split()
        alignment = align_overlap(previous_text, candidate.text, overlap_ratio)
        drop = alignment.right_prefix_tokens if alignment.score >= minimum_alignment else 0
        merged = " ".join(part for part in (merged, " ".join(tokens[drop:])) if part)
        previous_text = candidate.text
    return merged


class Vec2TextBackend:
    """Thin, lazy Vec2Text adapter kept separate from model-free reranking tests."""

    def __init__(
        self,
        model_name: str = "gtr-base",
        max_length: int = 32,
        sequence_beam_width: int = 2,
        device: str | None = None,
        inversion_checkpoint: str | None = None,
        corrector_checkpoint: str | None = None,
        inversion_revision: str | None = None,
        corrector_revision: str | None = None,
    ) -> None:
        import torch
        import vec2text

        self._torch = torch
        self._vec2text = vec2text
        self.max_length = max_length
        self.sequence_beam_width = sequence_beam_width
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but this PyTorch build cannot access a CUDA GPU; "
                "install the CUDA-enabled dependencies and verify the NVIDIA driver"
            )
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        if bool(inversion_checkpoint) != bool(corrector_checkpoint):
            raise ValueError("both inversion and corrector checkpoints are required")
        if inversion_checkpoint and corrector_checkpoint:
            inversion_model = vec2text.models.InversionModel.from_pretrained(
                inversion_checkpoint, revision=inversion_revision
            )
            corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
                corrector_checkpoint, revision=corrector_revision
            )
            self.corrector = vec2text.load_corrector(inversion_model, corrector_model)
        else:
            self.corrector = vec2text.load_pretrained_corrector(model_name)
        self.corrector.inversion_trainer.model.to(self.device)
        self.corrector.model.to(self.device)
        self.corrector.inversion_trainer.model.eval()
        self.corrector.model.eval()

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        encoded = self.corrector.embedder_tokenizer(
            list(texts), add_special_tokens=True, truncation=False
        )
        return [len(input_ids) for input_ids in encoded["input_ids"]]

    def validate_inputs(self, texts: Sequence[str]) -> list[int]:
        lengths = self.token_lengths(texts)
        too_long = [
            (index, length) for index, length in enumerate(lengths)
            if length > self.max_length
        ]
        if too_long:
            raise ValueError(
                f"GTR inputs exceed the {self.max_length}-subword checkpoint limit: "
                f"{too_long}"
            )
        return lengths

    def _embed(self, texts: Sequence[str]):
        encoded = self.corrector.embedder_tokenizer(
            list(texts),
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        ).to(self.device)
        with self._torch.no_grad():
            return self.corrector.inversion_trainer.call_embedding_model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
            )

    @property
    def embedder_tokenizer(self):
        return self.corrector.embedder_tokenizer

    def embed_raw_windows(
        self,
        texts: Sequence[str],
        *,
        expected_input_ids: Sequence[Sequence[int]] | None = None,
    ):
        """Embed independently tokenized raw windows once and freeze as FP32 CPU tensors."""

        if not texts:
            return self._torch.empty((0, 0), dtype=self._torch.float32)
        self.validate_inputs(texts)
        encoded = self.corrector.embedder_tokenizer(
            list(texts),
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        if expected_input_ids is not None:
            actual = encoded.input_ids.tolist()
            expected = [list(values) for values in expected_input_ids]
            if actual != expected:
                raise ValueError("raw-window tokenization differs from stable window IDs")
        encoded = encoded.to(self.device)
        with self._torch.inference_mode():
            embeddings = self.corrector.inversion_trainer.call_embedding_model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
            )
        frozen = embeddings.detach().to(device="cpu", dtype=self._torch.float32)
        if frozen.dtype != self._torch.float32:
            raise AssertionError("target embeddings must be frozen as FP32")
        return frozen

    def _invert_embeddings(
        self,
        embeddings,
        *,
        correction_steps: int,
        sequence_beam_width: int,
    ) -> list[str]:
        generation_kwargs = copy.copy(self.corrector.gen_kwargs)
        generation_kwargs.update({"min_length": 1, "max_length": self.max_length})
        embeddings = embeddings.to(self.device)
        with self._torch.inference_mode():
            if correction_steps == 0:
                generated = self.corrector.inversion_trainer.generate(
                    inputs={"frozen_embeddings": embeddings},
                    generation_kwargs=generation_kwargs,
                )
            else:
                self.corrector.return_best_hypothesis = sequence_beam_width > 0
                generated = self.corrector.generate(
                    inputs={"frozen_embeddings": embeddings},
                    generation_kwargs=generation_kwargs,
                    num_recursive_steps=correction_steps,
                    sequence_beam_width=max(1, sequence_beam_width),
                )
        return self.corrector.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )

    def _candidate_content_ids(self, text: str) -> tuple[int, ...]:
        encoded = self.corrector.embedder_tokenizer(
            text, add_special_tokens=False, truncation=False
        )["input_ids"]
        excluded = {
            value
            for value in (
                self.corrector.embedder_tokenizer.eos_token_id,
                self.corrector.embedder_tokenizer.pad_token_id,
            )
            if value is not None
        }
        return tuple(int(value) for value in encoded if int(value) not in excluded)

    def embed_candidate_texts(
        self, texts: Sequence[str], *, batch_size: int = 8
    ):
        """Embed candidate strings as FP32 CPU tensors for diagnostics."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return self._torch.empty((0, 0), dtype=self._torch.float32)
        batches = []
        for offset in range(0, len(texts), batch_size):
            batches.append(
                self._embed(texts[offset : offset + batch_size])
                .detach()
                .to(device="cpu", dtype=self._torch.float32)
            )
        return self._torch.cat(batches)

    @staticmethod
    def _normalized_candidate_key(text: str) -> str:
        return " ".join(text.casefold().split())

    def generate_final_sbeam_candidate_pools(
        self,
        target_embeddings,
        *,
        correction_steps: int = 50,
        sequence_beam_width: int = 8,
        top_n: int = 8,
    ) -> tuple[tuple[tuple[Candidate, ...], ...], tuple[dict[str, Any], ...]]:
        """Expose final sequence-beam expansions without changing Vec2Text search.

        The installed Vec2Text corrector embeds ``beam_width ** 2`` hypotheses at
        the final correction step and then collapses them to one best hypothesis.
        This method temporarily observes that existing embedding call, applies the
        same target-cosine ranking, removes normalized-string duplicates, and
        returns at most ``top_n`` candidates.  The public generation result must
        exactly equal the observed cosine Top-1 in token-ID space or the run fails.

        Batch size is deliberately fixed to one observation.  No generation,
        pruning, or ranking input is modified by the instrumentation.
        """

        if target_embeddings.dtype != self._torch.float32:
            raise ValueError("target embeddings must be FP32")
        if correction_steps < 1:
            raise ValueError("correction_steps must be positive")
        if sequence_beam_width < 2:
            raise ValueError("sequence_beam_width must be at least two")
        if top_n < 1 or top_n > sequence_beam_width:
            raise ValueError("top_n must be between one and sequence_beam_width")

        generation_kwargs_template = copy.copy(self.corrector.gen_kwargs)
        generation_kwargs_template.update(
            {"min_length": 1, "max_length": self.max_length}
        )
        pools: list[tuple[Candidate, ...]] = []
        records: list[dict[str, Any]] = []
        corrector = self.corrector

        for observation_index in range(len(target_embeddings)):
            target = target_embeddings[observation_index : observation_index + 1].to(
                self.device
            )
            captured: dict[str, Any] = {}
            call_count = 0
            embed_batch_sizes: list[int] = []
            original_embed = corrector.embed_generated_hypothesis
            had_instance_attribute = "embed_generated_hypothesis" in corrector.__dict__
            previous_instance_attribute = corrector.__dict__.get(
                "embed_generated_hypothesis"
            )

            def instrumented_embed_generated_hypothesis(*args, **kwargs):
                nonlocal call_count
                call_index = call_count
                call_count += 1
                result = original_embed(*args, **kwargs)
                embed_batch_sizes.append(int(result.shape[0]))
                if call_index == correction_steps:
                    input_ids = kwargs.get("input_ids")
                    if input_ids is None and args:
                        input_ids = args[0]
                    if input_ids is None:
                        raise RuntimeError(
                            "could not observe final sequence-beam token IDs"
                        )
                    scores = self._torch.nn.functional.cosine_similarity(
                        result,
                        target.expand_as(result),
                        dim=1,
                    )
                    captured.update(
                        {
                            "input_ids": input_ids.detach().to(device="cpu").clone(),
                            "embeddings": result.detach()
                            .to(device="cpu", dtype=self._torch.float32)
                            .clone(),
                            "scores": scores.detach()
                            .to(device="cpu", dtype=self._torch.float32)
                            .clone(),
                        }
                    )
                return result

            corrector.embed_generated_hypothesis = (
                instrumented_embed_generated_hypothesis
            )
            started = time.perf_counter()
            try:
                corrector.return_best_hypothesis = True
                with self._torch.inference_mode():
                    generated = corrector.generate(
                        inputs={"frozen_embeddings": target},
                        generation_kwargs=copy.copy(generation_kwargs_template),
                        num_recursive_steps=correction_steps,
                        sequence_beam_width=sequence_beam_width,
                    )
            finally:
                if had_instance_attribute:
                    corrector.embed_generated_hypothesis = previous_instance_attribute
                else:
                    del corrector.embed_generated_hypothesis
            elapsed = time.perf_counter() - started

            expected_calls = correction_steps + 1
            if call_count != expected_calls or not captured:
                raise RuntimeError(
                    "Vec2Text call structure changed: expected one initial-hypothesis "
                    f"embedding plus {correction_steps} correction embeddings, got "
                    f"{call_count} calls"
                )
            final_ids = captured["input_ids"]
            final_embeddings = captured["embeddings"]
            final_scores = captured["scores"]
            expected_raw = sequence_beam_width * sequence_beam_width
            if len(final_ids) != expected_raw:
                raise RuntimeError(
                    "Vec2Text final sequence-beam expansion changed: expected "
                    f"{expected_raw} hypotheses, got {len(final_ids)}"
                )
            if final_embeddings.shape[0] != len(final_ids):
                raise RuntimeError("captured hypothesis IDs and embeddings disagree")

            best_raw_index = int(self._torch.argmax(final_scores).item())
            public_ids = tuple(int(value) for value in generated[0].detach().cpu())
            observed_best_ids = tuple(
                int(value) for value in final_ids[best_raw_index]
            )
            if public_ids != observed_best_ids:
                raise RuntimeError(
                    "instrumented final-beam Top-1 differs from Vec2Text public Top-1"
                )

            decoded = corrector.tokenizer.batch_decode(
                final_ids, skip_special_tokens=True
            )
            order = sorted(
                range(len(final_ids)),
                key=lambda index: (-float(final_scores[index]), index),
            )
            unique_id_count = len(
                {tuple(int(value) for value in row) for row in final_ids}
            )
            unique_text_count = len(
                {self._normalized_candidate_key(text) for text in decoded}
            )
            seen_text: set[str] = set()
            candidates: list[Candidate] = []
            retained_raw_indices: list[int] = []
            for raw_index in order:
                text = decoded[raw_index].strip()
                key = self._normalized_candidate_key(text)
                if key in seen_text:
                    continue
                seen_text.add(key)
                rank = len(candidates) + 1
                candidates.append(
                    Candidate(
                        text=text,
                        local_score=float(final_scores[raw_index]),
                        correction_steps=correction_steps,
                        sequence_beam_width=sequence_beam_width,
                        generation_id=f"final_sbeam{sequence_beam_width}_rank{rank}",
                        content_ids=self._candidate_content_ids(text),
                    )
                )
                retained_raw_indices.append(raw_index)
                if len(candidates) == top_n:
                    break
            if not candidates:
                raise RuntimeError("final sequence beam produced no unique candidate")
            if retained_raw_indices[0] != best_raw_index:
                raise RuntimeError("deduplication unexpectedly changed the Top-1")

            pools.append(tuple(candidates))
            records.append(
                {
                    "observation_index": observation_index,
                    "elapsed_seconds": elapsed,
                    "embed_call_count": call_count,
                    "embed_batch_sizes": embed_batch_sizes,
                    "search_embedding_query_calls": call_count,
                    "search_hypothesis_rows_embedded": sum(embed_batch_sizes),
                    "inversion_generation_calls": 1,
                    "correction_generation_calls": correction_steps,
                    "raw_final_hypothesis_count": len(final_ids),
                    "expected_raw_final_hypothesis_count": expected_raw,
                    "unique_token_id_count_before_cap": unique_id_count,
                    "unique_normalized_text_count_before_cap": unique_text_count,
                    "returned_unique_count": len(candidates),
                    "duplicate_rate_by_normalized_text": (
                        1.0 - unique_text_count / len(final_ids)
                    ),
                    "retained_raw_indices": retained_raw_indices,
                    "public_top1_raw_index": best_raw_index,
                    "public_top1_exact_id_match": True,
                }
            )
        return tuple(pools), tuple(records)

    def generate_frozen_candidate_pools(
        self,
        target_embeddings,
        generation_settings: Sequence[dict[str, Any]],
        *,
        top_n: int,
        batch_size: int = 4,
    ) -> tuple[tuple[Candidate, ...], ...]:
        """Generate unique profile endpoints and retain the Top-N by fidelity.

        This pool is not the internal sequence beam. Each configured Vec2Text
        run contributes its final best endpoint, duplicates are collapsed, and
        the remaining endpoints are reranked by cosine similarity to the same
        frozen target embedding.
        """

        if top_n < 1 or batch_size < 1:
            raise ValueError("top_n and batch_size must be positive")
        if not generation_settings:
            raise ValueError("at least one generation setting is required")
        if target_embeddings.dtype != self._torch.float32:
            raise ValueError("target embeddings must be FP32")
        generated_by_observation: list[list[tuple[str, int, int, str]]] = [
            [] for _ in range(len(target_embeddings))
        ]
        for setting_index, raw_setting in enumerate(generation_settings):
            steps = int(raw_setting["correction_steps"])
            beam = int(raw_setting.get("sequence_beam_width", 0))
            generation_id = str(
                raw_setting.get("id", f"steps{steps}_sbeam{beam}_{setting_index}")
            )
            if steps < 0 or beam < 0:
                raise ValueError("correction steps and beam width cannot be negative")
            for offset in range(0, len(target_embeddings), batch_size):
                batch = target_embeddings[offset : offset + batch_size]
                texts = self._invert_embeddings(
                    batch,
                    correction_steps=steps,
                    sequence_beam_width=beam,
                )
                if len(texts) != len(batch):
                    raise RuntimeError("Vec2Text generation changed the batch size")
                for local_index, text in enumerate(texts):
                    generated_by_observation[offset + local_index].append(
                        (text.strip(), steps, beam, generation_id)
                    )

        pools: list[tuple[Candidate, ...]] = []
        cosine = self._torch.nn.functional.cosine_similarity
        for observation_index, generated in enumerate(generated_by_observation):
            unique: dict[str, tuple[str, int, int, str]] = {}
            for item in generated:
                key = " ".join(item[0].casefold().split())
                unique.setdefault(key, item)
            unique_items = list(unique.values())
            if not unique_items:
                raise RuntimeError("candidate generation produced an empty pool")
            candidate_embeddings = []
            candidate_texts = [item[0] for item in unique_items]
            for offset in range(0, len(candidate_texts), batch_size):
                candidate_embeddings.append(
                    self._embed(candidate_texts[offset : offset + batch_size])
                    .detach()
                    .to(device="cpu", dtype=self._torch.float32)
                )
            embedded = self._torch.cat(candidate_embeddings)
            target = target_embeddings[observation_index].unsqueeze(0).expand_as(embedded)
            scores = cosine(embedded, target, dim=1).tolist()
            candidates = [
                Candidate(
                    text=text,
                    local_score=float(score),
                    correction_steps=steps,
                    sequence_beam_width=beam,
                    generation_id=generation_id,
                    content_ids=self._candidate_content_ids(text),
                )
                for (text, steps, beam, generation_id), score in zip(
                    unique_items, scores
                )
            ]
            ordered = sorted(
                enumerate(candidates),
                key=lambda item: (-item[1].local_score, item[0]),
            )
            pools.append(tuple(candidate for _, candidate in ordered[:top_n]))
        return tuple(pools)

    @staticmethod
    def candidate_pool_hash(
        pools: Sequence[Sequence[Candidate]],
    ) -> str:
        payload = [[candidate.to_dict() for candidate in pool] for pool in pools]
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _invert(self, embeddings, steps: int) -> list[str]:
        gen_kwargs = copy.copy(self.corrector.gen_kwargs)
        gen_kwargs["min_length"] = 1
        gen_kwargs["max_length"] = self.max_length
        with self._torch.no_grad():
            if steps == 0:
                generated = self.corrector.inversion_trainer.generate(
                    inputs={"frozen_embeddings": embeddings},
                    generation_kwargs=gen_kwargs,
                )
            else:
                self.corrector.return_best_hypothesis = True
                generated = self.corrector.generate(
                    inputs={"frozen_embeddings": embeddings},
                    generation_kwargs=gen_kwargs,
                    num_recursive_steps=steps,
                    sequence_beam_width=self.sequence_beam_width,
                )
        return self.corrector.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )

    def generate_candidates(
        self, texts: Sequence[str], correction_steps: Sequence[int]
    ) -> list[list[Candidate]]:
        if not texts:
            return []
        if not correction_steps:
            raise ValueError("correction_steps must contain at least one value")
        if any(steps < 0 for steps in correction_steps):
            raise ValueError("correction_steps cannot contain negative values")
        self.validate_inputs(texts)
        target_embeddings = self._embed(texts)
        candidates_by_chunk: list[dict[str, Candidate]] = [dict() for _ in texts]
        cosine = self._torch.nn.CosineSimilarity(dim=1)

        for steps in correction_steps:
            generated_texts = self._invert(target_embeddings, steps)
            if len(generated_texts) != len(texts):
                raise RuntimeError(
                    "Vec2Text returned a different number of generations than inputs"
                )
            generated_embeddings = self._embed(generated_texts)
            local_scores = cosine(target_embeddings, generated_embeddings).tolist()
            for index, (text, score) in enumerate(zip(generated_texts, local_scores)):
                key = " ".join(text.casefold().split())
                candidate = Candidate(text=text.strip(), local_score=float(score), correction_steps=steps)
                previous = candidates_by_chunk[index].get(key)
                if previous is None or candidate.local_score > previous.local_score:
                    candidates_by_chunk[index][key] = candidate

        return [
            sorted(items.values(), key=lambda item: item.local_score, reverse=True)
            for items in candidates_by_chunk
        ]
