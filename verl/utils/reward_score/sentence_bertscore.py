r"""Sentence-level BERTScore coverage reward for static ConvAgent.

For one candidate answer \(p\) and one released answer reference \(g\), let
``P`` and ``G`` be their sentence lists.  The outcome score is

    mean_{g_j in G} max_{p_i in P} BERTScore_F1(p_i, g_j).

This is a gold-to-prediction semantic coverage score: every factual sentence
in the reference must be covered by at least one candidate sentence.  When a
static row has multiple released answer references, the reward is the maximum
coverage across those alternatives, matching the project-wide max-reference
F1 protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


# The static benchmarks are English.  Splitting at semicolons and line breaks
# also handles concise enumerations that encode several atomic facts in one
# generated paragraph, without requiring NLTK/Punkt downloads at runtime.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])(?=\s|$)|\n+")


def split_answer_sentences(value: Any) -> list[str]:
    """Split an answer into non-empty, semantically scoreable sentences."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return []
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    return sentences or [text]


@dataclass(frozen=True)
class SentenceBERTScore:
    """One candidate's sentence-coverage outcome and audit metadata."""

    reward: float
    predicted_sentence_count: int
    gold_sentence_count: int
    references_scored: int
    selected_reference_index: int


class SentenceBERTScoreReward:
    """Lazily-loaded, batched BERTScore sentence-coverage evaluator.

    The scorer remains local and frozen: it has no connection to the policy,
    gradients, tool environment, or generated rollout state beyond each final
    answer string.  Delayed construction avoids allocating RoBERTa during
    validation-only reward-manager creation.
    """

    def __init__(
        self,
        *,
        model: str = "roberta-large",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model = str(model or "roberta-large").strip()
        self.device = str(device or "cpu").strip()
        self.batch_size = int(batch_size)
        self._scorer: Any | None = None

        if not self.model:
            raise ValueError("sentence BERTScore reward requires a non-empty model name")
        if not self.device:
            raise ValueError("sentence BERTScore reward requires a device")
        if self.batch_size < 1:
            raise ValueError("sentence BERTScore batch_size must be positive")

    def _get_scorer(self) -> Any:
        if self._scorer is not None:
            return self._scorer
        try:
            from bert_score import BERTScorer
        except ImportError as error:  # pragma: no cover - guarded by runtime environment.
            raise RuntimeError(
                "Sentence BERTScore reward requires bert-score. Install "
                "bert-score==0.3.13 in the IGPO environment."
            ) from error

        # No baseline rescaling: reward values remain directly interpretable as
        # semantic F1 in roughly [0, 1], and are comparable to the former F1
        # outcome term.  The default CPU device prevents pressure on FSDP
        # actor/reference/critic workers.  A dedicated device is configurable.
        self._scorer = BERTScorer(
            model_type=self.model,
            lang="en",
            idf=False,
            rescale_with_baseline=False,
            device=self.device,
            batch_size=self.batch_size,
        )
        return self._scorer

    def _score_pairs(self, candidates: Sequence[str], references: Sequence[str]) -> list[float]:
        """Return pairwise BERTScore F1 values in input order."""

        if not candidates:
            return []
        if len(candidates) != len(references):
            raise ValueError("BERTScore candidates and references must have equal length")
        scorer = self._get_scorer()
        _, _, f1 = scorer.score(
            list(candidates),
            list(references),
            batch_size=self.batch_size,
            verbose=False,
        )
        values = [float(value) for value in f1.detach().cpu().tolist()]
        if len(values) != len(candidates):
            raise RuntimeError("BERTScore returned an unexpected number of pair scores")
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("BERTScore produced a non-finite sentence score")
        return values

    def score_records(self, records: Sequence[Mapping[str, Any]]) -> dict[int, SentenceBERTScore]:
        """Score a batch of policy answers indexed by ``record['row_index']``.

        Each record has a single policy ``predicted_answer`` and every released
        answer alternative in ``references``.  Pairwise sentence BERTScores are
        flattened into one batched call; each gold sentence then retains its
        highest score over candidate sentences.
        """

        candidate_texts: list[str] = []
        reference_texts: list[str] = []
        pair_locations: list[tuple[int, int, int]] = []
        record_metadata: dict[int, tuple[int, list[list[str]], int]] = {}
        best_by_gold_sentence: dict[tuple[int, int, int], float] = {}

        for record in records:
            row_index = int(record["row_index"])
            prediction_sentences = split_answer_sentences(record.get("predicted_answer", ""))
            references = [
                split_answer_sentences(reference)
                for reference in record.get("references", [])
                if str(reference or "").strip()
            ]
            references = [sentences for sentences in references if sentences]
            record_metadata[row_index] = (len(prediction_sentences), references, len(references))
            if not prediction_sentences or not references:
                continue

            for reference_index, gold_sentences in enumerate(references):
                for gold_index, gold_sentence in enumerate(gold_sentences):
                    location = (row_index, reference_index, gold_index)
                    best_by_gold_sentence[location] = 0.0
                    for prediction_sentence in prediction_sentences:
                        candidate_texts.append(prediction_sentence)
                        reference_texts.append(gold_sentence)
                        pair_locations.append(location)

        for location, score in zip(
            pair_locations,
            self._score_pairs(candidate_texts, reference_texts),
            strict=True,
        ):
            best_by_gold_sentence[location] = max(best_by_gold_sentence[location], score)

        results: dict[int, SentenceBERTScore] = {}
        for row_index, (prediction_count, references, reference_count) in record_metadata.items():
            candidates: list[tuple[float, int, int]] = []
            for reference_index, gold_sentences in enumerate(references):
                sentence_scores = [
                    best_by_gold_sentence.get((row_index, reference_index, gold_index), 0.0)
                    for gold_index in range(len(gold_sentences))
                ]
                coverage = float(sum(sentence_scores) / len(sentence_scores)) if sentence_scores else 0.0
                candidates.append((coverage, reference_index, len(gold_sentences)))

            if candidates:
                # ``max`` is stable: ties retain the first released reference.
                reward, selected_reference_index, gold_sentence_count = max(
                    candidates, key=lambda item: item[0]
                )
            else:
                reward, selected_reference_index, gold_sentence_count = 0.0, -1, 0
            results[row_index] = SentenceBERTScore(
                reward=float(reward),
                predicted_sentence_count=prediction_count,
                gold_sentence_count=gold_sentence_count,
                references_scored=reference_count,
                selected_reference_index=selected_reference_index,
            )
        return results
