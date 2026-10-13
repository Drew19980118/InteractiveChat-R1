r"""Whole-answer BERTScore reward for static ConvAgent.

This frozen reward replaces only ConvAgent's token-set F1 outcome term.  For
one policy answer ``p`` and the released answer alternatives ``G`` it returns

    max_{g in G} BERTScore_F1(p, g).

The maximum over references is deliberately the same convention used by the
project's static validation metrics.  Action and retrieval rewards remain
outside this scorer and are composed by the reward manager.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
from typing import Any


@dataclass(frozen=True)
class AnswerBERTScore:
    """One policy answer's max-reference BERTScore outcome and audit data."""

    reward: float
    references_scored: int
    selected_reference_index: int


class AnswerBERTScoreReward:
    """Lazily loaded, batched whole-answer BERTScore evaluator."""

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
            raise ValueError("answer BERTScore reward requires a non-empty model")
        if not self.device:
            raise ValueError("answer BERTScore reward requires a device")
        if self.batch_size < 1:
            raise ValueError("answer BERTScore batch_size must be positive")

    def _get_scorer(self) -> Any:
        if self._scorer is not None:
            return self._scorer
        try:
            from bert_score import BERTScorer
        except ImportError as error:  # pragma: no cover - runtime dependency.
            raise RuntimeError(
                "Answer BERTScore reward requires bert-score. Install "
                "bert-score==0.3.13 in the IGPO environment."
            ) from error

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
        if not candidates:
            return []
        if len(candidates) != len(references):
            raise ValueError("BERTScore candidates and references must have equal length")
        _, _, f1 = self._get_scorer().score(
            list(candidates),
            list(references),
            batch_size=self.batch_size,
            verbose=False,
        )
        values = [float(value) for value in f1.detach().cpu().tolist()]
        if len(values) != len(candidates):
            raise RuntimeError("BERTScore returned an unexpected number of pair scores")
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("BERTScore produced a non-finite answer score")
        return values

    def score_records(self, records: Sequence[Mapping[str, Any]]) -> dict[int, AnswerBERTScore]:
        """Score every non-empty released reference in one frozen batch."""

        candidates: list[str] = []
        references: list[str] = []
        locations: list[tuple[int, int]] = []
        record_references: dict[int, list[str]] = {}

        for record in records:
            row_index = int(record["row_index"])
            answer = str(record.get("predicted_answer") or "").strip()
            row_refs = [
                str(reference).strip()
                for reference in record.get("references", [])
                if str(reference or "").strip()
            ]
            record_references[row_index] = row_refs
            for reference_index, reference in enumerate(row_refs):
                candidates.append(answer)
                references.append(reference)
                locations.append((row_index, reference_index))

        scores_by_row: dict[int, list[tuple[float, int]]] = {
            row_index: [] for row_index in record_references
        }
        for (row_index, reference_index), score in zip(
            locations,
            self._score_pairs(candidates, references),
            strict=True,
        ):
            scores_by_row[row_index].append((score, reference_index))

        results: dict[int, AnswerBERTScore] = {}
        for row_index, row_refs in record_references.items():
            row_scores = scores_by_row[row_index]
            if row_scores:
                reward, selected_reference_index = max(row_scores, key=lambda item: item[0])
            else:
                reward, selected_reference_index = 0.0, -1
            results[row_index] = AnswerBERTScore(
                reward=float(reward),
                references_scored=len(row_refs),
                selected_reference_index=selected_reference_index,
            )
        return results
