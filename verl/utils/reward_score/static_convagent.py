"""Static ConvAgent-style reward utilities.

The original ConvAgent training data can provide several permissible terminal
actions for one context (for example, both an answer and a clarification).
This module keeps that supervision explicit rather than silently collapsing it
to one arbitrary label.  It also implements the paper-style direct evidence
coverage signal used after every executed search action.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from typing import Any


STATIC_ACTIONS = frozenset({"answer", "clarify", "nonanswer"})


def normalize_text(value: Any) -> str:
    """Lowercase text and normalize punctuation/whitespace for overlap tests."""
    text = "" if value is None else str(value).lower()
    text = text.translate(str.maketrans({character: " " for character in string.punctuation}))
    return re.sub(r"\s+", " ", text).strip()


def token_set_f1(prediction: Any, reference: Any) -> float:
    """Set-token F1 matching the rule reward used by the static benchmark."""
    prediction_tokens = set(normalize_text(prediction).split())
    reference_tokens = set(normalize_text(reference).split())
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = len(prediction_tokens & reference_tokens)
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _candidate_actions(ground_truth: Any) -> set[str]:
    if isinstance(ground_truth, Mapping):
        if "ground_truth" in ground_truth:
            return _candidate_actions(ground_truth["ground_truth"])
        action = str(ground_truth.get("action", "")).strip().lower()
        return {action} if action in STATIC_ACTIONS else set()
    if isinstance(ground_truth, Sequence) and not isinstance(ground_truth, (str, bytes, bytearray)):
        actions: set[str] = set()
        for candidate in ground_truth:
            if isinstance(candidate, Mapping):
                action = str(candidate.get("action", "")).strip().lower()
                if action in STATIC_ACTIONS:
                    actions.add(action)
        return actions
    return set()


def static_convagent_allowed_actions(ground_truth: Any, data_source: Any = None) -> set[str]:
    """Return all actions permitted by one static ConvAgent label.

    QReCC and CoRAL omit action labels because every target is answerable, so
    they retain the original implicit ``answer`` action.  For InsCiT and
    TopiOCQA, candidate labels are treated as a set of valid actions.
    """
    dataset = str(data_source or "").strip().lower()
    actions = _candidate_actions(ground_truth)
    if actions:
        return actions
    if dataset in {"qrecc", "coral"}:
        return {"answer"}
    return set()


def static_convagent_answer_and_passage_ids(ground_truth: Any) -> tuple[str, list[str]]:
    """Select the first answer candidate without discarding mixed-action rows."""
    if isinstance(ground_truth, Mapping):
        if "ground_truth" in ground_truth:
            return static_convagent_answer_and_passage_ids(ground_truth["ground_truth"])
        candidates: list[Mapping] = [ground_truth]
    elif isinstance(ground_truth, Sequence) and not isinstance(ground_truth, (str, bytes, bytearray)):
        candidates = [candidate for candidate in ground_truth if isinstance(candidate, Mapping)]
    elif isinstance(ground_truth, str):
        return ground_truth, []
    else:
        return "", []

    for candidate in candidates:
        action = str(candidate.get("action", "answer")).strip().lower()
        if action in {"", "answer"}:
            response = "" if candidate.get("response") is None else str(candidate.get("response"))
            passage_ids = candidate.get("passage_id", [])
            if isinstance(passage_ids, Sequence) and not isinstance(passage_ids, (str, bytes, bytearray)):
                return response, [str(value) for value in passage_ids if value is not None]
            return response, [] if passage_ids is None else [str(passage_ids)]
    return "", []


def direct_evidence_coverage(
    gold_answer: Any,
    passages: Sequence[Mapping[str, Any]] | Sequence[str],
    *,
    short_answer_token_threshold: int = 4,
) -> float:
    """Compute ConvAgent-style direct coverage of a gold answer by top-k text.

    Long references use the highest token-set F1 over the retrieved passages.
    For short factoid answers, the signal is exact normalized phrase coverage,
    which avoids rewarding a passage merely because it shares a common word.
    """
    gold = normalize_text(gold_answer)
    if not gold:
        return 0.0

    texts: list[str] = []
    for passage in passages:
        if isinstance(passage, Mapping):
            text = passage.get("passage_text", passage.get("quick_summary", ""))
        else:
            text = passage
        normalized = normalize_text(text)
        if normalized:
            texts.append(normalized)
    if not texts:
        return 0.0

    if len(gold.split()) <= short_answer_token_threshold:
        return float(any(gold in text for text in texts))
    return max(token_set_f1(text, gold) for text in texts)


def monitor_plateau_reached(
    scores: Sequence[float],
    *,
    patience: int,
    min_delta: float,
    stability_window: int,
    stability_tolerance: float,
) -> bool:
    """Return whether periodic validation has both plateaued and stabilized.

    ``scores`` contains one scalar from each *monitor* validation, not every
    optimizer update.  A stop is allowed only after (1) the most recent
    ``patience`` checks have not improved on the preceding best by
    ``min_delta`` and (2) the most recent window has a narrow range.  Keeping
    both conditions prevents an early stop on a temporary downward spike.
    """
    if patience < 1 or stability_window < 2:
        raise ValueError("patience must be >= 1 and stability_window must be >= 2")
    required = max(patience + 1, stability_window + 1)
    if len(scores) < required:
        return False

    recent = [float(value) for value in scores[-patience:]]
    historical_best = max(float(value) for value in scores[:-patience])
    no_recent_improvement = max(recent) <= historical_best + float(min_delta)
    stable_window = [float(value) for value in scores[-stability_window:]]
    is_stable = max(stable_window) - min(stable_window) <= float(stability_tolerance)
    return bool(no_recent_improvement and is_stable)
