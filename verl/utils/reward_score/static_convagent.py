"""Static ConvAgent-style reward utilities.

The released ConvAgent data may allow multiple terminal actions for a single
context (for example, both an answer and a clarification).  This module keeps
that supervision explicit instead of collapsing the label to an arbitrary
single action.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from typing import Any


STATIC_ACTIONS = frozenset({"answer", "clarify", "nonanswer"})
ANSWER_REFERENCE_SEPARATOR = "<|answer_split|>"


def normalize_text(value: Any) -> str:
    """Lowercase text and normalize punctuation and whitespace."""
    text = "" if value is None else str(value).lower()
    text = text.translate(str.maketrans({character: " " for character in string.punctuation}))
    return re.sub(r"\s+", " ", text).strip()


def token_set_f1(prediction: Any, reference: Any) -> float:
    """Set-token F1 used by the static benchmark utilities."""
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


def primary_answer_reference(references: Any) -> str:
    """Return the first released answer reference, if alternatives are encoded.

    This compatibility helper is used only by callers that explicitly request
    a canonical-reference analysis.  The shared baseline/Turn-PPO protocol
    uses :func:`token_set_f1_max_reference` instead.
    """
    return ("" if references is None else str(references)).split(
        ANSWER_REFERENCE_SEPARATOR, 1
    )[0].strip()


def token_set_f1_primary_reference(prediction: Any, references: Any) -> float:
    """Compute token-set F1 against the canonical first answer reference."""
    return token_set_f1(prediction, primary_answer_reference(references))


def token_set_f1_max_reference(prediction: Any, references: Any) -> float:
    """Return the best benchmark F1 across released answer references.

    Static conversational-search data stores alternate valid answers in a
    single string separated by ``<|answer_split|>``.  A policy still emits one
    answer, but receives credit for its best match to any released acceptable
    answer.  This is the shared reward and reporting rule for ConvAgent,
    ChatR1, and Turn-PPO.
    """
    if not prediction or not references:
        return 0.0
    return max(
        (token_set_f1(prediction, reference) for reference in str(references).split(ANSWER_REFERENCE_SEPARATOR)),
        default=0.0,
    )


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
    """Return every permissible terminal action for one static ConvAgent row."""
    dataset = str(data_source or "").strip().lower()
    actions = _candidate_actions(ground_truth)
    if actions:
        return actions
    # The released QReCC and CoRAL static rows are answer-only.
    if dataset in {"qrecc", "coral"}:
        return {"answer"}
    return set()


def static_convagent_answer_and_passage_ids(ground_truth: Any) -> tuple[str, list[str]]:
    """Select the answer candidate without discarding mixed-action rows."""
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


def static_convagent_answer_references(ground_truth: Any) -> list[str]:
    """Return every answer-only reference in source order.

    ConvAgent may expose a set of permissible terminal actions for the same
    context.  Its *outcome* reward is nevertheless an answer reward: a
    ``<clarify>`` or ``<nonanswer>`` prediction is evaluated only by the
    mixed-initiative action reward, never against clarification text.
    """
    if isinstance(ground_truth, Mapping):
        if "ground_truth" in ground_truth:
            return static_convagent_answer_references(ground_truth["ground_truth"])
        candidates: list[Mapping] = [ground_truth]
    elif isinstance(ground_truth, Sequence) and not isinstance(
        ground_truth, (str, bytes, bytearray)
    ):
        candidates = [candidate for candidate in ground_truth if isinstance(candidate, Mapping)]
    elif isinstance(ground_truth, str):
        return [ground_truth] if ground_truth.strip() else []
    else:
        return []

    references: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        action = str(candidate.get("action", "answer")).strip().lower()
        response = str(candidate.get("response", "") or "").strip()
        if action not in {"", "answer"} or not response:
            continue
        key = normalize_text(response)
        if key and key not in seen:
            seen.add(key)
            references.append(response)
    return references


def static_convagent_reference_string_and_passage_ids(ground_truth: Any) -> tuple[str, list[str]]:
    """Encode all released answer references and their answer evidence.

    The model still emits only one terminal ``<answer>``.  F1/BERTScore then
    score that one prediction against every released answer alternative and
    retain the maximum.  Clarification/non-answer candidates are deliberately
    excluded from this textual reference set; they remain action supervision.
    """
    if isinstance(ground_truth, Mapping):
        if "ground_truth" in ground_truth:
            return static_convagent_reference_string_and_passage_ids(ground_truth["ground_truth"])
        candidates: list[Mapping] = [ground_truth]
    elif isinstance(ground_truth, Sequence) and not isinstance(
        ground_truth, (str, bytes, bytearray)
    ):
        candidates = [candidate for candidate in ground_truth if isinstance(candidate, Mapping)]
    elif isinstance(ground_truth, str):
        return ground_truth.strip(), []
    else:
        return "", []

    references: list[str] = []
    reference_seen: set[str] = set()
    passage_ids: list[str] = []
    passage_seen: set[str] = set()
    for candidate in candidates:
        action = str(candidate.get("action", "answer")).strip().lower()
        response = str(candidate.get("response", "") or "").strip()
        if action not in {"", "answer"} or not response:
            continue
        normalized_response = normalize_text(response)
        if normalized_response and normalized_response not in reference_seen:
            references.append(response)
            reference_seen.add(normalized_response)
        raw_passages = candidate.get("passage_id", [])
        if isinstance(raw_passages, Sequence) and not isinstance(raw_passages, (str, bytes, bytearray)):
            values = raw_passages
        else:
            values = [] if raw_passages is None else [raw_passages]
        for value in values:
            passage_id = str(value or "").strip()
            if passage_id and passage_id not in passage_seen:
                passage_ids.append(passage_id)
                passage_seen.add(passage_id)
    return ANSWER_REFERENCE_SEPARATOR.join(references), passage_ids


def direct_evidence_coverage(
    gold_answer: Any,
    passages: Sequence[Mapping[str, Any]] | Sequence[str],
    *,
    short_answer_token_threshold: int = 4,
    concatenate_passages: bool = False,
) -> float:
    """Score whether a static ConvAgent search directly covers its answer.

    ``concatenate_passages=True`` reproduces ConvAgent's released reward
    implementation: concatenate the top-k passages returned for the search,
    then score the resulting evidence string against the answer.  The older
    static baseline used the maximum individual-passage F1 instead, so retain
    that behavior as the default for backward-compatible experiments.
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
        if concatenate_passages:
            return float(gold in " ".join(texts))
        return float(any(gold in text for text in texts))
    if concatenate_passages:
        return token_set_f1(" ".join(texts), gold)
    return max(token_set_f1(text, gold) for text in texts)


def monitor_plateau_reached(
    scores: Sequence[float],
    *,
    patience: int,
    min_delta: float,
    stability_window: int,
    stability_tolerance: float,
) -> bool:
    """Match InteractiveChat-R1's static-baseline stop criterion."""
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
