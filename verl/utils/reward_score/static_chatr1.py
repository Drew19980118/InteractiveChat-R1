"""Static ChatR1-style supervision helpers shared with InteractiveChat-R1.

The released ChatR1 Parquets retain one or more answer candidates per turn.
Each answer candidate can also contain the human standalone-query rewrite used
by the original method's intermediate query reward.  Validation needs the
same candidate normalization even when that training-only reward is disabled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .static_convagent import token_set_f1


def _as_python(value: Any) -> Any:
    """Convert Arrow/NumPy containers without changing ordinary Python data."""
    if hasattr(value, "as_py"):
        return value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return value.tolist()
    return value


def _candidates(ground_truth: Any) -> list[Mapping[str, Any]]:
    value = _as_python(ground_truth)
    if isinstance(value, Mapping):
        if "ground_truth" in value:
            return _candidates(value["ground_truth"])
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [candidate for candidate in value if isinstance(candidate, Mapping)]
    return []


def _answer_candidates(ground_truth: Any) -> list[Mapping[str, Any]]:
    # QReCC omits ``action`` because all rows are answerable.  InsCiT retains
    # it after ChatR1's clarification-turn filtering.
    return [
        candidate
        for candidate in _candidates(ground_truth)
        if str(candidate.get("action", "answer")).strip().lower() in {"", "answer"}
    ]


def static_chatr1_answer_candidates(ground_truth: Any) -> list[dict[str, Any]]:
    """Return copied, non-empty released answer candidates.

    The max-reference training view keeps these candidates together in one
    source row; the older expansion utility may still use this helper when an
    explicitly per-reference diagnostic dataset is desired.
    """
    candidates: list[dict[str, Any]] = []
    for candidate in _answer_candidates(ground_truth):
        response = str(candidate.get("response", "") or "").strip()
        if not response:
            continue
        copied = dict(candidate)
        copied["response"] = response
        candidates.append(copied)
    return candidates


def static_chatr1_answer_references(ground_truth: Any) -> list[str]:
    """Return released answer references in dataset order, de-duplicated."""
    references: list[str] = []
    seen: set[str] = set()
    for candidate in _answer_candidates(ground_truth):
        response = str(candidate.get("response", "") or "").strip()
        if response and response not in seen:
            references.append(response)
            seen.add(response)
    if not references and isinstance(_as_python(ground_truth), str):
        response = str(ground_truth).strip()
        if response:
            references.append(response)
    return references


def static_chatr1_reference_string(ground_truth: Any) -> str:
    """Encode references using the evaluator's established multi-answer tag."""
    return "<|answer_split|>".join(static_chatr1_answer_references(ground_truth))


def static_chatr1_primary_answer_and_passage_ids(ground_truth: Any) -> tuple[str, list[str]]:
    """Return a primary answer and the union of answer-candidate passages."""
    references = static_chatr1_answer_references(ground_truth)
    passage_ids: list[str] = []
    seen: set[str] = set()
    for candidate in _answer_candidates(ground_truth):
        value = _as_python(candidate.get("passage_id", []))
        values = (
            value
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
            else [value]
        )
        for passage_id in values:
            if passage_id is None:
                continue
            normalized = str(passage_id)
            if normalized and normalized not in seen:
                passage_ids.append(normalized)
                seen.add(normalized)
    return (references[0] if references else ""), passage_ids


def static_chatr1_rewrite(ground_truth: Any) -> str:
    """Return the first human standalone rewrite for a source turn."""
    for candidate in _answer_candidates(ground_truth):
        rewrite = str(candidate.get("rewrite", "") or "").strip()
        if rewrite:
            return rewrite
    return ""


def static_chatr1_intent_rewards(queries: Sequence[str | None], rewrite: str) -> list[float]:
    """Credit max query--rewrite F1 to the matching executed search action."""
    rewards = [0.0] * len(queries)
    if not rewrite:
        return rewards
    best_index = -1
    best_score = 0.0
    for index, query in enumerate(queries):
        score = token_set_f1(query or "", rewrite)
        if score > best_score:
            best_index, best_score = index, score
    if best_index >= 0:
        rewards[best_index] = float(best_score)
    return rewards
