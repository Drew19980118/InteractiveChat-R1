"""Static ChatR1-style supervision utilities.

ChatR1's released static Parquets retain one or more answer candidates per
turn.  Each candidate may also contain a human standalone query rewrite.  The
baseline objective is therefore deliberately different from the interactive
user-centric objective:

* outcome reward: maximum word-level answer F1 across answer candidates;
* intermediate reward: maximum word-level F1 between an executed search
  query and the human rewrite, credited to the best query in that trajectory.

The helpers accept Arrow/NumPy objects as well as ordinary Python mappings so
the same logic works before and after Parquet collation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .static_convagent import token_set_f1


def _as_python(value: Any) -> Any:
    """Convert Arrow/NumPy scalars or arrays without changing plain objects."""
    if hasattr(value, "as_py"):
        return value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return value.tolist()
    return value


def _candidates(ground_truth: Any) -> list[Mapping[str, Any]]:
    """Extract the released ChatR1 candidate list from nested reward metadata."""
    value = _as_python(ground_truth)
    if isinstance(value, Mapping):
        if "ground_truth" in value:
            return _candidates(value["ground_truth"])
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [candidate for candidate in value if isinstance(candidate, Mapping)]
    return []


def _answer_candidates(ground_truth: Any) -> list[Mapping[str, Any]]:
    candidates = _candidates(ground_truth)
    # QReCC omits ``action`` because every row is answerable.  InsCiT retains
    # the explicit answer tag after ChatR1's clarification-turn filtering.
    return [
        candidate
        for candidate in candidates
        if str(candidate.get("action", "answer")).strip().lower() in {"", "answer"}
    ]


def static_chatr1_answer_references(ground_truth: Any) -> list[str]:
    """Return all non-empty answer references in dataset order, de-duplicated."""
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
    """Encode all references using the scorer's existing multi-answer marker."""
    return "<|answer_split|>".join(static_chatr1_answer_references(ground_truth))


def static_chatr1_primary_answer_and_passage_ids(ground_truth: Any) -> tuple[str, list[str]]:
    """Return a primary answer for pseudo scoring plus the union of gold passages."""
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
    """Return the first human standalone rewrite supplied for the source turn."""
    for candidate in _answer_candidates(ground_truth):
        rewrite = str(candidate.get("rewrite", "") or "").strip()
        if rewrite:
            return rewrite
    return ""


def static_chatr1_intent_rewards(queries: Sequence[str | None], rewrite: str) -> list[float]:
    """Credit ChatR1's max query--rewrite score to the best executed query.

    The trace-level paper reward is ``max_q F1(q, rewrite)``.  Giving that
    value to only the maximizing action preserves the exact trace reward while
    avoiding duplicate credit when a policy issues several searches.
    """
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
