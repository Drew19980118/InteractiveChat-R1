"""Ground-truth normalization shared by ConvAgent data and IGPO rewards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .static_convagent import static_convagent_allowed_actions, static_convagent_answer_and_passage_ids


def _normalize_passage_ids(value: Any) -> list[str]:
    """Return a JSON/Parquet-friendly list for one answer candidate."""
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _candidate_answer_and_passages(candidate: Mapping) -> tuple[str, list[str]]:
    """Extract one candidate's text and its supporting passage identifiers."""
    response = candidate.get("response", "")
    answer = str(response) if response is not None else ""
    return answer, _normalize_passage_ids(candidate.get("passage_id"))


def _normalized_data_source(data_source: Any) -> str:
    return str(data_source or "").strip().lower()


def select_expected_action(ground_truth: Any, data_source: Any = None) -> str | None:
    """Return the terminal action expected by an action-labelled dataset.

    Every dataset participates in terminal-action supervision. QReCC and
    CoRAL always expect the only supported terminal action, ``answer``.  The
    action-labelled InSCIt/TopiOCQA parquets store their selected label
    explicitly; this helper also understands the original ConvAgent candidate
    list so conversion and runtime validation share one rule. ``clarify``
    deliberately has no mapping: this project does not expose a clarification
    action.
    """
    dataset = _normalized_data_source(data_source)
    if dataset in {"qrecc", "coral"}:
        return "answer"
    if dataset not in {"inscit", "topiocqa"}:
        return None

    if isinstance(ground_truth, Mapping):
        explicit = str(ground_truth.get("expected_action", "")).strip().lower()
        if explicit in {"answer", "nonanswer"}:
            return explicit
        if "ground_truth" in ground_truth:
            return select_expected_action(ground_truth["ground_truth"], data_source=dataset)
        action = str(ground_truth.get("action", "")).strip().lower()
        if action == "answer":
            return "answer"
        if action in {"nonanswer", "non_answer"}:
            return "nonanswer"
        return None

    if isinstance(ground_truth, Sequence) and not isinstance(ground_truth, (str, bytes, bytearray)):
        candidates = [candidate for candidate in ground_truth if isinstance(candidate, Mapping)]
        actions = {str(candidate.get("action", "")).strip().lower() for candidate in candidates}
        if actions & {"nonanswer", "non_answer"}:
            return "nonanswer"
        if "answer" in actions:
            return "answer"
    return None


def _has_disqualifying_action(candidates: list[Mapping], data_source: Any) -> bool:
    """Apply dataset-specific empty-target rules before choosing an answer."""
    actions = {str(candidate.get("action", "")).strip().lower() for candidate in candidates}
    dataset = _normalized_data_source(data_source)
    if dataset == "inscit":
        # A clarification/non-answer candidate makes the full InSCIt example
        # unsupervised, even if another candidate happens to be an answer.
        return bool(actions & {"clarify", "nonanswer", "non_answer"})
    if dataset == "topiocqa":
        return bool(actions & {"nonanswer", "non_answer"})
    return False


def select_answer_ground_truth_with_passage_ids(
    ground_truth: Any,
    data_source: Any = None,
) -> tuple[str, list[str]]:
    """Return IGPO's selected answer and the selected candidate's passages.

    The answer-selection rule is shared by conversion, rewards, and validation
    export: use the first explicit ``action == 'answer'`` candidate for
    action-labelled datasets, otherwise use the first candidate
    (CoRAL/QReCC). A sample with no answer candidate deliberately returns an
    empty answer and no passages.

    InSCIt samples containing any ``clarify`` or ``nonanswer`` candidate and
    TopiOCQA samples containing any ``nonanswer`` candidate are deliberately
    represented as empty targets, as requested by the dataset protocol.

    For TopiOCQA the upstream ``passage_id`` field contains the supporting
    passage *text*.  It is intentionally preserved verbatim here; validation
    uses the same representation for retrieved passages in that dataset.
    """
    if isinstance(ground_truth, str):
        return ground_truth, []

    if isinstance(ground_truth, Mapping):
        if "action" in ground_truth:
            action = str(ground_truth.get("action", "")).strip().lower()
            return _candidate_answer_and_passages(ground_truth) if action == "answer" else ("", [])
        if "ground_truth" in ground_truth:
            answer, nested_passage_ids = select_answer_ground_truth_with_passage_ids(
                ground_truth["ground_truth"], data_source=data_source
            )
            # Converted IGPO parquets store the selected passages on the outer
            # reward-model mapping, alongside a flat ground-truth string.
            if "ground_truth_passage_ids" in ground_truth:
                return answer, _normalize_passage_ids(ground_truth["ground_truth_passage_ids"])
            return answer, nested_passage_ids
        # A single unlabelled candidate follows the CoRAL/QReCC convention.
        if "response" in ground_truth:
            return _candidate_answer_and_passages(ground_truth)
        return "", []

    if isinstance(ground_truth, Sequence) and not isinstance(ground_truth, (bytes, bytearray)):
        candidates = [candidate for candidate in ground_truth if isinstance(candidate, Mapping)]
        if not candidates:
            return "", []

        if _has_disqualifying_action(candidates, data_source):
            return "", []

        # Action-bearing variants: only an explicit answer is supervised.
        if any("action" in candidate for candidate in candidates):
            for candidate in candidates:
                if str(candidate.get("action", "")).strip().lower() == "answer":
                    return _candidate_answer_and_passages(candidate)
            return "", []

        # CoRAL/QReCC candidates have no action field; all are answers.
        return _candidate_answer_and_passages(candidates[0])

    return "", []


def select_answer_ground_truth(ground_truth: Any, data_source: Any = None) -> str:
    """Return the first ConvAgent ground truth whose action is ``answer``.

    IGPO's outcome and information-gain rewards require one textual reference.
    ConvAgent's InSCIt/TopiOCQA variants store an action for each candidate.
    We deliberately use the first answer candidate (dataset order is the
    requested tie-breaker). A sample with only ``clarify``/``nonanswer``
    candidates has no supervised answer and is represented by ``""``;
    callers must assign it zero reward.

    ConvAgent's CoRAL/QReCC variants omit ``action`` because every candidate is
    an answer.  For those, use the first candidate response.

    Existing datasets already store a string and pass through unchanged.
    """
    return select_answer_ground_truth_with_passage_ids(ground_truth, data_source=data_source)[0]


def select_static_convagent_answer_ground_truth_with_passage_ids(
    ground_truth: Any,
    data_source: Any = None,
) -> tuple[str, list[str]]:
    """Select the static ConvAgent answer target.

    Unlike the interactive conversion, static ConvAgent rows with both answer
    and clarification candidates remain answer-supervised.  The action reward
    separately recognises every candidate action as valid.
    """
    del data_source  # Kept for the same call signature as the interactive helper.
    return static_convagent_answer_and_passage_ids(ground_truth)


def select_static_convagent_expected_actions(ground_truth: Any, data_source: Any = None) -> set[str]:
    """Return the permissible static ConvAgent terminal actions for one row."""
    return static_convagent_allowed_actions(ground_truth, data_source=data_source)
