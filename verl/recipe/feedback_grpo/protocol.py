"""Reward and grouping rules for alternating feedback/system GRPO.

This module contains no model or tensor operations. A feedback group has one
shared first response, several feedback samples, and one second response per
feedback. A system group instead contains responses to exactly one fixed
conditioning prompt. The coordinator is responsible for sampling those groups
and for keeping the other policy frozen during an update phase.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any, Sequence


ACTION_REWARD_WEIGHT = 0.5
CORRECT_ACTION_SCORE = 1.0
INCORRECT_ACTION_SCORE = -0.5

REANSWER_INSTRUCTION = (
    "Please respond to the current query again, taking the feedback above "
    "into account. Follow the same action format as before: "
    "use <think>...</think> followed by one allowed action per decision. "
    "Use <search>...</search> when needed, and finish with exactly one of "
    "<answer>...</answer>, <clarify>...</clarify>, or "
    "<nonanswer></nonanswer>."
)


@dataclass(frozen=True)
class StaticExample:
    """One original static query; both response rounds share its supervision."""

    uid: str
    prompt: list[dict[str, str]]
    history: str
    query: str
    ground_truth: Any
    data_source: str = "inscit"


@dataclass(frozen=True)
class Assessment:
    """A terminal response assessment; action_score is the *unweighted* MIA."""

    action: str
    format_valid: bool
    content: str
    f1: float
    action_score: float
    reward: float


@dataclass(frozen=True)
class FeedbackGroup:
    """Only valid second responses enter feedback GRPO normalization.

    Indices refer to the original feedback/second-response lists. Zero rewards
    remain in the group; filtering by reward truthiness would change GRPO.
    A skipped group has no rewards or advantages to accidentally optimize.
    """

    retained_indices: tuple[int, ...]
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    skipped: bool
    reason: str | None = None

    @property
    def zero_variance(self) -> bool:
        return bool(self.rewards) and max(self.rewards) == min(self.rewards)


def _python_ground_truth(value: Any) -> Any:
    """Accept the JSON and Arrow/NumPy wrappers used by static Parquet rows."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if isinstance(value, str) and value.lstrip().startswith(("[", "{")):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            pass
        else:
            if isinstance(parsed, (list, dict)):
                value = parsed
    if isinstance(value, dict):
        return {key: _python_ground_truth(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_python_ground_truth(item) for item in value]
    return value


def assess_response(
    response: str,
    ground_truth: Any,
    data_source: str = "inscit",
) -> Assessment:
    """Score one response using the static ConvAgent terminal-action protocol.

    Answer F1 is maximized over *answer* references only. Clarification and
    nonanswer text never receive F1. The total is F1 + 0.5 * MIA; malformed
    responses receive F1=0 and MIA=-0.5. No retrieval/format component is added.
    """
    # Lazy imports keep this module's data/grouping API tensor-independent.
    from verl.utils.reward_score.info_gain import extract_terminal_action
    from verl.utils.reward_score.static_convagent import (
        ANSWER_REFERENCE_SEPARATOR,
        static_convagent_allowed_actions,
        static_convagent_reference_string_and_passage_ids,
        token_set_f1_max_reference,
    )

    action, valid = extract_terminal_action(
        response, allow_clarify=True, allow_search=True
    )
    # A legal search is an intermediate decision, never a terminal answer.
    # Keep this explicit even if the shared parser later recognizes searches.
    if action not in {"answer", "clarify", "nonanswer"}:
        action, valid = "invalid", False
    final_turn = response.rsplit("\n<|im_start|>assistant\n", 1)[-1]
    # This separator encodes GT alternatives, never multiple policy answers.
    if ANSWER_REFERENCE_SEPARATOR in final_turn:
        action, valid = "invalid", False
    content = ""
    if valid:
        match = re.search(rf"<{action}>(.*?)</{action}>", final_turn, re.DOTALL)
        content = match.group(1).strip() if match else ""

    ground_truth = _python_ground_truth(ground_truth)
    permitted = static_convagent_allowed_actions(ground_truth, data_source)
    action_score = (
        CORRECT_ACTION_SCORE if valid and action in permitted else INCORRECT_ACTION_SCORE
    )
    f1 = 0.0
    if valid and action == "answer":
        references, _ = static_convagent_reference_string_and_passage_ids(ground_truth)
        f1 = token_set_f1_max_reference(content, references)
    return Assessment(
        action=action,
        format_valid=valid,
        content=content,
        f1=float(f1),
        action_score=action_score,
        reward=float(f1 + ACTION_REWARD_WEIGHT * action_score),
    )


def feedback_reward(first: Assessment, second: Assessment) -> float:
    """Conditional improvement reward specified for the feedback policy.

    If both actions are answer, use the full system-reward difference.
    Otherwise, use only the weighted action difference. Malformed outputs
    must be filtered by the caller instead of being scored as user feedback.
    """
    if not first.format_valid or not second.format_valid:
        raise ValueError("Feedback reward requires two format-valid terminal responses.")
    if first.action == second.action == "answer":
        return float(second.reward - first.reward)
    return float(ACTION_REWARD_WEIGHT * (second.action_score - first.action_score))


def group_advantages(scores: Sequence[float], epsilon: float = 1e-6) -> list[float]:
    """Normalize within one prompt group using population standard deviation.

    Absolute signs of rewards do not determine signs of advantages. Even an
    all-negative group favors its relatively better members. Equal scores
    yield zero advantages. Callers must not combine different second prompts.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("GRPO rewards must be finite.")
    if not values:
        return []
    mean = fmean(values)
    std = pstdev(values)
    if std == 0.0:
        return [0.0] * len(values)
    return [(value - mean) / (std + epsilon) for value in values]


def filter_feedback_group(
    first: Assessment,
    seconds: Sequence[Assessment],
    epsilon: float = 1e-6,
) -> FeedbackGroup:
    """Drop malformed second branches; skip when fewer than two remain."""
    if not first.format_valid:
        return FeedbackGroup((), (), (), True, "invalid_first_response")
    indices = tuple(index for index, second in enumerate(seconds) if second.format_valid)
    if len(indices) < 2:
        return FeedbackGroup(indices, (), (), True, "fewer_than_two_valid_feedbacks")
    rewards = tuple(feedback_reward(first, seconds[index]) for index in indices)
    advantages = tuple(group_advantages(rewards, epsilon=epsilon))
    return FeedbackGroup(indices, rewards, advantages, False)


def user_feedback_prompt(history: str, query: str, first: Assessment) -> str:
    """Expose only the terminal content to user; never reasoning, tools, or GT."""
    if not first.format_valid:
        raise ValueError("Cannot request feedback for a malformed first response.")
    content = "No answer was provided." if first.action == "nonanswer" else first.content
    return (
        "You are simulating a user in the conversation below.\n"
        "Give feedback on the system's response to your current query.\n"
        "Your feedback may describe what is helpful and what should be improved.\n\n"
        f"Conversation History:\n{history}\n\n"
        f"Your Current Query:\n{query}\n\n"
        f"System Response:\n{content}\n\n"
        "Feedback:"
    )
