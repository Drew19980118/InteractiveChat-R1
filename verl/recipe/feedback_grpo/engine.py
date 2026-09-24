"""Two-round collection and grouped credit assignment, independent of GPU runtime.

An episode contains the search decisions and final action for ONE system
response. The first and revised responses are distinct episodes with the same
labels. A Generation stores exactly the tokens sampled by one model; none of
the other model's text or manually appended instructions is an optimization
target. The backend must not update either policy during collection.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Protocol

from .protocol import (
    Assessment, StaticExample, REANSWER_INSTRUCTION, assess_response,
    feedback_reward, group_advantages, user_feedback_prompt,
)


@dataclass
class Generation:
    prompt_ids: list[int]
    response_ids: list[int]
    text: str


@dataclass
class Observation:
    content: str
    passages: list[dict[str, str]]


@dataclass
class SystemRequest:
    example: StaticExample
    messages: list[dict[str, str]] | None = None


@dataclass
class Episode:
    example: StaticExample
    messages: list[dict[str, str]]
    generations: list[Generation] = field(default_factory=list)
    raw_response: str = ""
    assessment: Assessment | None = None
    passages: list[dict[str, str]] = field(default_factory=list)
    tool_calls: int = 0


@dataclass
class TrainItem:
    generation: Generation
    advantage: float
    weight: float
    group_id: str
    reward: float
    stage: str


class Backend(Protocol):
    def generate(self, role: str, messages: list[list[dict[str, str]]], *,
                 greedy: bool, initial: bool = False) -> list[Generation]: ...

    def search(self, queries: list[tuple[StaticExample, str]]) -> list[Observation]: ...


def _summary(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {f"{prefix}/count": 0.0}
    return {f"{prefix}/mean": fmean(values), f"{prefix}/min": min(values),
            f"{prefix}/max": max(values), f"{prefix}/count": float(len(values))}


def _escape(text: str) -> str:
    return text.replace("<|im_start|>", "<|im_start_escaped|>").replace("<|im_end|>", "<|im_end_escaped|>")


def _search_query(text: str) -> str | None:
    # The shared terminal parser intentionally rejects search-only turns.
    # Recognize this intermediate decision separately, requiring exactly one
    # thought followed by one nonempty search and no terminal action mixed in.
    from verl.utils.reward_score.info_gain import check_tags_balance
    if not check_tags_balance(text, allow_search=True, allow_clarify=True):
        return None
    if any(text.count(f"<{tag}>") != 1 for tag in ("think", "search")):
        return None
    if any(f"<{tag}>" in text for tag in ("answer", "clarify", "nonanswer", "tool_call", "code")):
        return None
    match = re.fullmatch(r"\s*<think>.*?</think>\s*<search>(.*?)</search>\s*", text, re.DOTALL)
    return match.group(1).strip() if match and match.group(1).strip() else None


def _feedback_messages(example: StaticExample, first: Episode) -> list[dict[str, str]]:
    return [{"role": "user", "content": user_feedback_prompt(example.history, example.query, first.assessment)}]


def revision_request(first: Episode, feedback: Generation) -> SystemRequest:
    messages = copy.deepcopy(first.messages)
    messages.append({"role": "user", "content": (
        "Feedback:\n" + _escape(feedback.text.strip()) + "\n\n" + REANSWER_INSTRUCTION
    )})
    return SystemRequest(first.example, messages)


def _add_episode(items: list[TrainItem], episode: Episode, *, advantage: float,
                 coefficient: float, group_id: str, stage: str) -> None:
    tokens = sum(len(g.response_ids) for g in episode.generations)
    if tokens < 1:
        raise RuntimeError("A sampled system episode has no generated tokens")
    for generation in episode.generations:
        items.append(TrainItem(generation, advantage,
                               coefficient * len(generation.response_ids) / tokens,
                               group_id, episode.assessment.reward, stage))


def validate_train_items(items: list[TrainItem]) -> None:
    if not items:
        return
    if not math.isclose(sum(item.weight for item in items), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Training coefficients must sum to one over the effective batch")
    for item in items:
        if not item.generation.response_ids or not item.generation.prompt_ids:
            raise ValueError("Empty generation/prompt in training data")
        if not all(math.isfinite(x) for x in (item.advantage, item.reward, item.weight)) or item.weight < 0:
            raise ValueError("Non-finite or negative training coefficient")


class TwoRoundCollector:
    def __init__(self, backend: Backend, *, n: int = 8, max_turns: int = 4,
                 first_retries: int = 2, trace_samples: int = 2):
        if n < 2 or max_turns < 1 or first_retries < 0 or trace_samples < 0:
            raise ValueError("Invalid collector settings")
        self.backend, self.n, self.max_turns = backend, n, max_turns
        self.first_retries, self.trace_samples = first_retries, trace_samples

    def _first_rollout_with_retries(self, requests: list[SystemRequest]) -> tuple[list[Episode], int]:
        """Sample one first response per request, retrying only malformed rows.

        Both training phases deliberately share this behavior: a feedback or
        second-round group cannot be formed from a malformed terminal first
        response.  Retries are independent fresh rollouts, never continuations
        of an invalid response.
        """
        first = self.system_rollout(requests)
        retries = 0
        for _ in range(self.first_retries):
            invalid = [index for index, episode in enumerate(first) if not episode.assessment.format_valid]
            if not invalid:
                break
            replacements = self.system_rollout([requests[index] for index in invalid])
            retries += len(invalid)
            for index, replacement in zip(invalid, replacements):
                first[index] = replacement
        return first, retries

    def system_rollout(self, requests: list[SystemRequest], *, greedy: bool = False) -> list[Episode]:
        episodes = [Episode(request.example, copy.deepcopy(request.messages or request.example.prompt))
                    for request in requests]
        active = list(range(len(episodes)))
        for turn in range(self.max_turns):
            if not active:
                break
            # initial=True is only the original static context. Revised prompts
            # and tool observations may use the remaining 8192-token window.
            initial = turn == 0 and all(requests[index].messages is None for index in active)
            generated = self.backend.generate("system", [episodes[i].messages for i in active],
                                              greedy=greedy, initial=initial)
            if len(generated) != len(active):
                raise RuntimeError("System generation cardinality mismatch")
            searches: list[tuple[int, str]] = []
            for index, generation in zip(active, generated):
                episode = episodes[index]
                episode.generations.append(generation)
                episode.raw_response = generation.text
                episode.messages.append({"role": "assistant", "content": generation.text})
                episode.assessment = assess_response(generation.text, episode.example.ground_truth, episode.example.data_source)
                query = _search_query(generation.text)
                # A search without a later terminal action is invalid and earns
                # the usual action penalty, even when its own grammar is legal.
                if query is not None and turn + 1 < self.max_turns:
                    searches.append((index, query))
            observations = self.backend.search([(episodes[i].example, query) for i, query in searches]) if searches else []
            if len(observations) != len(searches):
                raise RuntimeError("Search observation cardinality mismatch")
            for (index, _), observation in zip(searches, observations):
                episode = episodes[index]
                if episode.tool_calls == 0:
                    episode.passages = observation.passages[:3]
                episode.tool_calls += 1
                episode.messages.append({"role": "user", "content": "<information>" + _escape(observation.content) + "</information>"})
            active = [index for index, _ in searches]
        return episodes

    def user_batch(self, examples: list[StaticExample], batch_id: str) -> tuple[list[TrainItem], dict[str, float], list[dict]]:
        first, retries = self._first_rollout_with_retries([SystemRequest(example) for example in examples])
        eligible = [i for i, episode in enumerate(first) if episode.assessment.format_valid]
        feedback_inputs = [_feedback_messages(examples[i], first[i]) for i in eligible for _ in range(self.n)]
        feedbacks = self.backend.generate("user", feedback_inputs, greedy=False) if feedback_inputs else []
        if len(feedbacks) != len(feedback_inputs):
            raise RuntimeError("Feedback cardinality mismatch")
        seconds = self.system_rollout([
            revision_request(first[i], feedbacks[k * self.n + j])
            for k, i in enumerate(eligible) for j in range(self.n)
        ])
        groups = []
        invalid_seconds, tiny_groups, zero_groups = 0, 0, 0
        traces: list[dict] = []
        for k, i in enumerate(eligible):
            valid = [j for j in range(self.n) if seconds[k * self.n + j].assessment.format_valid]
            invalid_seconds += self.n - len(valid)
            if len(valid) < 2:
                tiny_groups += 1
                continue
            rewards = [feedback_reward(first[i].assessment, seconds[k * self.n + j].assessment) for j in valid]
            advantages = group_advantages(rewards)
            zero_groups += int(max(rewards) == min(rewards))
            groups.append((k, i, valid, rewards, advantages))
            if len(traces) < self.trace_samples:
                traces.append({"source_id": examples[i].uid, "first": first[i].raw_response,
                               "first_reward": first[i].assessment.reward,
                               "feedback_group": [{"feedback": feedbacks[k*self.n+j].text,
                                   "second": seconds[k*self.n+j].raw_response,
                                   "reward": r, "advantage": a} for j, r, a in zip(valid, rewards, advantages)]})
        items = []
        for k, i, valid, rewards, advantages in groups:
            for j, reward, advantage in zip(valid, rewards, advantages):
                items.append(TrainItem(feedbacks[k * self.n + j], advantage,
                                       1.0 / (len(groups) * len(valid)),
                                       f"{batch_id}:user:{i}:{examples[i].uid}", reward, "feedback"))
        validate_train_items(items)
        values = [item.reward for item in items]
        metrics = {**_summary("user/reward", values),
                   "user/first_retry_attempts": float(retries),
                   "user/invalid_first_groups": float(len(examples) - len(eligible)),
                   "user/invalid_second_samples": float(invalid_seconds),
                   "user/too_small_groups": float(tiny_groups),
                   "user/valid_groups": float(len(groups)),
                   "user/zero_variance_group_rate": zero_groups / max(1, len(groups)),
                   "user/positive_improvement_rate": sum(v > 0 for v in values) / max(1, len(values)),
                   "user/negative_improvement_rate": sum(v < 0 for v in values) / max(1, len(values)),
                   "user/empty_feedback_rate": sum(not f.text.strip() for f in feedbacks) / max(1, len(feedbacks))}
        return items, metrics, traces

    def system_batch(self, examples: list[StaticExample], batch_id: str) -> tuple[list[TrainItem], dict[str, float], list[dict]]:
        """Collect one first response and one second-round GRPO group per source.

        The first response and learned-user feedback are environment/context
        samples for this phase.  Only the ``n`` revised responses sharing that
        exact prompt receive a System-policy gradient.  This preserves a valid
        same-prompt GRPO group while avoiding the previous n + n^2 expansion.
        """
        first, retries = self._first_rollout_with_retries([SystemRequest(example) for example in examples])
        eligible = [i for i, episode in enumerate(first) if episode.assessment.format_valid]
        feedbacks = self.backend.generate("user", [_feedback_messages(first[i].example, first[i]) for i in eligible], greedy=False) if eligible else []
        if len(feedbacks) != len(eligible):
            raise RuntimeError("Feedback cardinality mismatch")
        second = self.system_rollout([revision_request(first[i], feedbacks[k]) for k, i in enumerate(eligible) for _ in range(self.n)])
        items: list[TrainItem] = []
        groups = []
        invalid_seconds, tiny_groups, zero_groups = 0, 0, 0
        traces = []
        for k, i in enumerate(eligible):
            group = second[k*self.n:(k+1)*self.n]
            valid = [j for j, episode in enumerate(group) if episode.assessment.format_valid]
            invalid_seconds += self.n - len(valid)
            if len(valid) < 2:
                tiny_groups += 1
                continue
            rewards = [group[j].assessment.reward for j in valid]
            advantages = group_advantages(rewards)
            zero_groups += int(max(rewards) == min(rewards))
            groups.append((k, i, group, valid, rewards, advantages))

        for k, i, group, valid, rewards, advantages in groups:
            for j, advantage in zip(valid, advantages):
                _add_episode(items, group[j], advantage=advantage,
                             coefficient=1.0 / (len(groups) * len(valid)),
                             group_id=f"{batch_id}:second:{i}:{first[i].example.uid}", stage="second")
            if len(traces) < self.trace_samples:
                traces.append({"source_id": first[i].example.uid, "branch": i,
                    "first": first[i].raw_response, "feedback": feedbacks[k].text,
                    "second_group": [{"response": group[j].raw_response, "reward": reward, "advantage": advantage}
                                     for j, reward, advantage in zip(valid, rewards, advantages)]})
        if items:
            validate_train_items(items)
        metrics = {**_summary("system/round1/reward", [e.assessment.reward for e in first]),
                   **_summary("system/round2/reward", [e.assessment.reward for e in second]),
                   "system/round1/format_success_rate": len(eligible) / max(1, len(first)),
                   "system/round2/format_success_rate": sum(e.assessment.format_valid for e in second) / max(1, len(second)),
                   "system/round1/loss_weight": 0.0, "system/round2/loss_weight": float(bool(groups)),
                   "system/first_retry_attempts": float(retries),
                   "system/invalid_first_groups": float(len(examples) - len(eligible)),
                   "system/invalid_second_samples": float(invalid_seconds),
                   "system/too_small_groups": float(tiny_groups),
                   "system/valid_groups": float(len(groups)),
                   "system/round2/zero_variance_group_rate": zero_groups / max(1, len(groups)),
                   "system/round2/groups": float(len(groups))}
        return items, metrics, traces

    def evaluate_batch(self, examples: list[StaticExample]) -> tuple[list[Episode], list[Episode], list[str | None]]:
        # Evaluation never drops a source row or retries the first response.
        first = self.system_rollout([SystemRequest(example) for example in examples], greedy=True)
        eligible = [i for i, episode in enumerate(first) if episode.assessment.format_valid]
        feedbacks = self.backend.generate("user", [_feedback_messages(examples[i], first[i]) for i in eligible], greedy=True) if eligible else []
        if len(feedbacks) != len(eligible):
            raise RuntimeError("Feedback cardinality mismatch")
        revised = self.system_rollout([revision_request(first[i], feedback) for i, feedback in zip(eligible, feedbacks)], greedy=True)
        final = list(first)
        feedback_text: list[str | None] = [None] * len(examples)
        for i, episode, feedback in zip(eligible, revised, feedbacks):
            # Strict revised-response selection: do not fall back to a better
            # first answer if the revised response is malformed or worse.
            final[i] = episode
            feedback_text[i] = feedback.text
        return first, final, feedback_text
