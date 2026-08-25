"""Online simulated-user dialogue rollout for sparse multi-turn GRPO.

This module deliberately leaves :mod:`scrl.llm_agent.generation` untouched.
The legacy IGPO manager remains the default; this manager is selected only
when ``algorithm.simulated_user_enabled=true``.  It serializes policy actions
and environmental messages in the normal ChatML format so the existing actor
token mask continues to train assistant tokens only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch

from scrl.llm_agent.generation import LLMGenerationManager
from scrl.llm_agent.simulated_user import (
    AgentAction,
    RETRY_SYSTEM_PROMPT,
    UserSimulatorClient,
    build_simulated_user_system_prompt,
    keep_first_complete_action,
    normalize_text,
    parse_agent_action,
    parse_dialogue_payload,
)
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length


_COMPONENTS = (
    "action",
    "answer_f1",
    "evidence_utility",
    "search_efficiency",
    "uci",
    "clarity",
    "patience",
    "format",
    "clarify_f1",
)

# These two modes keep the task-control and user-satisfaction channels used by
# the proposed method.  ``uci_replaced_by_answer_f1`` changes exactly one
# answer-quality channel, rather than accidentally turning the comparison into
# a broad "F1 only" reward ablation.
_FULL_AUXILIARY_CHANNEL_MODES = frozenset({"full", "uci_replaced_by_answer_f1"})


def _token_f1(prediction: str, reference: str) -> float:
    """Small, deterministic lexical F1 used only for clarification reward."""
    pred = normalize_text(prediction).split()
    gold = normalize_text(reference).split()
    if not pred or not gold:
        return 0.0
    common: dict[str, int] = {}
    for token in pred:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2.0 * precision * recall / (precision + recall)


@dataclass
class SimulatedUserSettings:
    """Configuration owned by the new method rather than legacy IGPO."""

    allow_clarify: bool = True
    allow_nonanswer: bool = True
    # ``full`` uses UCI as the answer-quality channel.
    # ``uci_replaced_by_answer_f1`` is the controlled UCI ablation: it keeps
    # all other proposed-method channels but substitutes terminal answer F1
    # for UCI.  ``answer_f1_only`` is the intentionally much stronger
    # F1-only baseline, which sends no auxiliary reward channels to GRPO.
    reward_mode: str = "full"
    # The original UCI method does not use the answer-stripped evidence-only
    # channel or a repeated-search penalty.  Keep these independently
    # switchable so an ablation can remove the channels rather than merely
    # giving their normalized advantages a zero weight.
    enable_evidence_utility: bool = True
    enable_search_efficiency: bool = True
    # When disabled, a correct answer advances immediately: no remote user
    # simulator call, no clarity/patience reward, and no answer retry.  The
    # static-context ablation additionally resets each sub-task to its source
    # gold dialogue prefix.
    enable_user_feedback: bool = True
    use_static_gold_context: bool = False
    max_tool_calls: int = 4
    max_search_queries: int = 1
    search_top_k: int = 3
    max_answer_depth: int = 3
    # ``0`` means retain complete passages initially. Dynamic context
    # management first evicts older retrieval text and only compacts the newest
    # top-k when that newest evidence alone cannot fit in the model window.
    # A positive value remains available as an explicit debugging hard cap.
    tool_observation_token_cap: int = 0
    simulator_mode: str = "openai"
    simulator_base_url: Optional[str] = None
    simulator_model: Optional[str] = None
    simulator_timeout_seconds: int = 120
    # Caps apply only to the frozen user-simulator evaluation request. They do
    # not truncate policy trajectories, UCI inputs, or the actor update.
    simulator_question_token_cap: int = 384
    simulator_gold_response_token_cap: int = 768
    simulator_answer_token_cap: int = 768
    simulator_transcript_token_cap: int = 2048
    simulator_source_context_token_cap: int = 1024
    simulator_max_output_tokens: int = 128


@dataclass
class _Event:
    subtask_index: int
    response_depth: int
    action: str
    raw_response: str
    original_raw_response: str = ""
    trailing_content_discarded: bool = False
    components: dict[str, Optional[float]] = field(default_factory=dict)
    format_valid: bool = True
    fallback_reason: str = ""
    retrieved_passages: list[dict[str, str]] = field(default_factory=list)
    # These fields audit rare, context-driven shortening of the most recent
    # top-k. Passage IDs are always retained independently in
    # ``retrieved_passages`` for retrieval metrics.
    latest_retrieval_compacted: bool = False
    latest_retrieval_original_tokens: int = 0
    latest_retrieval_visible_tokens: int = 0
    simulator_level: Optional[int] = None
    simulator_feedback: str = ""
    simulator_status: str = ""
    # Audit-only snapshot of exactly what was supplied as public history to
    # the frozen simulator.  It intentionally excludes the candidate answer,
    # which is saved separately in ``raw_response``.
    simulator_public_transcript: str = ""
    # Mean old-policy log-prob of the canonical gold answer after removing the
    # entire terminal assistant response.  The remaining state contains only
    # dialogue history, private search queries, retrieval observations, and
    # any prior retry feedback.  It is a low-cost, trajectory-level proxy for
    # the utility of the retrieved evidence.
    evidence_utility: Optional[float] = None
    answer_f1: Optional[float] = None
    # Terminal-only cost for repeated retrieval within one original sub-task.
    # The first retrieval is free; each additional retrieval receives one unit
    # of penalty before same-depth sibling normalization.
    search_efficiency: Optional[float] = None
    uci: Optional[float] = None
    expected_action: str = ""
    # Exact live policy context immediately before this assistant action.  It
    # allows evidence to be compacted later without changing the log-prob
    # context used to train an earlier action.
    prompt_messages: list[dict[str, Any]] = field(default_factory=list)
    subtask_event_index: int = 0


@dataclass
class _State:
    rollout_index: int
    dialogue: dict[str, Any]
    initial_messages: list[dict[str, Any]]
    # The policy sees ``messages`` and therefore retains its private reasoning
    # trace and retrieval observations.  The frozen user simulator receives
    # only this separate public projection: user questions/feedback and final
    # assistant-facing answers (including environmental gold fallback).
    public_messages: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    # Snapshot immediately before every original sub-task is answered. A
    # snapshot contains prior policy/environment history and ends at the
    # current user question, so the sub-task can later become its own GRPO row.
    subtask_initial_messages: list[list[dict[str, Any]]] = field(default_factory=list)
    subtask_index: int = 0
    answer_depth: int = 1
    tool_calls: int = 0
    events: list[_Event] = field(default_factory=list)
    subtask_first_retrieval: list[dict[str, str]] = field(default_factory=list)
    # Tool observations in the active subtask only.  Each entry stores a
    # mutable policy message plus its id/title-only compact representation.
    active_tool_observations: list[dict[str, Any]] = field(default_factory=list)
    # Only used in an emergency when public dialogue alone is too long after
    # all private evidence has been compacted. The simulator still receives
    # its separately bounded full public projection.
    policy_history_compactions: int = 0
    # Filled just before every generation and copied into the resulting event.
    pending_action_prompt: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False

    @property
    def subtask(self) -> dict[str, Any]:
        return self.dialogue["subtasks"][self.subtask_index]


class SimulatedUserGenerationManager(LLMGenerationManager):
    """Collect full dialogue trajectories against a frozen user simulator.

    There are two intentionally separate phases at every policy answer:

    1. The pre-update actor scores UCI under ``torch.no_grad``.  The actor is
       not updated until all rollout scoring has completed, so this is the
       required frozen old-policy score without keeping a second model copy.
    2. The external Qwen32B simulator returns only a clarity level and short
       feedback.  It never becomes a policy assistant message.
    """

    def __init__(self, *args, settings: SimulatedUserSettings, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.settings = settings
        if settings.reward_mode not in {
            "full",
            "uci_replaced_by_answer_f1",
            "answer_f1_only",
        }:
            raise ValueError(
                "simulated_user_reward_mode must be 'full', "
                "'uci_replaced_by_answer_f1', or 'answer_f1_only', "
                f"got {settings.reward_mode!r}"
            )
        if settings.max_tool_calls < 1:
            raise ValueError("simulated_user_max_tool_calls must be >= 1")
        if settings.max_search_queries < 1:
            raise ValueError("simulated_user_max_search_queries must be >= 1")
        if settings.search_top_k < 1:
            raise ValueError("simulated_user_search_top_k must be >= 1")
        if settings.max_answer_depth < 1:
            raise ValueError("simulated_user_max_answer_depth must be >= 1")
        for name in (
            "simulator_question_token_cap",
            "simulator_gold_response_token_cap",
            "simulator_answer_token_cap",
            "simulator_transcript_token_cap",
            "simulator_source_context_token_cap",
            "simulator_max_output_tokens",
        ):
            if int(getattr(settings, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        self.simulator: Optional[UserSimulatorClient] = None
        if settings.enable_user_feedback:
            self.simulator = UserSimulatorClient(
                base_url=settings.simulator_base_url,
                model=settings.simulator_model,
                timeout_seconds=settings.simulator_timeout_seconds,
                max_output_tokens=settings.simulator_max_output_tokens,
                mode=settings.simulator_mode,
            )
        self.system_prompt = build_simulated_user_system_prompt(
            self.system_prompt,
            allow_clarify=settings.allow_clarify,
            allow_nonanswer=settings.allow_nonanswer,
            max_search_queries=settings.max_search_queries,
        )

    def _static_subtask_messages(self, subtask: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Build a static canonical dialogue prefix ending at ``subtask``.

        ``source_context`` is derived from the original dataset and alternates
        user/assistant utterances.  It contains gold prior answers, so the
        no-feedback ablation never conditions a later source sub-task on a
        sampled answer, simulator feedback, or a fallback side effect.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        source_context = subtask.get("source_context", [])
        if isinstance(source_context, list):
            for index, value in enumerate(source_context):
                content = str(value or "").strip()
                if content:
                    messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": content})
        question = str(subtask.get("question", "")).strip()
        if not question:
            raise ValueError("simulated-user subtask has an empty question")
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != question:
            messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _public_messages_from_policy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"role": str(message["role"]), "content": str(message.get("content", ""))}
            for message in messages
            if message.get("role") in {"user", "assistant"}
        ]

    def _new_state(self, rollout_index: int, dialogue: dict[str, Any]) -> _State:
        first_subtask = dialogue["subtasks"][0]
        if self.settings.use_static_gold_context:
            initial_messages = self._static_subtask_messages(first_subtask)
            public_messages = self._public_messages_from_policy_messages(initial_messages)
        else:
            first_question = first_subtask["question"]
            initial_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": first_question},
            ]
            public_messages = [{"role": "user", "content": first_question}]
        return _State(
            rollout_index=rollout_index,
            dialogue=dialogue,
            initial_messages=[dict(message) for message in initial_messages],
            public_messages=public_messages,
            messages=[dict(message) for message in initial_messages],
            subtask_initial_messages=[[dict(message) for message in initial_messages]],
        )

    def _tool_observation_pages(self, content: Any) -> tuple[str, list[Mapping[str, Any]]]:
        """Extract the configured leading query and top-k pages once."""
        structured = content
        if isinstance(structured, str):
            try:
                structured = json.loads(structured)
            except json.JSONDecodeError:
                structured = None
        if not isinstance(structured, list):
            return "", []

        pages: list[Mapping[str, Any]] = []
        query = ""
        for query_result in structured[: self.settings.max_search_queries]:
            if not isinstance(query_result, Mapping):
                continue
            if not query:
                query = str(query_result.get("search_query", ""))
            candidates = query_result.get("web_page_info_list", [])
            if isinstance(candidates, list):
                pages.extend(page for page in candidates if isinstance(page, Mapping))
        return query, pages[: self.settings.search_top_k]

    def _truncate_tool_observation(self, content: Any, *, token_cap: Optional[int] = None) -> str:
        """Render a fresh retrieval observation for the live policy.

        The normal setting is ``cap=0``: the newest top-k is inserted in full.
        Dynamic context management later replaces older observations by their
        compact id/title form and, only if unavoidable, uses ``token_cap`` to
        fit the newest top-k itself. A positive configured cap is retained as
        a manual debugging escape hatch, never as the default training policy.
        """
        cap = int(self.settings.tool_observation_token_cap) if token_cap is None else max(0, int(token_cap))
        query, pages = self._tool_observation_pages(content)
        if cap > 0 and pages:
            # Distribute the available budget across the complete top-k rather
            # than silently showing only rank 1.  A final exact token trim
            # guards against tokenizer boundary effects in the concatenation.
            header = f"Search query: {query}\nRetrieved passages (top-{len(pages)}):"
            header_ids = self.tokenizer(header, add_special_tokens=False)["input_ids"][:cap]
            header = self.tokenizer.decode(header_ids, skip_special_tokens=False)
            remaining = max(0, cap - len(header_ids))
            per_passage_budget = max(1, remaining // len(pages))
            excerpts = [header]
            for rank, page in enumerate(pages, start=1):
                passage_id = str(page.get("passage_id", ""))
                title = str(page.get("title", ""))
                text = str(page.get("quick_summary", page.get("passage_text", "")))
                passage = (
                    f"\n[Passage {rank}] id={passage_id}\n"
                    f"title={title}\n"
                    f"text={text}"
                )
                token_ids = self.tokenizer(passage, add_special_tokens=False)["input_ids"]
                excerpts.append(
                    self.tokenizer.decode(token_ids[:per_passage_budget], skip_special_tokens=False)
                )
            rendered = self._escape_tool_observation("\n".join(excerpts))
            rendered_ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
            return self.tokenizer.decode(rendered_ids[:cap], skip_special_tokens=False)

        # ``cap=0`` preserves the handler's complete top-k payload.  It is
        # not sent to the simulator; it is policy-private evidence.
        text = self._escape_tool_observation(content)
        if cap <= 0:
            return text
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= cap:
            return text
        return self.tokenizer.decode(token_ids[:cap], skip_special_tokens=False) + "\n[retrieval observation truncated]"

    def _compact_tool_observation(self, content: Any) -> str:
        """Keep retrievability metadata while evicting old passage text."""
        query, pages = self._tool_observation_pages(content)
        if not pages:
            return "[Earlier retrieval omitted to preserve context budget.]"
        lines = [f"Earlier search query: {query}", f"Earlier retrieved passages (top-{len(pages)}; text omitted):"]
        for rank, page in enumerate(pages, start=1):
            lines.append(
                f"[Passage {rank}] id={str(page.get('passage_id', ''))} "
                f"title={str(page.get('title', ''))}"
            )
        return self._escape_tool_observation("\n".join(lines))

    def _compact_one_old_policy_history_message(self, state: _State) -> bool:
        """Compress one old public turn only after all old evidence is gone.

        This is an emergency context safeguard, not the normal dialogue
        policy. The simulator remains unaffected because it owns a separate,
        bounded public transcript. We retain the most recent two public
        messages and compact the earliest older turn first.
        """
        for message in state.messages[1:-2]:
            if str(message.get("role", "")).lower() not in {"user", "assistant"}:
                continue
            content = str(message.get("content", "") or "")
            if content.startswith("[Earlier dialogue turn compacted"):
                continue
            role = str(message.get("role", "conversation"))
            message["content"] = f"[Earlier dialogue turn compacted; role={role}.]"
            state.policy_history_compactions += 1
            return True
        return False

    def _fit_newest_observation_to_budget(
        self,
        state: _State,
        observation: dict[str, Any],
        budget: int,
    ) -> bool:
        """Fit the latest top-k to the *remaining* live prompt budget.

        The full newest observation is first swapped out for its id/title
        representation to measure the non-evidence prompt exactly. The
        remaining tokens are then shared across the newest top-k passages.
        This makes compaction adaptive to the actual dialogue length instead
        of relying on a fixed 512/1536 cap.
        """
        message = observation["message"]
        full_content = str(observation["full_content"])
        message["content"] = observation["compact_content"]
        base_tokens = self._policy_prompt_token_count(state)
        available = max(1, budget - base_tokens)

        # Iterate once or twice because chat-template/tokenizer boundaries can
        # add a few tokens after an observation is rendered and escaped.
        visible = self._truncate_tool_observation(observation["raw_content"], token_cap=available)
        for _ in range(3):
            message["content"] = visible
            overflow = self._policy_prompt_token_count(state) - budget
            if overflow <= 0:
                break
            next_cap = max(1, available - overflow - 8)
            if next_cap == available:
                break
            available = next_cap
            visible = self._truncate_tool_observation(observation["raw_content"], token_cap=available)

        observation["latest_compacted"] = True
        observation["event"].latest_retrieval_compacted = True
        observation["event"].latest_retrieval_original_tokens = len(
            self.tokenizer(full_content, add_special_tokens=False)["input_ids"]
        )
        observation["event"].latest_retrieval_visible_tokens = len(
            self.tokenizer(str(message["content"]), add_special_tokens=False)["input_ids"]
        )
        return self._policy_prompt_token_count(state) <= budget

    def _append_policy_message(self, state: _State, message: Mapping[str, Any]) -> dict[str, Any]:
        """Append an environment/policy message to the live policy context."""
        copied = dict(message)
        state.messages.append(copied)
        return copied

    def _policy_prompt_token_count(self, state: _State) -> int:
        serialized = self.tokenizer.apply_chat_template(state.messages, add_generation_prompt=True, tokenize=False)
        if isinstance(serialized, list):
            if len(serialized) != 1:
                raise RuntimeError("simulated-user expected one serialized live policy prompt")
            serialized = serialized[0]
        serialized = str(serialized) + "<think>"
        return len(self.tokenizer(serialized, add_special_tokens=False)["input_ids"])

    def _enforce_live_policy_context_budget(self, state: _State) -> None:
        """Keep every live policy prompt inside the model window.

        Full newest top-k evidence is retained whenever possible. When it is
        intrinsically too large, it is adaptively shortened only after all
        older evidence has been compacted; the event records that exception.
        Exact prompt snapshots are saved per assistant action, so modifying a
        live context here cannot retroactively change PPO log-probs for an
        earlier action.
        """
        budget = self._prompt_token_budget()
        if budget is None:
            return
        # ``_build_rollings_from_messages`` appends the opening ``<think>``
        # after computing its generic response reserve. Keep a tiny explicit
        # cushion so an exact event prompt never needs a second implicit left
        # truncation during actor log-prob recomputation.
        budget = max(1, budget - 8)
        while self._policy_prompt_token_count(state) > budget:
            candidate = next(
                (item for item in state.active_tool_observations[:-1] if not bool(item["compacted"])),
                None,
            )
            if candidate is not None:
                candidate["message"]["content"] = candidate["compact_content"]
                candidate["compacted"] = True
                continue

            newest = state.active_tool_observations[-1] if state.active_tool_observations else None
            if newest is not None and not bool(newest.get("latest_compacted", False)):
                # If even id/title-only newest evidence cannot coexist with
                # the old public dialogue, compact that old dialogue first.
                # This avoids needlessly reducing newest evidence to one token
                # and then discovering that history compaction freed space.
                newest_message = newest["message"]
                original_content = newest_message["content"]
                newest_message["content"] = newest["compact_content"]
                base_over_budget = self._policy_prompt_token_count(state) > budget
                newest_message["content"] = original_content
                if base_over_budget and self._compact_one_old_policy_history_message(state):
                    continue
                if self._fit_newest_observation_to_budget(state, newest, budget):
                    continue

            # This branch is uncommon: all private retrieval evidence is
            # already as small as possible but a very long multi-subtask
            # public dialogue still exceeds the policy window.  Compact only
            # the oldest completed public turns, never the current question.
            if self._compact_one_old_policy_history_message(state):
                continue

            # A single current question and the fixed system contract should
            # be far below 8k. If a malformed source row violates that
            # assumption, preserve its beginning (where the user request
            # normally appears) rather than crashing an entire 128-context
            # update. This event has no prior action yet, so there is no
            # historical PPO prompt whose consistency could be broken.
            current_user = next(
                (message for message in reversed(state.messages) if message.get("role") == "user"),
                None,
            )
            if current_user is not None and not str(current_user.get("content", "")).endswith(
                "[current question compacted]"
            ):
                content_ids = self.tokenizer(str(current_user.get("content", "")), add_special_tokens=False)["input_ids"]
                excess = self._policy_prompt_token_count(state) - budget
                keep = max(1, len(content_ids) - excess - 16)
                current_user["content"] = (
                    self.tokenizer.decode(content_ids[:keep], skip_special_tokens=False)
                    + "\n[current question compacted]"
                )
                continue

            # Do not stop the entire update for a pathological row. The
            # generic batching path already has configured left truncation;
            # this final guard records the condition to make it auditable.
            # It should only be reachable if the fixed system prompt itself
            # is longer than the model context.
            print(
                "[SimUser] warning: fixed policy prompt remains over budget after adaptive compaction; "
                "configured left truncation will be used for this generation.",
                flush=True,
            )
            return

    def _generation_batch(self, states: list[_State]) -> DataProto:
        messages = [state.messages for state in states]
        active = self._build_rollings_from_messages(
            messages,
            self._prompt_token_budget(),
        )
        active.meta_info.update(
            {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": not self.is_validation,
            }
        )
        return active

    def _score_uci_batch(
        self,
        requests: list[tuple[_State, list[dict[str, Any]], str]],
    ) -> tuple[list[float], list[float]]:
        """Return UCI and answer-stripped evidence utility for each request.

        ``uci`` is ``logP(gold|history+answer)-logP(gold|history)``.  The
        accompanying evidence utility is the latter absolute score only:
        after removing the *whole* terminal assistant response, it measures
        how likely the old policy finds the gold under the preceding dialogue,
        private queries, and retrieved passages.  It is independently
        normalized among same-dialogue/same-subtask/same-depth siblings, so
        an absolute log-prob becomes a relative positive/negative reward.

        Both score states receive the same invisible evaluator prompt.  The
        prompt is never appended to a policy rollout; this is a no-gradient
        old-policy probe after generation and before the actor update.
        """
        if not requests:
            return [], []
        prompt_messages: list[list[dict[str, Any]]] = []
        targets: list[list[int]] = []
        lookup: list[tuple[int, bool]] = []
        probe = (
            "Internal evaluator request: give the canonical answer to the current user question "
            "in one concise response."
        )
        for request_index, (_state, before, gold) in enumerate(requests):
            if not gold.strip():
                lookup.extend(((request_index, False), (request_index, True)))
                prompt_messages.extend((before, before))
                targets.extend(([], []))
                continue
            after = [dict(message) for message in before]
            # ``before`` passed in already includes the generated assistant answer.
            # Its companion history removes that final policy assistant message.
            history = [dict(message) for message in before[:-1]]
            for include_answer, messages in ((False, history), (True, after)):
                scored_messages = [dict(message) for message in messages]
                scored_messages.append({"role": "user", "content": probe})
                prompt_messages.append(scored_messages)
                target = self.tokenizer(
                    gold + "\n<|im_end|>", add_special_tokens=False
                )["input_ids"]
                targets.append(target)
                lookup.append((request_index, include_answer))

        nonempty = [index for index, target in enumerate(targets) if target]
        scores = [0.0] * len(targets)
        if nonempty:
            prompts = self._build_rollings_from_messages(
                [prompt_messages[index] for index in nonempty],
                self._prompt_token_budget(max(len(targets[index]) for index in nonempty)),
            )
            pseudo = self.pseudo_generate_sequences(
                prompts,
                [targets[index] for index in nonempty],
            )
            padded, pad_size = pad_dataproto_to_divisor(pseudo, self.actor_rollout_wg.world_size)
            log_probs = self.actor_rollout_wg.compute_log_prob(padded)
            log_probs = unpad_dataproto(log_probs, pad_size=pad_size)
            for local_index, global_index in enumerate(nonempty):
                token_count = len(targets[global_index])
                values = log_probs.batch["old_log_probs"][local_index, :token_count]
                mean = float(values.mean().item()) if token_count else 0.0
                scores[global_index] = mean if math.isfinite(mean) else 0.0

        per_request: list[dict[bool, float]] = [dict() for _ in requests]
        for score, (request_index, include_answer) in zip(scores, lookup):
            per_request[request_index][include_answer] = score
        evidence_utilities = [parts.get(False, 0.0) for parts in per_request]
        uci_values = [
            parts.get(True, 0.0) - parts.get(False, 0.0)
            for parts in per_request
        ]
        return uci_values, evidence_utilities

    @staticmethod
    def _append_public_assistant_message(state: _State, content: Any) -> None:
        """Add one user-visible assistant utterance to the simulator view."""
        text = str(content or "").strip()
        if text:
            state.public_messages.append({"role": "assistant", "content": text})

    def _public_transcript(self, state: _State) -> str:
        """Render a safe simulator-only view of the dialogue.

        This never serializes ``state.messages`` because that object contains
        private ``<think>`` content, tool calls, retrieved passages and system
        instructions.  The candidate answer is provided to the simulator in a
        separate field, so it is not duplicated in this history.
        """
        rendered: list[str] = []
        for message in state.public_messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            rendered.append(f"{role.title()}: {content}")
        return self._truncate_simulator_text(
            "\n".join(rendered),
            self.settings.simulator_transcript_token_cap,
            keep_tail=True,
        )

    def _append_environmental_gold(self, state: _State, reason: str) -> None:
        """Advance with canonical gold as non-trainable assistant context.

        The failed policy action remains in the serialized rollout and keeps
        its negative reward boundary.  The following subtask, however, sees a
        canonical assistant resolution in its prompt.  That gold text is part
        of the *prompt only* and is never treated as an actor action.
        """
        del reason  # The fallback cause is retained in the rollout record, not exposed as dialogue text.
        gold = str(state.subtask["gold_response"] or "").strip()
        if not gold:
            gold = "I cannot provide an answer to the previous request."
        self._append_policy_message(state, {"role": "assistant", "content": gold})
        self._append_public_assistant_message(state, gold)

    def _reset_policy_to_public_dialogue(self, state: _State) -> None:
        """Drop completed subtask's private trace before the next user task."""
        state.messages = [{"role": "system", "content": self.system_prompt}]
        state.messages.extend(dict(message) for message in state.public_messages)

    def _move_to_next_subtask(self, state: _State, *, fallback_reason: str = "") -> None:
        if self.settings.use_static_gold_context:
            # The static ablation intentionally uses the dataset's canonical
            # prefix for every original task.  In particular, a sampled answer
            # or format/action fallback from task t cannot alter task t+1.
            state.subtask_index += 1
            if state.subtask_index >= len(state.dialogue["subtasks"]):
                state.finished = True
                return
            state.answer_depth = 1
            state.tool_calls = 0
            state.subtask_first_retrieval = []
            state.active_tool_observations = []
            state.messages = self._static_subtask_messages(state.subtask)
            state.public_messages = self._public_messages_from_policy_messages(state.messages)
            state.subtask_initial_messages.append([dict(message) for message in state.messages])
            return
        if fallback_reason:
            self._append_environmental_gold(state, fallback_reason)
        state.subtask_index += 1
        if state.subtask_index >= len(state.dialogue["subtasks"]):
            state.finished = True
            return
        state.answer_depth = 1
        state.tool_calls = 0
        state.subtask_first_retrieval = []
        state.active_tool_observations = []
        next_question = state.subtask["question"]
        state.public_messages.append({"role": "user", "content": next_question})
        # The model starts the next original task from public dialogue only.
        # Previous think/tool/passage content stays in the immutable event
        # trace for PPO, but cannot consume the next subtask's context window.
        self._reset_policy_to_public_dialogue(state)
        state.subtask_initial_messages.append([dict(message) for message in state.messages])

    def _event(
        self,
        state: _State,
        *,
        action: str,
        raw_response: str,
        original_raw_response: Optional[str] = None,
        trailing_content_discarded: bool = False,
        format_valid: bool = True,
        fallback_reason: str = "",
    ) -> _Event:
        event = _Event(
            subtask_index=state.subtask_index,
            response_depth=state.answer_depth,
            action=action,
            raw_response=raw_response,
            original_raw_response=original_raw_response if original_raw_response is not None else raw_response,
            trailing_content_discarded=trailing_content_discarded,
            format_valid=format_valid,
            fallback_reason=fallback_reason,
            expected_action=state.subtask["expected_action"],
            prompt_messages=[dict(message) for message in state.pending_action_prompt],
            subtask_event_index=sum(
                prior.subtask_index == state.subtask_index for prior in state.events
            ),
        )
        if self.settings.reward_mode in _FULL_AUXILIARY_CHANNEL_MODES:
            # Format is a response-level channel. Including valid ``0`` events
            # is essential: otherwise a group containing only invalid ``-1``
            # events has zero variance and produces no GRPO learning signal.
            event.components["format"] = 0.0 if format_valid else -1.0
        state.events.append(event)
        return event

    @staticmethod
    def _success_scale(depth: int) -> float:
        return {1: 1.0, 2: 0.5, 3: 0.25}.get(depth, 0.0)

    @staticmethod
    def _search_efficiency_reward(tool_calls: int) -> float:
        """Leave one retrieval free and penalize each additional retrieval."""
        return -float(max(0, int(tool_calls) - 1))

    def _truncate_simulator_text(self, value: Any, token_cap: int, *, keep_tail: bool) -> str:
        """Token-truncate a field sent only to the Qwen32B evaluator.

        Qwen2.5-3B and Qwen2.5-32B use compatible tokenization.  Applying the
        cap here is therefore materially safer than a character heuristic and
        leaves the serialized policy trajectory completely untouched.
        """
        text = str(value or "")
        if not text:
            return ""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= token_cap:
            return text
        retained = token_ids[-token_cap:] if keep_tail else token_ids[:token_cap]
        return self.tokenizer.decode(retained, skip_special_tokens=True)

    def _bounded_simulator_source_context(self, source_context: Any) -> list[str]:
        if not isinstance(source_context, list):
            return []
        # The most recent turns are the useful ones for resolving references in
        # the current question. Keep them as a single bounded field so JSON
        # framing and a long list cannot defeat the token budget.
        source_text = "\n".join(str(item) for item in source_context if str(item).strip())
        bounded = self._truncate_simulator_text(
            source_text,
            self.settings.simulator_source_context_token_cap,
            keep_tail=True,
        )
        return [bounded] if bounded else []

    def _process_terminal_action(
        self,
        state: _State,
        action: AgentAction,
        raw_response: str,
        original_raw_response: str,
        trailing_content_discarded: bool,
        pending_uci: list[tuple[_Event, _State, list[dict[str, Any]], str]],
        pending_judgements: list[tuple[_State, _Event, str]],
    ) -> None:
        expected = state.subtask["expected_action"]
        event = self._event(
            state,
            action=action.kind,
            raw_response=raw_response,
            original_raw_response=original_raw_response,
            trailing_content_discarded=trailing_content_discarded,
        )
        # Action reward is collected only at the initial terminal decision.
        # The controlled UCI→F1 ablation keeps this channel; the F1-only
        # baseline deliberately omits it together with all other auxiliaries.
        if (
            self.settings.reward_mode in _FULL_AUXILIARY_CHANNEL_MODES
            and state.answer_depth == 1
        ):
            event.components["action"] = 1.0 if action.kind == expected else -1.0
        if action.kind != expected:
            event.fallback_reason = "action_mismatch"
            self._move_to_next_subtask(state, fallback_reason=event.fallback_reason)
            return

        if action.kind == "nonanswer":
            self._move_to_next_subtask(state)
            return
        if action.kind == "clarify":
            if self.settings.reward_mode in _FULL_AUXILIARY_CHANNEL_MODES:
                event.components["clarify_f1"] = _token_f1(
                    action.content, state.subtask["gold_response"]
                )
            self._append_public_assistant_message(state, action.content)
            self._move_to_next_subtask(state)
            return

        if self.settings.reward_mode == "answer_f1_only":
            # Preserve retrieval, simulator feedback/retries, and fallback
            # mechanics, while supplying only direct answer-vs-gold F1 to
            # sparse GRPO in this ablation.
            event.answer_f1 = _token_f1(action.content, state.subtask["gold_response"])
            event.components["answer_f1"] = event.answer_f1
            pending_judgements.append((state, event, action.content))
            return

        if self.settings.reward_mode == "uci_replaced_by_answer_f1":
            # Controlled ablation: all task-control, formatting and simulator
            # satisfaction channels remain active, but direct answer-vs-gold
            # F1 replaces the UCI channel.  No frozen-model UCI forward is
            # scheduled in this branch.
            event.answer_f1 = _token_f1(action.content, state.subtask["gold_response"])
            event.components["answer_f1"] = event.answer_f1
            if self.settings.enable_search_efficiency:
                event.search_efficiency = self._search_efficiency_reward(state.tool_calls)
                event.components["search_efficiency"] = event.search_efficiency
            pending_judgements.append((state, event, action.content))
            return

        # The first retrieval is normally necessary, so charge only the second
        # and later calls.  Placing this on the terminal answer lets sparse
        # reward-to-go teach all earlier query actions without extra scoring
        # forwards per tool call.
        if self.settings.enable_search_efficiency:
            event.search_efficiency = self._search_efficiency_reward(state.tool_calls)
            event.components["search_efficiency"] = event.search_efficiency
        # Correct answer: score UCI before any user-simulator feedback is appended.
        pending_uci.append((event, state, [dict(message) for message in state.messages], state.subtask["gold_response"]))

    def _handle_judgement(self, state: _State, event: _Event, answer: str) -> None:
        if not self.settings.enable_user_feedback:
            # This is the static-gold-context ablation: an answer gets exactly
            # one user-facing attempt and then the environment moves directly
            # to the next source sub-task.  No simulator-derived reward or
            # feedback text is produced.
            self._append_public_assistant_message(state, answer)
            self._move_to_next_subtask(state)
            return
        if self.simulator is None:
            raise RuntimeError("simulated-user feedback is enabled but no simulator client was created")
        subtask = state.subtask
        public_transcript = self._public_transcript(state)
        judgement = self.simulator.judge_answer(
            dialogue_id=state.dialogue["dialogue_id"],
            subtask_index=state.subtask_index,
            question=self._truncate_simulator_text(
                subtask["question"], self.settings.simulator_question_token_cap, keep_tail=False
            ),
            public_transcript=public_transcript,
            answer=self._truncate_simulator_text(
                answer, self.settings.simulator_answer_token_cap, keep_tail=False
            ),
        )
        event.simulator_level = judgement.level
        event.simulator_feedback = judgement.feedback
        event.simulator_status = judgement.source
        event.simulator_public_transcript = public_transcript
        # The user has now seen this answer.  Future clarity judgements and
        # retry turns retain only this public answer, never its think/tool
        # trace or retrieved passages.
        self._append_public_assistant_message(state, answer)
        if judgement.level == 1:
            if self.settings.reward_mode in _FULL_AUXILIARY_CHANNEL_MODES:
                event.components.update(
                    {
                        "clarity": self._success_scale(state.answer_depth),
                        # A satisfied user has no patience penalty. Keeping
                        # this explicit zero lets the channel contrast it with
                        # L2/L3 in the full method.
                        "patience": 0.0,
                    }
                )
            self._move_to_next_subtask(state)
            return
        if self.settings.reward_mode in _FULL_AUXILIARY_CHANNEL_MODES:
            if judgement.level == 2:
                event.components.update({"clarity": -1.0, "patience": -0.5})
            else:
                event.components.update({"clarity": -2.0, "patience": -1.0})
        if state.answer_depth >= self.settings.max_answer_depth:
            event.fallback_reason = "answer_retry_exhausted"
            self._move_to_next_subtask(state, fallback_reason=event.fallback_reason)
            return
        state.answer_depth += 1
        feedback = "User feedback: " + (judgement.feedback or "Please clarify your answer.")
        self._append_policy_message(state, {"role": "user", "content": feedback})
        state.public_messages.append({"role": "user", "content": feedback})
        self._append_policy_message(state, {"role": "system", "content": RETRY_SYSTEM_PROMPT})

    def _extract_dialogues(self, reward_models: Iterable[Any]) -> list[dict[str, Any]]:
        dialogues: list[dict[str, Any]] = []
        for index, reward_model in enumerate(reward_models):
            payload = reward_model
            if isinstance(reward_model, Mapping):
                payload = reward_model.get("simulated_dialogue", reward_model.get("dialogue", reward_model))
            dialogues.append(parse_dialogue_payload(payload, fallback_id=f"dialogue-{index}"))
        return dialogues

    def _response_tensors_and_events(self, states: list[_State]) -> tuple[DataProto, list[dict[str, Any]]]:
        """Flatten to exact per-action policy snapshots.

        A live policy context may compact an old retrieval after a later search.
        Consequently one immutable ``prompt + whole-subtask-response`` string
        cannot reproduce every action's original context.  We instead retain
        the exact prompt immediately before every action.  Sparse GRPO still
        computes reward-to-go within each ``(dialogue, rollout, subtask)``
        event chain, so tool-call tokens receive the later terminal reward.
        """
        initial_prompts: list[str] = []
        response_texts: list[str] = []
        response_event_positions: list[list[tuple[_Event, int]]] = []
        records: list[dict[str, Any]] = []
        row_uids: list[str] = []
        row_uidrs: list[int] = []
        row_subtasks: list[int] = []
        row_event_orders: list[int] = []
        for state in states:
            for event in state.events:
                if not event.prompt_messages:
                    raise RuntimeError("simulated-user event is missing its exact policy prompt snapshot")
                initial = self.tokenizer.apply_chat_template(
                    event.prompt_messages, add_generation_prompt=True, tokenize=False
                )
                full = self.tokenizer.apply_chat_template(
                    [*event.prompt_messages, {"role": "assistant", "content": event.raw_response}],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                if isinstance(initial, list) or isinstance(full, list):
                    raise RuntimeError("simulated-user expected scalar ChatML serialization for one event")
                if not full.startswith(initial):
                    raise RuntimeError("simulated-user action prompt is not a prefix of its serialized response")
                serialized_response = full[len(initial):]
                if not serialized_response:
                    raise RuntimeError("simulated-user serialized an empty policy action")
                initial_prompts.append(initial)
                response_texts.append(serialized_response)
                response_event_positions.append([(event, len(serialized_response))])
                records.append(
                    self._state_record(
                        state,
                        serialized_response,
                        subtask_index=event.subtask_index,
                        events=[event],
                    )
                )
                row_uids.append(str(state.dialogue["dialogue_id"]))
                row_uidrs.append(int(state.rollout_index % self.config.n))
                row_subtasks.append(int(event.subtask_index))
                row_event_orders.append(int(event.subtask_event_index))

        if not records:
            raise RuntimeError("simulated-user rollout produced no sub-task response records")

        response_encoded = self.tokenizer(
            response_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        responses = response_encoded["input_ids"]
        response_mask = response_encoded["attention_mask"]
        bsz, response_length = responses.shape
        context_window = self._model_context_window()
        # Every row now contains exactly one policy action. Its prompt is the
        # exact rollout-time snapshot, so left truncation is a final safety
        # assertion rather than a replacement for context management.
        prompt_token_budget = self._prompt_token_budget()
        if context_window is not None:
            longest_response = int(response_mask.sum(dim=1).max().item())
            if longest_response >= context_window:
                raise RuntimeError(
                    "simulated-user current sub-task response alone exceeds model context: "
                    f"response_tokens={longest_response}, max_model_len={context_window}. "
                    "Reduce simulated_user_max_tool_calls, simulated_user_max_answer_depth, "
                    "or data.max_response_length."
                )
            dynamic_prompt_budget = context_window - longest_response
            prompt_token_budget = (
                dynamic_prompt_budget
                if prompt_token_budget is None
                else min(prompt_token_budget, dynamic_prompt_budget)
            )
        prompt_encoded = self._tokenize_with_left_truncation(
            initial_prompts,
            max_length=prompt_token_budget,
        )
        prompts = prompt_encoded["input_ids"]
        prompt_mask = prompt_encoded["attention_mask"]
        sort = prompt_mask.to(torch.int64).argsort(dim=1, stable=True)
        prompts = prompts.gather(1, sort)
        prompt_mask = prompt_mask.gather(1, sort)
        if context_window is not None:
            sequence_lengths = prompt_mask.sum(dim=1) + response_mask.sum(dim=1)
            largest_sequence = int(sequence_lengths.max().item())
            if largest_sequence > context_window:
                raise RuntimeError(
                    "simulated-user dynamic prompt truncation failed to fit the model context: "
                    f"max_seq_len={largest_sequence}, max_model_len={context_window}."
                )
        boundary = torch.zeros((bsz, response_length), dtype=torch.long)
        subtask_ids = torch.full((bsz, response_length), -1, dtype=torch.long)
        depths = torch.full((bsz, response_length), -1, dtype=torch.long)
        component_values = {name: torch.zeros((bsz, response_length), dtype=torch.float32) for name in _COMPONENTS}
        component_masks = {name: torch.zeros((bsz, response_length), dtype=torch.bool) for name in _COMPONENTS}
        offsets = response_encoded["offset_mapping"].tolist()
        for row, event_positions in enumerate(response_event_positions):
            for event_index, (event, end_char) in enumerate(event_positions):
                candidates = [
                    index for index, (_start, end) in enumerate(offsets[row])
                    if end > 0 and end <= end_char and bool(response_mask[row, index])
                ]
                if not candidates:
                    raise RuntimeError("could not map a simulated-user event boundary to response tokens")
                column = candidates[-1]
                boundary[row, column] = 1
                records[row]["subtasks"][event_index]["response_boundary_token"] = int(column)
                records[row]["subtasks"][event_index]["normalized_advantage"] = None
                subtask_ids[row, column] = event.subtask_index
                depths[row, column] = event.response_depth
                for name, value in event.components.items():
                    if name not in component_values or value is None:
                        continue
                    if not math.isfinite(float(value)):
                        raise RuntimeError(f"simulated-user reward {name!r} is not finite")
                    component_values[name][row, column] = float(value)
                    component_masks[name][row, column] = True

        attention = torch.cat((prompt_mask, response_mask), dim=-1)
        positions = self.tensor_fn.create_position_ids(attention)
        loss_mask = attention.clone()
        batch = {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": attention,
            "position_ids": positions,
            "loss_mask": loss_mask,
            "sim_user_turn_boundary_mask": boundary,
            "sim_user_subtask_ids": subtask_ids,
            "sim_user_response_depths": depths,
            "sim_user_event_order": torch.tensor(row_event_orders, dtype=torch.long),
            "sim_user_row_subtask": torch.tensor(row_subtasks, dtype=torch.long),
        }
        for name in _COMPONENTS:
            batch[f"sim_user_{name}_values"] = component_values[name]
            batch[f"sim_user_{name}_mask"] = component_masks[name].to(torch.long)
        output = DataProto.from_dict(batch)
        output.non_tensor_batch["simulated_user_records"] = np.array(records, dtype=object)
        output.non_tensor_batch["uid"] = np.array(row_uids, dtype=object)
        output.non_tensor_batch["uidr"] = np.array(row_uidrs, dtype=np.int64)
        return output, records

    def _event_record(self, state: _State, event: _Event) -> dict[str, Any]:
        subtask = state.dialogue["subtasks"][event.subtask_index]
        return {
            "subtask_index": event.subtask_index,
            "question": subtask["question"],
            "expected_action": event.expected_action,
            "predicted_action": event.action,
            "action_correct": event.action == event.expected_action and event.format_valid,
            "response_depth": event.response_depth,
            "subtask_event_index": event.subtask_event_index,
            "raw_response": event.raw_response,
            "original_raw_response": event.original_raw_response,
            "trailing_content_discarded": event.trailing_content_discarded,
            "ground_truth_answer": subtask["gold_response"],
            "ground_truth_passage_ids": subtask.get("ground_truth_passage_ids", []),
            "ground_truth_passage_texts": subtask.get("ground_truth_passage_texts", []),
            "ground_truth_passage_titles": subtask.get("ground_truth_passage_titles", []),
            "predicted_passage_ids": [p.get("passage_id", "") for p in event.retrieved_passages],
            # TopiOCQA's source ``Gold_passage.id`` values do not share the
            # local retriever's passage-ID namespace.  Preserve the returned
            # document text as well, so its retrieval metric can compare the
            # canonical passage contents instead of incomparable IDs.
            "predicted_passage_texts": [p.get("passage_text", "") for p in event.retrieved_passages],
            "latest_retrieval_compacted": event.latest_retrieval_compacted,
            "latest_retrieval_original_tokens": event.latest_retrieval_original_tokens,
            "latest_retrieval_visible_tokens": event.latest_retrieval_visible_tokens,
            "format_valid": event.format_valid,
            "fallback_reason": event.fallback_reason,
            "simulator_level": event.simulator_level,
            "simulator_feedback": event.simulator_feedback,
            "simulator_status": event.simulator_status,
            "simulator_public_transcript": event.simulator_public_transcript,
            "raw_rewards": event.components,
            "answer_f1": event.answer_f1,
            "evidence_utility": event.evidence_utility,
            "search_efficiency": event.search_efficiency,
            "uci": event.uci,
        }

    def _state_record(
        self,
        state: _State,
        serialized_response: str,
        *,
        subtask_index: Optional[int] = None,
        events: Optional[Iterable[_Event]] = None,
    ) -> dict[str, Any]:
        subtasks = []
        selected_events = list(events) if events is not None else state.events
        for event in selected_events:
            subtasks.append(self._event_record(state, event))
        return {
            "dialogue_id": state.dialogue["dialogue_id"],
            "data_source": state.dialogue.get("data_source", "inscit"),
            "serialized_response": serialized_response,
            "context_subtask_index": subtask_index,
            "rollout_index": state.rollout_index % self.config.n,
            "subtasks": subtasks,
        }

    def _dialogue_record(self, state: _State) -> dict[str, Any]:
        """Full event trace returned to validation, one record per rollout."""
        return self._state_record(
            state,
            "\n".join(event.raw_response for event in state.events),
            events=state.events,
        )

    def run_dialogue_loop(
        self,
        *,
        gen_batch: DataProto,
        reward_models: Iterable[Any],
    ) -> tuple[DataProto, list[dict[str, Any]]]:
        if self.client is None:
            raise RuntimeError("simulated-user rollout requires the normal IGPO MessageClient")
        dialogues = self._extract_dialogues(reward_models)
        if len(dialogues) != len(gen_batch):
            raise ValueError(f"dialogue count {len(dialogues)} does not match batch size {len(gen_batch)}")
        states = []
        for dialogue in dialogues:
            for _ in range(self.config.n):
                # The tool handler uses this field as a request/result key. It
                # must be unique across the full batch, not merely per dialogue.
                states.append(self._new_state(len(states), dialogue))
        safety_limit = sum(len(dialogue["subtasks"]) for dialogue in dialogues) * self.config.n * (
            self.settings.max_tool_calls + self.settings.max_answer_depth + 2
        )
        iterations = 0
        while any(not state.finished for state in states):
            iterations += 1
            if iterations > safety_limit:
                raise RuntimeError("simulated-user rollout exceeded its structural safety limit")
            active = [state for state in states if not state.finished]
            for state in active:
                self._enforce_live_policy_context_budget(state)
                state.pending_action_prompt = [dict(message) for message in state.messages]
            generation_input = self._generation_batch(active)
            generated = self._generate_with_gpu_padding(generation_input)
            decoded = self.tokenizer.batch_decode(generated.batch["responses"], skip_special_tokens=False)
            pending_tools: list[tuple[_State, _Event, AgentAction]] = []
            pending_uci: list[tuple[_Event, _State, list[dict[str, Any]], str]] = []
            pending_judgements: list[tuple[_State, _Event, str]] = []
            for state, decoded_response in zip(active, decoded):
                generated_text = decoded_response.replace("<|endoftext|>", "").strip()
                while generated_text.endswith("<|im_end|>"):
                    generated_text = generated_text[: -len("<|im_end|>")].rstrip()
                original_raw = "<think>" + generated_text
                raw, trailing_content_discarded = keep_first_complete_action(original_raw)
                retry_mode = state.answer_depth > 1
                action = parse_agent_action(
                    raw,
                    allow_clarify=self.settings.allow_clarify and state.tool_calls == 0 and state.answer_depth == 1,
                    allow_nonanswer=self.settings.allow_nonanswer,
                    force_answer=retry_mode,
                    allow_tool_call_in_answer_locked_retry=retry_mode,
                    max_search_queries=self.settings.max_search_queries,
                )
                self._append_policy_message(state, {"role": "assistant", "content": raw})
                if not action.valid:
                    event = self._event(
                        state,
                        action="invalid",
                        raw_response=raw,
                        original_raw_response=original_raw,
                        trailing_content_discarded=trailing_content_discarded,
                        format_valid=False,
                        fallback_reason=action.error,
                    )
                    self._move_to_next_subtask(state, fallback_reason="format_invalid")
                    continue
                if action.kind == "tool_call":
                    if state.tool_calls >= self.settings.max_tool_calls:
                        event = self._event(
                            state,
                            action="invalid",
                            raw_response=raw,
                            original_raw_response=original_raw,
                            trailing_content_discarded=trailing_content_discarded,
                            format_valid=False,
                            fallback_reason="tool_call_not_allowed",
                        )
                        self._move_to_next_subtask(state, fallback_reason=event.fallback_reason)
                        continue
                    event = self._event(
                        state,
                        action="tool_call",
                        raw_response=raw,
                        original_raw_response=original_raw,
                        trailing_content_discarded=trailing_content_discarded,
                    )
                    pending_tools.append((state, event, action))
                    continue
                self._process_terminal_action(
                    state,
                    action,
                    raw,
                    original_raw,
                    trailing_content_discarded,
                    pending_uci,
                    pending_judgements,
                )

            if pending_tools:
                request_list = [
                    (state.rollout_index, state.subtask["question"], action.think, action.tool_call)
                    for state, _event, action in pending_tools
                ]
                results = self.execute_predictions(request_list, len(states))
                if len(results) != len(pending_tools):
                    raise RuntimeError("tool server returned a mismatched number of simulated-user results")
                for (state, event, _action), result in zip(pending_tools, results):
                    content = result.get("content", "") if isinstance(result, Mapping) else ""
                    event.retrieved_passages = self._extract_top_retrieved_passages(content)
                    if not state.subtask_first_retrieval:
                        state.subtask_first_retrieval = list(event.retrieved_passages)
                    full_observation = self._truncate_tool_observation(content)
                    event.latest_retrieval_original_tokens = len(
                        self.tokenizer(full_observation, add_special_tokens=False)["input_ids"]
                    )
                    event.latest_retrieval_visible_tokens = event.latest_retrieval_original_tokens
                    tool_message = self._append_policy_message(
                        state,
                        {"role": "tool", "name": "web_search", "content": full_observation},
                    )
                    state.active_tool_observations.append(
                        {
                            "message": tool_message,
                            "raw_content": content,
                            "full_content": full_observation,
                            "compact_content": self._compact_tool_observation(content),
                            "compacted": False,
                            "latest_compacted": False,
                            "event": event,
                        }
                    )
                    state.tool_calls += 1
                    self._enforce_live_policy_context_budget(state)

            if pending_uci:
                uci_values, evidence_utilities = self._score_uci_batch(
                    [(state, messages, gold) for event, state, messages, gold in pending_uci]
                )
                for (event, state, _messages, _gold), uci, evidence_utility in zip(
                    pending_uci, uci_values, evidence_utilities
                ):
                    # This value is scored after removing the complete final
                    # policy response.  It cannot be inflated by answer text
                    # or private terminal <think> content.
                    if self.settings.enable_evidence_utility:
                        event.evidence_utility = evidence_utility
                        event.components["evidence_utility"] = evidence_utility
                    event.uci = uci
                    event.components["uci"] = uci
                    answer = parse_agent_action(
                        event.raw_response,
                        allow_clarify=self.settings.allow_clarify,
                        allow_nonanswer=self.settings.allow_nonanswer,
                        force_answer=event.response_depth > 1,
                        max_search_queries=self.settings.max_search_queries,
                    ).content
                    pending_judgements.append((state, event, answer))
            for state, event, answer in pending_judgements:
                self._handle_judgement(state, event, answer)

        output, _training_records = self._response_tensors_and_events(states)
        output.meta_info["simulated_user"] = True
        # Training consumes the per-action records held in ``output``.  The
        # caller receives one aggregate record per completed rollout so
        # validation can still score each original subtask as one dialogue.
        return output, [self._dialogue_record(state) for state in states]
