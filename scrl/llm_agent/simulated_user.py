"""Utilities for goal-conditioned simulated-user multi-turn RAG.

This module intentionally contains no Ray/FSDP code.  It owns the stable
dialogue schema, the strict action parser, and the OpenAI-compatible user
simulator client.  Keeping those pieces pure makes canonical-label selection
and reward-format handling testable without GPUs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


_SPACE_RE = re.compile(r"\s+")
_ACTION_TYPES = {
    "directanswer": "answer",
    "noanswerbutrelevantinfo": "answer",
    "clarification": "clarify",
    "noanswernorelevantinfo": "nonanswer",
    "answer": "answer",
    "clarify": "clarify",
    "nonanswer": "nonanswer",
}


def normalize_text(value: Any) -> str:
    """Normalize text solely for matching source dialogue labels."""
    return _SPACE_RE.sub(" ", str(value or "")).strip().casefold()


def response_type_to_action(response_type: Any) -> Optional[str]:
    return _ACTION_TYPES.get(normalize_text(response_type).replace(" ", ""))


def _label_response(label: Mapping[str, Any]) -> str:
    return str(label.get("response", label.get("ground_truth", "")) or "").strip()


def _label_passage_ids(label: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for evidence in label.get("evidence", []) or []:
        if isinstance(evidence, Mapping) and evidence.get("passage_id") is not None:
            values.append(str(evidence["passage_id"]))
    raw_ids = label.get("passage_id", [])
    if isinstance(raw_ids, (str, int)):
        raw_ids = [raw_ids]
    for value in raw_ids or []:
        values.append(str(value))
    # Preserve source order while removing duplicates.
    return list(dict.fromkeys(value for value in values if value))


def _label_passage_texts(label: Mapping[str, Any]) -> list[str]:
    """Extract source evidence text while preserving evidence order."""
    values: list[str] = []
    for evidence in label.get("evidence", []) or []:
        if isinstance(evidence, Mapping):
            text = str(evidence.get("passage_text", "") or "").strip()
            if text:
                values.append(text)
    raw_texts = label.get("passage_text", [])
    if isinstance(raw_texts, str):
        raw_texts = [raw_texts]
    for value in raw_texts or []:
        text = str(value or "").strip()
        if text:
            values.append(text)
    return values


@dataclass(frozen=True)
class CanonicalSubtask:
    question: str
    expected_action: str
    gold_response: str
    ground_truth_passage_ids: tuple[str, ...]
    selection_source: str
    source_context: tuple[str, ...]
    ground_truth_passage_texts: tuple[str, ...] = ()
    ground_truth_passage_titles: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "expected_action": self.expected_action,
            "gold_response": self.gold_response,
            "ground_truth_passage_ids": list(self.ground_truth_passage_ids),
            "ground_truth_passage_texts": list(self.ground_truth_passage_texts),
            "ground_truth_passage_titles": list(self.ground_truth_passage_titles),
            "selection_source": self.selection_source,
            "source_context": list(self.source_context),
        }


def _last_turn_candidate(labels: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], str]:
    """Use the approved answer -> clarify -> nonanswer final-turn policy."""
    labels = [label for label in labels if isinstance(label, Mapping)]
    for action in ("answer", "clarify", "nonanswer"):
        for label in labels:
            if response_type_to_action(label.get("responseType", label.get("action"))) == action:
                return label, "last_turn_priority"
    raise ValueError("A dialogue turn has no supported responseType/response label")


def canonicalize_inscit_dialogue(dialogue_id: str, turns: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Convert an original InsCiT dialogue into canonical online subtasks.

    For all but the final turn, the response actually used by the source
    dialogue is recovered from ``next_turn.context[-2]``.  This follows the
    approved rule rather than blindly trusting candidate order or
    ``more_comprehensive_label``.  A mismatch is represented explicitly in
    the output so callers can audit it before training.
    """
    source_turns = list(turns)
    if not source_turns:
        raise ValueError(f"Dialogue {dialogue_id!r} has no turns")

    subtasks: list[CanonicalSubtask] = []
    for turn_index, turn in enumerate(source_turns):
        context = turn.get("context", []) if isinstance(turn, Mapping) else []
        if not isinstance(context, list) or not context:
            raise ValueError(f"Dialogue {dialogue_id!r} turn {turn_index} has no context/question")
        question = str(context[-1]).strip()
        labels = turn.get("labels", []) if isinstance(turn, Mapping) else []
        if not isinstance(labels, list):
            raise ValueError(f"Dialogue {dialogue_id!r} turn {turn_index} has malformed labels")

        selected: Optional[Mapping[str, Any]] = None
        selection_source = ""
        if turn_index + 1 < len(source_turns):
            next_context = source_turns[turn_index + 1].get("context", [])
            if isinstance(next_context, list) and len(next_context) >= 2:
                used_response = normalize_text(next_context[-2])
                matches = [label for label in labels if normalize_text(_label_response(label)) == used_response]
                if matches:
                    selected = matches[0]
                    selection_source = "next_context_match"
            if selected is None:
                selected, selection_source = _last_turn_candidate(labels)
                selection_source = f"fallback_{selection_source}"
        else:
            selected, selection_source = _last_turn_candidate(labels)

        action = response_type_to_action(selected.get("responseType", selected.get("action")))
        if action is None:
            raise ValueError(
                f"Dialogue {dialogue_id!r} turn {turn_index} selected unsupported response type: "
                f"{selected.get('responseType', selected.get('action'))!r}"
            )
        subtasks.append(
            CanonicalSubtask(
                question=question,
                expected_action=action,
                gold_response=_label_response(selected),
                ground_truth_passage_ids=tuple(_label_passage_ids(selected)),
                selection_source=selection_source,
                source_context=tuple(str(value) for value in context),
                ground_truth_passage_texts=tuple(_label_passage_texts(selected)),
            )
        )

    return {
        "dialogue_id": str(dialogue_id),
        "data_source": "inscit",
        "subtasks": [subtask.as_dict() for subtask in subtasks],
    }


def parse_dialogue_payload(payload: Any, *, fallback_id: str = "") -> dict[str, Any]:
    """Accept either a serialized canonical dialogue or original InsCiT turns."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise TypeError(f"simulated dialogue must be a mapping/string, got {type(payload).__name__}")
    if isinstance(payload.get("subtasks"), list):
        subtasks = []
        for position, value in enumerate(payload["subtasks"]):
            if not isinstance(value, Mapping):
                raise TypeError(f"subtasks[{position}] is not a mapping")
            action = str(value.get("expected_action", "")).strip().lower()
            if action not in {"answer", "clarify", "nonanswer"}:
                raise ValueError(f"subtasks[{position}] has unsupported expected_action={action!r}")
            subtasks.append(
                {
                    "question": str(value.get("question", "")).strip(),
                    "expected_action": action,
                    "gold_response": str(value.get("gold_response", "")).strip(),
                    "ground_truth_passage_ids": [str(x) for x in value.get("ground_truth_passage_ids", []) or []],
                    "ground_truth_passage_texts": [
                        str(item) for item in value.get("ground_truth_passage_texts", []) or []
                    ],
                    "ground_truth_passage_titles": [
                        str(item) for item in value.get("ground_truth_passage_titles", []) or []
                    ],
                    "selection_source": str(value.get("selection_source", "canonical")),
                    "source_context": [str(item) for item in value.get("source_context", []) or []],
                }
            )
        if not subtasks or any(not subtask["question"] for subtask in subtasks):
            raise ValueError("canonical simulated dialogue must contain non-empty questions")
        return {
            "dialogue_id": str(payload.get("dialogue_id", fallback_id)),
            "data_source": str(payload.get("data_source", "")),
            "subtasks": subtasks,
        }
    if isinstance(payload.get("turns"), list):
        return canonicalize_inscit_dialogue(str(payload.get("dialogue_id", fallback_id)), payload["turns"])
    raise ValueError("simulated dialogue needs either `subtasks` or original InsCiT `turns`")


@dataclass(frozen=True)
class AgentAction:
    valid: bool
    kind: str
    think: str = ""
    content: str = ""
    tool_call: Optional[dict[str, Any]] = None
    error: str = ""


def parse_agent_action(
    response: str,
    *,
    allow_clarify: bool,
    allow_nonanswer: bool,
    force_answer: bool = False,
    allow_tool_call_in_answer_locked_retry: bool = False,
    max_search_queries: int = 1,
) -> AgentAction:
    """Parse exactly one policy action from one simulated-user generation.

    ``force_answer`` is used after the user simulator has judged an answer as
    unclear.  In that answer-locked retry state, the policy may optionally
    retrieve more evidence, but it may only *terminate* with ``answer``.  A
    tool call is still one standalone policy action: the environment returns
    its observation before the next generation begins.
    """
    content = str(response or "").replace("<|endoftext|>", "").strip()
    actions = ("answer", "tool_call", "clarify", "nonanswer")
    present = [name for name in actions if f"<{name}>" in content or f"</{name}>" in content]
    if content.count("<think>") != 1 or content.count("</think>") != 1 or len(present) != 1:
        return AgentAction(False, "invalid", error="response must contain one think block and one action")
    action = present[0]
    if force_answer and action != "answer":
        if not (allow_tool_call_in_answer_locked_retry and action == "tool_call"):
            return AgentAction(
                False,
                "invalid",
                error="answer-locked retry permits only tool_call or answer",
            )
    if action == "clarify" and not allow_clarify:
        return AgentAction(False, "invalid", error="clarify is disabled for this dataset")
    if action == "nonanswer" and not allow_nonanswer:
        return AgentAction(False, "invalid", error="nonanswer is disabled for this dataset")

    think_match = re.fullmatch(r"<think>(.*?)</think>\s*", content.split(f"<{action}>", 1)[0], flags=re.DOTALL)
    if think_match is None:
        return AgentAction(False, "invalid", error="think block must immediately precede the action")
    block_match = re.fullmatch(
        rf"<think>(.*?)</think>\s*<{action}>(.*?)</{action}>\s*",
        content,
        flags=re.DOTALL,
    )
    if block_match is None:
        return AgentAction(False, "invalid", error="unexpected text or malformed action block")
    think, body = block_match.group(1), block_match.group(2)

    if action == "nonanswer":
        if body.strip():
            return AgentAction(False, "invalid", error="nonanswer must be empty")
        return AgentAction(True, action, think=think)
    if action in {"answer", "clarify"}:
        if not body.strip():
            return AgentAction(False, "invalid", error=f"{action} must be non-empty")
        return AgentAction(True, action, think=think, content=body.strip())

    try:
        tool_call = json.loads(body)
    except json.JSONDecodeError as exc:
        return AgentAction(False, "invalid", error=f"tool_call is not JSON: {exc.msg}")
    if not isinstance(tool_call, dict) or tool_call.get("name") != "web_search" or not isinstance(tool_call.get("arguments"), dict):
        return AgentAction(False, "invalid", error="tool_call must be a web_search object with arguments")
    queries = tool_call["arguments"].get("query")
    if isinstance(queries, str):
        queries = [queries]
        tool_call["arguments"]["query"] = queries
    if not isinstance(queries, list) or not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
        return AgentAction(False, "invalid", error="web_search requires at least one non-empty query")
    if max_search_queries < 1:
        raise ValueError("max_search_queries must be >= 1")
    # Keep the policy/environment contract robust for an untrained model: the
    # prompt asks for one query, but an accidental extra query must not turn an
    # otherwise executable action into a format failure.  The canonical tool
    # call, saved rollout trace, retrieval observation, and NDCG computation
    # therefore all use only the configured leading queries (one by default).
    tool_call["arguments"]["query"] = queries[:max_search_queries]
    return AgentAction(True, action, think=think, tool_call=tool_call)


_FIRST_COMPLETE_ACTION_RE = re.compile(
    r"^(<think>.*?</think>\s*<(tool_call|answer|clarify|nonanswer)>.*?</\2>)",
    flags=re.DOTALL,
)


def keep_first_complete_action(response: str) -> tuple[str, bool]:
    """Keep a completed first action and drop accidental continuation text.

    The rollout prompt already opens ``<think>``.  Some base models emit a
    valid action and then continue with a ChatML role marker or a second
    action.  The environment must execute only the first completed action;
    retaining a continuation would make an otherwise valid action fail the
    strict parser and would incorrectly train the policy on a fictitious
    multi-action turn.  Incomplete blocks are deliberately left untouched so
    they remain format errors.
    """
    content = str(response or "").replace("<|endoftext|>", "").strip()
    match = _FIRST_COMPLETE_ACTION_RE.match(content)
    if match is None:
        return content, False
    first_action = match.group(1).strip()
    return first_action, first_action != content


def build_simulated_user_system_prompt(
    base_prompt: str,
    *,
    allow_clarify: bool,
    allow_nonanswer: bool,
    max_search_queries: int,
) -> str:
    """Extend the existing strict IGPO prompt without weakening its contract."""
    if max_search_queries < 1:
        raise ValueError("max_search_queries must be >= 1")
    query_description = (
        "exactly one non-empty query string"
        if max_search_queries == 1
        else f"at most {max_search_queries} non-empty query strings"
    )
    legal = [
        "<think>YOUR THINKING PROCESS</think>\n<tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": [\"ONE QUERY\"]}}</tool_call>",
        "<think>YOUR THINKING PROCESS</think>\n<answer>YOUR ANSWER</answer>",
    ]
    if allow_clarify:
        legal.append("<think>YOUR THINKING PROCESS</think>\n<clarify>ONE CLEAR CLARIFYING QUESTION</clarify>")
    if allow_nonanswer:
        legal.append("<think>YOUR THINKING PROCESS</think>\n<nonanswer></nonanswer>")
    return (
        f"{base_prompt.rstrip()}\n\n"
        "Simulated-user task contract: output exactly ONE of the following forms, with no text outside it:\n"
        + "\n\nOR\n\n".join(legal)
        + "\n\nA clarification is legal only before any tool call or answer in the current user task. "
        "An answer and nonanswer terminate the current user task. "
        f"Every web_search tool call must contain {query_description}. "
        "The runtime has already written the opening `<think>` immediately before your generation. "
        "Therefore generate only the thinking content, then exactly one `</think>` and one action block; "
        "never emit another `<think>`, a new ChatML role marker such as `<|im_start|>`, a second action, or explanatory text after "
        "the closing action tag. Keep the thinking concise. After a tool call the environment will return passages "
        "and ask for the next action."
    )


RETRY_SYSTEM_PROMPT = (
    "RETRY MODE — you are repairing the same user task after user feedback. Your final terminal "
    "action for this task must be a non-empty `<answer>...</answer>`. In this retry state, each "
    "generation may be either exactly `<think>...</think><tool_call>...</tool_call>` to obtain more "
    "evidence, or exactly `<think>...</think><answer>...</answer>` to finish. The runtime already "
    "supplied the opening `<think>`, so do not emit another `<think>` or a new ChatML role marker. Do not "
    "output clarify, nonanswer, multiple actions, or text after the closing action tag."
)


@dataclass(frozen=True)
class UserJudgement:
    level: int
    feedback: str
    # ``simulator`` is a normal first-pass judgement; the other values make a
    # rare malformed/failed simulator response visible in rollout artefacts
    # and aggregate metrics instead of letting it silently affect training.
    source: str = "simulator"


def build_user_simulator_request_payload(
    *,
    dialogue_id: str,
    subtask_index: int,
    question: str,
    public_transcript: str,
    answer: str,
) -> dict[str, Any]:
    """Build the intentionally minimal request sent to the user simulator.

    Keeping this construction pure makes the privacy boundary easy to test:
    no hidden label, policy trace, system prompt, tool call, or retrieved
    passage can enter the request through this API.
    """
    return {
        "dialogue_id": dialogue_id,
        "subtask_index": subtask_index,
        "current_question": question,
        "public_dialogue_history": public_transcript,
        "assistant_final_answer": answer,
    }


def parse_user_simulator_judgement(content: Any) -> tuple[int, str]:
    """Parse the simulator's small JSON response with narrow recoveries.

    Qwen occasionally writes an apostrophe in a JSON string as ``\\'``.  That
    is accepted by some programming-language string literals but is *not*
    valid JSON, so Python correctly rejects it. It also occasionally omits the
    opening quote of the ``feedback`` value while retaining the two required
    field names. Recover only these observed slips, while still requiring an
    explicit level in {1, 2, 3} and an explicit feedback field. Other malformed
    output is not silently accepted.
    """
    raw = str(content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as first_error:
        repaired = raw.replace("\\'", "'")
        if repaired != raw:
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None
        if parsed is None:
            # Accept precisely ``{"level": 1|2|3, "feedback": text}``
            # when the model omitted only the quote around ``text``. This is
            # intentionally not a general best-effort JSON parser.
            level_match = re.search(r'"level"\s*:\s*([123])(?:\s*,|\s*})', repaired)
            # Covers the two observed malformed forms:
            #   "feedback": plain text}
            #   "feedback: "plain text"}
            # It deliberately does not accept unrelated/renamed fields.
            feedback_match = re.search(
                r'"feedback\s*:?[ \t]*"\s*:?[ \t]*"?(.*?)"?\s*}\s*$',
                repaired,
                flags=re.DOTALL,
            )
            if level_match is None or feedback_match is None:
                raise first_error
            feedback = feedback_match.group(1).strip()
            if len(feedback) >= 2 and feedback[0] == '"' and feedback[-1] == '"':
                # The normal JSON path would already have handled this. Keep
                # this guard for a mixed malformed response only.
                feedback = feedback[1:-1]
            if not feedback:
                raise first_error
            return int(level_match.group(1)), feedback
    if not isinstance(parsed, Mapping):
        raise ValueError("user simulator response must be a JSON object")
    level = int(parsed["level"])
    feedback = str(parsed.get("feedback", "")).strip()
    return level, feedback


class UserSimulatorClient:
    """Frozen Qwen user simulator through an OpenAI-compatible endpoint.

    ``mode=mock`` is deliberately simple and intended only for unit/smoke tests
    without a 32B server.  Real experiments must use ``mode=openai``.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str],
        model: Optional[str],
        timeout_seconds: int = 120,
        max_output_tokens: int = 128,
        mode: str = "openai",
    ) -> None:
        self.base_url = (base_url or os.environ.get("USER_SIMULATOR_BASE_URL", "")).rstrip("/")
        # vLLM exposes OpenAI routes under /v1. Accept either its host root
        # (http://host:8010) or the full OpenAI base URL (.../v1).
        if self.base_url and not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.model = model or os.environ.get("USER_SIMULATOR_MODEL", "")
        self.timeout_seconds = int(timeout_seconds)
        self.max_output_tokens = int(max_output_tokens)
        self.mode = str(mode).lower()
        if self.mode not in {"openai", "mock"}:
            raise ValueError("user simulator mode must be 'openai' or 'mock'")
        if self.mode == "openai" and (not self.base_url or not self.model):
            raise ValueError("OpenAI user simulator needs base_url and model")
        if self.max_output_tokens < 1:
            raise ValueError("user simulator max_output_tokens must be >= 1")

    def judge_answer(
        self,
        *,
        dialogue_id: str,
        subtask_index: int,
        question: str,
        public_transcript: str,
        answer: str,
    ) -> UserJudgement:
        """Judge only the user-visible part of a policy answer.

        ``public_transcript`` is deliberately a projection of the dialogue,
        not the policy's ChatML trajectory.  In particular it must not contain
        chain-of-thought, tool calls, retrieved passages, system prompts, or a
        hidden gold response.  The user simulator is a clarity/follow-up
        model, not a reference-answer judge.
        """
        if self.mode == "mock":
            return UserJudgement(
                level=1 if answer.strip() else 3,
                feedback="Please answer the question clearly.",
                source="mock",
            )

        import requests

        request_payload = build_user_simulator_request_payload(
            dialogue_id=dialogue_id,
            subtask_index=subtask_index,
            question=question,
            public_transcript=public_transcript,
            answer=answer,
        )
        system = (
            "You are a frozen simulated user evaluator. You receive only a public conversation history, "
            "the current user question, and the assistant's final answer. Do not answer the task yourself. "
            "Judge only whether that final answer is understandable and on-topic for the current user question; "
            "do not score factual correctness or rely on information not present in the public conversation. "
            "Return JSON only: {\"level\": 1|2|3, \"feedback\": \"short feedback\"}. "
            "Level 1: clear and understandable; level 2: unclear and needs a focused follow-up; "
            "level 3: off-topic or unintelligible. Feedback must be short, specific, and must not introduce new facts. "
            "Use plain text in feedback: do not use backslashes or quotation marks."
        )
        request_json = {
            "model": self.model,
            "temperature": 0.0,
            # The simulator only needs a small JSON judgement.  An explicit
            # limit reserves almost all of its 8192-token context window
            # for the bounded evaluation input below.
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
            ],
        }

        def request_judgement(payload: Mapping[str, Any]) -> tuple[Optional[tuple[int, str]], str]:
            """Make one bounded request and classify failure without leaking it.

            A rare malformed decode must not discard a 128-context online
            update.  The returned status is persisted on the corresponding
            event, so a run with non-negligible fallback usage is immediately
            identifiable and is not accidentally reported as a clean run.
            """
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=dict(payload),
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                return None, f"request_exception_{type(exc).__name__.lower()}"
            if not response.ok:
                return None, f"http_{response.status_code}"
            try:
                content = response.json()["choices"][0]["message"]["content"]
                level, feedback = parse_user_simulator_judgement(content)
                if level not in {1, 2, 3}:
                    return None, "invalid_level"
                if not feedback.strip():
                    return None, "empty_feedback"
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                return None, "invalid_output"
            return (level, feedback[:600]), "ok"

        result, status = request_judgement(request_json)
        if result is not None:
            return UserJudgement(level=result[0], feedback=result[1], source="simulator")

        # Do not use OpenAI ``response_format`` here.  With this Qwen/vLLM
        # deployment it occasionally degenerates into a bare ``{\"``.  A
        # second, short, explicit request is more reliable and keeps exactly
        # the same privacy-bounded user payload.
        retry_system = (
            "Return exactly one valid JSON object and nothing else. "
            "It must have exactly these keys: level and feedback. "
            "level must be 1, 2, or 3. feedback must be a short plain-text string. "
            'Example: {"level": 2, "feedback": "Please clarify the answer."}'
        )
        retry_json = dict(request_json)
        retry_json["max_tokens"] = min(self.max_output_tokens, 64)
        retry_json["messages"] = [
            {"role": "system", "content": retry_system},
            {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
        ]
        result, retry_status = request_judgement(retry_json)
        if result is not None:
            return UserJudgement(level=result[0], feedback=result[1], source="retry_success")

        # Conservative fallback: level 2 makes the environment ask the
        # policy to clarify/retry, rather than falsely rewarding a potentially
        # unreadable answer.  It is deliberately auditable via ``source``.
        return UserJudgement(
            level=2,
            feedback="Please clarify your answer for the user.",
            source=f"fallback_after_{status}_then_{retry_status}",
        )
