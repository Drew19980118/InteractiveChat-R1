"""Information-grounded target (IGT) reward for static ConvAgent.

The scorer is intentionally a *frozen external* language model.  For a
candidate policy answer, it compares the likelihood of reconstructing each
released gold answer when the simulator sees (a) that gold answer and (b) the
candidate answer as its reference.  It is therefore a reward-only component:
no simulator output, hidden gold label, or gradient is ever sent to the
policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Iterable, Mapping, Sequence


IGT_SIMULATOR_SYSTEM_PROMPT = (
    "You are simulating a user with the following conversation history and "
    "a current query."
)

_CONTEXT_PATTERN = re.compile(
    r"(?:Context Begin:|Conversation context:)\s*<context>\s*(.*?)\s*</context>",
    re.DOTALL | re.IGNORECASE,
)
_FALLBACK_CONTEXT_PATTERN = re.compile(r"<context>\s*(.*?)\s*</context>", re.DOTALL | re.IGNORECASE)
_QUERY_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Your Current Query|Current User Query|User query|Question)\s*:\s*"
    r"(.*?)(?=\n\s*(?:Available actions|Actions|Output format|Response format|"
    r"Instructions|Context End)\s*:|\n\s*<[^>]+>|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_USER_TURN_PATTERN = re.compile(
    r"(?:^|\n)User:\s*(.*?)(?=\n(?:Assistant|User):|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_static_convagent_public_state(prompt: str) -> tuple[str, str]:
    """Extract the public history and current query from a static prompt.

    ConvAgent source rows use ``Context Begin: <context>...</context>`` plus a
    current ``Question:``/``User query:`` field.  We fail closed when the
    current query cannot be recovered instead of silently placing hidden
    training metadata in the IGT prompt.
    """

    text = str(prompt or "")
    context_match = _CONTEXT_PATTERN.search(text) or _FALLBACK_CONTEXT_PATTERN.search(text)
    history = _normalize_whitespace(context_match.group(1)) if context_match else ""

    search_text = text[context_match.end() :] if context_match else text
    query_matches = list(_QUERY_PATTERN.finditer(search_text))
    query = _normalize_whitespace(query_matches[-1].group(1)) if query_matches else ""

    # Some released rows place the current turn inside the context and omit a
    # separate question marker.  The final public User turn is then the only
    # safe fallback; it still never reads reward-model labels or tool traces.
    if not query and history:
        user_turns = list(_USER_TURN_PATTERN.finditer(history))
        if user_turns:
            query = _normalize_whitespace(user_turns[-1].group(1))

    if not query:
        raise ValueError(
            "IGT reward could not extract the public current query from the static ConvAgent prompt. "
            "Expected a Context Begin/Conversation context block plus Question: or User query:."
        )
    return history, query


def build_igt_user_prompt(*, history: str, query: str, reference_answer: str) -> str:
    """Return the user-visible IGT scoring prompt accepted in the protocol."""

    return (
        "Conversation History:\n"
        "<conversation_history>\n"
        f"{history.strip()}\n"
        "</conversation_history>\n\n"
        "Your Current Query:\n"
        "<current_query>\n"
        f"{query.strip()}\n"
        "</current_query>\n\n"
        "Reference Answer:\n"
        "<reference_answer>\n"
        f"{reference_answer.strip()}\n"
        "</reference_answer>\n\n"
        "Task:\n"
        "Infer the answer to your current query using the conversation history,\n"
        "your current query, and the Reference Answer.\n\n"
        "Do not use any external knowledge beyond the information provided above.\n\n"
        "Give one direct, concise answer and nothing else.\n\n"
        "Predict Answer:\n"
    )


@dataclass(frozen=True)
class IGTScore:
    """Per-policy-answer IGT diagnostics used in rollout traces and W&B."""

    reward: float
    gap: float
    gold_logprob: float
    predicted_logprob: float
    references_scored: int


class IGTUserSimulatorScorer:
    """Batch, teacher-forced target scorer backed by a vLLM Completions API.

    vLLM 0.6.3 exposes ``prompt_logprobs`` on ``/v1/completions``.  We append
    the gold target to a completed chat prompt and read only those suffix token
    log-probabilities.  The simulator is never sampled for a free-form answer.
    """

    def __init__(
        self,
        *,
        tokenizer,
        base_url: str,
        model: str,
        timeout_seconds: int = 300,
        batch_size: int = 32,
        tau: float = 0.5,
        reward_floor: float = 0.0,
        reward_ceiling: float = 1.2,
    ) -> None:
        self.tokenizer = tokenizer
        self.base_url = str(base_url or "").rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = str(model or "").strip()
        self.timeout_seconds = int(timeout_seconds)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.reward_floor = float(reward_floor)
        self.reward_ceiling = float(reward_ceiling)
        self._gold_baseline_cache: dict[str, float] = {}

        if not self.base_url or not self.model:
            raise ValueError("IGT reward requires a simulator base URL and served model name")
        if self.timeout_seconds < 1 or self.batch_size < 1:
            raise ValueError("IGT timeout_seconds and batch_size must be positive")
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("IGT tau must be finite and > 0")
        if not math.isfinite(self.reward_floor) or not math.isfinite(self.reward_ceiling):
            raise ValueError("IGT reward bounds must be finite")
        if self.reward_ceiling < self.reward_floor:
            raise ValueError("IGT reward_ceiling must be >= reward_floor")

    def _chat_prefix(self, *, history: str, query: str, reference_answer: str) -> str:
        user_prompt = build_igt_user_prompt(
            history=history,
            query=query,
            reference_answer=reference_answer,
        )
        messages = [
            {"role": "system", "content": IGT_SIMULATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_template is None:
            # Qwen is expected in production.  This fallback makes pure unit
            # tests possible without weakening the scoring boundary.
            return f"{IGT_SIMULATOR_SYSTEM_PROMPT}\n\n{user_prompt}"
        return str(apply_template(messages, tokenize=False, add_generation_prompt=True))

    def _target_token_ids(self, target: str) -> list[int]:
        token_ids = self.tokenizer.encode(str(target), add_special_tokens=False)
        if not token_ids:
            raise ValueError("IGT target answer tokenized to an empty sequence")
        return [int(token_id) for token_id in token_ids]

    @staticmethod
    def _cache_key(prefix: str, target: str) -> str:
        return hashlib.sha256(f"{prefix}\0{target}".encode("utf-8")).hexdigest()

    @staticmethod
    def _as_logprob(value: Any) -> float | None:
        if isinstance(value, Mapping):
            value = value.get("logprob")
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _selected_prompt_logprob(self, entry: Any, token_id: int) -> float:
        if not isinstance(entry, Mapping):
            raise RuntimeError("vLLM returned a non-mapping prompt_logprobs entry")
        selected = entry.get(token_id)
        if selected is None:
            selected = entry.get(str(token_id))
        value = self._as_logprob(selected)
        if value is None:
            available = list(entry)[:5]
            raise RuntimeError(
                "vLLM prompt_logprobs did not include the forced target token "
                f"id={token_id}; available keys begin {available}."
            )
        return value

    def _score_suffix_batch(self, requests_to_score: Sequence[tuple[str, str]]) -> list[float]:
        """Return mean log-probability for each ``(prefix, target)`` pair."""

        if not requests_to_score:
            return []
        try:
            import requests
        except ImportError as error:  # pragma: no cover - production env has requests.
            raise RuntimeError("IGT reward requires the requests package") from error

        all_scores: list[float] = []
        for start in range(0, len(requests_to_score), self.batch_size):
            chunk = list(requests_to_score[start : start + self.batch_size])
            payload = {
                "model": self.model,
                "prompt": [prefix + target for prefix, target in chunk],
                # One generated token is required by the Completions API but
                # ignored. The target itself belongs to the scored prompt.
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "logprobs": 0,
                "prompt_logprobs": 1,
                "add_special_tokens": False,
            }
            try:
                response = requests.post(
                    f"{self.base_url}/v1/completions",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as error:
                raise RuntimeError(f"IGT simulator request failed: {type(error).__name__}") from error
            if not response.ok:
                raise RuntimeError(
                    "IGT simulator returned HTTP "
                    f"{response.status_code}: {response.text[:800]}"
                )
            try:
                choices = response.json()["choices"]
                choices = sorted(choices, key=lambda choice: int(choice.get("index", 0)))
            except (TypeError, KeyError, ValueError) as error:
                raise RuntimeError("IGT simulator returned an invalid Completions response") from error
            if len(choices) != len(chunk):
                raise RuntimeError(
                    "IGT simulator returned an unexpected number of choices: "
                    f"expected={len(chunk)} actual={len(choices)}"
                )

            for (prefix, target), choice in zip(chunk, choices):
                target_ids = self._target_token_ids(target)
                prompt_logprobs = choice.get("prompt_logprobs")
                if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) < len(target_ids):
                    raise RuntimeError(
                        "IGT simulator did not return enough prompt_logprobs for the forced target. "
                        "Ensure the server is vLLM >= 0.6 with prompt_logprobs support."
                    )
                suffix_entries = prompt_logprobs[-len(target_ids) :]
                token_logprobs = [
                    self._selected_prompt_logprob(entry, token_id)
                    for entry, token_id in zip(suffix_entries, target_ids)
                ]
                all_scores.append(float(sum(token_logprobs) / len(token_logprobs)))
        return all_scores

    def probe(self) -> float:
        """Perform one end-to-end prompt-logprob request before training starts."""

        prefix = self._chat_prefix(
            history="User: What is the capital of France?",
            query="What is the capital of France?",
            reference_answer="Paris is the capital of France.",
        )
        return self._score_suffix_batch([(prefix, "Paris")])[0]

    def score_records(self, records: Sequence[Mapping[str, Any]]) -> dict[int, IGTScore]:
        """Score valid terminal answer records and return diagnostics by row index.

        Each record must expose ``row_index``, ``history``, ``query``,
        ``predicted_answer`` and ``references``. Gold-reference scores are
        memoized across the n=8 rollouts and across future replays of a row.
        """

        pending: dict[str, tuple[str, str]] = {}
        # Keep this set separate from ``pending``.  The latter also contains
        # sampled actor answers and must not become a long-lived cache.
        missing_gold_keys: set[str] = set()
        record_pairs: dict[int, list[tuple[str, str]]] = {}
        for record in records:
            row_index = int(record["row_index"])
            history = str(record["history"])
            query = str(record["query"])
            predicted_answer = str(record["predicted_answer"])
            references = [str(value).strip() for value in record["references"] if str(value).strip()]
            pairs: list[tuple[str, str]] = []
            for reference in references:
                gold_prefix = self._chat_prefix(
                    history=history, query=query, reference_answer=reference
                )
                pred_prefix = self._chat_prefix(
                    history=history, query=query, reference_answer=predicted_answer
                )
                gold_key = self._cache_key(gold_prefix, reference)
                pred_key = self._cache_key(pred_prefix, reference)
                if gold_key not in self._gold_baseline_cache:
                    pending.setdefault(gold_key, (gold_prefix, reference))
                    missing_gold_keys.add(gold_key)
                pending.setdefault(pred_key, (pred_prefix, reference))
                pairs.append((gold_key, pred_key))
            if pairs:
                record_pairs[row_index] = pairs

        resolved_scores: dict[str, float] = {}
        if pending:
            values = self._score_suffix_batch(list(pending.values()))
            resolved_scores = dict(zip(pending, values))
            # Gold baselines are the only values deliberately retained across
            # updates. Candidate scores change with the policy and remain
            # local to this reward computation.
            self._gold_baseline_cache.update(
                {key: resolved_scores[key] for key in missing_gold_keys}
            )

        results: dict[int, IGTScore] = {}
        for row_index, pairs in record_pairs.items():
            candidates: list[IGTScore] = []
            for gold_key, pred_key in pairs:
                gold_logprob = resolved_scores.get(
                    gold_key, self._gold_baseline_cache[gold_key]
                )
                try:
                    predicted_logprob = resolved_scores[pred_key]
                except KeyError as error:
                    raise RuntimeError("IGT candidate score unexpectedly missing") from error
                gap = gold_logprob - predicted_logprob
                reward = min(
                    self.reward_ceiling,
                    max(self.reward_floor, 1.0 - gap / self.tau),
                )
                candidates.append(
                    IGTScore(
                        reward=float(reward),
                        gap=float(gap),
                        gold_logprob=float(gold_logprob),
                        predicted_logprob=float(predicted_logprob),
                        references_scored=len(pairs),
                    )
                )
            # Maximum IGT over released answer references matches max-F1.
            results[row_index] = max(candidates, key=lambda value: value.reward)

        return results
