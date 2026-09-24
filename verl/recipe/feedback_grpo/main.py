"""Train the alternating learned-feedback GRPO recipe on static ConvAgent data.

This entry point deliberately owns *coordination* only.  Search/action rollout
and grouped rewards live in :mod:`engine`; policy-gradient loss and FSDP work
live in the independent Ray workers.  Keeping the coordinator small makes the
two important invariants easy to audit:

* user feedback is sampled while the system policy is fixed, and its reward is
  the resulting improvement;
* one system first response and its learned-user feedback condition a same-
  prompt revised-answer GRPO group while the user policy is fixed.

The recipe is an experimental alternating conditional policy-gradient method,
not a claim that a two-player objective has a globally convergent optimum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
from statistics import fmean
from typing import Any, Iterable

from .artifacts import PairCheckpointStore, evaluate_jsonl
from .engine import Episode, SystemRequest, TwoRoundCollector
from .protocol import StaticExample


RECIPE_VERSION = 4
SYSTEM_UPDATE_MODE = "singleton_first_second_only_v1"
SELECTION_METRICS = ("f1", "bertscore_f1", "ndcg_at_3")
SELECTION_SETTINGS = {
    # This is the only setting whose test result is directly comparable with
    # the one-response ConvAgent/ChatR1 baselines.
    "direct-response": "first_response",
    # This selects the complete learned-feedback method. It must be reported
    # as a two-round interaction, not as a single-response baseline result.
    "feedback-refinement": "feedback_round_two",
}

# Keep this byte-for-byte-compatible in behavior with
# scripts/prepare_static_monitor_split.py.  The recipe must be runnable as a
# module even when ``scripts`` is not importable (for example, package-style
# server launches where it is not a Python namespace package).
_CONTEXT_PATTERN = re.compile(
    r"(?:Context Begin:|Conversation context:)\s*<context>(.*?)</context>",
    re.DOTALL | re.IGNORECASE,
)
_FIRST_USER_PATTERN = re.compile(
    r"(?:^|\n)User:\s*(.*?)(?=\nAssistant:|\nUser:|$)", re.DOTALL | re.IGNORECASE
)
_QUESTION_PATTERN = re.compile(
    r"(?:^|\n)(?:Question|User query):\s*(.*?)(?:\n|$)", re.DOTALL | re.IGNORECASE
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-model", required=True, type=Path)
    parser.add_argument("--user-model", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--selection-setting", choices=tuple(SELECTION_SETTINGS), default="direct-response",
                        help="one experimental setting per run; determines holdout selection and early stopping")
    parser.add_argument("--n-gpus", type=int, default=2)
    parser.add_argument("--sequence-parallel", type=int, default=1,
                        help="must be 1: system and learned-user each own one GPU")
    parser.add_argument("--n", type=int, default=8, help="GRPO candidates per source group")
    parser.add_argument("--train-batch-size", type=int, default=128,
                        help="source static rows per effective update")
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--user-updates-per-phase", type=int, default=5)
    parser.add_argument("--system-updates-per-phase", type=int, default=5)
    parser.add_argument("--max-system-updates", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=5,
                        help="completed system updates between full holdout checks")
    parser.add_argument("--patience", type=int, default=3,
                        help="strict non-improving validation checks before stopping")
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-response-length", type=int, default=500)
    parser.add_argument("--feedback-length", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.001)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--logprob-batch-size", type=int, default=8)
    parser.add_argument("--rollout-memory", type=float, default=0.15)
    parser.add_argument("--trace-samples", type=int, default=2)
    parser.add_argument("--max-empty-user-batches", type=int, default=20)
    parser.add_argument("--max-empty-system-batches", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/feedback_grpo"))
    parser.add_argument("--eval-root", type=Path, default=Path("eval_log/feedback_grpo"))
    parser.add_argument("--export-root", type=Path, default=Path("exports/feedback_grpo"))
    parser.add_argument("--bert-score-model", default="roberta-large")
    parser.add_argument("--bert-score-device", default="cpu")
    parser.add_argument("--bert-score-batch-size", type=int, default=8)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="interactivechat-r1")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--system-repo-id", default="")
    parser.add_argument("--user-repo-id", default="")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    _validate_args(arguments)
    return arguments


def _validate_args(args: argparse.Namespace) -> None:
    if not args.experiment or args.experiment.startswith(".") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in args.experiment):
        raise ValueError("--experiment must be a non-hidden simple directory name")
    positive = (
        "n_gpus", "sequence_parallel", "n", "train_batch_size", "val_batch_size",
        "user_updates_per_phase", "system_updates_per_phase", "max_system_updates",
        "validate_every", "patience", "max_prompt_length", "max_response_length",
        "feedback_length", "max_model_len", "max_turns", "rollout_batch_size",
        "logprob_batch_size", "bert_score_batch_size", "max_empty_user_batches",
        "max_empty_system_batches",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.n < 2:
        raise ValueError("--n must be at least 2 for GRPO")
    if args.sequence_parallel != 1:
        raise ValueError(
            "Feedback GRPO uses one whole GPU per independent policy; "
            "set --sequence-parallel=1 (ULYSSES_SEQUENCE_PARALLEL_SIZE=1)."
        )
    if not 0 < args.holdout_fraction < 1:
        raise ValueError("--holdout-fraction must be strictly between zero and one")
    if not 0 < args.rollout_memory < 1:
        raise ValueError("--rollout-memory must be strictly between zero and one")
    if args.learning_rate <= 0 or args.kl_coef < 0 or args.entropy_coef < 0:
        raise ValueError("learning rate must be positive; KL and entropy coefficients cannot be negative")
    if args.max_prompt_length + max(args.max_response_length, args.feedback_length) + 32 > args.max_model_len:
        raise ValueError("prompt/response limits do not fit in --max-model-len")
    for model in (args.system_model, args.user_model):
        if not (model / "config.json").is_file():
            raise FileNotFoundError(f"Model config.json is missing: {model}")
    for source in (args.train_file, args.test_file):
        if not source.is_file():
            raise FileNotFoundError(f"Static source data is missing: {source}")


def _as_python(value: Any) -> Any:
    """Convert Arrow/numpy scalar containers without altering user text."""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _as_python(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_as_python(item) for item in value]
    if isinstance(value, list):
        return [_as_python(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Parquet object columns can be JSON strings.  Decode only structures;
    # ordinary answer strings must remain exactly as released.
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _maybe_json(value: Any) -> Any:
    value = _as_python(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in {"[", "{"}:
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return value


def _messages(prompt: Any) -> list[dict[str, str]]:
    prompt = _maybe_json(prompt)
    if isinstance(prompt, dict):
        prompt = [prompt]
    if isinstance(prompt, (list, tuple)):
        result = []
        for item in prompt:
            item = _maybe_json(item)
            if not isinstance(item, dict) or "content" not in item:
                raise ValueError("Static prompt message must be a {role, content} mapping")
            role = str(item.get("role", "user") or "user")
            content = str(item.get("content", "") or "")
            result.append({"role": role, "content": content})
        if result:
            return result
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt}]
    raise ValueError("Static source row has an empty prompt")


def _prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def _ground_truth(row: dict[str, Any]) -> Any:
    reward_model = _maybe_json(row.get("reward_model"))
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return _maybe_json(reward_model["ground_truth"])
    if "ground_truth" in row:
        return _maybe_json(row["ground_truth"])
    raise ValueError("Static source row lacks reward_model.ground_truth")


def _data_source(row: dict[str, Any], fallback: str) -> str:
    value = row.get("data_source", fallback)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else fallback
    return str(value or fallback).strip().lower() or fallback


def examples_from_frame(frame: Any, *, prefix: str, fallback_data_source: str) -> list[StaticExample]:
    """Build source examples without adding gold labels to model prompts."""
    from verl.utils.reward_score.igt import extract_static_convagent_public_state

    examples: list[StaticExample] = []
    for position, (_, series) in enumerate(frame.iterrows()):
        row = {str(key): _as_python(value) for key, value in series.to_dict().items()}
        messages = _messages(row.get("prompt"))
        history, query = extract_static_convagent_public_state(_prompt_text(messages))
        examples.append(StaticExample(
            uid=f"{prefix}:{position}", prompt=messages, history=history, query=query,
            ground_truth=_ground_truth(row), data_source=_data_source(row, fallback_data_source),
        ))
    if not examples:
        raise ValueError(f"{prefix} source partition is empty")
    return examples


class SourceBatchStream:
    """Deterministic, resumable source-row batching without materializing epochs."""

    def __init__(self, examples: list[StaticExample], *, seed: int):
        self.examples, self.seed = examples, int(seed)
        self.epoch, self.offset = 0, 0

    def _order(self) -> list[int]:
        order = list(range(len(self.examples)))
        random.Random(self.seed + self.epoch).shuffle(order)
        return order

    def next(self, size: int) -> list[StaticExample]:
        result: list[StaticExample] = []
        while len(result) < size:
            order = self._order()
            remaining = len(order) - self.offset
            take = min(size - len(result), remaining)
            result.extend(self.examples[index] for index in order[self.offset:self.offset + take])
            self.offset += take
            if self.offset == len(order):
                self.epoch += 1
                self.offset = 0
        return result

    def state(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}

    def restore(self, state: dict[str, Any]) -> None:
        epoch, offset = state.get("epoch"), state.get("offset")
        if isinstance(epoch, bool) or isinstance(offset, bool) or not isinstance(epoch, int) or not isinstance(offset, int):
            raise ValueError("Invalid source stream state")
        if epoch < 0 or not 0 <= offset < len(self.examples):
            raise ValueError("Source stream state is out of range")
        self.epoch, self.offset = epoch, offset


def _dialogue_key_from_prompt(prompt: Any) -> str:
    """The exact deterministic conversation key used by all static baselines."""
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)):
        text = "\n".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in prompt)
    elif isinstance(prompt, dict):
        text = str(prompt.get("content", ""))
    else:
        text = str(prompt or "")
    context_match = _CONTEXT_PATTERN.search(text)
    context = context_match.group(1) if context_match else text
    user_match = _FIRST_USER_PATTERN.search(context)
    if user_match:
        value = user_match.group(1)
    else:
        question_match = _QUESTION_PATTERN.search(text)
        value = question_match.group(1) if question_match else text
    return re.sub(r"\s+", " ", value).strip().casefold()


def _baseline_matched_split(frame: Any, *, holdout_fraction: float, seed: int) -> tuple[Any, Any, dict[str, Any]]:
    """Conversation-disjoint static split, matching the baseline helper."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be strictly between 0 and 1")
    if "prompt" not in frame.columns:
        raise ValueError("static baseline Parquet must contain a 'prompt' column")
    dialogue_keys = [_dialogue_key_from_prompt(prompt) for prompt in frame["prompt"]]
    assignments: dict[str, bool] = {}
    for key in sorted(set(dialogue_keys)):
        digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
        assignments[key] = int.from_bytes(digest[:8], "big") / 2**64 < holdout_fraction
    monitor_mask = [assignments[key] for key in dialogue_keys]
    train = frame.loc[[not value for value in monitor_mask]].reset_index(drop=True)
    monitor = frame.loc[monitor_mask].reset_index(drop=True)
    if train.empty or monitor.empty:
        raise RuntimeError("conversation split produced an empty partition; change --seed or --holdout-fraction")
    return train, monitor, {
        "split_unit": "source_conversation", "seed": seed, "holdout_fraction": holdout_fraction,
        "source_rows": int(len(frame)), "source_dialogues": int(len(assignments)),
        "train_rows": int(len(train)), "monitor_rows": int(len(monitor)),
        "train_dialogues": int(sum(not value for value in assignments.values())),
        "monitor_dialogues": int(sum(assignments.values())),
    }


def _split_examples(args: argparse.Namespace) -> tuple[list[StaticExample], list[StaticExample], list[StaticExample], dict[str, Any]]:
    import pandas as pd

    source = pd.read_parquet(args.train_file)
    train, monitor, split = _baseline_matched_split(source, holdout_fraction=args.holdout_fraction, seed=args.seed)
    test = pd.read_parquet(args.test_file)
    split = {**split, "input": str(args.train_file.resolve()), "test_input": str(args.test_file.resolve())}
    return (
        examples_from_frame(train, prefix="train", fallback_data_source="inscit"),
        examples_from_frame(monitor, prefix="monitor", fallback_data_source="inscit"),
        examples_from_frame(test, prefix="test", fallback_data_source="inscit"),
        split,
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _configuration_signature(args: argparse.Namespace, split: dict[str, Any]) -> str:
    value = {
        "recipe_version": RECIPE_VERSION,
        "system_model": str(args.system_model.resolve()), "user_model": str(args.user_model.resolve()),
        "n": args.n, "train_batch_size": args.train_batch_size,
        "holdout_fraction": args.holdout_fraction, "seed": args.seed,
        "max_turns": args.max_turns, "selection_setting": args.selection_setting,
        "system_update_mode": SYSTEM_UPDATE_MODE,
        "max_empty_system_batches": args.max_empty_system_batches,
        "split": split,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _episode_row(example: StaticExample, episode: Episode, *, evaluation_mode: str,
                 inherited_passages: list[dict[str, str]] | None = None,
                 feedback: str | None = None) -> dict[str, Any]:
    from verl.utils.reward_score.static_convagent import (
        static_convagent_allowed_actions,
        static_convagent_reference_string_and_passage_ids,
    )

    assessment = episode.assessment
    if assessment is None:
        raise RuntimeError("Cannot evaluate an unfinished episode")
    answer_references, passage_ids = static_convagent_reference_string_and_passage_ids(example.ground_truth)
    passages = episode.passages or inherited_passages or []
    # The shared evaluator deliberately extracts ``<answer>`` rather than
    # trusting a free-text field. Preserve the full terminal model response
    # here; passing ``assessment.content`` would silently turn every F1 and
    # BERTScore value into zero because it contains no answer tags.
    selected = episode.raw_response if assessment.format_valid and assessment.action == "answer" else ""
    return {
        "source_id": example.uid,
        "data_source": example.data_source,
        "evaluation_mode": evaluation_mode,
        "raw_response": episode.raw_response,
        "answer_raw_response": selected,
        "answer_selection_source": "terminal" if selected else "missing",
        "ground_truth_answer": answer_references,
        "ground_truth_passage_ids": passage_ids,
        "predicted_passage_ids": [str(p.get("passage_id", "")) for p in passages if p.get("passage_id")],
        "predicted_passage_texts": [str(p.get("passage_text", "")) for p in passages if p.get("passage_text")],
        "permissible_actions": sorted(static_convagent_allowed_actions(example.ground_truth, example.data_source)),
        "expected_action": sorted(static_convagent_allowed_actions(example.ground_truth, example.data_source)),
        "predicted_action": assessment.action,
        "format_valid": assessment.format_valid,
        "reward": assessment.reward,
        "f1_reward_component": assessment.f1,
        "action_reward_component": assessment.action_score,
        "tool_calls": episode.tool_calls,
        "feedback": feedback,
    }


def _metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in SELECTION_METRICS:
        value = metrics.get(name)
        if value is None or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
            raise ValueError(f"Evaluation summary has invalid {name}: {value!r}")
        result[name] = float(value)
    return result


def _selection_score(summary: dict[str, Any]) -> float:
    # All three values are bounded [0, 1] by their metric definitions, so this
    # is the requested normalized composite (not a post-hoc fitted weighting).
    values = _metric_summary(summary["metrics"])
    return fmean(max(0.0, min(1.0, values[name])) for name in SELECTION_METRICS)


def _selection_summary(setting: str, first_summary: dict[str, Any],
                       revised_summary: dict[str, Any]) -> dict[str, Any]:
    """Return the one holdout summary allowed to select this run's pair."""
    try:
        mode = SELECTION_SETTINGS[setting]
    except KeyError as error:
        raise ValueError(f"Unknown selection setting: {setting}") from error
    return first_summary if mode == "first_response" else revised_summary


def evaluate_pair(collector: TwoRoundCollector, examples: list[StaticExample], *,
                  batch_size: int, output_dir: Path, label: str,
                  bert_model: str, bert_device: str, bert_batch_size: int,
                  selection_setting: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    """Evaluate first-response and strict revised-response modes on every row."""
    output_dir.mkdir(parents=True, exist_ok=True)
    first_rows, revised_rows = [], []
    revised_deltas, first_valid, revised_valid = [], 0, 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        first, final, feedbacks = collector.evaluate_batch(batch)
        for example, initial, revised, feedback in zip(batch, first, final, feedbacks):
            first_rows.append(_episode_row(example, initial, evaluation_mode="first_response"))
            revised_rows.append(_episode_row(
                example, revised, evaluation_mode="feedback_round_two",
                inherited_passages=initial.passages, feedback=feedback,
            ))
            first_valid += int(initial.assessment.format_valid)
            revised_valid += int(revised.assessment.format_valid)
            revised_deltas.append(revised.assessment.reward - initial.assessment.reward)
    first_path = output_dir / "first_response.jsonl"
    revised_path = output_dir / "feedback_round_two.jsonl"
    for path, rows in ((first_path, first_rows), (revised_path, revised_rows)):
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    first_summary = evaluate_jsonl(first_path, output_dir / "first_response_metrics", bert_model, bert_device, bert_batch_size)
    revised_summary = evaluate_jsonl(revised_path, output_dir / "feedback_round_two_metrics", bert_model, bert_device, bert_batch_size)
    diagnostics = {
        f"{label}/feedback/reward_delta_mean": fmean(revised_deltas) if revised_deltas else 0.0,
        f"{label}/feedback/reward_improvement_rate": sum(delta > 0 for delta in revised_deltas) / max(1, len(revised_deltas)),
        f"{label}/first/format_success_rate": first_valid / max(1, len(first_rows)),
        f"{label}/round_two/format_success_rate": revised_valid / max(1, len(revised_rows)),
        f"{label}/sources": float(len(examples)),
    }
    _atomic_json(output_dir / "evaluation_index.json", {
        "first_response": str((output_dir / "first_response_metrics" / "metrics_summary.json").resolve()),
        "feedback_round_two": str((output_dir / "feedback_round_two_metrics" / "metrics_summary.json").resolve()),
        "selection_setting": selection_setting,
        "selection_evaluation_mode": SELECTION_SETTINGS[selection_setting],
        "diagnostics": diagnostics,
    })
    return first_summary, revised_summary, diagnostics


def _log(tracker: Any | None, values: dict[str, float], step: int) -> None:
    if tracker is not None:
        tracker.log(values, step=step)
    concise = {key: round(value, 6) for key, value in values.items()
               if key.endswith(("/mean", "/f1", "/bertscore_f1", "/ndcg_at_3", "/composite"))}
    if concise:
        print("[FeedbackGRPO]", json.dumps(concise, sort_keys=True), flush=True)


def _coordinator_state(*, args: argparse.Namespace, signature: str, stream: SourceBatchStream,
                       phase: str, user_in_phase: int, system_in_phase: int,
                       user_updates: int, system_updates: int, validation_checks: int,
                       best_score: float | None, non_improving: int, event_step: int) -> dict[str, Any]:
    return {
        "recipe": "feedback_grpo", "version": RECIPE_VERSION,
        "configuration_signature": signature, "system_update_mode": SYSTEM_UPDATE_MODE,
        "phase": phase,
        "user_in_phase": user_in_phase, "system_in_phase": system_in_phase,
        "user_updates": user_updates, "system_updates": system_updates,
        "validation_checks": validation_checks, "best_score": best_score,
        "non_improving_checks": non_improving, "event_step": event_step,
        "source_stream": stream.state(),
        "selection": {
            "setting": args.selection_setting,
            "evaluation_mode": SELECTION_SETTINGS[args.selection_setting],
            "metrics": list(SELECTION_METRICS),
            "strict_improvement": True,
            "patience": args.patience,
            "best_score": best_score,
            "best_system_update": system_updates if best_score is not None and non_improving == 0 else None,
        },
    }


def _write_traces(path: Path, phase: str, update: int, traces: list[dict[str, Any]]) -> None:
    if not traces:
        return
    target = path / "traces" / f"{phase}_{update:06d}.json"
    _atomic_json(target, {"phase": phase, "update": update, "traces": traces})


def _metadata(args: argparse.Namespace, checkpoint: Path, state: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    return {
        "recipe": "feedback_grpo", "recipe_version": RECIPE_VERSION,
        "pairing": "The system and user directories are one selected checkpoint pair; use them together.",
        "selected_checkpoint": str(checkpoint),
        "system_repository": args.system_repo_id or None,
        "user_repository": args.user_repo_id or None,
        "selection": state["selection"], "selected_state": state,
        "split": split,
        "reward": {
            "system": "max-reference answer token-set F1 + 0.5 * action score; action score is +1 permissible/correct and -0.5 wrong/malformed; retrieval reward is zero",
            "user": "if first and revised actions are both answer: system_reward(revised)-system_reward(first); otherwise 0.5*(action_score(revised)-action_score(first))",
        },
        "system_update": (
            "one retried first response and one frozen-user feedback condition n revised responses; "
            "only the same-prompt revised-response group updates the system policy"
        ),
        "evaluation": "Both official-test modes use maximum released answer references. Direct-response is comparable to ordinary static baselines; feedback-refinement is explicitly a two-round interaction result.",
    }


def _export_pair(backend: Any, args: argparse.Namespace, *, checkpoint: Path,
                 state: dict[str, Any], split: dict[str, Any]) -> Path:
    root = (args.export_root / args.experiment).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing export directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        backend.export_pair(root)
        metadata = _metadata(args, checkpoint, state, split)
        for role in ("system", "user"):
            role_metadata = {**metadata, "role": role, "counterpart_role": "user" if role == "system" else "system"}
            _atomic_json(root / role / "feedback_grpo_pair.json", role_metadata)
            (root / role / "README.md").write_text(
                "# Feedback-GRPO paired policy\n\n"
                f"This directory is the **{role}** role of a selected learned-feedback GRPO pair. "
                "Load it only with the matching counterpart model described in `feedback_grpo_pair.json`.\n",
                encoding="utf-8",
            )
        _atomic_json(root / "feedback_grpo_pair.json", metadata)
    except Exception:
        # An incomplete export is never a valid upload artifact. It is local,
        # recipe-owned output, so removing it is safe after explicit failure.
        if root.exists():
            shutil.rmtree(root)
        raise
    return root


def run(args: argparse.Namespace) -> None:
    from .backend import RayBackend
    from verl.utils.tracking import Tracking

    train, monitor, test, split = _split_examples(args)
    experiment_root = (args.output_root / args.experiment).resolve()
    store = PairCheckpointStore(experiment_root)
    manifest = store.root / "split_manifest.json"
    signature = _configuration_signature(args, split)
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("configuration_signature") != signature:
            raise ValueError("Existing experiment directory has a different static split/configuration; choose a new --experiment")
    else:
        _atomic_json(manifest, {"configuration_signature": signature, **split})

    tracker = Tracking(
        project_name=args.wandb_project,
        experiment_name=args.experiment,
        default_backend=["console", "wandb"] if args.wandb else ["console"],
        config={"recipe": "feedback_grpo", "n": args.n, "train_batch_size": args.train_batch_size,
                "val_batch_size": args.val_batch_size, "seed": args.seed, "split": split,
                "selection_setting": args.selection_setting,
                "system_update_mode": SYSTEM_UPDATE_MODE,
                "system_model": str(args.system_model), "user_model": str(args.user_model)},
    )
    stream = SourceBatchStream(train, seed=args.seed)
    phase, user_in_phase, system_in_phase = "user", 0, 0
    user_updates = system_updates = validation_checks = event_step = 0
    best_score: float | None = None
    non_improving = 0
    backend = None
    try:
        backend = RayBackend(args)
        collector = TwoRoundCollector(backend, n=args.n, max_turns=args.max_turns, first_retries=2,
                                      trace_samples=args.trace_samples)
        if args.resume:
            checkpoint, saved = store.load_state()
            if saved.get("configuration_signature") != signature:
                raise ValueError("Refusing resume: paired checkpoint was created with a different configuration")
            selection = saved.get("selection")
            if not isinstance(selection, dict) or selection.get("setting") != args.selection_setting:
                raise ValueError("Refusing resume: checkpoint belongs to a different selection setting")
            backend.load_pair(checkpoint)
            stream.restore(saved["source_stream"])
            phase = saved["phase"]
            user_in_phase, system_in_phase = int(saved["user_in_phase"]), int(saved["system_in_phase"])
            user_updates, system_updates = int(saved["user_updates"]), int(saved["system_updates"])
            validation_checks, event_step = int(saved["validation_checks"]), int(saved["event_step"])
            best_score, non_improving = saved.get("best_score"), int(saved["non_improving_checks"])
            if best_score is not None and (not isinstance(best_score, (int, float)) or not math.isfinite(float(best_score))):
                raise ValueError("Resume checkpoint has an invalid best selection score")
            best_score = None if best_score is None else float(best_score)
            if non_improving < 0:
                raise ValueError("Resume checkpoint has an invalid non-improving count")
            print(f"[FeedbackGRPO] resumed selected pair at {checkpoint}; system_updates={system_updates}", flush=True)
        elif (store.root / "final_checkpoint.txt").exists():
            raise FileExistsError("Experiment already has a selected pair; use --resume or a new --experiment")

        empty_user_batches = empty_system_batches = 0
        stopped_early = False
        while system_updates < args.max_system_updates and not stopped_early:
            examples = stream.next(args.train_batch_size)
            if phase == "user":
                items, metrics, traces = collector.user_batch(examples, f"u{user_updates + 1}")
                if not items:
                    empty_user_batches += 1
                    event_step += 1
                    _log(tracker, {**metrics, "user/empty_effective_batch": 1.0}, event_step)
                    _write_traces(experiment_root, "user_empty", user_updates + 1, traces)
                    if empty_user_batches >= args.max_empty_user_batches:
                        raise RuntimeError("Too many consecutive empty user batches; inspect format/retriever traces")
                    continue
                empty_user_batches = 0
                update_metrics = backend.update("user", items)
                user_updates += 1
                user_in_phase += 1
                event_step += 1
                _log(tracker, {**metrics, **update_metrics, "train/user_updates": float(user_updates)}, event_step)
                _write_traces(experiment_root, "user", user_updates, traces)
                if user_in_phase == args.user_updates_per_phase:
                    phase, user_in_phase = "system", 0
                    print("[FeedbackGRPO] switching to frozen-user system phase", flush=True)
                continue

            items, metrics, traces = collector.system_batch(examples, f"s{system_updates + 1}")
            if not items:
                empty_system_batches += 1
                event_step += 1
                _log(tracker, {**metrics, "system/empty_effective_batch": 1.0}, event_step)
                _write_traces(experiment_root, "system_empty", system_updates + 1, traces)
                if empty_system_batches >= args.max_empty_system_batches:
                    raise RuntimeError("Too many consecutive empty system batches; inspect format/retriever traces")
                continue
            empty_system_batches = 0
            update_metrics = backend.update("system", items)
            system_updates += 1
            system_in_phase += 1
            event_step += 1
            _log(tracker, {**metrics, **update_metrics, "train/system_updates": float(system_updates)}, event_step)
            _write_traces(experiment_root, "system", system_updates, traces)
            if system_in_phase == args.system_updates_per_phase:
                phase, system_in_phase = "user", 0
                print("[FeedbackGRPO] switching to frozen-system user phase", flush=True)

            if system_updates % args.validate_every:
                continue
            validation_checks += 1
            validation_dir = (args.eval_root / args.experiment / "monitor" / f"system_step_{system_updates}").resolve()
            first_summary, revised_summary, diagnostics = evaluate_pair(
                collector, monitor, batch_size=args.val_batch_size, output_dir=validation_dir,
                label="val", bert_model=args.bert_score_model, bert_device=args.bert_score_device,
                bert_batch_size=args.bert_score_batch_size, selection_setting=args.selection_setting,
            )
            score = _selection_score(_selection_summary(args.selection_setting, first_summary, revised_summary))
            values = {
                **{f"val/direct_response/{key}": value for key, value in _metric_summary(first_summary["metrics"]).items()},
                "val/direct_response/composite": _selection_score(first_summary),
                **{f"val/feedback_refinement/{key}": value for key, value in _metric_summary(revised_summary["metrics"]).items()},
                "val/feedback_refinement/composite": _selection_score(revised_summary),
                "val/selection/composite": score,
                **diagnostics,
            }
            event_step += 1
            _log(tracker, values, event_step)
            if best_score is None or score > best_score:
                best_score, non_improving = score, 0
                state = _coordinator_state(
                    args=args, signature=signature, stream=stream, phase=phase,
                    user_in_phase=user_in_phase, system_in_phase=system_in_phase,
                    user_updates=user_updates, system_updates=system_updates,
                    validation_checks=validation_checks, best_score=best_score,
                    non_improving=non_improving, event_step=event_step,
                )
                checkpoint = store.save(system_updates, state, backend.save_policy)
                print(
                    f"[FeedbackGRPO] new {args.selection_setting} paired best ({score:.6f}) saved: {checkpoint}",
                    flush=True,
                )
            else:
                non_improving += 1
                print(
                    f"[FeedbackGRPO] {args.selection_setting} monitor did not improve "
                    f"({score:.6f} <= {best_score:.6f}); {non_improving}/{args.patience}",
                    flush=True,
                )
                if non_improving >= args.patience:
                    stopped_early = True

        if best_score is None:
            # The user requested selection every five system batches. If the
            # budget is shorter, still make one full, explicit selection.
            validation_checks += 1
            validation_dir = (args.eval_root / args.experiment / "monitor" / f"system_step_{system_updates}_final").resolve()
            first_summary, revised_summary, diagnostics = evaluate_pair(
                collector, monitor, batch_size=args.val_batch_size, output_dir=validation_dir,
                label="val", bert_model=args.bert_score_model, bert_device=args.bert_score_device,
                bert_batch_size=args.bert_score_batch_size, selection_setting=args.selection_setting,
            )
            best_score = _selection_score(_selection_summary(args.selection_setting, first_summary, revised_summary))
            state = _coordinator_state(
                args=args, signature=signature, stream=stream, phase=phase, user_in_phase=user_in_phase,
                system_in_phase=system_in_phase, user_updates=user_updates, system_updates=system_updates,
                validation_checks=validation_checks, best_score=best_score, non_improving=0, event_step=event_step,
            )
            checkpoint = store.save(system_updates, state, backend.save_policy)
            print(f"[FeedbackGRPO] initial {args.selection_setting} paired best ({best_score:.6f}) saved: {checkpoint}", flush=True)
            _log(tracker, {
                **{f"val/direct_response/{key}": value for key, value in _metric_summary(first_summary["metrics"]).items()},
                "val/direct_response/composite": _selection_score(first_summary),
                **{f"val/feedback_refinement/{key}": value for key, value in _metric_summary(revised_summary["metrics"]).items()},
                "val/feedback_refinement/composite": _selection_score(revised_summary),
                "val/selection/composite": best_score,
                **diagnostics,
            }, event_step + 1)

        checkpoint, selected_state = store.load_state()
        backend.load_pair(checkpoint)
        test_dir = (args.eval_root / args.experiment / "official_test").resolve()
        first_summary, revised_summary, diagnostics = evaluate_pair(
            collector, test, batch_size=args.val_batch_size, output_dir=test_dir, label="test",
            bert_model=args.bert_score_model, bert_device=args.bert_score_device,
            bert_batch_size=args.bert_score_batch_size, selection_setting=args.selection_setting,
        )
        event_step += 1
        _log(tracker, {
            **{f"test/direct_response/{key}": value for key, value in _metric_summary(first_summary["metrics"]).items()},
            **{f"test/feedback_refinement/{key}": value for key, value in _metric_summary(revised_summary["metrics"]).items()},
            **diagnostics,
        }, event_step)
        export = _export_pair(backend, args, checkpoint=checkpoint, state=selected_state, split=split)
        print("[FeedbackGRPO] complete", json.dumps({
            "selection_setting": args.selection_setting,
            "selected_pair": str(checkpoint),
            "direct_test_metrics": str(test_dir / "first_response_metrics" / "metrics_summary.json"),
            "feedback_test_metrics": str(test_dir / "feedback_round_two_metrics" / "metrics_summary.json"),
            "export": str(export),
        }, ensure_ascii=False), flush=True)
    finally:
        if backend is not None:
            backend.close()


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
