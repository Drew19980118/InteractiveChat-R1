#!/usr/bin/env python3
"""Compute requested answer/retrieval metrics for one InteractiveChat-R1 validation JSONL.

For simulated-user validation, ``answer_raw_response`` is the last valid
user-visible answer within the original sub-task.  Otherwise ``raw_response``
is used.  Prompts are intentionally never inspected because they contain
format examples.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
import string
from pathlib import Path
from statistics import fmean
from typing import Any


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Validation JSONL.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bert-score-model", default="roberta-large")
    parser.add_argument("--bert-score-batch-size", default=16, type=int)
    parser.add_argument("--bert-score-device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument(
        "--simulated-user-batches-evaluated",
        type=int,
        default=None,
        help="Explicit batch count for legacy simulated-user JSONL that lacks per-row batch metadata.",
    )
    parser.add_argument(
        "--simulated-user-validation-truncated",
        action="store_const",
        const=True,
        default=None,
        help="Mark a legacy simulated-user JSONL as intentionally validation-capped.",
    )
    return parser.parse_args()


def as_text(value: Any) -> str:
    return "" if value is None else str(value)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_string_list(value: Any) -> list[str]:
    """Convert JSON scalars/lists to strings; do not fuzzy-normalize IDs."""
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [as_text(value)]
    if isinstance(value, (list, tuple, set)):
        return [as_text(item) for item in value if item is not None]
    return [as_text(value)]


def extract_last_answer(raw_response: Any) -> str:
    matches = ANSWER_PATTERN.findall(as_text(raw_response))
    return matches[-1].strip() if matches else ""


def select_text_answer_raw_response(row: dict[str, Any]) -> tuple[str, str]:
    """Return the text-metric answer source, including old SimUser JSONL.

    New validation records carry ``answer_raw_response`` explicitly.  Older
    records retain their per-subtask ``events`` list, from which the same
    chronological last-valid-answer fallback can be reconstructed.
    """
    if "answer_raw_response" in row:
        return as_text(row.get("answer_raw_response")), as_text(
            row.get("answer_selection_source", "terminal")
        )

    events = row.get("events")
    if isinstance(events, list) and events:
        terminal = events[-1] if isinstance(events[-1], dict) else {}
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            if event.get("predicted_action") != "answer" or not bool(event.get("format_valid", False)):
                continue
            raw_response = as_text(event.get("raw_response"))
            if extract_last_answer(raw_response):
                return raw_response, "terminal" if event is terminal else "last_valid_answer"
        return "", "missing"

    return as_text(row.get("raw_response", "")), "terminal"


def normalize_for_f1(text: str) -> str:
    # Same normalization as verl.utils.reward_score.info_gain.compute_f1.
    text = text.lower()
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " ")
    return re.sub(r"\s+", " ", text).strip()


def answer_f1(predicted_answer: str, ground_truth_answer: str) -> float:
    """Token-set F1, taking the maximum across <|answer_split|> labels."""
    if not predicted_answer or not ground_truth_answer:
        return 0.0
    predicted_tokens = set(normalize_for_f1(predicted_answer).split())
    if not predicted_tokens:
        return 0.0

    best = 0.0
    for reference in ground_truth_answer.split("<|answer_split|>"):
        reference_tokens = set(normalize_for_f1(reference).split())
        if not reference_tokens:
            continue
        overlap = len(predicted_tokens & reference_tokens)
        if not overlap:
            continue
        precision = overlap / len(predicted_tokens)
        recall = overlap / len(reference_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _canonical_passage_text(value: Any) -> str:
    """Normalize presentation-only text differences for TopiOCQA relevance."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def ndcg_at_3(
    predicted_passages: Any,
    ground_truth_passages: Any,
    *,
    match_by_text: bool = False,
) -> float:
    """Binary NDCG@3 with exact IDs or normalized passage-text relevance."""
    canonicalize = _canonical_passage_text if match_by_text else lambda value: str(value).strip()
    labels = {
        canonicalize(value)
        for value in as_string_list(ground_truth_passages)
        if canonicalize(value) != ""
    }
    if not labels:
        return 0.0

    dcg = 0.0
    matched_labels: set[str] = set()
    for rank, passage in enumerate(as_string_list(predicted_passages)[:3]):
        passage_key = canonicalize(passage)
        if passage_key in labels and passage_key not in matched_labels:
            dcg += 1.0 / math.log2(rank + 2)
            matched_labels.add(passage_key)

    ideal_count = min(3, len(labels))
    ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def resolve_bertscore_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def calculate_bertscore(records: list[dict[str, Any]], model: str, batch_size: int, device: str) -> None:
    valid_indices = [
        index
        for index, record in enumerate(records)
        if record["predicted_answer"] and record["ground_truth_answer"]
    ]
    if not valid_indices:
        return

    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore is not installed. Run: python -m pip install 'bert-score==0.3.13'"
        ) from exc

    candidates = [records[index]["predicted_answer"] for index in valid_indices]
    references = [records[index]["ground_truth_answer"] for index in valid_indices]
    _, _, f1_values = bert_score(
        candidates,
        references,
        model_type=model,
        lang="en",
        idf=False,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=False,
        verbose=True,
    )
    for index, value in zip(valid_indices, f1_values.tolist(), strict=True):
        records[index]["bertscore_f1"] = float(value)


def read_jsonl(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Validation JSONL does not exist: {input_path}")
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {input_path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}: {input_path}")
            rows.append(record)
    if not rows:
        raise ValueError(f"Validation JSONL is empty: {input_path}")
    return rows


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    per_sample: list[dict[str, Any]] = []

    for sample_index, row in enumerate(rows):
        terminal_raw_response = row.get("raw_response", "")
        answer_raw_response, answer_selection_source = select_text_answer_raw_response(row)
        predicted_answer = extract_last_answer(answer_raw_response)
        terminal_predicted_answer = extract_last_answer(terminal_raw_response)
        ground_truth_answer = as_text(row.get("ground_truth_answer", row.get("ground_truth", "")))
        ground_truth_passage_ids = as_string_list(row.get("ground_truth_passage_ids", []))
        ground_truth_passage_texts = as_string_list(row.get("ground_truth_passage_texts", []))
        predicted_passage_ids = as_string_list(row.get("predicted_passage_ids", []))
        predicted_passage_texts = as_string_list(row.get("predicted_passage_texts", []))
        data_source = as_text(row.get("data_source", "")).strip().lower()
        evaluation_mode = as_text(
            row.get(
                "evaluation_mode",
                "simulated_user" if isinstance(row.get("events"), list) else "static",
            )
        ).strip().lower()
        retrieval_match_mode = "passage_text" if data_source == "topiocqa" and ground_truth_passage_texts else "passage_id"
        if retrieval_match_mode == "passage_text" and "predicted_passage_texts" not in row:
            raise ValueError(
                "TopiOCQA validation JSONL lacks predicted_passage_texts, so text-based NDCG@3 "
                "cannot be computed. Re-run validation with the updated simulated-user rollout code."
            )
        predicted_passages = predicted_passage_texts if retrieval_match_mode == "passage_text" else predicted_passage_ids
        ground_truth_passages = (
            ground_truth_passage_texts if retrieval_match_mode == "passage_text" else ground_truth_passage_ids
        )
        expected_action = as_text(row.get("expected_action", "")).strip().lower()
        predicted_action = as_text(row.get("predicted_action", "")).strip().lower()
        format_valid = bool(row.get("format_valid", False)) if expected_action else None
        action_correct = (
            bool(format_valid and predicted_action == expected_action)
            if expected_action in {"answer", "clarify", "nonanswer"}
            else None
        )
        per_sample.append(
            {
                "sample_index": sample_index,
                "predicted_answer": predicted_answer,
                "terminal_predicted_answer": terminal_predicted_answer,
                "answer_selection_source": answer_selection_source,
                "ground_truth_answer": ground_truth_answer,
                "has_terminal_answer": bool(terminal_predicted_answer),
                "has_selected_answer": bool(predicted_answer),
                "ground_truth_passage_ids": ground_truth_passage_ids,
                "ground_truth_passage_texts": ground_truth_passage_texts,
                "predicted_passage_ids": predicted_passage_ids,
                "predicted_passage_texts": predicted_passage_texts,
                "retrieval_match_mode": retrieval_match_mode,
                "evaluation_mode": evaluation_mode,
                "f1": answer_f1(predicted_answer, ground_truth_answer),
                "bertscore_f1": 0.0,
                "ndcg_at_3": ndcg_at_3(
                    predicted_passages,
                    ground_truth_passages,
                    match_by_text=retrieval_match_mode == "passage_text",
                ),
                "expected_action": expected_action or None,
                "predicted_action": predicted_action or None,
                "format_valid": format_valid,
                "action_accuracy": float(action_correct) if action_correct is not None else None,
                "simulator_level": row.get("simulator_level"),
                "simulator_status": as_text(row.get("simulator_status", "")),
                "simulator_feedback_enabled": bool(row.get("simulator_feedback_enabled", False)),
                "simulator_satisfaction_assessed": bool(
                    row.get("simulator_satisfaction_assessed", row.get("simulator_feedback_enabled", False))
                ),
                "simulator_judgement_count": as_float(row.get("simulator_judgement_count")) or 0.0,
                "simulator_fallback_count": as_float(row.get("simulator_fallback_count")) or 0.0,
                "retry_depth": as_float(row.get("retry_depth")),
                "tool_call_count": as_float(row.get("tool_call_count")),
                "validation_batch_index": row.get("validation_batch_index"),
                "validation_truncated": row.get("validation_truncated"),
            }
        )

    bertscore_device = resolve_bertscore_device(args.bert_score_device)
    calculate_bertscore(per_sample, args.bert_score_model, args.bert_score_batch_size, bertscore_device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.output_dir / "metrics_per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for record in per_sample:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    action_labeled = [record for record in per_sample if record["action_accuracy"] is not None]
    # A nonanswer/clarify target has no answer text to score, so it must not
    # dilute answer F1, BERTScore, or answer retrieval NDCG. This also handles
    # static ConvAgent rows whose unsupported clarification target is exported
    # with an empty selected answer and no expected terminal action.
    answer_labeled = [
        record
        for record in per_sample
        if record["expected_action"] in {None, "answer"} and bool(record["ground_truth_answer"])
    ]
    simulated_user_records = [
        record for record in per_sample if record["evaluation_mode"] == "simulated_user"
    ]
    online_metrics: dict[str, float] = {}
    if simulated_user_records:
        feedback_enabled = any(
            bool(record["simulator_feedback_enabled"]) for record in simulated_user_records
        )
        satisfaction_assessed = any(
            bool(record["simulator_satisfaction_assessed"])
            for record in simulated_user_records
        )
        available_batch_indices = {
            int(record["validation_batch_index"])
            for record in simulated_user_records
            if record["validation_batch_index"] is not None
        }
        reported_truncation = [
            bool(record["validation_truncated"])
            for record in simulated_user_records
            if record["validation_truncated"] is not None
        ]
        if args.simulated_user_batches_evaluated is not None:
            if args.simulated_user_batches_evaluated < 1:
                raise ValueError("--simulated-user-batches-evaluated must be >= 1")
            batches_evaluated = float(args.simulated_user_batches_evaluated)
        elif available_batch_indices:
            batches_evaluated = float(len(available_batch_indices))
        else:
            batches_evaluated = None
        if args.simulated_user_validation_truncated is not None:
            validation_truncated = float(args.simulated_user_validation_truncated)
        elif reported_truncation:
            validation_truncated = float(any(reported_truncation))
        else:
            validation_truncated = None
        online_metrics = {
            "format_success_rate": fmean(
                bool(record["format_valid"]) for record in simulated_user_records
            ),
            "mean_tool_calls": fmean(
                record["tool_call_count"]
                for record in simulated_user_records
                if record["tool_call_count"] is not None
            )
            if any(record["tool_call_count"] is not None for record in simulated_user_records)
            else 0.0,
            "subtasks": float(len(simulated_user_records)),
            # Old JSONL files do not contain per-row batch metadata, so this
            # is omitted rather than guessed from the sub-task count.
            **({"batches_evaluated": batches_evaluated} if batches_evaluated is not None else {}),
            **({"validation_truncated": validation_truncated} if validation_truncated is not None else {}),
        }
        if satisfaction_assessed:
            satisfaction_rate = fmean(
                record["simulator_level"] == 1 for record in answer_labeled
            ) if answer_labeled else 0.0
            online_metrics.update(
                {
                    "level_1_rate": satisfaction_rate,
                    "user_satisfaction_rate": satisfaction_rate,
                    "simulator_fallback_rate": (
                        sum(record["simulator_fallback_count"] for record in simulated_user_records)
                        / sum(record["simulator_judgement_count"] for record in simulated_user_records)
                        if sum(record["simulator_judgement_count"] for record in simulated_user_records)
                        else 0.0
                    ),
                }
            )
            if feedback_enabled:
                online_metrics["mean_retry_depth"] = fmean(
                    record["retry_depth"] for record in answer_labeled
                ) if answer_labeled else 0.0

    metric_definitions = {
        "f1": "Answer-only token-set F1 of the terminal valid <answer>, otherwise the last valid answer in the same sub-task; no valid answer is 0.",
        "bertscore_f1": "Answer-only BERTScore F1 using the same terminal-or-last-valid answer fallback; no valid answer or empty label is 0.",
        "ndcg_at_3": "Answer-only binary NDCG@3. TopiOCQA uses normalized exact passage-text matching because its source and retriever IDs differ; other datasets use strict passage-ID matching. Empty labels are 0.",
        "action_accuracy": "Exact expected answer/clarify/nonanswer terminal-tag accuracy on action-labelled datasets only.",
    }
    if simulated_user_records:
        metric_definitions.update(
            {
                "format_success_rate": "Fraction of simulated-user sub-tasks whose terminal event is format-valid.",
                "mean_tool_calls": "Mean number of valid executed web-search tool calls per original simulated-user sub-task.",
                "subtasks": "Number of simulated-user source sub-tasks included in this validation JSONL.",
                "batches_evaluated": "Number of packed validation batches represented in this JSONL; absent for old files that lack batch metadata.",
                "validation_truncated": "1.0 when validation was intentionally capped for a smoke test, otherwise 0.0.",
            }
        )
        if any(bool(record["simulator_satisfaction_assessed"]) for record in simulated_user_records):
            metric_definitions.update(
                {
                    "level_1_rate": "Legacy alias of user_satisfaction_rate.",
                    "user_satisfaction_rate": "Among answer-labelled sub-tasks, fraction whose terminal simulator assessment is level 1 (satisfied).",
                    "simulator_fallback_rate": "Among answer-labelled sub-tasks with a simulator judgement, fraction using the conservative fallback after two failed simulator calls.",
                }
            )
        if any(bool(record["simulator_feedback_enabled"]) for record in simulated_user_records):
            metric_definitions["mean_retry_depth"] = (
                "Mean terminal response depth among answer-labelled simulated-user sub-tasks; depth 1 means no retry."
            )

    summary = {
        "input_jsonl": str(args.input),
        "sample_count": len(per_sample),
        "terminal_answer_count": sum(record["has_terminal_answer"] for record in per_sample),
        "selected_answer_count": sum(record["has_selected_answer"] for record in per_sample),
        "last_valid_answer_fallback_count": sum(
            record["answer_selection_source"] == "last_valid_answer" for record in per_sample
        ),
        "answer_label_count": sum(bool(record["ground_truth_answer"]) for record in per_sample),
        "retrieval_label_count": sum(bool(record["ground_truth_passage_ids"]) for record in per_sample),
        "action_label_count": len(action_labeled),        "bertscore_model": args.bert_score_model,
        "bertscore_device": bertscore_device,
        "metrics": {
            "f1": fmean(record["f1"] for record in answer_labeled) if answer_labeled else 0.0,
            "bertscore_f1": fmean(record["bertscore_f1"] for record in answer_labeled)
            if answer_labeled
            else 0.0,
            "ndcg_at_3": fmean(record["ndcg_at_3"] for record in answer_labeled) if answer_labeled else 0.0,
            **({"action_accuracy": fmean(record["action_accuracy"] for record in action_labeled)} if action_labeled else {}),
            **online_metrics,
        },
        "metric_definitions": metric_definitions,
    }
    summary_path = args.output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Per-sample metrics: {per_sample_path}")
    print(f"Metric summary: {summary_path}")
    print(json.dumps(summary["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
