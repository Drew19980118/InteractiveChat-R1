"""Validation/reporting helpers for simulated-user dialogue rollouts."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from verl import DataProto


def _answer(raw_response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", str(raw_response), flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def select_answer_for_text_metrics(
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str, str]:
    """Select the last user-visible, valid answer within one sub-task.

    Action accuracy remains an end-to-end terminal-event metric.  Text quality
    is different: after a user has already received a complete answer, a later
    malformed retry must not erase that answer from F1/BERTScore.  We therefore
    prefer the terminal answer when it is valid and otherwise fall back to the
    most recent valid answer in the same source sub-task.  This is deliberately
    chronological rather than an oracle max-F1 selection.
    """
    if not events:
        return None, "", "missing"
    terminal = events[-1]
    for event in reversed(events):
        if event.get("predicted_action") != "answer" or not bool(event.get("format_valid", False)):
            continue
        answer = _answer(event.get("raw_response", ""))
        if answer:
            source = "terminal" if event is terminal else "last_valid_answer"
            return event, answer, source
    return None, "", "missing"


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = str(prediction).lower().split()
    reference_tokens = str(reference).lower().split()
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _canonical_passage_text(value: Any) -> str:
    """Normalize only presentation differences before exact text matching."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def _ndcg_at_3(
    predicted_passages: list[Any],
    gold_passages: list[Any],
    *,
    match_by_text: bool = False,
) -> float:
    """Binary NDCG@3 with no duplicate credit.

    TopiOCQA uses canonical ``Gold_passage.text`` for relevance because the
    source IDs (e.g. ``wiki:...``) are not the IDs emitted by the local
    retriever.  Other datasets continue to use stable passage IDs.
    """
    canonicalize = _canonical_passage_text if match_by_text else lambda value: str(value).strip()
    gold = {canonicalize(value) for value in gold_passages or [] if canonicalize(value)}
    if not gold:
        return 0.0

    dcg = 0.0
    matched: set[str] = set()
    for rank, passage in enumerate(predicted_passages[:3], start=1):
        key = canonicalize(passage)
        if key in gold and key not in matched:
            dcg += float(1.0 / np.log2(rank + 1))
            matched.add(key)
    ideal_count = min(3, len(gold))
    ideal_dcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return float(dcg / ideal_dcg) if ideal_dcg else 0.0


def validate_simulated_user(
    trainer,
    *,
    max_batches_override: int | None = None,
) -> dict[str, float]:
    """Validate one online trajectory per dialogue and write subtask JSONL.

    ``trainer`` is intentionally duck-typed to avoid coupling this helper to
    the very large legacy RayPPOTrainer module.
    """
    manager = trainer._make_simulated_user_manager(n=1, is_validation=True)
    feedback_enabled = bool(manager.settings.enable_user_feedback)
    rows: list[dict[str, Any]] = []
    try:
        total_batches = len(trainer.val_dataloader)
    except TypeError:
        total_batches = None
    configured_max_batches = (
        max_batches_override
        if max_batches_override is not None
        else trainer.config.algorithm.get("simulated_user_validation_max_batches", None)
    )
    max_batches = None if configured_max_batches is None else int(configured_max_batches)
    if max_batches is not None and max_batches < 1:
        raise ValueError("simulated_user_validation_max_batches must be >= 1 or null")
    batches_to_evaluate = (
        min(total_batches, max_batches)
        if total_batches is not None and max_batches is not None
        else total_batches
    )
    validation_truncated = bool(
        max_batches is not None
        and total_batches is not None
        and total_batches > max_batches
    )
    if max_batches is not None:
        print(
            "[SimUser Validation] smoke-test cap enabled: "
            f"evaluating at most {max_batches} packed validation batches",
            flush=True,
        )
    completed = 0
    evaluated_batches = 0
    for batch_index, batch_dict in enumerate(trainer.val_dataloader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        evaluated_batches += 1
        batch = DataProto.from_single_dict(batch_dict)
        reward_models = list(batch.non_tensor_batch["reward_model"])
        input_count = len(batch)
        batch_keys = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_keys = ["raw_prompt_ids"]
        if "multi_modal_inputs" in batch.non_tensor_batch:
            non_tensor_keys.extend(["multi_modal_data", "multi_modal_inputs"])
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_keys.append("tools_kwargs")
        gen_batch = batch.pop(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys)
        gen_batch.meta_info = {
            "eos_token_id": trainer.tokenizer.eos_token_id,
            "pad_token_id": trainer.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": False,
            "validate": True,
        }
        _output, dialogue_records = manager.run_dialogue_loop(
            gen_batch=gen_batch,
            reward_models=reward_models,
        )
        for dialogue_record in dialogue_records:
            by_subtask: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for event in dialogue_record["subtasks"]:
                by_subtask[int(event["subtask_index"])].append(event)
            for subtask_index, events in sorted(by_subtask.items()):
                terminal = events[-1]
                first_retrieval = next((event for event in events if event["predicted_passage_ids"]), None)
                predicted_ids = first_retrieval["predicted_passage_ids"] if first_retrieval else []
                predicted_texts = (
                    first_retrieval.get("predicted_passage_texts", []) if first_retrieval else []
                )
                gold_texts = terminal.get("ground_truth_passage_texts", [])
                is_topiocqa = str(dialogue_record.get("data_source", "")).strip().lower() == "topiocqa"
                # TopiOCQA source IDs and local-retriever IDs are incompatible;
                # use canonical passage text for its retrieval relevance.
                retrieval_match_mode = "passage_text" if is_topiocqa and gold_texts else "passage_id"
                predicted_passages = predicted_texts if retrieval_match_mode == "passage_text" else predicted_ids
                gold_passages = gold_texts if retrieval_match_mode == "passage_text" else terminal["ground_truth_passage_ids"]
                answer_event, prediction, answer_selection_source = select_answer_for_text_metrics(events)
                answer_raw_response = answer_event["raw_response"] if answer_event is not None else ""
                terminal_prediction = _answer(terminal["raw_response"])
                expected = terminal["expected_action"]
                predicted = terminal["predicted_action"]
                # A level-2/3 answer can have an earlier simulator judgement
                # before a later retry terminates this original sub-task.
                # Account for every judgement rather than only the terminal
                # event, otherwise fallback use would be under-reported.
                simulator_statuses = [
                    str(event.get("simulator_status", ""))
                    for event in events
                    if event.get("simulator_status")
                ]
                f1 = (
                    _token_f1(prediction, terminal["ground_truth_answer"])
                    if expected == "answer"
                    else 0.0
                )
                rows.append(
                    {
                        "dialogue_id": dialogue_record["dialogue_id"],
                        "subtask_index": subtask_index,
                        "data_source": dialogue_record["data_source"],
                        # Keep terminal action fields for action/format metrics;
                        # use answer_raw_response only for text metrics.
                        "raw_response": terminal["raw_response"],
                        "answer_raw_response": answer_raw_response,
                        "answer_selection_source": answer_selection_source,
                        "terminal_predicted_answer": terminal_prediction,
                        "predicted_answer": prediction,
                        "ground_truth_answer": terminal["ground_truth_answer"],
                        "ground_truth_passage_ids": terminal["ground_truth_passage_ids"],
                        "ground_truth_passage_texts": terminal.get("ground_truth_passage_texts", []),
                        "ground_truth_passage_titles": terminal.get("ground_truth_passage_titles", []),
                        "predicted_passage_ids": predicted_ids,
                        "predicted_passage_texts": predicted_texts,
                        "retrieval_match_mode": retrieval_match_mode,
                        "expected_action": expected,
                        "predicted_action": predicted,
                        "action_correct": bool(terminal["action_correct"]),
                        "format_valid": bool(terminal["format_valid"]),
                        "f1": f1,
                        "ndcg_at_3": _ndcg_at_3(
                            predicted_passages,
                            gold_passages,
                            match_by_text=retrieval_match_mode == "passage_text",
                        ),
                        "simulator_level": terminal["simulator_level"],
                        "simulator_feedback": terminal["simulator_feedback"],
                        "simulator_status": terminal.get("simulator_status", ""),
                        "simulator_feedback_enabled": feedback_enabled,
                        "simulator_judgement_count": len(simulator_statuses),
                        "simulator_fallback_count": sum(
                            status.startswith("fallback_") for status in simulator_statuses
                        ),
                        "retry_depth": terminal["response_depth"],
                        # Count only tool actions that survived the strict
                        # parser and actually entered the trajectory.  This
                        # is an interaction-efficiency statistic, not a
                        # reward component.
                        "tool_call_count": sum(
                            event["predicted_action"] == "tool_call"
                            for event in events
                        ),
                        # Repeat run-level metadata on each record so a
                        # standalone metrics script can report whether this
                        # was a complete or capped smoke-test validation.
                        "validation_batch_index": batch_index,
                        "validation_batches_evaluated": batches_to_evaluate,
                        "validation_truncated": validation_truncated,
                        "events": events,
                    }
                )
        completed += input_count
        total_text = str(batches_to_evaluate) if batches_to_evaluate is not None else "?"
        print(
            f"[SimUser Validation] completed batch {batch_index}/{total_text}; "
            f"dialogues={completed}; subtasks={len(rows)}",
            flush=True,
        )

    validation_dir = trainer.config.trainer.get("validation_data_dir", None)
    if validation_dir:
        os.makedirs(validation_dir, exist_ok=True)
        path = os.path.join(validation_dir, f"{trainer.global_steps}.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[SimUser Validation] wrote {len(rows)} subtask records to {path}", flush=True)

    answer_rows = [row for row in rows if row["expected_action"] == "answer"]
    metrics = {
        "val/sim_user/f1": float(np.mean([row["f1"] for row in answer_rows])) if answer_rows else 0.0,
        "val/sim_user/ndcg_at_3": float(np.mean([row["ndcg_at_3"] for row in answer_rows])) if answer_rows else 0.0,
        "val/sim_user/action_accuracy": float(np.mean([row["action_correct"] for row in rows])) if rows else 0.0,
        "val/sim_user/format_success_rate": float(np.mean([row["format_valid"] for row in rows])) if rows else 0.0,
        "val/sim_user/mean_tool_calls": float(np.mean([row["tool_call_count"] for row in rows])) if rows else 0.0,
        "val/sim_user/subtasks": float(len(rows)),
        "val/sim_user/batches_evaluated": float(evaluated_batches),
        "val/sim_user/validation_truncated": float(validation_truncated),
    }
    if feedback_enabled:
        simulator_judgement_count = sum(row["simulator_judgement_count"] for row in rows)
        simulator_fallback_count = sum(row["simulator_fallback_count"] for row in rows)
        metrics.update(
            {
                "val/sim_user/level_1_rate": float(
                    np.mean([row["simulator_level"] == 1 for row in answer_rows])
                )
                if answer_rows
                else 0.0,
                "val/sim_user/simulator_fallback_rate": float(
                    simulator_fallback_count / simulator_judgement_count
                )
                if simulator_judgement_count
                else 0.0,
                "val/sim_user/mean_retry_depth": float(
                    np.mean([row["retry_depth"] for row in answer_rows])
                )
                if answer_rows
                else 0.0,
            }
        )
    print("[SimUser Validation] metrics: " + json.dumps(metrics, sort_keys=True), flush=True)
    return metrics
