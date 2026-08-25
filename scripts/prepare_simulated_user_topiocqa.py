#!/usr/bin/env python3
"""Convert original TopiOCQA turn JSON/JSONL into simulated-user Parquet.

TopiOCQA source records are grouped by ``Conversation_no`` and sorted by
``Turn_no``.  The literal answer ``UNANSWERABLE`` is the only nonanswer
label; all other non-empty ``Answer`` values are answer labels.  Gold passage
id, title, and text are retained for downstream retrieval evaluation.

Examples
--------
    python scripts/prepare_simulated_user_topiocqa.py \
      --input data/topiocqa_train.json \
      --output data/sim_user_topiocqa_train.parquet --split train
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


_UNANSWERABLE = "unanswerable"

SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string()),
        pa.field(
            "prompt",
            pa.list_(pa.struct([pa.field("content", pa.string()), pa.field("role", pa.string())])),
        ),
        pa.field("ability", pa.string()),
        pa.field(
            "reward_model",
            pa.struct(
                [
                    pa.field("ground_truth", pa.string()),
                    pa.field("ground_truth_passage_ids", pa.list_(pa.string())),
                    pa.field("expected_action", pa.string()),
                    pa.field("simulated_dialogue", pa.string()),
                    pa.field("style", pa.string()),
                ]
            ),
        ),
        pa.field("extra_info", pa.struct([pa.field("index", pa.string()), pa.field("split", pa.string())])),
    ]
)


def _turn_order(value: Any, *, conversation_id: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TopiOCQA conversation={conversation_id!r} has invalid Turn_no={value!r}") from exc


def _required_text(value: Any, *, field: str, conversation_id: str, turn_no: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(
            f"TopiOCQA conversation={conversation_id!r}, turn={turn_no!r} has missing/empty {field}"
        )
    return text


def _source_context(value: Any, *, question: str, conversation_id: str, turn_no: int) -> list[str]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise TypeError(
            f"TopiOCQA conversation={conversation_id!r}, turn={turn_no!r} has malformed Context"
        )
    context = [str(item) for item in value if str(item).strip()]
    if not context or context[-1].strip() != question:
        context.append(question)
    return context


def _gold_passages(value: Any, *, conversation_id: str, turn_no: int) -> tuple[list[str], list[str], list[str]]:
    """Return ids, texts, titles in source order for one/many Gold_passage mappings."""
    if value is None:
        return [], [], []
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        raise TypeError(
            f"TopiOCQA conversation={conversation_id!r}, turn={turn_no!r} has malformed Gold_passage"
        )
    ids: list[str] = []
    texts: list[str] = []
    titles: list[str] = []
    for position, passage in enumerate(value):
        if not isinstance(passage, Mapping):
            raise TypeError(
                f"TopiOCQA conversation={conversation_id!r}, turn={turn_no!r}, Gold_passage[{position}] is not an object"
            )
        passage_id = str(passage.get("id", "") or "").strip()
        if passage_id:
            ids.append(passage_id)
        text = str(passage.get("text", "") or "").strip()
        if text:
            texts.append(text)
        title = str(passage.get("title", "") or "").strip()
        if title:
            titles.append(title)
    return ids, texts, titles


def _load_records(input_path: Path) -> list[Mapping[str, Any]]:
    """Accept regular JSON arrays and JSONL without guessing malformed JSON."""
    raw = input_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        records: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{input_path} is neither valid JSON nor valid JSONL; invalid record at line {line_number}"
                ) from exc
            if not isinstance(record, Mapping):
                raise TypeError(f"{input_path} JSONL line {line_number} is not an object")
            records.append(record)
        if not records:
            raise ValueError(f"{input_path} contains no TopiOCQA records")
        return records
    if isinstance(payload, Mapping):
        for key in ("data", "records", "examples"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise TypeError("Original TopiOCQA input must be a list of source turn objects")
    return payload


def canonicalize_topiocqa_records(
    records: Iterable[Mapping[str, Any]],
    *,
    audit_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, list[tuple[int, int, Mapping[str, Any]]]] = OrderedDict()
    for source_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"TopiOCQA records[{source_index}] is not an object")
        if "Conversation_no" not in record:
            raise ValueError("TopiOCQA source row is missing Conversation_no")
        conversation_id = str(record["Conversation_no"]).strip()
        if not conversation_id:
            raise ValueError(f"TopiOCQA records[{source_index}] has empty Conversation_no")
        grouped.setdefault(conversation_id, []).append(
            (_turn_order(record.get("Turn_no"), conversation_id=conversation_id), source_index, record)
        )

    dialogues: list[dict[str, Any]] = []
    for conversation_id, turns in grouped.items():
        turns.sort(key=lambda item: (item[0], item[1]))
        seen_turn_numbers: set[int] = set()
        subtasks: list[dict[str, Any]] = []
        for turn_no, _source_index, turn in turns:
            if turn_no in seen_turn_numbers:
                raise ValueError(f"TopiOCQA conversation={conversation_id!r} has duplicate Turn_no={turn_no}")
            seen_turn_numbers.add(turn_no)
            question = _required_text(
                turn.get("Question"), field="Question", conversation_id=conversation_id, turn_no=turn_no
            )
            answer = str(turn.get("Answer") or "").strip()
            if answer.casefold() == _UNANSWERABLE:
                expected_action = "nonanswer"
                gold_response = ""
            elif answer:
                expected_action = "answer"
                gold_response = answer
            else:
                # This is neither a valid answer nor the explicit TopiOCQA
                # UNANSWERABLE label, so it cannot support UCI/fallback.
                if audit_rows is not None:
                    audit_rows.append(
                        {
                            "conversation_no": conversation_id,
                            "turn_no": turn_no,
                            "question": question,
                            "selection_source": "skipped_empty_answer",
                            "reason": "Answer is missing/empty and not UNANSWERABLE",
                        }
                    )
                continue
            passage_ids, passage_texts, passage_titles = _gold_passages(
                turn.get("Gold_passage"), conversation_id=conversation_id, turn_no=turn_no
            )
            subtasks.append(
                {
                    "question": question,
                    "expected_action": expected_action,
                    "gold_response": gold_response,
                    "ground_truth_passage_ids": passage_ids,
                    "ground_truth_passage_texts": passage_texts,
                    "ground_truth_passage_titles": passage_titles,
                    "selection_source": "topiocqa_answer" if expected_action == "answer" else "topiocqa_unanswerable",
                    "source_context": _source_context(
                        turn.get("Context", []),
                        question=question,
                        conversation_id=conversation_id,
                        turn_no=turn_no,
                    ),
                    # Preserve alternatives for auditing/future max-over-golds
                    # evaluation. UCI deliberately uses the primary Answer.
                    "additional_gold_answers": [
                        str(item).strip() for item in turn.get("Additional_answers", []) or [] if str(item).strip()
                    ],
                    "source_turn_id": turn_no,
                }
            )
        if not subtasks:
            if audit_rows is not None:
                audit_rows.append(
                    {
                        "conversation_no": conversation_id,
                        "selection_source": "dropped_empty_dialogue",
                        "reason": "all source turns have missing/empty Answer",
                    }
                )
            continue
        dialogues.append({"dialogue_id": conversation_id, "data_source": "topiocqa", "subtasks": subtasks})
    if not dialogues:
        raise ValueError("No usable TopiOCQA dialogues were found")
    return dialogues


def convert(input_path: Path, output_path: Path, split: str) -> tuple[int, int, int, Path]:
    audit_rows: list[dict[str, Any]] = []
    dialogues = canonicalize_topiocqa_records(_load_records(input_path), audit_rows=audit_rows)
    rows = [
        {
            "data_source": "topiocqa",
            "prompt": [{"role": "user", "content": f"Simulated TopiOCQA dialogue {dialogue['dialogue_id']}"}],
            "ability": "multi_turn_rag",
            "reward_model": {
                "ground_truth": "",
                "ground_truth_passage_ids": [],
                "expected_action": "",
                "simulated_dialogue": json.dumps(dialogue, ensure_ascii=False),
                "style": "rule",
            },
            "extra_info": {"index": dialogue["dialogue_id"], "split": str(split)},
        }
        for dialogue in dialogues
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), output_path)
    audit_path = output_path.with_name(output_path.stem + ".canonical_audit.jsonl")
    with audit_path.open("w", encoding="utf-8") as handle:
        for record in audit_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(dialogues), sum(len(dialogue["subtasks"]) for dialogue in dialogues), len(audit_rows), audit_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original TopiOCQA JSON or JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output one-dialogue-per-row Parquet")
    parser.add_argument("--split", choices=("train", "test", "validation"), required=True)
    args = parser.parse_args()
    dialogues, subtasks, skipped, audit_path = convert(args.input, args.output, args.split)
    print(
        f"Wrote {args.output}: {dialogues} dialogues, {subtasks} TopiOCQA subtasks. "
        f"Skipped {skipped} source turns/dialogues with invalid empty Answer. Audit file: {audit_path}"
    )


if __name__ == "__main__":
    main()
