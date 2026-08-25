#!/usr/bin/env python3
"""Convert original QReCC turn JSON into simulated-user dialogue Parquet.

QReCC is stored as one JSON object per source turn.  ``Conversation_no``
identifies which turns belong to one dialogue and ``Turn_no`` defines their
original order.  This adapter groups those source turns into one Parquet row
per complete dialogue, ready for the simulated-user sparse-GRPO rollout.

Unlike InsCiT, QReCC has only answerable turns: every canonical subtask has
``expected_action='answer'``.  The original history is retained in
``source_context`` for the frozen user simulator, while the online rollout
itself uses its own generated/fallback history.

Examples
--------
    python scripts/prepare_simulated_user_qrecc.py \
      --input data/qrecc_train.json \
      --output data/sim_user_qrecc_train.parquet --split train
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


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


def _as_nonempty_text(value: Any, *, field: str, conversation_id: str, turn_no: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(
            f"QReCC conversation={conversation_id!r}, turn={turn_no!r} has missing/empty {field}"
        )
    return text


def _passage_ids(value: Any, *, conversation_id: str, turn_no: Any) -> list[str]:
    """Normalize QReCC ``Truth_passages`` without changing source order."""
    if value is None:
        return []
    if isinstance(value, (str, int)):
        value = [value]
    if not isinstance(value, list):
        raise TypeError(
            f"QReCC conversation={conversation_id!r}, turn={turn_no!r} has malformed Truth_passages"
        )
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _turn_order(value: Any, *, conversation_id: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"QReCC conversation={conversation_id!r} has invalid Turn_no={value!r}") from exc


def canonicalize_qrecc_records(
    records: Iterable[Mapping[str, Any]],
    *,
    audit_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Group QReCC source rows into canonical answer-only dialogues.

    The resulting payload directly satisfies ``parse_dialogue_payload`` in
    ``scrl.llm_agent.simulated_user`` and intentionally makes no attempt to
    infer clarify/nonanswer labels that QReCC does not contain.
    """
    grouped: OrderedDict[str, list[tuple[int, int, Mapping[str, Any]]]] = OrderedDict()
    for source_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"QReCC records[{source_index}] is not an object")
        if "Conversation_no" not in record:
            raise ValueError(
                "QReCC source row is missing Conversation_no; this does not look like the original QReCC JSON format"
            )
        conversation_id = str(record["Conversation_no"]).strip()
        if not conversation_id:
            raise ValueError(f"QReCC records[{source_index}] has empty Conversation_no")
        turn_no = _turn_order(record.get("Turn_no"), conversation_id=conversation_id)
        grouped.setdefault(conversation_id, []).append((turn_no, source_index, record))

    dialogues: list[dict[str, Any]] = []
    for conversation_id, turns in grouped.items():
        turns.sort(key=lambda item: (item[0], item[1]))
        seen_turn_numbers: set[int] = set()
        subtasks: list[dict[str, Any]] = []
        for turn_no, _source_index, turn in turns:
            if turn_no in seen_turn_numbers:
                raise ValueError(
                    f"QReCC conversation={conversation_id!r} has duplicate Turn_no={turn_no}; refusing ambiguous order"
                )
            seen_turn_numbers.add(turn_no)
            question = _as_nonempty_text(
                turn.get("Question"), field="Question", conversation_id=conversation_id, turn_no=turn_no
            )
            gold_response = str(turn.get("Truth_answer") or "").strip()
            if not gold_response:
                # The simulated-user method needs a non-empty gold response
                # for both fallback history and UCI.  QReCC has no separate
                # nonanswer label, so skip this unscorable source turn instead
                # of reclassifying or fabricating an answer.
                if audit_rows is not None:
                    audit_rows.append(
                        {
                            "conversation_no": conversation_id,
                            "turn_no": turn_no,
                            "question": question,
                            "selection_source": "skipped_empty_truth_answer",
                            "reason": "Truth_answer is missing/empty",
                        }
                    )
                continue
            selection_source = "qrecc_truth"
            source_context = turn.get("Context", [])
            if source_context is None:
                source_context = []
            if not isinstance(source_context, list):
                raise TypeError(
                    f"QReCC conversation={conversation_id!r}, turn={turn_no!r} has malformed Context"
                )
            # Match InsCiT's convention: source_context contains the historical
            # conversation followed by the current source question.
            source_context = [str(value) for value in source_context if str(value).strip()]
            if not source_context or source_context[-1].strip() != question:
                source_context.append(question)
            subtasks.append(
                {
                    "question": question,
                    "expected_action": "answer",
                    "gold_response": gold_response,
                    "ground_truth_passage_ids": _passage_ids(
                        turn.get("Truth_passages", []), conversation_id=conversation_id, turn_no=turn_no
                    ),
                    "selection_source": selection_source,
                    "source_context": source_context,
                }
            )
        if not subtasks:
            if audit_rows is not None:
                audit_rows.append(
                    {
                        "conversation_no": conversation_id,
                        "selection_source": "dropped_empty_dialogue",
                        "reason": "all source turns have missing/empty Truth_answer",
                    }
                )
            continue
        dialogues.append(
            {
                "dialogue_id": conversation_id,
                "data_source": "qrecc",
                "subtasks": subtasks,
            }
        )
    if not dialogues:
        raise ValueError("No QReCC source records were found")
    return dialogues


def _load_records(input_path: Path) -> list[Mapping[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        for key in ("data", "records", "examples"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise TypeError("Original QReCC JSON must be a list of turn records (or a mapping containing data/records/examples)")
    return payload


def convert(input_path: Path, output_path: Path, split: str) -> tuple[int, int, Path, int]:
    audit_rows: list[dict[str, Any]] = []
    dialogues = canonicalize_qrecc_records(_load_records(input_path), audit_rows=audit_rows)
    rows = [
        {
            "data_source": "qrecc",
            # The rollout manager rebuilds the actual prompt from
            # reward_model.simulated_dialogue.  This only satisfies RLHFDataset.
            "prompt": [{"role": "user", "content": f"Simulated QReCC dialogue {dialogue['dialogue_id']}"}],
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
    return len(dialogues), sum(len(dialogue["subtasks"]) for dialogue in dialogues), audit_path, len(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original per-turn QReCC JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output one-dialogue-per-row Parquet")
    parser.add_argument("--split", choices=("train", "test", "validation"), required=True)
    args = parser.parse_args()
    dialogues, subtasks, audit_path, fallback_count = convert(args.input, args.output, args.split)
    print(
        f"Wrote {args.output}: {dialogues} dialogues, {subtasks} answer-only QReCC subtasks. "
        f"Skipped {fallback_count} source turns/dialogues with missing or empty Truth_answer. "
        f"Audit file: {audit_path}"
    )


if __name__ == "__main__":
    main()
