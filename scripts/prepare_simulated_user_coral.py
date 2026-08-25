#!/usr/bin/env python3
"""Convert original CoRAL dialogue JSON into simulated-user dialogue Parquet.

CoRAL stores a full conversation under each ``conv_id``.  Every source turn is
answer-only; ``turn_id`` establishes its order, ``response`` is its canonical
gold answer, and ``golden_docs_pids``/``golden_docs_text`` are retained as
retrieval labels and metadata for later evaluation.

Examples
--------
    python scripts/prepare_simulated_user_coral.py \
      --input data/coral_train.json \
      --output data/sim_user_coral_train.parquet --split train
"""

from __future__ import annotations

import argparse
import json
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


def _turn_order(value: Any, *, conv_id: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CoRAL conv_id={conv_id!r} has invalid turn_id={value!r}") from exc


def _required_text(value: Any, *, field: str, conv_id: str, turn_id: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"CoRAL conv_id={conv_id!r}, turn_id={turn_id!r} has missing/empty {field}")
    return text


def _string_list(value: Any, *, field: str, conv_id: str, turn_id: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        value = [value]
    if not isinstance(value, list):
        raise TypeError(f"CoRAL conv_id={conv_id!r}, turn_id={turn_id!r} has malformed {field}")
    # Keep source order. Do not deduplicate text: two different pids may happen
    # to have identical passage contents and must remain aligned with the pids.
    return [str(item).strip() for item in value if str(item).strip()]


def _iter_dialogues(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        for key in ("data", "dialogues", "conversations"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise TypeError("Original CoRAL JSON must be a list of conv_id dialogue objects")
    for index, dialogue in enumerate(payload):
        if not isinstance(dialogue, Mapping):
            raise TypeError(f"CoRAL dialogues[{index}] is not an object")
        yield dialogue


def canonicalize_coral_dialogues(
    source_dialogues: Iterable[Mapping[str, Any]],
    *,
    audit_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Canonicalize CoRAL dialogues, skipping only turns with empty response."""
    dialogues: list[dict[str, Any]] = []
    seen_conv_ids: set[str] = set()
    for source_index, dialogue in enumerate(source_dialogues):
        conv_id = str(dialogue.get("conv_id", "")).strip()
        if not conv_id:
            raise ValueError(f"CoRAL dialogues[{source_index}] has missing/empty conv_id")
        if conv_id in seen_conv_ids:
            raise ValueError(f"CoRAL has duplicate conv_id={conv_id!r}")
        seen_conv_ids.add(conv_id)
        turns = dialogue.get("turns", [])
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"CoRAL conv_id={conv_id!r} has no turns")

        ordered_turns: list[tuple[int, int, Mapping[str, Any]]] = []
        for source_turn_index, turn in enumerate(turns):
            if not isinstance(turn, Mapping):
                raise TypeError(f"CoRAL conv_id={conv_id!r}, turns[{source_turn_index}] is not an object")
            ordered_turns.append((_turn_order(turn.get("turn_id"), conv_id=conv_id), source_turn_index, turn))
        ordered_turns.sort(key=lambda item: (item[0], item[1]))

        history: list[str] = []
        subtasks: list[dict[str, Any]] = []
        seen_turn_ids: set[int] = set()
        for turn_id, _source_turn_index, turn in ordered_turns:
            if turn_id in seen_turn_ids:
                raise ValueError(f"CoRAL conv_id={conv_id!r} has duplicate turn_id={turn_id}")
            seen_turn_ids.add(turn_id)
            question = _required_text(turn.get("question"), field="question", conv_id=conv_id, turn_id=turn_id)
            response = str(turn.get("response") or "").strip()
            if not response:
                if audit_rows is not None:
                    audit_rows.append(
                        {
                            "conv_id": conv_id,
                            "turn_id": turn_id,
                            "question": question,
                            "selection_source": "skipped_empty_response",
                            "reason": "response is missing/empty",
                        }
                    )
                continue
            subtasks.append(
                {
                    "question": question,
                    "expected_action": "answer",
                    "gold_response": response,
                    "ground_truth_passage_ids": _string_list(
                        turn.get("golden_docs_pids", []),
                        field="golden_docs_pids",
                        conv_id=conv_id,
                        turn_id=turn_id,
                    ),
                    "ground_truth_passage_texts": _string_list(
                        turn.get("golden_docs_text", []),
                        field="golden_docs_text",
                        conv_id=conv_id,
                        turn_id=turn_id,
                    ),
                    "selection_source": "coral_response",
                    # The source dialogue has no explicit Context field, so
                    # reconstruct it from preceding source question/answers.
                    "source_context": [*history, question],
                    "golden_rewrite": str(turn.get("golden_rewrite", "") or "").strip(),
                    "source_turn_id": turn_id,
                }
            )
            history.extend((question, response))

        if not subtasks:
            if audit_rows is not None:
                audit_rows.append(
                    {
                        "conv_id": conv_id,
                        "selection_source": "dropped_empty_dialogue",
                        "reason": "all source turns have missing/empty response",
                    }
                )
            continue
        dialogues.append({"dialogue_id": conv_id, "data_source": "coral", "subtasks": subtasks})
    if not dialogues:
        raise ValueError("No usable CoRAL dialogues were found")
    return dialogues


def convert(input_path: Path, output_path: Path, split: str) -> tuple[int, int, int, Path]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    audit_rows: list[dict[str, Any]] = []
    dialogues = canonicalize_coral_dialogues(_iter_dialogues(payload), audit_rows=audit_rows)
    rows = [
        {
            "data_source": "coral",
            "prompt": [{"role": "user", "content": f"Simulated CoRAL dialogue {dialogue['dialogue_id']}"}],
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
    parser.add_argument("--input", type=Path, required=True, help="Original CoRAL dialogue JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output one-dialogue-per-row Parquet")
    parser.add_argument("--split", choices=("train", "test", "validation"), required=True)
    args = parser.parse_args()
    dialogues, subtasks, skipped, audit_path = convert(args.input, args.output, args.split)
    print(
        f"Wrote {args.output}: {dialogues} dialogues, {subtasks} answer-only CoRAL subtasks. "
        f"Skipped {skipped} source turns/dialogues with missing/empty response. Audit file: {audit_path}"
    )


if __name__ == "__main__":
    main()
