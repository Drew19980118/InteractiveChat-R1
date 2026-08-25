#!/usr/bin/env python3
"""Build one-row-per-dialogue InsCiT parquet files for simulated-user GRPO.

The existing ``convagent_igpo.py`` emits one last-turn RAG item per row.  This
adapter instead stores the complete original dialogue as JSON in
``reward_model.simulated_dialogue``.  The training manager reconstructs the
canonical source label for every original user turn at rollout time.

Examples
--------
    python scripts/prepare_simulated_user_inscit.py \
      --input collection/raw_inscit/inscit_train.json \
      --output data/sim_user_inscit_train.parquet --split train
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "igpo_simulated_user_adapter",
    ROOT / "scrl" / "llm_agent" / "simulated_user.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load simulated-user canonicalization helper")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
canonicalize_inscit_dialogue = MODULE.canonicalize_inscit_dialogue


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


def _iter_dialogues(payload: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Accept common InsCiT JSON containers without silently discarding rows."""
    if isinstance(payload, Mapping):
        if isinstance(payload.get("dialogues"), list):
            payload = payload["dialogues"]
        elif isinstance(payload.get("data"), list):
            payload = payload["data"]
        elif "turns" in payload:
            yield str(payload.get("dialogue_id", payload.get("id", "dialogue-0"))), payload
            return
        else:
            for dialogue_id, dialogue in payload.items():
                if not isinstance(dialogue, Mapping) or "turns" not in dialogue:
                    raise ValueError(f"Top-level item {dialogue_id!r} is not an InsCiT dialogue with turns")
                yield str(dialogue_id), dialogue
            return
    if isinstance(payload, list):
        for index, dialogue in enumerate(payload):
            if not isinstance(dialogue, Mapping) or "turns" not in dialogue:
                raise ValueError(f"dialogues[{index}] is not an InsCiT dialogue with turns")
            yield str(dialogue.get("dialogue_id", dialogue.get("id", f"dialogue-{index}"))), dialogue
        return
    raise TypeError(f"Unsupported JSON root: {type(payload).__name__}")


def convert(input_path: Path, output_path: Path, split: str) -> tuple[int, int, Path]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    audit_rows = []
    fallback_count = 0
    for dialogue_id, dialogue in _iter_dialogues(payload):
        canonical = canonicalize_inscit_dialogue(dialogue_id, dialogue["turns"])
        for subtask_index, subtask in enumerate(canonical["subtasks"]):
            if str(subtask["selection_source"]).startswith("fallback_"):
                fallback_count += 1
                audit_rows.append(
                    {
                        "dialogue_id": dialogue_id,
                        "subtask_index": subtask_index,
                        "question": subtask["question"],
                        "selection_source": subtask["selection_source"],
                        "gold_response": subtask["gold_response"],
                    }
                )
        rows.append(
            {
                "data_source": "inscit",
                # The real rollout prompt is rebuilt from simulated_dialogue.
                # This harmless placeholder only satisfies RLHFDataset tokenization.
                "prompt": [{"role": "user", "content": f"Simulated dialogue {dialogue_id}"}],
                "ability": "multi_turn_rag",
                "reward_model": {
                    "ground_truth": "",
                    "ground_truth_passage_ids": [],
                    "expected_action": "",
                    "simulated_dialogue": json.dumps(canonical, ensure_ascii=False),
                    "style": "rule",
                },
                "extra_info": {"index": str(dialogue_id), "split": str(split)},
            }
        )
    if not rows:
        raise ValueError("No dialogues were found in input")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), output_path)
    audit_path = output_path.with_name(output_path.stem + ".canonical_audit.jsonl")
    with audit_path.open("w", encoding="utf-8") as handle:
        for record in audit_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(rows), fallback_count, audit_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original full-dialogue InsCiT JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output one-dialogue-per-row Parquet file")
    parser.add_argument("--split", choices=("train", "test", "validation"), required=True)
    args = parser.parse_args()
    count, fallbacks, audit_path = convert(args.input, args.output, args.split)
    print(
        f"Wrote {args.output}: {count} dialogues. "
        f"Canonical-label next-context mismatches requiring auditable fallback: {fallbacks}. "
        f"Audit file: {audit_path}"
    )


if __name__ == "__main__":
    main()
