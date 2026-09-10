#!/usr/bin/env python3
"""Create a dialogue-disjoint train/monitor split for online user-simulator RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("inscit", "qrecc"))
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-key",
        choices=("dialogue_id", "first_question"),
        default="dialogue_id",
        help=(
            "Dialogue key used for deterministic assignment.  first_question "
            "matches prepare_static_monitor_split.py's source-conversation "
            "key for InsCiT protocol-matched comparisons."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def dialogue_payload(row: pd.Series, row_index: int) -> dict[str, Any]:
    """Read the original/canonical dialogue embedded in a rollout row."""
    payload: Any = row.get("reward_model", {})
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"row {row_index} has no mapping reward_model payload")
    dialogue: Any = payload.get("simulated_dialogue", payload.get("dialogue", payload))
    if isinstance(dialogue, str):
        dialogue = json.loads(dialogue)
    if not isinstance(dialogue, dict) or not str(dialogue.get("dialogue_id", "")).strip():
        raise ValueError(
            f"row {row_index} has no simulated_dialogue.dialogue_id; "
            "refuse to split individual turns across monitor/train"
        )
    return dialogue


def first_question_key(dialogue: dict[str, Any], row_index: int) -> str:
    """Match the static baseline's normalized first-user-question split key."""
    subtasks = dialogue.get("subtasks")
    question: Any = None
    if isinstance(subtasks, list) and subtasks and isinstance(subtasks[0], dict):
        question = subtasks[0].get("question")
    if question is None:
        turns = dialogue.get("turns")
        if isinstance(turns, list) and turns and isinstance(turns[0], dict):
            context = turns[0].get("context")
            if isinstance(context, list) and context:
                question = context[-1]
    key = re.sub(r"\s+", " ", str(question or "")).strip().casefold()
    if not key:
        raise ValueError(
            f"row {row_index} has no first source user question; "
            "cannot make a static-protocol-matched split"
        )
    return key


def dialogue_key(row: pd.Series, row_index: int, *, split_key: str) -> str:
    dialogue = dialogue_payload(row, row_index)
    if split_key == "dialogue_id":
        return str(dialogue["dialogue_id"])
    if split_key == "first_question":
        return first_question_key(dialogue, row_index)
    raise ValueError(f"Unsupported split key: {split_key}")


def split_frame(
    frame: pd.DataFrame, *, holdout_fraction: float, seed: int, split_key: str = "dialogue_id"
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be strictly between 0 and 1")
    if "reward_model" not in frame.columns:
        raise ValueError("simulated-user Parquet must contain a 'reward_model' column")

    keys = [
        dialogue_key(row, index, split_key=split_key)
        for index, (_, row) in enumerate(frame.iterrows())
    ]
    assignments: dict[str, bool] = {}
    for key in sorted(set(keys)):
        digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
        assignments[key] = int.from_bytes(digest[:8], "big") / 2**64 < holdout_fraction

    monitor_mask = [assignments[key] for key in keys]
    train = frame.loc[[not value for value in monitor_mask]].reset_index(drop=True)
    monitor = frame.loc[monitor_mask].reset_index(drop=True)
    if train.empty or monitor.empty:
        raise RuntimeError("dialogue split produced an empty partition; change --seed or --holdout-fraction")
    return train, monitor, {
        "split_unit": (
            "simulated_dialogue.dialogue_id"
            if split_key == "dialogue_id"
            else "first_source_user_question"
        ),
        "split_key": split_key,
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "source_rows": int(len(frame)),
        "source_dialogues": int(len(assignments)),
        "train_rows": int(len(train)),
        "monitor_rows": int(len(monitor)),
        "train_dialogues": int(sum(not value for value in assignments.values())),
        "monitor_dialogues": int(sum(assignments.values())),
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input Parquet not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.dataset}_train.parquet"
    monitor_path = args.output_dir / f"{args.dataset}_monitor.parquet"
    manifest_path = args.output_dir / f"{args.dataset}_split_manifest.json"
    outputs = (train_path, monitor_path, manifest_path)
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError("split outputs already exist; use --force only when intentionally rebuilding them")

    train, monitor, manifest = split_frame(
        pd.read_parquet(args.input),
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        split_key=args.split_key,
    )
    train.to_parquet(train_path, index=False)
    monitor.to_parquet(monitor_path, index=False)
    manifest["input"] = str(args.input.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": str(train_path), "monitor": str(monitor_path), **manifest}))


if __name__ == "__main__":
    main()
