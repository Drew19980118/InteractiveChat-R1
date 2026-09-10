"""Build max-reference ChatR1 PPO training and monitor Parquets.

Each retained source subtask has one answer-only row containing all released
answer references.  ChatR1 still generates one terminal answer; its reward and
reported F1/BERTScore retain the maximum over those references.  Whole source
conversations are assigned to the same deterministic 90/10 partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from prepare_chatr1_paper_data import answer_only_frame
from prepare_static_monitor_split import dialogue_key_from_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("inscit", "qrecc"))
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _assignment(key: str, *, seed: int, holdout_fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < holdout_fraction


def _monitor_mask(frame: pd.DataFrame, *, seed: int, holdout_fraction: float) -> tuple[list[bool], int]:
    if "prompt" not in frame.columns:
        raise ValueError("ChatR1 source Parquet must contain a 'prompt' column")
    keys = [dialogue_key_from_prompt(prompt) for prompt in frame["prompt"]]
    assignments = {
        key: _assignment(key, seed=seed, holdout_fraction=holdout_fraction)
        for key in sorted(set(keys))
    }
    return [assignments[key] for key in keys], len(assignments)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be strictly between 0 and 1")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input Parquet not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.dataset}_train.parquet"
    monitor_path = args.output_dir / f"{args.dataset}_monitor.parquet"
    manifest_path = args.output_dir / f"{args.dataset}_split_manifest.json"
    outputs = (train_path, monitor_path, manifest_path)
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError("split outputs already exist; use --force only when intentionally rebuilding them")

    source = pd.read_parquet(args.input)
    answer_only, answer_only_stats = answer_only_frame(source)
    monitor_mask, source_dialogues = _monitor_mask(
        answer_only, seed=args.seed, holdout_fraction=args.holdout_fraction
    )
    train = answer_only.loc[[not value for value in monitor_mask]].reset_index(drop=True)
    monitor = answer_only.loc[monitor_mask].reset_index(drop=True)
    if train.empty or monitor.empty:
        raise RuntimeError("conversation split produced an empty train or monitor partition")

    train.to_parquet(train_path, index=False)
    monitor.to_parquet(monitor_path, index=False)
    manifest = {
        "input": str(args.input.resolve()),
        "split_unit": "source_conversation",
        "split_key": "first_source_user_question",
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "source_dialogues": source_dialogues,
        "training_view": "one answer-only row with all released answer references",
        "monitor_view": "one answer-only row with all released answer references",
        "train_rows": int(len(train)),
        "monitor_rows": int(len(monitor)),
        **answer_only_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": str(train_path), "monitor": str(monitor_path), **manifest}))


if __name__ == "__main__":
    main()
