#!/usr/bin/env python3
"""Create a conversation-disjoint train/monitor split for static agent data.

The input is the original static baseline Parquet.  It is not an online
benchmark conversion: prompts and labels are copied verbatim.  The split key
is a stable hash of the first user message in the gold conversation history,
so no turn from one source dialogue can appear in both files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


CONTEXT_PATTERN = re.compile(
    r"(?:Context Begin:|Conversation context:)\s*<context>(.*?)</context>",
    re.DOTALL | re.IGNORECASE,
)
FIRST_USER_PATTERN = re.compile(r"(?:^|\n)User:\s*(.*?)(?=\nAssistant:|\nUser:|$)", re.DOTALL | re.IGNORECASE)
QUESTION_PATTERN = re.compile(r"(?:^|\n)(?:Question|User query):\s*(.*?)(?:\n|$)", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("inscit", "qrecc"))
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _as_prompt_text(prompt: Any) -> str:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)):
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(prompt, dict):
        return str(prompt.get("content", ""))
    return str(prompt or "")


def dialogue_key_from_prompt(prompt: Any) -> str:
    """Return a content key for a complete source dialogue.

    ConvAgent and ChatR1 static prompts contain a gold ``<context>`` block.
    Its first user turn identifies the original conversation across all
    derived rows.
    The conservative fallback is the current question, which preserves
    determinism for malformed external data while remaining auditable.
    """
    text = _as_prompt_text(prompt)
    context_match = CONTEXT_PATTERN.search(text)
    context = context_match.group(1) if context_match else text
    user_match = FIRST_USER_PATTERN.search(context)
    if user_match:
        value = user_match.group(1)
    else:
        question_match = QUESTION_PATTERN.search(text)
        value = question_match.group(1) if question_match else text
    return re.sub(r"\s+", " ", value).strip().casefold()


def split_frame(
    frame: pd.DataFrame,
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be strictly between 0 and 1")
    if "prompt" not in frame.columns:
        raise ValueError("static baseline Parquet must contain a 'prompt' column")

    dialogue_keys = [dialogue_key_from_prompt(prompt) for prompt in frame["prompt"]]
    assignments: dict[str, bool] = {}
    for key in sorted(set(dialogue_keys)):
        digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
        assignments[key] = int.from_bytes(digest[:8], "big") / 2**64 < holdout_fraction

    monitor_mask = [assignments[key] for key in dialogue_keys]
    train = frame.loc[[not value for value in monitor_mask]].reset_index(drop=True)
    monitor = frame.loc[monitor_mask].reset_index(drop=True)
    if train.empty or monitor.empty:
        raise RuntimeError(
            "conversation split produced an empty partition; change --seed or --holdout-fraction"
        )
    manifest = {
        "split_unit": "source_conversation",
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "source_rows": int(len(frame)),
        "source_dialogues": int(len(assignments)),
        "train_rows": int(len(train)),
        "monitor_rows": int(len(monitor)),
        "train_dialogues": int(sum(not value for value in assignments.values())),
        "monitor_dialogues": int(sum(assignments.values())),
    }
    return train, monitor, manifest


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
        raise FileExistsError(
            "split outputs already exist; use --force only when intentionally rebuilding them"
        )

    frame = pd.read_parquet(args.input)
    train, monitor, manifest = split_frame(
        frame, holdout_fraction=args.holdout_fraction, seed=args.seed
    )
    train.to_parquet(train_path, index=False)
    monitor.to_parquet(monitor_path, index=False)
    manifest.update({"input": str(args.input.resolve())})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": str(train_path), "monitor": str(monitor_path), **manifest}))


if __name__ == "__main__":
    main()
