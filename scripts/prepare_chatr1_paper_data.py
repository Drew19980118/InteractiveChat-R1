"""Prepare ChatR1 static training rows exactly as described in the paper.

ChatR1 removes non-answer (for example, clarification) turns from RL training
and treats every released answer reference as an independent training sample.
The original Parquet remains untouched; this script writes an expanded copy
that is used only by the paper-faithful PPO launcher.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from verl.utils.reward_score.static_chatr1 import static_chatr1_answer_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _as_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: _as_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_python(item) for item in value]
    if isinstance(value, tuple):
        return [_as_python(item) for item in value]
    return value


def _reward_model(value: Any) -> dict[str, Any]:
    value = _as_python(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("reward_model is a non-JSON string") from exc
    if not isinstance(value, dict) or "ground_truth" not in value:
        raise ValueError("each ChatR1 row must contain reward_model.ground_truth")
    return value


def expand_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if "reward_model" not in frame.columns:
        raise ValueError("ChatR1 source Parquet must contain a 'reward_model' column")

    rows: list[dict[str, Any]] = []
    dropped_nonanswer_rows = 0
    for source_row in frame.to_dict(orient="records"):
        reward_model = _reward_model(source_row["reward_model"])
        ground_truth = _as_python(reward_model["ground_truth"])
        candidates = static_chatr1_answer_candidates(ground_truth)
        if not candidates and isinstance(ground_truth, str) and ground_truth.strip():
            candidates = [{"action": "answer", "response": ground_truth.strip()}]
        if not candidates:
            dropped_nonanswer_rows += 1
            continue

        # The paper trains each gold answer independently.  Keep source order
        # and intentionally do not de-duplicate references.
        for candidate in candidates:
            row = copy.deepcopy(source_row)
            row_reward_model = copy.deepcopy(reward_model)
            row_reward_model["ground_truth"] = [candidate]
            row["reward_model"] = row_reward_model
            rows.append(row)

    if not rows:
        raise RuntimeError("no answer-supervised rows remain after ChatR1 filtering")
    return pd.DataFrame(rows), {
        "source_rows": int(len(frame)),
        "paper_training_rows": int(len(rows)),
        "dropped_nonanswer_rows": int(dropped_nonanswer_rows),
    }


def primary_answer_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep exactly the first released answer candidate for each source row.

    ChatR1's PPO training data are intentionally expanded by
    :func:`expand_frame`.  This companion view is solely for fair checkpoint
    selection and reporting, where each original subtask must contribute one
    canonical answer reference rather than one vote per alternative answer.
    """
    if "reward_model" not in frame.columns:
        raise ValueError("ChatR1 source Parquet must contain a 'reward_model' column")

    rows: list[dict[str, Any]] = []
    dropped_nonanswer_rows = 0
    for source_row in frame.to_dict(orient="records"):
        reward_model = _reward_model(source_row["reward_model"])
        ground_truth = _as_python(reward_model["ground_truth"])
        candidates = static_chatr1_answer_candidates(ground_truth)
        if not candidates and isinstance(ground_truth, str) and ground_truth.strip():
            candidates = [{"action": "answer", "response": ground_truth.strip()}]
        if not candidates:
            dropped_nonanswer_rows += 1
            continue

        row = copy.deepcopy(source_row)
        row_reward_model = copy.deepcopy(reward_model)
        row_reward_model["ground_truth"] = [candidates[0]]
        row["reward_model"] = row_reward_model
        rows.append(row)

    if not rows:
        raise RuntimeError("no answer-supervised rows remain after ChatR1 filtering")
    return pd.DataFrame(rows), {
        "source_rows": int(len(frame)),
        "primary_reference_rows": int(len(rows)),
        "dropped_nonanswer_rows": int(dropped_nonanswer_rows),
    }


def answer_only_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep one answer-only row with all released answer references per source row.

    The policy emits one answer, while its PPO outcome reward takes the maximum
    F1 across these references.  This is the common max-reference protocol
    used by ConvAgent and Turn-PPO, and deliberately differs from the older
    independent-reference expansion retained above for archival reproduction.
    """
    if "reward_model" not in frame.columns:
        raise ValueError("ChatR1 source Parquet must contain a 'reward_model' column")

    rows: list[dict[str, Any]] = []
    dropped_nonanswer_rows = 0
    for source_row in frame.to_dict(orient="records"):
        reward_model = _reward_model(source_row["reward_model"])
        ground_truth = _as_python(reward_model["ground_truth"])
        candidates = static_chatr1_answer_candidates(ground_truth)
        if not candidates and isinstance(ground_truth, str) and ground_truth.strip():
            candidates = [{"action": "answer", "response": ground_truth.strip()}]
        if not candidates:
            dropped_nonanswer_rows += 1
            continue

        row = copy.deepcopy(source_row)
        row_reward_model = copy.deepcopy(reward_model)
        row_reward_model["ground_truth"] = candidates
        row["reward_model"] = row_reward_model
        rows.append(row)

    if not rows:
        raise RuntimeError("no answer-supervised rows remain after ChatR1 filtering")
    return pd.DataFrame(rows), {
        "source_rows": int(len(frame)),
        "answer_only_rows": int(len(rows)),
        "dropped_nonanswer_rows": int(dropped_nonanswer_rows),
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input Parquet not found: {args.input}")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}; use --force to rebuild")

    frame = pd.read_parquet(args.input)
    expanded, stats = expand_frame(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_parquet(args.output, index=False)

    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
                "method": "ChatR1 paper answer-only filtering and independent references",
                **stats,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "manifest": str(manifest), **stats}))


if __name__ == "__main__":
    main()
