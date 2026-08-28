#!/usr/bin/env python3
"""Materialize one benchmark/split from the combined ``DrewZhang/conv`` release.

The Hugging Face release is a *combined* Parquet dataset.  Each row carries a
``data_source`` (InsCiT, QReCC, TopiOCQA, or CoRAL) and the original source
split in ``extra_info['split']``.  This utility filters those metadata fields
without rewriting the ConvAgent prompt or reward labels, then produces the
per-benchmark files consumed by the static baseline launchers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _row_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        # ``DrewZhang/conv`` also contains the ChatR1 and IGPO releases.
        # When its repository root is supplied, consume only the explicitly
        # released ConvAgent data rather than silently mixing (or duplicating)
        # rows from the other method folders.
        convagent_root = path / "ConvAgent"
        search_root = convagent_root if convagent_root.is_dir() else path
        files = sorted(search_root.rglob("*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"No Parquet files found under: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Downloaded DrewZhang/conv Parquet file or its containing directory.",
    )
    parser.add_argument("--dataset", choices=("inscit", "qrecc", "topiocqa", "coral"), required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = _parquet_files(args.input)
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    required = {"prompt", "data_source", "extra_info", "reward_model"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"DrewZhang/conv rows are missing expected columns: {sorted(missing)}")

    source = frame["data_source"].astype(str).str.strip().str.casefold()
    split = frame["extra_info"].map(lambda item: str(_row_dict(item).get("split", "")).casefold())
    selected = frame.loc[(source == args.dataset) & (split == args.split)].reset_index(drop=True)
    if selected.empty:
        observed_sources = sorted(source.dropna().unique().tolist())
        observed_splits = sorted(split.dropna().unique().tolist())
        raise ValueError(
            f"No rows for data_source={args.dataset!r}, split={args.split!r}. "
            f"Observed data_source values: {observed_sources}; observed source splits: {observed_splits}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)
    print(
        json.dumps(
            {
                "input_files": [str(path) for path in files],
                "dataset": args.dataset,
                "split": args.split,
                "rows": int(len(selected)),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
