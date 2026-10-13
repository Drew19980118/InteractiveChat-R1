#!/usr/bin/env python3
"""Create identical conversation-disjoint train/monitor splits for paired views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-train", type=Path, required=True)
    parser.add_argument("--dynamic-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 < args.holdout_fraction < 1.0:
        parser.error("--holdout-fraction must be strictly between 0 and 1")
    return args


def _info(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, Mapping):
            return item
    raise ValueError(f"extra_info is not a mapping: {type(value).__name__}")


def _metadata(frame: pd.DataFrame) -> list[tuple[str, str]]:
    if "extra_info" not in frame:
        raise ValueError("paired context Parquet lacks extra_info")
    values: list[tuple[str, str]] = []
    for index, raw in enumerate(frame["extra_info"]):
        info = _info(raw)
        pair_id = str(info.get("pair_id", "")).strip()
        conversation_id = str(info.get("conversation_id", "")).strip()
        if not pair_id or not conversation_id:
            raise ValueError(f"row {index} lacks pair_id or conversation_id")
        values.append((pair_id, conversation_id))
    return values


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def split_paired_frames(
    static: pd.DataFrame,
    dynamic: pd.DataFrame,
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], dict[str, Any]]:
    static_meta = _metadata(static)
    dynamic_meta = _metadata(dynamic)
    if [value[0] for value in static_meta] != [value[0] for value in dynamic_meta]:
        raise ValueError("static and dynamic pair_id ordering differs")
    if [value[1] for value in static_meta] != [value[1] for value in dynamic_meta]:
        raise ValueError("static and dynamic conversation_id ordering differs")

    assignments: dict[str, bool] = {}
    for conversation_id in sorted({value[1] for value in static_meta}):
        digest = hashlib.sha256(f"{seed}\0{conversation_id}".encode("utf-8")).digest()
        assignments[conversation_id] = int.from_bytes(digest[:8], "big") / 2**64 < holdout_fraction
    monitor_mask = [assignments[conversation_id] for _, conversation_id in static_meta]
    if not any(monitor_mask) or all(monitor_mask):
        raise RuntimeError("conversation split produced an empty partition; change seed/fraction")

    result: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for variant, frame in (("static", static), ("dynamic", dynamic)):
        train = frame.loc[[not value for value in monitor_mask]].reset_index(drop=True)
        monitor = frame.loc[monitor_mask].reset_index(drop=True)
        result[variant] = train, monitor
    manifest = {
        "split_unit": "conversation_id",
        "paired_views": True,
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "source_rows": len(static),
        "source_conversations": len(assignments),
        "train_rows": len(result["static"][0]),
        "monitor_rows": len(result["static"][1]),
        "train_conversations": sum(not value for value in assignments.values()),
        "monitor_conversations": sum(assignments.values()),
        "assignments_sha256": hashlib.sha256(
            json.dumps(assignments, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    return result, manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = {
        variant: {
            split: args.output_dir / variant / f"inscit_{split}.parquet"
            for split in ("train", "monitor")
        }
        for variant in ("static", "dynamic")
    }
    manifest_path = args.output_dir / "split_manifest.json"
    outputs = [manifest_path] + [path for mapping in paths.values() for path in mapping.values()]
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError("split outputs already exist; use --force to rebuild intentionally")
    for path in (args.static_train, args.dynamic_train):
        if not path.is_file():
            raise FileNotFoundError(path)

    result, manifest = split_paired_frames(
        pd.read_parquet(args.static_train),
        pd.read_parquet(args.dynamic_train),
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    for variant, (train, monitor) in result.items():
        _atomic_parquet(train, paths[variant]["train"])
        _atomic_parquet(monitor, paths[variant]["monitor"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.update(
        {
            "static_source": str(args.static_train.resolve()),
            "dynamic_source": str(args.dynamic_train.resolve()),
            "outputs": {
                variant: {split: str(path.resolve()) for split, path in mapping.items()}
                for variant, mapping in paths.items()
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
