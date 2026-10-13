#!/usr/bin/env python3
"""Summarize the paired 2x2 InSCIT context-ablation evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence


METRICS = (
    "f1",
    "bertscore_f1",
    "ndcg_at_3",
    "action_accuracy",
    "terminal_success_rate",
    "format_success_rate",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-train-static-test", type=Path, required=True, dest="ss")
    parser.add_argument("--static-train-dynamic-test", type=Path, required=True, dest="sd")
    parser.add_argument("--dynamic-train-static-test", type=Path, required=True, dest="ds")
    parser.add_argument("--dynamic-train-dynamic-test", type=Path, required=True, dest="dd")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be >= 1")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def load_cell(directory: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(directory / "metrics_per_sample.jsonl")
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id") or "").strip()
        if not pair_id:
            raise ValueError(f"Missing pair_id in {directory}; re-run evaluation with metadata export")
        if pair_id in mapped:
            raise ValueError(f"Duplicate pair_id {pair_id} in {directory}")
        row = dict(row)
        row["terminal_success_rate"] = float(bool(row.get("has_terminal_answer")))
        row["format_success_rate"] = float(bool(row.get("format_valid")))
        mapped[pair_id] = row
    return mapped


def subset_predicates() -> dict[str, Callable[[Mapping[str, Any]], bool]]:
    return {
        "all": lambda row: True,
        "history_bearing": lambda row: int(row.get("turn_index") or 0) > 0,
        "dependency_required": lambda row: row.get("dependency_class") == "dependent_sufficient",
        "independent": lambda row: row.get("dependency_class") == "independent",
    }


def metric_mean(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    values = [row.get(metric) for row in rows]
    values = [float(value) for value in values if value is not None]
    return fmean(values) if values else 0.0


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    differences = [float(left[pair_id][metric]) - float(right[pair_id][metric]) for pair_id in ids]
    if not differences:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    boot = [
        fmean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(samples)
    ]
    return {
        "estimate": fmean(differences),
        "ci_low": percentile(boot, 0.025),
        "ci_high": percentile(boot, 0.975),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cells = {name: load_cell(getattr(args, name)) for name in ("ss", "sd", "ds", "dd")}
    pair_sets = {name: set(rows) for name, rows in cells.items()}
    if len({frozenset(values) for values in pair_sets.values()}) != 1:
        details = {name: len(values) for name, values in pair_sets.items()}
        raise ValueError(f"The four evaluation cells are not pair-aligned: {details}")
    pair_ids = sorted(next(iter(pair_sets.values())))

    subsets: dict[str, Any] = {}
    for subset_name, predicate in subset_predicates().items():
        ids = [pair_id for pair_id in pair_ids if predicate(cells["ss"][pair_id])]
        cell_summary = {}
        for cell_name, rows in cells.items():
            selected = [rows[pair_id] for pair_id in ids]
            cell_summary[cell_name] = {
                "sample_count": len(selected),
                "terminal_answer_count": sum(bool(row.get("has_terminal_answer")) for row in selected),
                "metrics": {metric: metric_mean(selected, metric) for metric in METRICS},
            }
        contrasts = {}
        for contrast_name, left_name, right_name in (
            ("dynamic_vs_static_train_on_static_test", "ds", "ss"),
            ("dynamic_vs_static_train_on_dynamic_test", "dd", "sd"),
            ("dynamic_vs_static_test_for_static_train", "sd", "ss"),
            ("dynamic_vs_static_test_for_dynamic_train", "dd", "ds"),
        ):
            contrasts[contrast_name] = {
                metric: paired_bootstrap(
                    cells[left_name],
                    cells[right_name],
                    ids,
                    metric,
                    samples=args.bootstrap_samples,
                    seed=args.seed + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }
        subsets[subset_name] = {"cells": cell_summary, "contrasts": contrasts}

    summary = {
        "cell_names": {
            "ss": "static train -> static test",
            "sd": "static train -> dynamic test",
            "ds": "dynamic train -> static test",
            "dd": "dynamic train -> dynamic test",
        },
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "subsets": subsets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "context_ablation_2x2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (args.output_dir / "context_ablation_2x2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subset", "cell", "sample_count", *METRICS])
        for subset_name, subset in subsets.items():
            for cell_name, cell in subset["cells"].items():
                writer.writerow(
                    [subset_name, cell_name, cell["sample_count"]]
                    + [cell["metrics"][metric] for metric in METRICS]
                )

    lines = [
        "# InSCIT context ablation: 2x2 evaluation",
        "",
        "Values are percentages. SS/SD/DS/DD denote train-context/test-context.",
        "",
    ]
    for subset_name, subset in subsets.items():
        lines.extend(
            [
                f"## {subset_name}",
                "",
                "| Cell | N | F1 | BERTScore F1 | NDCG@3 | Action accuracy | Terminal success | Format success |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cell_name in ("ss", "sd", "ds", "dd"):
            cell = subset["cells"][cell_name]
            values = cell["metrics"]
            lines.append(
                f"| {cell_name.upper()} | {cell['sample_count']} | "
                + " | ".join(f"{100 * values[metric]:.2f}" for metric in METRICS)
                + " |"
            )
        lines.append("")
    (args.output_dir / "context_ablation_2x2.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "pairs": len(pair_ids)}, indent=2))


if __name__ == "__main__":
    main()
