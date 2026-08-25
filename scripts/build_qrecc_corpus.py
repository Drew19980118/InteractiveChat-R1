#!/usr/bin/env python3
"""Build the JSONL corpus aligned with downloaded QReCC passage shards.

This performs no network access.  It preserves input shard order and writes
only ``passage_id`` and ``passage_text`` so the resulting row positions stay
aligned with the companion FAISS index.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    collection_dir = args.collection_dir
    shards = sorted(collection_dir.glob("part_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No part_*.parquet files in {collection_dir}")
    output = args.output or collection_dir / "qrecc_index.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("wb") as destination:
        for shard in shards:
            reader = pq.ParquetFile(shard)
            for batch in reader.iter_batches(
                batch_size=50_000, columns=["passage_id", "passage_text"]
            ):
                rows = batch.to_pylist()
                destination.write(b"\n".join(orjson.dumps(row) for row in rows))
                destination.write(b"\n")
                count += len(rows)
    print(f"Wrote {output}: {count} passages from {len(shards)} shards.")


if __name__ == "__main__":
    main()
