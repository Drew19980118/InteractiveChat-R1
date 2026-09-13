#!/usr/bin/env python3
"""Build the JSONL passage corpus aligned with the released InsCiT E5 index.

The Hub release stores the passages in ``part_*.parquet`` shards and the
corresponding FAISS index in separately downloaded shards.  The retriever
addresses passages by their *row position*, so this utility preserves the
lexicographic shard order and writes one JSON object per original row.

It accepts the common field names used by released passage collections and
normalizes the output records to ``passage_id`` and ``passage_text``.

Example
-------
    python scripts/build_inscit_corpus.py --collection-dir collection/inscit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

try:
    import orjson
except ImportError:  # Keep the one-off setup utility usable in minimal envs.
    import json

    def serialize_json(value: dict[str, str]) -> bytes:
        return json.dumps(value, ensure_ascii=False).encode("utf-8")

else:

    def serialize_json(value: dict[str, str]) -> bytes:
        return orjson.dumps(value)


ID_CANDIDATES = ("passage_id", "id", "docid", "doc_id")
TEXT_CANDIDATES = ("passage_text", "text", "contents", "content")


def resolve_column(columns: set[str], candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    available = ", ".join(sorted(columns))
    expected = ", ".join(candidates)
    raise ValueError(
        f"Could not find an {label} column. Expected one of [{expected}], "
        f"but the Parquet schema contains [{available}]."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    collection_dir = args.collection_dir
    shards = sorted(collection_dir.glob("part_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No part_*.parquet files in {collection_dir}")

    first_schema = set(pq.ParquetFile(shards[0]).schema_arrow.names)
    id_column = resolve_column(first_schema, ID_CANDIDATES, "passage-id")
    text_column = resolve_column(first_schema, TEXT_CANDIDATES, "passage-text")
    output = args.output or collection_dir / "inscit_index.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("wb") as destination:
        for shard in shards:
            reader = pq.ParquetFile(shard)
            schema_columns = set(reader.schema_arrow.names)
            if id_column not in schema_columns or text_column not in schema_columns:
                raise ValueError(
                    f"{shard} does not contain the expected columns "
                    f"{id_column!r}, {text_column!r}."
                )
            for batch in reader.iter_batches(
                batch_size=50_000, columns=[id_column, text_column]
            ):
                for row in batch.to_pylist():
                    destination.write(
                        serialize_json(
                            {
                                "passage_id": str(row[id_column]),
                                "passage_text": str(row[text_column]),
                            }
                        )
                    )
                    destination.write(b"\n")
                    count += 1

    print(f"Wrote {output}: {count} passages from {len(shards)} shards.")


if __name__ == "__main__":
    main()
