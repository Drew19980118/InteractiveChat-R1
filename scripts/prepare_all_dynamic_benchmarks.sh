#!/usr/bin/env bash
# Convert the four original static benchmark JSON files into one-dialogue-per-
# row dynamic-environment Parquet files.  No model or retrieval operation is
# performed in this step.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RAW_DATA_DIR="${RAW_DATA_DIR:-$PROJECT_ROOT/data/raw}"
OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-$PROJECT_ROOT/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

datasets=(inscit qrecc coral topiocqa)
for dataset in "${datasets[@]}"; do
  for split in train test; do
    input="$RAW_DATA_DIR/${dataset}_${split}.json"
    output="$OUTPUT_DATA_DIR/sim_user_${dataset}_${split}.parquet"
    if [[ ! -f "$input" ]]; then
      echo "ERROR: missing source JSON: $input" >&2
      exit 2
    fi
    "$PYTHON_BIN" "scripts/prepare_simulated_user_${dataset}.py" \
      --input "$input" --output "$output" --split "$split"
  done
done

echo "Dynamic benchmark Parquet files are in: $OUTPUT_DATA_DIR"
