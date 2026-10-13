#!/usr/bin/env bash
# Train one paired InSCIT context variant with the full ConvAgent reward.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${CONTEXT_VARIANT:?Set CONTEXT_VARIANT to static or dynamic.}"
: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-3B-Instruct.}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"
case "$CONTEXT_VARIANT" in static|dynamic) ;; *) echo "ERROR: invalid CONTEXT_VARIANT" >&2; exit 2;; esac

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/inscit_context_ablation/v1}"
SPLIT_ROOT="${SPLIT_ROOT:-$DATA_ROOT/splits_seed42}"
HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-convagent_inscit_${CONTEXT_VARIANT}_context_seed42}"
SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-interactivechat-r1}"
SWANLAB_RUN_GROUP="${SWANLAB_RUN_GROUP:-inscit_context_ablation_seed42}"

for path in \
  "$DATA_ROOT/static/inscit_train.parquet" \
  "$DATA_ROOT/dynamic/inscit_train.parquet" \
  "$DATA_ROOT/static/inscit_test.parquet" \
  "$DATA_ROOT/dynamic/inscit_test.parquet"; do
  [[ -f "$path" ]] || { echo "ERROR: missing paired dataset $path" >&2; exit 2; }
done

if [[ ! -f "$SPLIT_ROOT/split_manifest.json" ]]; then
  python -u scripts/prepare_inscit_context_ablation_splits.py \
    --static-train "$DATA_ROOT/static/inscit_train.parquet" \
    --dynamic-train "$DATA_ROOT/dynamic/inscit_train.parquet" \
    --output-dir "$SPLIT_ROOT" \
    --holdout-fraction "$HOLDOUT_FRACTION" \
    --seed "$SPLIT_SEED"
fi

export STATIC_BASELINE=convagent DATASET=inscit
export STATIC_TRAIN_SOURCE="$DATA_ROOT/$CONTEXT_VARIANT/inscit_train.parquet"
export STATIC_TRAIN_FILE="$SPLIT_ROOT/$CONTEXT_VARIANT/inscit_train.parquet"
export STATIC_MONITOR_FILE="$SPLIT_ROOT/$CONTEXT_VARIANT/inscit_monitor.parquet"
export STATIC_SPLIT_DIR="$SPLIT_ROOT/$CONTEXT_VARIANT"
export STATIC_SPLIT_MANIFEST="$SPLIT_ROOT/split_manifest.json"
export HOLDOUT_FRACTION SPLIT_SEED EXPERIMENT_NAME CONTEXT_VARIANT
export TRAINING_SEED="$SPLIT_SEED"
export TRACKING_BACKEND=swanlab SWANLAB_ENABLED=true SWANLAB_PROJ_NAME SWANLAB_RUN_GROUP
export WANDB_ENABLED=false

# Full paper ConvAgent reward: max-reference F1 + 0.5*(retrieval + action).
export CONVAGENT_RETRIEVAL_REWARD=true
export CONVAGENT_ACTION_REWARD=true
export CONVAGENT_FORMAT_REWARD=false
export CONVAGENT_IGT_REWARD=false
export CONVAGENT_SENTENCE_BERTSCORE_REWARD=false
export CONVAGENT_ANSWER_BERTSCORE_REWARD=false

exec bash scripts/run_paper_static_baseline_train.sh

