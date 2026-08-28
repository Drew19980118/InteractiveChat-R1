#!/usr/bin/env bash
# Train static ConvAgent on QReCC, then evaluate the selected checkpoint on
# the QReCC static test set.  CoRAL evaluation must be launched only after
# the local retriever has been restarted with the CoRAL collection.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-3B/7B-Instruct.}"
: "${CUDA_VISIBLE_DEVICES:?Set training GPUs.}"

MODEL_TAG="${MODEL_TAG:-qwen25_3b}"
TRAIN_EXPERIMENT_NAME="${TRAIN_EXPERIMENT_NAME:-static_convagent_qrecc_${MODEL_TAG}}"

DATASET=qrecc \
EXPERIMENT_NAME="$TRAIN_EXPERIMENT_NAME" \
bash scripts/run_static_convagent_train.sh

CHECKPOINT_PATH="$(< "$PROJECT_ROOT/outputs/static_convagent/$TRAIN_EXPERIMENT_NAME/final_checkpoint.txt")"
EXPERIMENT_NAME="${TRAIN_EXPERIMENT_NAME}_to_qrecc" \
TRAIN_DATASET=qrecc \
EVAL_DATASET=qrecc \
CHECKPOINT_PATH="$CHECKPOINT_PATH" \
bash scripts/run_static_convagent_eval.sh

echo "QReCC source evaluation completed."
echo "For CoRAL, restart the retriever with collection/coral, then run:"
echo "  TRAIN_DATASET=qrecc EVAL_DATASET=coral CHECKPOINT_PATH=$CHECKPOINT_PATH EXPERIMENT_NAME=${TRAIN_EXPERIMENT_NAME}_to_coral bash scripts/run_static_convagent_eval.sh"
