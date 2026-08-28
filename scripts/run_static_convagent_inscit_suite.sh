#!/usr/bin/env bash
# Train static ConvAgent on InsCiT, then evaluate the selected checkpoint on
# the InsCiT static test set.  TopiOCQA evaluation must be launched only
# after the local retriever has been restarted with the TopiOCQA collection.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-3B/7B-Instruct.}"
: "${CUDA_VISIBLE_DEVICES:?Set training GPUs.}"

MODEL_TAG="${MODEL_TAG:-qwen25_3b}"
TRAIN_EXPERIMENT_NAME="${TRAIN_EXPERIMENT_NAME:-static_convagent_inscit_${MODEL_TAG}}"

DATASET=inscit \
EXPERIMENT_NAME="$TRAIN_EXPERIMENT_NAME" \
bash scripts/run_static_convagent_train.sh

CHECKPOINT_PATH="$(< "$PROJECT_ROOT/outputs/static_convagent/$TRAIN_EXPERIMENT_NAME/final_checkpoint.txt")"
EXPERIMENT_NAME="${TRAIN_EXPERIMENT_NAME}_to_inscit" \
TRAIN_DATASET=inscit \
EVAL_DATASET=inscit \
CHECKPOINT_PATH="$CHECKPOINT_PATH" \
bash scripts/run_static_convagent_eval.sh

echo "InsCiT source evaluation completed."
echo "For TopiOCQA, restart the retriever with collection/topiocqa, then run:"
echo "  TRAIN_DATASET=inscit EVAL_DATASET=topiocqa CHECKPOINT_PATH=$CHECKPOINT_PATH EXPERIMENT_NAME=${TRAIN_EXPERIMENT_NAME}_to_topiocqa bash scripts/run_static_convagent_eval.sh"
