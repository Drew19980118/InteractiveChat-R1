#!/usr/bin/env bash
# Serial InsCiT Qwen2.5-7B static-baseline suite:
#   1. static ConvAgent-style training + selected-checkpoint evaluation on
#      the InsCiT static test set;
#   2. static ChatR1-style training + selected-checkpoint evaluation on
#      the same InsCiT static test set.
#
# Both stages use the unchanged static-baseline budget and early-stopping
# monitor from their respective launchers.  Keep the retriever on the InsCiT
# collection throughout.  No user simulator is used by either baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to models/Qwen2.5-7B-Instruct.}"
: "${CUDA_VISIBLE_DEVICES:?Set the training GPU ids.}"

MODEL_TAG="${MODEL_TAG:-qwen25_7b}"
CONVAGENT_EXPERIMENT_NAME="${CONVAGENT_EXPERIMENT_NAME:-static_convagent_inscit_${MODEL_TAG}}"
CHATR1_EXPERIMENT_NAME="${CHATR1_EXPERIMENT_NAME:-static_chatr1_inscit_${MODEL_TAG}}"

echo "===== [1/2] Static ConvAgent: InsCiT train + InsCiT static validation ====="
TRAIN_EXPERIMENT_NAME="$CONVAGENT_EXPERIMENT_NAME" \
  MODEL_TAG="$MODEL_TAG" \
  bash "$SCRIPT_DIR/run_static_convagent_inscit_suite.sh"

echo "===== [2/2] Static ChatR1: InsCiT train + InsCiT static validation ====="
TRAIN_EXPERIMENT_NAME="$CHATR1_EXPERIMENT_NAME" \
  MODEL_TAG="$MODEL_TAG" \
  bash "$SCRIPT_DIR/run_static_chatr1_inscit_suite.sh"

echo "===== Serial InsCiT-7B static-baseline suite completed ====="
echo "ConvAgent metrics: $PROJECT_ROOT/eval_log/static_convagent/${CONVAGENT_EXPERIMENT_NAME}_to_inscit/metrics_summary.json"
echo "ChatR1 metrics:    $PROJECT_ROOT/eval_log/static_chatr1/${CHATR1_EXPERIMENT_NAME}_to_inscit/metrics_summary.json"
