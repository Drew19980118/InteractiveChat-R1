#!/usr/bin/env bash
# Serial InsCiT-3B comparison suite:
#   1. static ConvAgent-style train + source static validation;
#   2. static ChatR1-style train + source static validation;
#   3. InteractiveChat-R1 w/o level-wise satisfaction reward (patience-only)
#      train + final full online validation.
#
# The first two runs require the InsCiT retriever collection.  The third also
# needs the frozen user-simulator endpoint.  Cross-domain TopiOCQA validation
# is deliberately left to the dedicated evaluation launchers, because it
# requires restarting the retriever with the TopiOCQA collection.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to models/Qwen2.5-3B-Instruct.}"
: "${CUDA_VISIBLE_DEVICES:?Set the training GPU ids.}"

MODEL_TAG="${MODEL_TAG:-qwen25_3b}"
CONVAGENT_EXPERIMENT_NAME="${CONVAGENT_EXPERIMENT_NAME:-static_convagent_inscit_${MODEL_TAG}}"
CHATR1_EXPERIMENT_NAME="${CHATR1_EXPERIMENT_NAME:-static_chatr1_inscit_${MODEL_TAG}}"
ABLATION_EXPERIMENT_NAME="${ABLATION_EXPERIMENT_NAME:-interactivechat_r1_inscit_3b_wo_clarity_reward_15steps}"

echo "===== [1/3] Static ConvAgent: InsCiT train + InsCiT static validation ====="
TRAIN_EXPERIMENT_NAME="$CONVAGENT_EXPERIMENT_NAME" \
  MODEL_TAG="$MODEL_TAG" \
  bash "$SCRIPT_DIR/run_static_convagent_inscit_suite.sh"

echo "===== [2/3] Static ChatR1: InsCiT train + InsCiT static validation ====="
TRAIN_EXPERIMENT_NAME="$CHATR1_EXPERIMENT_NAME" \
  MODEL_TAG="$MODEL_TAG" \
  bash "$SCRIPT_DIR/run_static_chatr1_inscit_suite.sh"

echo "===== [3/3] InteractiveChat-R1: w/o level-wise satisfaction reward (patience-only) ====="
EXPERIMENT_NAME="$ABLATION_EXPERIMENT_NAME" \
  bash "$SCRIPT_DIR/run_inscit_3b_wo_clarity_reward_train.sh"

echo "===== Serial InsCiT-3B suite completed ====="
echo "ConvAgent metrics: $PROJECT_ROOT/eval_log/static_convagent/${CONVAGENT_EXPERIMENT_NAME}_to_inscit/metrics_summary.json"
echo "ChatR1 metrics:    $PROJECT_ROOT/eval_log/static_chatr1/${CHATR1_EXPERIMENT_NAME}_to_inscit/metrics_summary.json"
echo "Ablation metrics:  $PROJECT_ROOT/eval_log/inscit/${ABLATION_EXPERIMENT_NAME}/metrics_summary.json"
