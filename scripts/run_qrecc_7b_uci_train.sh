#!/usr/bin/env bash
# Canonical InteractiveChat-R1 run: QReCC train -> QReCC online test, Qwen2.5-7B.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET=qrecc
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-7B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_qrecc_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_qrecc_test.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-interactivechat_r1_qrecc_7b_uci_30steps}"
export TOTAL_STEPS="${TOTAL_STEPS:-30}"

source "$SCRIPT_DIR/experiment_common.sh"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.04}"
exec bash "$SCRIPT_DIR/run_simulated_user_train.sh"
