#!/usr/bin/env bash
# Re-evaluate a QReCC-trained Qwen2.5-7B FSDP checkpoint on QReCC online test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export DATASET=qrecc
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-7B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_qrecc_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_qrecc_test.parquet}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/outputs/qrecc/interactivechat_r1_qrecc_7b_uci_30steps/global_step_29}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-interactivechat_r1_qrecc_7b_uci_step29_val}"
source "$SCRIPT_DIR/experiment_common.sh"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.04}"
exec bash "$SCRIPT_DIR/run_simulated_user_val.sh"
