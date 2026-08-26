#!/usr/bin/env bash
# Evaluate a QReCC-trained Qwen2.5-3B full checkpoint on QReCC online test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export DATASET=qrecc
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_qrecc_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_qrecc_test.parquet}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/outputs/qrecc/interactivechat_r1_qrecc_3b_full_f1_satisfaction_30steps/global_step_29}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-interactivechat_r1_qrecc_3b_full_f1_satisfaction_step29_val}"
source "$SCRIPT_DIR/experiment_common.sh"
exec bash "$SCRIPT_DIR/run_simulated_user_val.sh"
