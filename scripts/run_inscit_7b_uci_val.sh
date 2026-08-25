#!/usr/bin/env bash
# Re-evaluate an InsCiT-trained Qwen2.5-7B FSDP checkpoint on InsCiT online test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export DATASET=inscit
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-7B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_inscit_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_inscit_test.parquet}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/outputs/inscit/interactivechat_r1_inscit_7b_uci_15steps/global_step_14}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-interactivechat_r1_inscit_7b_uci_step14_val}"
source "$SCRIPT_DIR/experiment_common.sh"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.04}"
exec bash "$SCRIPT_DIR/run_simulated_user_val.sh"
