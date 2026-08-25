#!/usr/bin/env bash
# Re-evaluate an InsCiT-trained Qwen2.5-3B FSDP checkpoint on InsCiT online test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export DATASET=inscit
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_inscit_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_inscit_test.parquet}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/outputs/inscit/interactivechat_r1_inscit_3b_uci_15steps/global_step_14}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-interactivechat_r1_inscit_3b_uci_step14_val}"
source "$SCRIPT_DIR/experiment_common.sh"
exec bash "$SCRIPT_DIR/run_simulated_user_val.sh"
