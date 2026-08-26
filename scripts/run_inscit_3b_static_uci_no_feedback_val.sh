#!/usr/bin/env bash
# Standalone online-environment evaluation for the static/no-feedback ablation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET=inscit
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_inscit_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_inscit_test.parquet}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/outputs/inscit/sim_user_inscit_3b_static_no_feedback_f1_15steps/global_step_14}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sim_user_inscit_3b_static_no_feedback_f1_step14_val}"

source "$SCRIPT_DIR/experiment_common.sh"
export SIMULATED_USER_ENABLE_FEEDBACK=false
export SIMULATED_USER_STATIC_GOLD_CONTEXT=true
export MAX_ANSWER_DEPTH=1
exec bash "$SCRIPT_DIR/run_simulated_user_val.sh"
