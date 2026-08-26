#!/usr/bin/env bash
# Transfer: InsCiT-trained Qwen2.5-3B full checkpoint -> TopiOCQA online test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export DATASET=topiocqa
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_inscit_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_topiocqa_test.parquet}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/outputs/inscit/interactivechat_r1_inscit_3b_full_f1_satisfaction_15steps/global_step_14}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-interactivechat_r1_topiocqa_from_inscit_3b_full_f1_satisfaction_step14}"
source "$SCRIPT_DIR/experiment_common.sh"
exec bash "$SCRIPT_DIR/run_simulated_user_val.sh"
