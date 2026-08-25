#!/usr/bin/env bash
# Controlled UCI ablation: replace UCI only with answer-vs-gold token F1,
# retaining action, clarity, patience, format, and clarification-F1 channels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET=inscit
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_inscit_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_inscit_test.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sim_user_inscit_3b_uci_to_f1_15steps}"
export TOTAL_STEPS="${TOTAL_STEPS:-15}"

source "$SCRIPT_DIR/experiment_common.sh"
export SIMULATED_USER_REWARD_MODE=uci_replaced_by_answer_f1
export SIMULATED_USER_ANSWER_F1_WEIGHT=1.0
export SIMULATED_USER_UCI_WEIGHT=0.0
exec bash "$SCRIPT_DIR/run_simulated_user_train.sh"
