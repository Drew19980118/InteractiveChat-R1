#!/usr/bin/env bash
# Canonical InteractiveChat-R1: InsCiT Qwen2.5-3B, F1 + user satisfaction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export DATASET=inscit
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_inscit_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_inscit_test.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-interactivechat_r1_inscit_3b_full_f1_satisfaction_15steps}"
export TOTAL_STEPS="${TOTAL_STEPS:-15}"
source "$SCRIPT_DIR/experiment_common.sh"
exec bash "$SCRIPT_DIR/run_simulated_user_train.sh"
