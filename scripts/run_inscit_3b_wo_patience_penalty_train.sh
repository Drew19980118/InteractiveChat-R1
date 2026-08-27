#!/usr/bin/env bash
# InsCiT Qwen2.5-3B ablation: retain clarity reward and interactive feedback,
# but remove the explicit level-2/3 patience penalty.  All non-patience
# channels and dynamic-context mechanics match the full method.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET=inscit
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_inscit_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_inscit_test.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-interactivechat_r1_inscit_3b_wo_patience_penalty_15steps}"
export TOTAL_STEPS="${TOTAL_STEPS:-15}"

source "$SCRIPT_DIR/experiment_common.sh"
export SIMULATED_USER_REWARD_MODE=full
export SIMULATED_USER_ENABLE_FEEDBACK=true
export SIMULATED_USER_ASSESS_SATISFACTION=false
export SIMULATED_USER_STATIC_GOLD_CONTEXT=false
export SIMULATED_USER_PATIENCE_WEIGHT=0.0
exec bash "$SCRIPT_DIR/run_simulated_user_train.sh"
