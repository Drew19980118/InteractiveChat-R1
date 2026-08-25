#!/usr/bin/env bash
# Static-context / no-feedback ablation on InsCiT with Qwen2.5-3B.
#
# Each original sub-task starts from the data's canonical gold dialogue prefix;
# sampled answers never become context for a later source sub-task.  The user
# simulator, clarity, patience, evidence-utility, and search-efficiency terms
# are disabled.  UCI, action, format, and clarification-F1 remain active.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET=inscit
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_inscit_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_inscit_test.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sim_user_inscit_3b_static_uci_no_feedback_15steps}"
export TOTAL_STEPS="${TOTAL_STEPS:-15}"

source "$SCRIPT_DIR/experiment_common.sh"
export SIMULATED_USER_REWARD_MODE=full
export SIMULATED_USER_ENABLE_FEEDBACK=false
export SIMULATED_USER_STATIC_GOLD_CONTEXT=true
export SIMULATED_USER_ENABLE_EVIDENCE_UTILITY=false
export SIMULATED_USER_ENABLE_SEARCH_EFFICIENCY=false
export SIMULATED_USER_ANSWER_F1_WEIGHT=0.0
export SIMULATED_USER_EVIDENCE_UTILITY_WEIGHT=0.0
export SIMULATED_USER_SEARCH_EFFICIENCY_WEIGHT=0.0
export SIMULATED_USER_ACTION_WEIGHT=1.0
export SIMULATED_USER_UCI_WEIGHT=1.0
export SIMULATED_USER_CLARITY_WEIGHT=0.0
export SIMULATED_USER_PATIENCE_WEIGHT=0.0
export SIMULATED_USER_FORMAT_WEIGHT=1.0
export SIMULATED_USER_CLARIFY_F1_WEIGHT=1.0
export MAX_ANSWER_DEPTH=1
exec bash "$SCRIPT_DIR/run_simulated_user_train.sh"
