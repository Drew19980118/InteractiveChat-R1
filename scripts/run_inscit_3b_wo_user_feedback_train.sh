#!/usr/bin/env bash
# InsCiT Qwen2.5-3B ablation: remove user feedback and answer retries while
# retaining the online, model-generated dialogue context.  Later source
# subtasks observe the policy answer or an environmental gold fallback, never
# the dataset's canonical gold prefix.  Validation makes one hidden simulator
# judgement per answer to report user satisfaction only; it is not feedback.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET=inscit
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_inscit_train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_inscit_test.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-interactivechat_r1_inscit_3b_wo_user_feedback_15steps}"
export TOTAL_STEPS="${TOTAL_STEPS:-15}"

source "$SCRIPT_DIR/experiment_common.sh"
export SIMULATED_USER_REWARD_MODE=full
export SIMULATED_USER_ENABLE_FEEDBACK=false
export SIMULATED_USER_ASSESS_SATISFACTION=true
export SIMULATED_USER_STATIC_GOLD_CONTEXT=false
export SIMULATED_USER_CLARITY_WEIGHT=0.0
export SIMULATED_USER_PATIENCE_WEIGHT=0.0
export MAX_ANSWER_DEPTH=1
exec bash "$SCRIPT_DIR/run_simulated_user_train.sh"
