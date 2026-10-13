#!/usr/bin/env bash
# Evaluate both selected policies on both paired test-context views.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-3B-Instruct.}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/inscit_context_ablation/v1}"
SPLIT_ROOT="${SPLIT_ROOT:-$DATA_ROOT/splits_seed42}"
STATIC_EXPERIMENT="${STATIC_EXPERIMENT:-convagent_inscit_static_context_seed42}"
DYNAMIC_EXPERIMENT="${DYNAMIC_EXPERIMENT:-convagent_inscit_dynamic_context_seed42}"
STATIC_CHECKPOINT="${STATIC_CHECKPOINT:-}"
DYNAMIC_CHECKPOINT="${DYNAMIC_CHECKPOINT:-}"
SUMMARY_DIR="${SUMMARY_DIR:-$PROJECT_ROOT/eval_results/inscit_context_ablation_seed42}"

if [[ -z "$STATIC_CHECKPOINT" ]]; then
  pointer="$PROJECT_ROOT/outputs/paper_static_convagent/$STATIC_EXPERIMENT/final_checkpoint.txt"
  [[ -f "$pointer" ]] || { echo "ERROR: missing $pointer" >&2; exit 2; }
  STATIC_CHECKPOINT="$(<"$pointer")"
fi
if [[ -z "$DYNAMIC_CHECKPOINT" ]]; then
  pointer="$PROJECT_ROOT/outputs/paper_static_convagent/$DYNAMIC_EXPERIMENT/final_checkpoint.txt"
  [[ -f "$pointer" ]] || { echo "ERROR: missing $pointer" >&2; exit 2; }
  DYNAMIC_CHECKPOINT="$(<"$pointer")"
fi

run_cell() {
  local train_variant="$1" test_variant="$2" checkpoint="$3" cell="$4"
  local experiment="inscit_context_ablation_${cell}_seed42"
  STATIC_BASELINE=convagent \
  TRAIN_DATASET=inscit \
  EVAL_DATASET=inscit \
  TRAIN_FILE="$SPLIT_ROOT/$train_variant/inscit_train.parquet" \
  VAL_FILE="$DATA_ROOT/$test_variant/inscit_test.parquet" \
  CHECKPOINT_PATH="$checkpoint" \
  EXPERIMENT_NAME="$experiment" \
    bash scripts/run_paper_static_baseline_eval.sh
}

run_cell static static "$STATIC_CHECKPOINT" ss
run_cell static dynamic "$STATIC_CHECKPOINT" sd
run_cell dynamic static "$DYNAMIC_CHECKPOINT" ds
run_cell dynamic dynamic "$DYNAMIC_CHECKPOINT" dd

python -u scripts/summarize_inscit_context_ablation.py \
  --static-train-static-test "$PROJECT_ROOT/eval_log/paper_static_eval/inscit_context_ablation_ss_seed42" \
  --static-train-dynamic-test "$PROJECT_ROOT/eval_log/paper_static_eval/inscit_context_ablation_sd_seed42" \
  --dynamic-train-static-test "$PROJECT_ROOT/eval_log/paper_static_eval/inscit_context_ablation_ds_seed42" \
  --dynamic-train-dynamic-test "$PROJECT_ROOT/eval_log/paper_static_eval/inscit_context_ablation_dd_seed42" \
  --output-dir "$SUMMARY_DIR" \
  --bootstrap-samples 1000 \
  --seed 42

echo "2x2 summary: $SUMMARY_DIR/context_ablation_2x2.md"
