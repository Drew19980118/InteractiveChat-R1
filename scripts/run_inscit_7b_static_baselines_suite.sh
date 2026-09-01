#!/usr/bin/env bash
# Serial InsCiT Qwen2.5-7B static-baseline suite:
#   1. static ConvAgent-style training + selected-checkpoint evaluation on
#      the InsCiT static test set;
#   2. static ChatR1-style training + selected-checkpoint evaluation on
#      the same InsCiT static test set.
#
# Both stages use the unchanged static-baseline budget and early-stopping
# monitor from their respective launchers.  Keep the retriever on the InsCiT
# collection throughout.  No user simulator is used by either baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to models/Qwen2.5-7B-Instruct.}"
: "${CUDA_VISIBLE_DEVICES:?Set the training GPU ids.}"

MODEL_TAG="${MODEL_TAG:-qwen25_7b}"
CONVAGENT_EXPERIMENT_NAME="${CONVAGENT_EXPERIMENT_NAME:-static_convagent_inscit_${MODEL_TAG}}"
CHATR1_EXPERIMENT_NAME="${CHATR1_EXPERIMENT_NAME:-static_chatr1_inscit_${MODEL_TAG}}"
EXPORT_ACTOR_ONLY="${EXPORT_ACTOR_ONLY:-false}"
HF_UPLOAD_ACTOR_ONLY="${HF_UPLOAD_ACTOR_ONLY:-false}"
HF_CONVAGENT_REPO_ID="${HF_CONVAGENT_REPO_ID:-}"
HF_CHATR1_REPO_ID="${HF_CHATR1_REPO_ID:-}"
HF_UPLOAD_NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-8}"

if [[ "$HF_UPLOAD_ACTOR_ONLY" == "true" && "$EXPORT_ACTOR_ONLY" != "true" ]]; then
  echo "ERROR: HF_UPLOAD_ACTOR_ONLY=true requires EXPORT_ACTOR_ONLY=true." >&2
  exit 2
fi

export_selected_actor() {
  local baseline="$1"
  local experiment_name="$2"
  local train_file="$3"
  local repo_id="$4"
  local checkpoint_path
  local export_dir

  [[ "$EXPORT_ACTOR_ONLY" == "true" ]] || return 0

  checkpoint_path="$(< "$PROJECT_ROOT/outputs/$baseline/$experiment_name/final_checkpoint.txt")"
  export_dir="$PROJECT_ROOT/exports/${experiment_name}_actor_hf"
  echo "===== Exporting $baseline actor-only checkpoint: $checkpoint_path ====="

  CHECKPOINT_PATH="$checkpoint_path" \
    MODEL_PATH="$MODEL_PATH" \
    TRAIN_FILE="$train_file" \
    ACTOR_EXPORT_DIR="$export_dir" \
    N_GPUS="${N_GPUS:-2}" \
    ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-${N_GPUS:-2}}" \
    HF_UPLOAD_ACTOR_ONLY="$HF_UPLOAD_ACTOR_ONLY" \
    HF_REPO_ID="$repo_id" \
    HF_UPLOAD_NUM_WORKERS="$HF_UPLOAD_NUM_WORKERS" \
    bash "$SCRIPT_DIR/export_and_upload_actor_policy.sh"
}

echo "===== [1/2] Static ConvAgent: InsCiT train + InsCiT static validation ====="
TRAIN_EXPERIMENT_NAME="$CONVAGENT_EXPERIMENT_NAME" \
  MODEL_TAG="$MODEL_TAG" \
  bash "$SCRIPT_DIR/run_static_convagent_inscit_suite.sh"

export_selected_actor \
  "static_convagent" \
  "$CONVAGENT_EXPERIMENT_NAME" \
  "$PROJECT_ROOT/data/static_convagent_splits/inscit/inscit_train.parquet" \
  "$HF_CONVAGENT_REPO_ID"

echo "===== [2/2] Static ChatR1: InsCiT train + InsCiT static validation ====="
TRAIN_EXPERIMENT_NAME="$CHATR1_EXPERIMENT_NAME" \
  MODEL_TAG="$MODEL_TAG" \
  bash "$SCRIPT_DIR/run_static_chatr1_inscit_suite.sh"

export_selected_actor \
  "static_chatr1" \
  "$CHATR1_EXPERIMENT_NAME" \
  "$PROJECT_ROOT/data/static_chatr1_splits/inscit/inscit_train.parquet" \
  "$HF_CHATR1_REPO_ID"

echo "===== Serial InsCiT-7B static-baseline suite completed ====="
echo "ConvAgent metrics: $PROJECT_ROOT/eval_log/static_convagent/${CONVAGENT_EXPERIMENT_NAME}_to_inscit/metrics_summary.json"
echo "ChatR1 metrics:    $PROJECT_ROOT/eval_log/static_chatr1/${CHATR1_EXPERIMENT_NAME}_to_inscit/metrics_summary.json"
if [[ "$EXPORT_ACTOR_ONLY" == "true" ]]; then
  echo "ConvAgent actor: $PROJECT_ROOT/exports/${CONVAGENT_EXPERIMENT_NAME}_actor_hf"
  echo "ChatR1 actor:    $PROJECT_ROOT/exports/${CHATR1_EXPERIMENT_NAME}_actor_hf"
fi
