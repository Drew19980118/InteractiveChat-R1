#!/usr/bin/env bash
# Serial QReCC static-baseline suite:
#   1. Qwen2.5-3B static ConvAgent train + QReCC source validation + export/upload;
#   2. Qwen2.5-3B static ChatR1 train + QReCC source validation + export/upload;
#   3. Qwen2.5-7B static ConvAgent train + QReCC source validation + export/upload;
#   4. Qwen2.5-7B static ChatR1 train + QReCC source validation + export/upload.
#
# Each baseline retains its existing static-data reward, deterministic monitor
# split, plateau-based final-checkpoint selection, and source-test protocol.
# Keep the local retriever on the QReCC collection for this entire suite.
# CoRAL transfer validation is intentionally separate because it requires
# restarting that retriever with collection/coral.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${CUDA_VISIBLE_DEVICES:?Set the training GPU ids.}"

MODEL_3B_PATH="${MODEL_3B_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
MODEL_7B_PATH="${MODEL_7B_PATH:-$PROJECT_ROOT/models/Qwen2.5-7B-Instruct}"
MODEL_3B_TAG="${MODEL_3B_TAG:-qwen25_3b}"
MODEL_7B_TAG="${MODEL_7B_TAG:-qwen25_7b}"

CONVAGENT_3B_EXPERIMENT_NAME="${CONVAGENT_3B_EXPERIMENT_NAME:-static_convagent_qrecc_${MODEL_3B_TAG}}"
CHATR1_3B_EXPERIMENT_NAME="${CHATR1_3B_EXPERIMENT_NAME:-static_chatr1_qrecc_${MODEL_3B_TAG}}"
CONVAGENT_7B_EXPERIMENT_NAME="${CONVAGENT_7B_EXPERIMENT_NAME:-static_convagent_qrecc_${MODEL_7B_TAG}}"
CHATR1_7B_EXPERIMENT_NAME="${CHATR1_7B_EXPERIMENT_NAME:-static_chatr1_qrecc_${MODEL_7B_TAG}}"

EXPORT_ACTOR_ONLY="${EXPORT_ACTOR_ONLY:-false}"
HF_UPLOAD_ACTOR_ONLY="${HF_UPLOAD_ACTOR_ONLY:-false}"
HF_CONVAGENT_3B_REPO_ID="${HF_CONVAGENT_3B_REPO_ID:-}"
HF_CHATR1_3B_REPO_ID="${HF_CHATR1_3B_REPO_ID:-}"
HF_CONVAGENT_7B_REPO_ID="${HF_CONVAGENT_7B_REPO_ID:-}"
HF_CHATR1_7B_REPO_ID="${HF_CHATR1_7B_REPO_ID:-}"
HF_UPLOAD_NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-8}"

if [[ "$HF_UPLOAD_ACTOR_ONLY" == "true" && "$EXPORT_ACTOR_ONLY" != "true" ]]; then
  echo "ERROR: HF_UPLOAD_ACTOR_ONLY=true requires EXPORT_ACTOR_ONLY=true." >&2
  exit 2
fi
if [[ "$HF_UPLOAD_ACTOR_ONLY" == "true" ]]; then
  for repo_id in \
    "$HF_CONVAGENT_3B_REPO_ID" \
    "$HF_CHATR1_3B_REPO_ID" \
    "$HF_CONVAGENT_7B_REPO_ID" \
    "$HF_CHATR1_7B_REPO_ID"; do
    if [[ -z "$repo_id" ]]; then
      echo "ERROR: Set all four HF_*_REPO_ID variables when HF_UPLOAD_ACTOR_ONLY=true." >&2
      exit 2
    fi
  done
fi

export_selected_actor() {
  local baseline="$1"
  local experiment_name="$2"
  local model_path="$3"
  local train_file="$4"
  local repo_id="$5"
  local checkpoint_path
  local export_dir

  [[ "$EXPORT_ACTOR_ONLY" == "true" ]] || return 0

  checkpoint_path="$(< "$PROJECT_ROOT/outputs/$baseline/$experiment_name/final_checkpoint.txt")"
  export_dir="$PROJECT_ROOT/exports/${experiment_name}_actor_hf"
  echo "===== Exporting $baseline actor-only checkpoint: $checkpoint_path ====="

  CHECKPOINT_PATH="$checkpoint_path" \
    MODEL_PATH="$model_path" \
    TRAIN_FILE="$train_file" \
    ACTOR_EXPORT_DIR="$export_dir" \
    N_GPUS="${N_GPUS:-2}" \
    ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-${N_GPUS:-2}}" \
    HF_UPLOAD_ACTOR_ONLY="$HF_UPLOAD_ACTOR_ONLY" \
    HF_REPO_ID="$repo_id" \
    HF_UPLOAD_NUM_WORKERS="$HF_UPLOAD_NUM_WORKERS" \
    bash "$SCRIPT_DIR/export_and_upload_actor_policy.sh"
}

echo "===== [1/4] Static ConvAgent: QReCC 3B train + QReCC static validation ====="
MODEL_PATH="$MODEL_3B_PATH" \
  MODEL_TAG="$MODEL_3B_TAG" \
  TRAIN_EXPERIMENT_NAME="$CONVAGENT_3B_EXPERIMENT_NAME" \
  bash "$SCRIPT_DIR/run_static_convagent_qrecc_suite.sh"
export_selected_actor \
  "static_convagent" \
  "$CONVAGENT_3B_EXPERIMENT_NAME" \
  "$MODEL_3B_PATH" \
  "$PROJECT_ROOT/data/static_convagent_splits/qrecc/qrecc_train.parquet" \
  "$HF_CONVAGENT_3B_REPO_ID"

echo "===== [2/4] Static ChatR1: QReCC 3B train + QReCC static validation ====="
MODEL_PATH="$MODEL_3B_PATH" \
  MODEL_TAG="$MODEL_3B_TAG" \
  TRAIN_EXPERIMENT_NAME="$CHATR1_3B_EXPERIMENT_NAME" \
  bash "$SCRIPT_DIR/run_static_chatr1_qrecc_suite.sh"
export_selected_actor \
  "static_chatr1" \
  "$CHATR1_3B_EXPERIMENT_NAME" \
  "$MODEL_3B_PATH" \
  "$PROJECT_ROOT/data/static_chatr1_splits/qrecc/qrecc_train.parquet" \
  "$HF_CHATR1_3B_REPO_ID"

echo "===== [3/4] Static ConvAgent: QReCC 7B train + QReCC static validation ====="
MODEL_PATH="$MODEL_7B_PATH" \
  MODEL_TAG="$MODEL_7B_TAG" \
  TRAIN_EXPERIMENT_NAME="$CONVAGENT_7B_EXPERIMENT_NAME" \
  bash "$SCRIPT_DIR/run_static_convagent_qrecc_suite.sh"
export_selected_actor \
  "static_convagent" \
  "$CONVAGENT_7B_EXPERIMENT_NAME" \
  "$MODEL_7B_PATH" \
  "$PROJECT_ROOT/data/static_convagent_splits/qrecc/qrecc_train.parquet" \
  "$HF_CONVAGENT_7B_REPO_ID"

echo "===== [4/4] Static ChatR1: QReCC 7B train + QReCC static validation ====="
MODEL_PATH="$MODEL_7B_PATH" \
  MODEL_TAG="$MODEL_7B_TAG" \
  TRAIN_EXPERIMENT_NAME="$CHATR1_7B_EXPERIMENT_NAME" \
  bash "$SCRIPT_DIR/run_static_chatr1_qrecc_suite.sh"
export_selected_actor \
  "static_chatr1" \
  "$CHATR1_7B_EXPERIMENT_NAME" \
  "$MODEL_7B_PATH" \
  "$PROJECT_ROOT/data/static_chatr1_splits/qrecc/qrecc_train.parquet" \
  "$HF_CHATR1_7B_REPO_ID"

echo "===== Serial QReCC 3B -> 7B static-baseline suite completed ====="
echo "ConvAgent 3B metrics: $PROJECT_ROOT/eval_log/static_convagent/${CONVAGENT_3B_EXPERIMENT_NAME}_to_qrecc/metrics_summary.json"
echo "ChatR1 3B metrics:    $PROJECT_ROOT/eval_log/static_chatr1/${CHATR1_3B_EXPERIMENT_NAME}_to_qrecc/metrics_summary.json"
echo "ConvAgent 7B metrics: $PROJECT_ROOT/eval_log/static_convagent/${CONVAGENT_7B_EXPERIMENT_NAME}_to_qrecc/metrics_summary.json"
echo "ChatR1 7B metrics:    $PROJECT_ROOT/eval_log/static_chatr1/${CHATR1_7B_EXPERIMENT_NAME}_to_qrecc/metrics_summary.json"
if [[ "$EXPORT_ACTOR_ONLY" == "true" ]]; then
  echo "ConvAgent 3B actor: $PROJECT_ROOT/exports/${CONVAGENT_3B_EXPERIMENT_NAME}_actor_hf"
  echo "ChatR1 3B actor:    $PROJECT_ROOT/exports/${CHATR1_3B_EXPERIMENT_NAME}_actor_hf"
  echo "ConvAgent 7B actor: $PROJECT_ROOT/exports/${CONVAGENT_7B_EXPERIMENT_NAME}_actor_hf"
  echo "ChatR1 7B actor:    $PROJECT_ROOT/exports/${CHATR1_7B_EXPERIMENT_NAME}_actor_hf"
fi
