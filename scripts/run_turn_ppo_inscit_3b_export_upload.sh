#!/usr/bin/env bash
# Train stable Turn-PPO on InsCiT, run its dynamic and static final tests,
# export the selected actor, and upload that actor-only HF model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen2.5-3B-Instruct directory.}"
: "${USER_SIMULATOR_BASE_URL:?Example: http://127.0.0.1:8010}"
: "${USER_SIMULATOR_MODEL:?Set the served Qwen32B user-simulator name.}"
: "${CUDA_VISIBLE_DEVICES:?Set the two training GPUs.}"

INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-${IGPO_CONDA_ENV:-interactivechat-r1}}"
IGPO_CONDA_ENV="$INTERACTIVECHAT_CONDA_ENV"
N_GPUS="${N_GPUS:-2}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-turn_ppo_inscit_qwen25_3b_stable_v1}"
STATIC_EXPERIMENT_NAME="${STATIC_EXPERIMENT_NAME:-${EXPERIMENT_NAME}_static_inscit}"
SPLIT_DIR="${SPLIT_DIR:-$PROJECT_ROOT/data/sim_user_turn_ppo_splits/inscit_static_matched}"
TRAIN_FILE="$SPLIT_DIR/inscit_train.parquet"
ACTOR_EXPORT_DIR="${ACTOR_EXPORT_DIR:-$PROJECT_ROOT/exports/${EXPERIMENT_NAME}_actor_hf}"
HF_UPLOAD_ACTOR_ONLY="${HF_UPLOAD_ACTOR_ONLY:-true}"
HF_TURN_PPO_REPO_ID="${HF_TURN_PPO_REPO_ID:-DrewZhang/interactivechat-r1-turn-ppo-inscit-qwen25-3b}"
HF_UPLOAD_NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-4}"
# Must match the static-baseline suite's selection split parameters.
HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
# Safety gate only: override a threshold explicitly for a deliberate ablation.
DYNAMIC_MIN_F1_FOR_UPLOAD="${DYNAMIC_MIN_F1_FOR_UPLOAD:-0.10}"
DYNAMIC_MIN_ACTION_ACCURACY_FOR_UPLOAD="${DYNAMIC_MIN_ACTION_ACCURACY_FOR_UPLOAD:-0.50}"

if (( N_GPUS != 2 || ULYSSES_SEQUENCE_PARALLEL_SIZE != 2 )); then
  echo "ERROR: this FSDP export/upload suite requires N_GPUS=2 and ULYSSES_SEQUENCE_PARALLEL_SIZE=2." >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: MODEL_PATH is not a local Hugging Face model directory: $MODEL_PATH" >&2
  exit 2
fi
if [[ "$HF_UPLOAD_ACTOR_ONLY" != "true" && "$HF_UPLOAD_ACTOR_ONLY" != "false" ]]; then
  echo "ERROR: HF_UPLOAD_ACTOR_ONLY must be true or false." >&2
  exit 2
fi

echo "===== Turn-PPO InsCiT 3B: train and final validation ====="
MODEL_PATH="$MODEL_PATH" \
  USER_SIMULATOR_BASE_URL="$USER_SIMULATOR_BASE_URL" \
  USER_SIMULATOR_MODEL="$USER_SIMULATOR_MODEL" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  N_GPUS="$N_GPUS" \
  ULYSSES_SEQUENCE_PARALLEL_SIZE="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  IGPO_CONDA_ENV="$IGPO_CONDA_ENV" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  STATIC_EXPERIMENT_NAME="$STATIC_EXPERIMENT_NAME" \
  SPLIT_DIR="$SPLIT_DIR" \
  HOLDOUT_FRACTION="$HOLDOUT_FRACTION" \
  SPLIT_SEED="$SPLIT_SEED" \
  bash "$SCRIPT_DIR/run_simulated_user_inscit_3b_turn_ppo_train.sh"

CHECKPOINT_PATH="$(< "$PROJECT_ROOT/outputs/inscit/$EXPERIMENT_NAME/final_checkpoint.txt")"
DYNAMIC_SUMMARY="$PROJECT_ROOT/eval_log/inscit/${EXPERIMENT_NAME}_test/metrics_summary.json"
STATIC_SUMMARY="$PROJECT_ROOT/eval_log/inscit/$STATIC_EXPERIMENT_NAME/metrics_summary.json"
if [[ ! -d "$CHECKPOINT_PATH" || ! -f "$DYNAMIC_SUMMARY" || ! -f "$STATIC_SUMMARY" ]]; then
  echo "ERROR: Turn-PPO train/final validation did not produce all required artifacts." >&2
  echo "checkpoint=$CHECKPOINT_PATH dynamic_summary=$DYNAMIC_SUMMARY static_summary=$STATIC_SUMMARY" >&2
  exit 3
fi

DYNAMIC_SUMMARY="$DYNAMIC_SUMMARY" \
  DYNAMIC_MIN_F1_FOR_UPLOAD="$DYNAMIC_MIN_F1_FOR_UPLOAD" \
  DYNAMIC_MIN_ACTION_ACCURACY_FOR_UPLOAD="$DYNAMIC_MIN_ACTION_ACCURACY_FOR_UPLOAD" \
  python - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ["DYNAMIC_SUMMARY"])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
metrics = summary.get("metrics", {})
f1 = float(metrics.get("f1", 0.0))
action_accuracy = float(metrics.get("action_accuracy", 0.0))
minimum_f1 = float(os.environ["DYNAMIC_MIN_F1_FOR_UPLOAD"])
minimum_action_accuracy = float(os.environ["DYNAMIC_MIN_ACTION_ACCURACY_FOR_UPLOAD"])
print(
    "[TurnPPO source health gate] "
    f"dynamic_f1={f1:.6f} (minimum {minimum_f1:.6f}); "
    f"action_accuracy={action_accuracy:.6f} "
    f"(minimum {minimum_action_accuracy:.6f})"
)
if f1 < minimum_f1 or action_accuracy < minimum_action_accuracy:
    raise SystemExit(
        "ERROR: dynamic source-policy health gate failed. Refusing Hub upload; "
        "inspect the run or explicitly lower a threshold for an ablation."
    )
PY

if [[ -e "$ACTOR_EXPORT_DIR" && -n "$(find "$ACTOR_EXPORT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: actor export directory already has files: $ACTOR_EXPORT_DIR" >&2
  echo "Choose a new EXPERIMENT_NAME or move the completed export before restarting." >&2
  exit 3
fi

echo "===== Turn-PPO InsCiT 3B: actor-only export and Hub upload ====="
CHECKPOINT_PATH="$CHECKPOINT_PATH" \
  MODEL_PATH="$MODEL_PATH" \
  TRAIN_FILE="$TRAIN_FILE" \
  ACTOR_EXPORT_DIR="$ACTOR_EXPORT_DIR" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  N_GPUS="$N_GPUS" \
  ULYSSES_SEQUENCE_PARALLEL_SIZE="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  IGPO_CONDA_ENV="$IGPO_CONDA_ENV" \
  HF_UPLOAD_ACTOR_ONLY="$HF_UPLOAD_ACTOR_ONLY" \
  HF_REPO_ID="$HF_TURN_PPO_REPO_ID" \
  HF_UPLOAD_NUM_WORKERS="$HF_UPLOAD_NUM_WORKERS" \
  bash "$SCRIPT_DIR/export_and_upload_actor_policy.sh"

echo "===== Turn-PPO InsCiT 3B completed ====="
echo "Final checkpoint: $CHECKPOINT_PATH"
echo "Dynamic test metrics: $DYNAMIC_SUMMARY"
echo "Static test metrics: $STATIC_SUMMARY"
echo "Actor-only export: $ACTOR_EXPORT_DIR"
if [[ "$HF_UPLOAD_ACTOR_ONLY" == "true" ]]; then
  echo "Hub model: https://huggingface.co/$HF_TURN_PPO_REPO_ID"
fi
