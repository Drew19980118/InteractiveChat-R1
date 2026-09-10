#!/usr/bin/env bash
# Latest protocol-matched static baseline lifecycle, executed serially:
#   ConvAgent 3B -> source test -> actor-only export -> optional Hub upload
#   ConvAgent 7B -> source test -> actor-only export -> optional Hub upload
#   ChatR1    3B -> source test -> actor-only export -> optional Hub upload
#   ChatR1    7B -> source test -> actor-only export -> optional Hub upload
#
# DATASET is either inscit or qrecc.  The underlying trainer fixes the shared
# budget to n=8, train batch=128, monitor/evaluation batch=256, prompt=4096,
# response=500, model context=8192, PPO mini-batch=64, micro-batch=1, and
# validates every five completed updates.  Checkpoint selection is the exact
# max-reference composite (F1 + BERTScore-F1 + NDCG@3) / 3, with early stop
# after three consecutive non-improvements.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATASET:?Set DATASET to inscit or qrecc.}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to two idle GPUs.}"

case "$DATASET" in
  inscit|qrecc) ;;
  *) echo "ERROR: DATASET must be inscit or qrecc." >&2; exit 2 ;;
esac

INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-${IGPO_CONDA_ENV:-interactivechat-r1}}"
N_GPUS="${N_GPUS:-2}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
MODEL_3B_PATH="${MODEL_3B_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
MODEL_7B_PATH="${MODEL_7B_PATH:-$PROJECT_ROOT/models/Qwen2.5-7B-Instruct}"
HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
HF_UPLOAD_ACTOR_ONLY="${HF_UPLOAD_ACTOR_ONLY:-true}"
HF_UPLOAD_NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-4}"
SOURCE_MIN_F1_FOR_EXPORT="${SOURCE_MIN_F1_FOR_EXPORT:-0.10}"
SOURCE_MIN_TERMINAL_ANSWER_RATE_FOR_EXPORT="${SOURCE_MIN_TERMINAL_ANSWER_RATE_FOR_EXPORT:-0.50}"

HF_CONVAGENT_3B_REPO_ID="${HF_CONVAGENT_3B_REPO_ID:-DrewZhang/interactivechat-r1-static-convagent-${DATASET}-qwen25-3b}"
HF_CONVAGENT_7B_REPO_ID="${HF_CONVAGENT_7B_REPO_ID:-DrewZhang/interactivechat-r1-static-convagent-${DATASET}-qwen25-7b}"
HF_CHATR1_3B_REPO_ID="${HF_CHATR1_3B_REPO_ID:-DrewZhang/interactivechat-r1-static-chatr1-${DATASET}-qwen25-3b}"
HF_CHATR1_7B_REPO_ID="${HF_CHATR1_7B_REPO_ID:-DrewZhang/interactivechat-r1-static-chatr1-${DATASET}-qwen25-7b}"

RUN_CONVAGENT_3B="${RUN_CONVAGENT_3B:-true}"
RUN_CONVAGENT_7B="${RUN_CONVAGENT_7B:-true}"
RUN_CHATR1_3B="${RUN_CHATR1_3B:-true}"
RUN_CHATR1_7B="${RUN_CHATR1_7B:-true}"

if (( N_GPUS != 2 || ULYSSES_SEQUENCE_PARALLEL_SIZE != 2 )); then
  echo "ERROR: this serial FSDP suite requires N_GPUS=2 and ULYSSES_SEQUENCE_PARALLEL_SIZE=2." >&2
  exit 2
fi
if [[ "$HF_UPLOAD_ACTOR_ONLY" != true && "$HF_UPLOAD_ACTOR_ONLY" != false ]]; then
  echo "ERROR: HF_UPLOAD_ACTOR_ONLY must be true or false." >&2
  exit 2
fi
for flag in "$RUN_CONVAGENT_3B" "$RUN_CONVAGENT_7B" "$RUN_CHATR1_3B" "$RUN_CHATR1_7B"; do
  [[ "$flag" == true || "$flag" == false ]] || {
    echo "ERROR: each RUN_* switch must be true or false." >&2; exit 2;
  }
done
for model_path in "$MODEL_3B_PATH" "$MODEL_7B_PATH"; do
  [[ -f "$model_path/config.json" ]] || {
    echo "ERROR: missing local Hugging Face model directory: $model_path" >&2; exit 2;
  }
done

run_one() {
  local baseline="$1" model_path="$2" model_tag="$3" experiment_name="$4" repo_id="$5"
  local project_name="paper_static_${baseline}"
  local checkpoint_path train_file export_dir summary_path

  echo "===== ${baseline} ${model_tag}: ${DATASET} training ====="
  STATIC_BASELINE="$baseline" \
    DATASET="$DATASET" \
    MODEL_PATH="$model_path" \
    CHATR1_CRITIC_MODEL_PATH="$model_path" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    N_GPUS="$N_GPUS" \
    ULYSSES_SEQUENCE_PARALLEL_SIZE="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
    INTERACTIVECHAT_CONDA_ENV="$INTERACTIVECHAT_CONDA_ENV" \
    EXPERIMENT_NAME="$experiment_name" \
    HOLDOUT_FRACTION="$HOLDOUT_FRACTION" \
    SPLIT_SEED="$SPLIT_SEED" \
    bash "$SCRIPT_DIR/run_paper_static_baseline_train.sh"

  checkpoint_path="$(< "$PROJECT_ROOT/outputs/$project_name/$experiment_name/final_checkpoint.txt")"
  [[ -d "$checkpoint_path" ]] || {
    echo "ERROR: selected checkpoint is missing: $checkpoint_path" >&2; exit 3;
  }

  echo "===== ${baseline} ${model_tag}: ${DATASET} static test ====="
  STATIC_BASELINE="$baseline" \
    TRAIN_DATASET="$DATASET" \
    EVAL_DATASET="$DATASET" \
    MODEL_PATH="$model_path" \
    CHATR1_CRITIC_MODEL_PATH="$model_path" \
    CHECKPOINT_PATH="$checkpoint_path" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    N_GPUS="$N_GPUS" \
    ULYSSES_SEQUENCE_PARALLEL_SIZE="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
    INTERACTIVECHAT_CONDA_ENV="$INTERACTIVECHAT_CONDA_ENV" \
    EXPERIMENT_NAME="${experiment_name}_to_${DATASET}" \
    bash "$SCRIPT_DIR/run_paper_static_baseline_eval.sh"

  summary_path="$PROJECT_ROOT/eval_log/paper_static_eval/${experiment_name}_to_${DATASET}/metrics_summary.json"
  SUMMARY_PATH="$summary_path" \
    SOURCE_MIN_F1_FOR_EXPORT="$SOURCE_MIN_F1_FOR_EXPORT" \
    SOURCE_MIN_TERMINAL_ANSWER_RATE_FOR_EXPORT="$SOURCE_MIN_TERMINAL_ANSWER_RATE_FOR_EXPORT" \
    python - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["SUMMARY_PATH"]).read_text(encoding="utf-8"))
sample_count = int(summary.get("sample_count", 0))
terminal_count = int(summary.get("terminal_answer_count", 0))
f1 = float(summary.get("metrics", {}).get("f1", 0.0))
terminal_rate = terminal_count / sample_count if sample_count else 0.0
minimum_f1 = float(os.environ["SOURCE_MIN_F1_FOR_EXPORT"])
minimum_terminal_rate = float(os.environ["SOURCE_MIN_TERMINAL_ANSWER_RATE_FOR_EXPORT"])
print(
    "[Source health gate] "
    f"f1={f1:.6f} (minimum {minimum_f1:.6f}); "
    f"terminal-answer-rate={terminal_rate:.3%} (minimum {minimum_terminal_rate:.3%})"
)
if f1 < minimum_f1 or terminal_rate < minimum_terminal_rate:
    raise SystemExit(
        "ERROR: source-policy health gate failed; refusing actor export/upload. "
        "Inspect the selected checkpoint or intentionally override the threshold."
    )
PY

  if [[ "$baseline" == convagent ]]; then
    train_file="$PROJECT_ROOT/data/paper_static_convagent_splits/$DATASET/${DATASET}_train.parquet"
  else
    train_file="$PROJECT_ROOT/data/paper_static_chatr1_splits_max_reference/$DATASET/${DATASET}_train.parquet"
  fi
  [[ -f "$train_file" ]] || {
    echo "ERROR: prepared training file is missing for export: $train_file" >&2; exit 3;
  }

  export_dir="$PROJECT_ROOT/exports/${experiment_name}_actor_hf"
  if [[ -e "$export_dir" && -n "$(find "$export_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: export directory is non-empty: $export_dir" >&2
    echo "Choose a new experiment name or move the completed export before rerunning." >&2
    exit 3
  fi

  echo "===== ${baseline} ${model_tag}: actor-only export and Hub upload ====="
  CHECKPOINT_PATH="$checkpoint_path" \
    MODEL_PATH="$model_path" \
    TRAIN_FILE="$train_file" \
    ACTOR_EXPORT_DIR="$export_dir" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    N_GPUS="$N_GPUS" \
    ULYSSES_SEQUENCE_PARALLEL_SIZE="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
    INTERACTIVECHAT_CONDA_ENV="$INTERACTIVECHAT_CONDA_ENV" \
    HF_UPLOAD_ACTOR_ONLY="$HF_UPLOAD_ACTOR_ONLY" \
    HF_REPO_ID="$repo_id" \
    HF_UPLOAD_NUM_WORKERS="$HF_UPLOAD_NUM_WORKERS" \
    bash "$SCRIPT_DIR/export_and_upload_actor_policy.sh"

  echo "Completed ${baseline} ${model_tag}."
  echo "  final checkpoint: $checkpoint_path"
  echo "  source-test metrics: $summary_path"
  echo "  actor-only export: $export_dir"
  [[ "$HF_UPLOAD_ACTOR_ONLY" == true ]] && echo "  Hub model: https://huggingface.co/$repo_id"
}

if [[ "$RUN_CONVAGENT_3B" == true ]]; then
  run_one convagent "$MODEL_3B_PATH" qwen25_3b \
    "${CONVAGENT_3B_EXPERIMENT_NAME:-latest_static_convagent_${DATASET}_qwen25_3b}" \
    "$HF_CONVAGENT_3B_REPO_ID"
fi
if [[ "$RUN_CONVAGENT_7B" == true ]]; then
  run_one convagent "$MODEL_7B_PATH" qwen25_7b \
    "${CONVAGENT_7B_EXPERIMENT_NAME:-latest_static_convagent_${DATASET}_qwen25_7b}" \
    "$HF_CONVAGENT_7B_REPO_ID"
fi
if [[ "$RUN_CHATR1_3B" == true ]]; then
  run_one chatr1 "$MODEL_3B_PATH" qwen25_3b \
    "${CHATR1_3B_EXPERIMENT_NAME:-latest_static_chatr1_${DATASET}_qwen25_3b}" \
    "$HF_CHATR1_3B_REPO_ID"
fi
if [[ "$RUN_CHATR1_7B" == true ]]; then
  run_one chatr1 "$MODEL_7B_PATH" qwen25_7b \
    "${CHATR1_7B_EXPERIMENT_NAME:-latest_static_chatr1_${DATASET}_qwen25_7b}" \
    "$HF_CHATR1_7B_REPO_ID"
fi

echo "===== Latest static ${DATASET} baseline suite completed ====="
