#!/usr/bin/env bash
# Evaluate a selected static ConvAgent FSDP checkpoint on one static benchmark.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${TRAIN_DATASET:?Set TRAIN_DATASET to inscit or qrecc.}"
: "${EVAL_DATASET:?Set EVAL_DATASET to inscit, topiocqa, qrecc, or coral.}"
: "${MODEL_PATH:?Set MODEL_PATH to the same base Qwen2.5 Instruct model directory.}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the selected global_step_* directory.}"
: "${CUDA_VISIBLE_DEVICES:?Reserve validation GPUs through CUDA_VISIBLE_DEVICES.}"

case "$TRAIN_DATASET" in inscit|qrecc) ;; *) echo "ERROR: TRAIN_DATASET must be inscit or qrecc." >&2; exit 2;; esac
case "$EVAL_DATASET" in inscit|topiocqa|qrecc|coral) ;; *) echo "ERROR: unsupported EVAL_DATASET." >&2; exit 2;; esac

INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-interactivechat-r1}"
N_GPUS="${N_GPUS:-4}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.08}"
BERT_SCORE_DEVICE="${BERT_SCORE_DEVICE:-cuda}"
BERT_SCORE_BATCH_SIZE="${BERT_SCORE_BATCH_SIZE:-64}"
# Directly consume the unmodified `ConvAgent/` static Parquets.
STATIC_SOURCE_ROOT="${STATIC_SOURCE_ROOT:-$PROJECT_ROOT/data/static_convagent_raw/ConvAgent}"
STATIC_SPLIT_DIR="${STATIC_SPLIT_DIR:-$PROJECT_ROOT/data/static_convagent_splits/$TRAIN_DATASET}"
TRAIN_FILE="${TRAIN_FILE:-$STATIC_SPLIT_DIR/${TRAIN_DATASET}_train.parquet}"
VAL_FILE="${VAL_FILE:-$STATIC_SOURCE_ROOT/$EVAL_DATASET/${EVAL_DATASET}_test.parquet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME for this evaluation.}"

if (( TRAIN_BATCH_SIZE != 128 || PPO_MINI_BATCH_SIZE != 64 || VAL_BATCH_SIZE != 256 )); then
  echo "ERROR: static baseline evaluation requires train_batch=128, ppo_mini_batch=64, val_batch=256." >&2
  exit 2
fi
if (( N_GPUS < 1 || ULYSSES_SEQUENCE_PARALLEL_SIZE < 1 || N_GPUS % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "ERROR: invalid N_GPUS / ULYSSES_SEQUENCE_PARALLEL_SIZE." >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT_PATH" || "$(basename "$CHECKPOINT_PATH")" != global_step_* ]]; then
  echo "ERROR: CHECKPOINT_PATH must be an existing global_step_* directory." >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "ERROR: missing static parquet: TRAIN_FILE=$TRAIN_FILE VAL_FILE=$VAL_FILE" >&2
  exit 2
fi
EVAL_STEP="${CHECKPOINT_PATH##*global_step_}"
if ! [[ "$EVAL_STEP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not parse checkpoint step from $CHECKPOINT_PATH" >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$INTERACTIVECHAT_CONDA_ENV"
export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0 VLLM_ATTENTION_BACKEND=XFORMERS
export IGPO_MAX_SEARCH_QUERIES=1 IGPO_SEARCH_TOP_K=3

python - <<'PY'
import requests
response = requests.post(
    "http://127.0.0.1:8002/retrieve",
    json={"queries": ["static ConvAgent retriever readiness probe"], "topk": 3, "return_scores": True},
    timeout=60,
)
response.raise_for_status()
print("Local retriever API: ready")
PY

OUTPUT_DIR="$PROJECT_ROOT/outputs/static_convagent/$EXPERIMENT_NAME"
EVAL_DIR="$PROJECT_ROOT/eval_log/static_convagent/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR" "$EVAL_DIR" "$PROJECT_ROOT/cache/task_queue"

echo "[StaticConvAgent Eval] train=$TRAIN_DATASET eval=$EVAL_DATASET checkpoint=$CHECKPOINT_PATH val_batch=256"

python -u -m verl.trainer.main_ppo \
  "data.train_files=$TRAIN_FILE" \
  "data.val_files=$VAL_FILE" \
  "data.train_batch_size=$TRAIN_BATCH_SIZE" \
  "data.val_batch_size=$VAL_BATCH_SIZE" \
  "data.max_prompt_length=4096" \
  "data.truncation=left" \
  "data.max_response_length=500" \
  "+data.max_model_len=8192" \
  "+data.data_writing_path=$PROJECT_ROOT/cache/task_queue/" \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  "actor_rollout_ref.model.use_remove_padding=true" \
  "actor_rollout_ref.actor.optim.lr=0" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192" \
  "actor_rollout_ref.actor.use_dynamic_bsz=true" \
  "actor_rollout_ref.actor.fsdp_config.param_offload=true" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true" \
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192" \
  "actor_rollout_ref.ref.fsdp_config.param_offload=true" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
  "actor_rollout_ref.rollout.dtype=bfloat16" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=8192" \
  "actor_rollout_ref.rollout.max_model_len=8192" \
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192" \
  "actor_rollout_ref.rollout.temperature=1.0" \
  "critic.model.path=$MODEL_PATH" \
  "critic.optim.lr=0" \
  "critic.ppo_micro_batch_size_per_gpu=1" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.gamma=1.0" \
  "+algorithm.info_gain_norm_mode=separate" \
  "algorithm.query_group_advantage=disabled" \
  "algorithm.max_search_queries=1" \
  "algorithm.allow_nonanswer_action=true" \
  "algorithm.use_action_reward=true" \
  "algorithm.action_incorrect_reward=-0.5" \
  "algorithm.static_convagent_mode=true" \
  "algorithm.static_convagent_direct_evidence_reward=false" \
  "algorithm.static_convagent_info_gain_weight=0.5" \
  "algorithm.static_convagent_action_weight=0.5" \
  "algorithm.simulated_user_enabled=false" \
  "trainer.logger=['console']" \
  "trainer.project_name=static_convagent" \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  "trainer.default_hdfs_dir=null" \
  "trainer.default_local_dir=$OUTPUT_DIR" \
  "trainer.validation_data_dir=$EVAL_DIR" \
  "trainer.val_before_train=true" \
  "+trainer.val_only=true" \
  "trainer.resume_mode=resume_path" \
  "trainer.resume_from_path=$CHECKPOINT_PATH" \
  "trainer.n_gpus_per_node=$N_GPUS" \
  "trainer.nnodes=1" \
  "trainer.total_training_steps=$((EVAL_STEP + 1))" \
  "trainer.total_epochs=1" \
  "trainer.save_freq=-1" \
  "trainer.test_freq=-1" \
  "agent_grpo.n=1" \
  "max_turns=4" \
  "search_engine=local_retriever" \
  "codeact_env_disabled=true"

VALIDATION_JSON="$EVAL_DIR/$EVAL_STEP.jsonl"
if [[ ! -f "$VALIDATION_JSON" ]]; then
  echo "ERROR: validation JSONL was not created: $VALIDATION_JSON" >&2
  exit 3
fi
python -u scripts/compute_convagent_eval_metrics.py \
  --input "$VALIDATION_JSON" \
  --output-dir "$EVAL_DIR" \
  --bert-score-device "$BERT_SCORE_DEVICE" \
  --bert-score-batch-size "$BERT_SCORE_BATCH_SIZE"

echo "Completed static ConvAgent evaluation: $EXPERIMENT_NAME"
echo "Metric summary: $EVAL_DIR/metrics_summary.json"
