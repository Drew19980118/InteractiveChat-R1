#!/usr/bin/env bash
# Evaluate a full checkpoint produced by run_paper_static_baseline_train.sh
# using the same protocol-matched static rollout budget as its holdout monitor.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${STATIC_BASELINE:?Set STATIC_BASELINE to convagent or chatr1.}"
: "${TRAIN_DATASET:?Set TRAIN_DATASET to inscit or qrecc.}"
: "${EVAL_DATASET:?Set EVAL_DATASET to inscit, topiocqa, qrecc, or coral.}"
: "${MODEL_PATH:?Set MODEL_PATH to the base Qwen model directory.}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the selected global_step_* checkpoint.}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"

case "$STATIC_BASELINE" in convagent|chatr1) ;; *) echo "ERROR: invalid STATIC_BASELINE." >&2; exit 2;; esac
case "$TRAIN_DATASET" in inscit|qrecc) ;; *) echo "ERROR: invalid TRAIN_DATASET." >&2; exit 2;; esac
case "$EVAL_DATASET" in inscit|topiocqa|qrecc|coral) ;; *) echo "ERROR: invalid EVAL_DATASET." >&2; exit 2;; esac
if [[ ! -d "$CHECKPOINT_PATH" || "$(basename "$CHECKPOINT_PATH")" != global_step_* ]]; then
  echo "ERROR: CHECKPOINT_PATH must be an existing global_step_* directory." >&2
  exit 2
fi

INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-${IGPO_CONDA_ENV:-interactivechat-r1}}"
IGPO_CONDA_ENV="$INTERACTIVECHAT_CONDA_ENV"
N_GPUS="${N_GPUS:-2}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.15}"
PPO_MICRO_BATCH_PER_GPU="${PPO_MICRO_BATCH_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
REF_LOG_PROB_MICRO_BATCH_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_PER_GPU:-1}"
ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-true}"
MAX_TURNS="${MAX_TURNS:-4}"
MAX_SEARCH_QUERIES="${MAX_SEARCH_QUERIES:-1}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
STATIC_SOURCE_ROOT="${STATIC_SOURCE_ROOT:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-paper_${STATIC_BASELINE}_${TRAIN_DATASET}_to_${EVAL_DATASET}}"
EVAL_STEP="${CHECKPOINT_PATH##*global_step_}"

if (( VAL_BATCH_SIZE != 256 )); then
  echo "ERROR: the requested protocol fixes VAL_BATCH_SIZE=256." >&2
  exit 2
fi
if (( MAX_PROMPT_LENGTH != 4096 || MAX_RESPONSE_LENGTH != 500 || MAX_MODEL_LEN != 8192 || PPO_MINI_BATCH_SIZE != 64 || PPO_MICRO_BATCH_PER_GPU != 1 || REF_LOG_PROB_MICRO_BATCH_PER_GPU != 1 || PPO_MAX_TOKEN_LEN_PER_GPU != 8192 || MAX_TURNS != 4 || MAX_SEARCH_QUERIES != 1 )); then
  echo "ERROR: protocol-matched evaluation requires prompt=4096, response=500, model_len=8192, ppo_mini=64, ppo/ref_micro=1, ppo_token_len=8192, max_turns=4, and max_search_queries=1." >&2
  exit 2
fi
if [[ "$ACTOR_USE_DYNAMIC_BSZ" != "true" || "$ROLLOUT_TOP_P" != "1.0" ]]; then
  echo "ERROR: protocol-matched evaluation requires ACTOR_USE_DYNAMIC_BSZ=true and ROLLOUT_TOP_P=1.0." >&2
  exit 2
fi
if (( N_GPUS < 1 || ULYSSES_SEQUENCE_PARALLEL_SIZE < 1 || N_GPUS % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "ERROR: ULYSSES_SEQUENCE_PARALLEL_SIZE must be a positive divisor of N_GPUS." >&2
  exit 2
fi

if [[ "$STATIC_BASELINE" == "convagent" ]]; then
  STATIC_SOURCE_ROOT="${STATIC_SOURCE_ROOT:-$PROJECT_ROOT/data/static_convagent_raw/ConvAgent}"
  DEFAULT_TRAIN_FILE="$STATIC_SOURCE_ROOT/$TRAIN_DATASET/${TRAIN_DATASET}_train.parquet"
  # Shared protocol variables are validated above.
  ROLLOUT_MAX_NUM_SEQS="${CONVAGENT_ROLLOUT_MAX_NUM_SEQS:-256}"
  BASELINE_ARGS=(
    "algorithm.adv_estimator=grpo"
    "algorithm.gamma=1.0"
    "algorithm.query_group_advantage=disabled"
    "algorithm.allow_nonanswer_action=true"
    "algorithm.use_action_reward=true"
    "algorithm.action_incorrect_reward=-0.5"
    "algorithm.static_convagent_mode=true"
    "algorithm.static_convagent_direct_evidence_reward=false"
    "algorithm.static_convagent_paper_reward=false"
    "algorithm.static_chatr1_mode=false"
    "algorithm.static_chatr1_paper_reward=false"
    "critic.optim.lr=0"
  )
  METRIC_ARGS=(--multi-reference)
else
  STATIC_SOURCE_ROOT="${STATIC_SOURCE_ROOT:-$PROJECT_ROOT/data/static_chatr1_raw/ChatR1}"
  # Restore against the actual answer-only max-reference training partition.
  DEFAULT_TRAIN_FILE="$PROJECT_ROOT/data/paper_static_chatr1_splits_max_reference/$TRAIN_DATASET/${TRAIN_DATASET}_train.parquet"
  CHATR1_CRITIC_MODEL_PATH="${CHATR1_CRITIC_MODEL_PATH:-$MODEL_PATH}"
  ROLLOUT_MAX_NUM_SEQS="${CHATR1_ROLLOUT_MAX_NUM_SEQS:-256}"
  BASELINE_ARGS=(
    "algorithm.adv_estimator=gae"
    "algorithm.gamma=1.0"
    "algorithm.lam=1.0"
    "algorithm.query_group_advantage=disabled"
    "algorithm.allow_nonanswer_action=false"
    "algorithm.use_action_reward=false"
    "algorithm.static_convagent_mode=false"
    "algorithm.static_convagent_paper_reward=false"
    "algorithm.static_chatr1_mode=true"
    "algorithm.static_chatr1_intent_reward=false"
    # This is validation-only: the flag permits native multi-turn PPO/GAE
    # configuration and has no reward effect while is_validation=true.
    "algorithm.static_chatr1_paper_reward=true"
    "critic.optim.lr=0"
  )
  # The policy emits one answer; metrics retain its maximum score over every
  # released answer reference, matching ConvAgent and Turn-PPO.
  METRIC_ARGS=(--multi-reference)
fi

TRAIN_FILE="${TRAIN_FILE:-$DEFAULT_TRAIN_FILE}"
VAL_FILE="${VAL_FILE:-$STATIC_SOURCE_ROOT/$EVAL_DATASET/${EVAL_DATASET}_test.parquet}"
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "ERROR: missing static parquet: TRAIN_FILE=$TRAIN_FILE VAL_FILE=$VAL_FILE" >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$IGPO_CONDA_ENV"
export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PET_NODE_RANK="${PET_NODE_RANK:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAY_memory_monitor_refresh_ms=0 VLLM_ATTENTION_BACKEND=XFORMERS
export IGPO_MAX_SEARCH_QUERIES="$MAX_SEARCH_QUERIES" IGPO_SEARCH_TOP_K=3

python - <<'PY'
import requests
response = requests.post(
    "http://127.0.0.1:8002/retrieve",
    json={"queries": ["paper static baseline evaluation readiness probe"], "topk": 3, "return_scores": True},
    timeout=60,
)
response.raise_for_status()
print("Local retriever API: ready")
PY

OUTPUT_DIR="$PROJECT_ROOT/outputs/paper_static_eval/$EXPERIMENT_NAME"
EVAL_DIR="$PROJECT_ROOT/eval_log/paper_static_eval/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR" "$EVAL_DIR" "$PROJECT_ROOT/cache/task_queue"

echo "[Protocol-matched $STATIC_BASELINE Eval] train=$TRAIN_DATASET eval=$EVAL_DATASET checkpoint=$CHECKPOINT_PATH"
echo "[Protocol-matched $STATIC_BASELINE Eval] prompt=4096 response=500 model_len=8192 ppo_mini=64 ppo/ref_micro=1 dynamic_bsz=true turns=4 searches=1 topk=3; no simulator or feedback"

python -u -m verl.trainer.main_ppo \
  "data.train_files=$TRAIN_FILE" \
  "data.val_files=$VAL_FILE" \
  "data.train_batch_size=128" \
  "data.val_batch_size=$VAL_BATCH_SIZE" \
  "data.max_prompt_length=$MAX_PROMPT_LENGTH" \
  "data.truncation=left" \
  "data.max_response_length=$MAX_RESPONSE_LENGTH" \
  "+data.max_model_len=$MAX_MODEL_LEN" \
  "+data.data_writing_path=$PROJECT_ROOT/cache/task_queue/" \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  "actor_rollout_ref.model.use_remove_padding=true" \
  "actor_rollout_ref.actor.optim.lr=0" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU" \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU" \
  "actor_rollout_ref.actor.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ" \
  "actor_rollout_ref.actor.fsdp_config.param_offload=true" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true" \
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_PER_GPU" \
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU" \
  "actor_rollout_ref.ref.fsdp_config.param_offload=true" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
  "actor_rollout_ref.rollout.dtype=bfloat16" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN" \
  "actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS" \
  "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN" \
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU" \
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU" \
  "actor_rollout_ref.rollout.temperature=1.0" \
  "actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P" \
  "critic.model.path=${CHATR1_CRITIC_MODEL_PATH:-$MODEL_PATH}" \
  "critic.model.use_remove_padding=true" \
  "critic.model.fsdp_config.param_offload=true" \
  "critic.model.fsdp_config.optimizer_offload=true" \
  "critic.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
  "critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU" \
  "critic.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU" \
  "critic.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ" \
  "critic.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  "${BASELINE_ARGS[@]}" \
  "algorithm.max_search_queries=$MAX_SEARCH_QUERIES" \
  "algorithm.simulated_user_enabled=false" \
  "trainer.logger=['console']" \
  "trainer.project_name=paper_static_eval" \
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
  "max_turns=$MAX_TURNS" \
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
  --bert-score-device "${BERT_SCORE_DEVICE:-cuda}" \
  --bert-score-batch-size "${BERT_SCORE_BATCH_SIZE:-64}" \
  "${METRIC_ARGS[@]}"

echo "Completed protocol-matched $STATIC_BASELINE evaluation."
echo "Metric summary: $EVAL_DIR/metrics_summary.json"
