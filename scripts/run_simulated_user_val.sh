#!/usr/bin/env bash
# Evaluate one simulated-user FSDP checkpoint or HuggingFace model on a target dialogue dataset.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATASET:?Set DATASET to inscit, topiocqa, qrecc, or coral.}"
: "${MODEL_PATH:?Set MODEL_PATH to the matching base Qwen2.5 Instruct directory.}"
: "${CUDA_VISIBLE_DEVICES:?Set validation GPUs not used by the simulator/retriever.}"

case "$DATASET" in
  inscit)
    DEFAULT_ALLOW_NONANSWER=true
    DEFAULT_ALLOW_CLARIFY=true
    ;;
  topiocqa)
    DEFAULT_ALLOW_NONANSWER=true
    DEFAULT_ALLOW_CLARIFY=false
    ;;
  qrecc|coral)
    DEFAULT_ALLOW_NONANSWER=false
    DEFAULT_ALLOW_CLARIFY=false
    ;;
  *)
    echo "ERROR: unsupported validation DATASET=$DATASET" >&2
    exit 2
    ;;
esac

# Keep IGPO_CONDA_ENV as a backwards-compatible alias for earlier runs.
INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-${IGPO_CONDA_ENV:-interactivechat-r1}}"
N_GPUS="${N_GPUS:-4}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
# In FSDP-resume mode this must be the source training parquet because the
# checkpoint restores its train-dataloader state.  In MODEL_ONLY mode the
# trainer still constructs this dataloader, but does not restore its state.
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to the source simulated-user train parquet.}"
VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_${DATASET}_test.parquet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME for this evaluation.}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
ALLOW_NONANSWER_ACTION="${ALLOW_NONANSWER_ACTION:-$DEFAULT_ALLOW_NONANSWER}"
SIMULATED_USER_ALLOW_CLARIFY="${SIMULATED_USER_ALLOW_CLARIFY:-$DEFAULT_ALLOW_CLARIFY}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-4}"
MAX_SEARCH_QUERIES="${MAX_SEARCH_QUERIES:-1}"
SEARCH_TOP_K="${SEARCH_TOP_K:-3}"
MAX_ANSWER_DEPTH="${MAX_ANSWER_DEPTH:-3}"
SIMULATED_USER_ENABLE_FEEDBACK="${SIMULATED_USER_ENABLE_FEEDBACK:-true}"
SIMULATED_USER_STATIC_GOLD_CONTEXT="${SIMULATED_USER_STATIC_GOLD_CONTEXT:-false}"
# Dynamic-evidence default: retain a fresh top-3 in full, compact old
# retrievals only if the live policy context requires it.
TOOL_OBSERVATION_TOKEN_CAP="${TOOL_OBSERVATION_TOKEN_CAP:-0}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.15}"
# MODEL_ONLY=true evaluates a complete HuggingFace model directory directly.
# The default, false, preserves FSDP global_step_* checkpoint validation.
MODEL_ONLY="${MODEL_ONLY:-false}"

if (( TRAIN_BATCH_SIZE != 128 || PPO_MINI_BATCH_SIZE != 64 )); then
  echo "ERROR: formal validation requires TRAIN_BATCH_SIZE=128 and PPO_MINI_BATCH_SIZE=64." >&2
  exit 2
fi
if (( VAL_BATCH_SIZE != 256 )); then
  echo "ERROR: formal validation requires VAL_BATCH_SIZE=256." >&2
  exit 2
fi
if (( N_GPUS < 1 || ULYSSES_SEQUENCE_PARALLEL_SIZE < 1 || N_GPUS % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "ERROR: N_GPUS must be positive and ULYSSES_SEQUENCE_PARALLEL_SIZE must be its positive divisor." >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "ERROR: missing parquet: TRAIN_FILE=$TRAIN_FILE VAL_FILE=$VAL_FILE" >&2
  exit 2
fi
for boolean_name in SIMULATED_USER_ENABLE_FEEDBACK SIMULATED_USER_STATIC_GOLD_CONTEXT; do
  boolean_value="${!boolean_name}"
  if [[ "$boolean_value" != "true" && "$boolean_value" != "false" ]]; then
    echo "ERROR: $boolean_name must be true or false." >&2
    exit 2
  fi
done
if [[ "$SIMULATED_USER_ENABLE_FEEDBACK" == "true" ]]; then
  : "${USER_SIMULATOR_BASE_URL:?Example: http://127.0.0.1:8010}"
  : "${USER_SIMULATOR_MODEL:?Set the served Qwen32B user-simulator name.}"
fi
if [[ "$SIMULATED_USER_STATIC_GOLD_CONTEXT" == "true" && "$SIMULATED_USER_ENABLE_FEEDBACK" != "false" ]]; then
  echo "ERROR: static gold context requires SIMULATED_USER_ENABLE_FEEDBACK=false." >&2
  exit 2
fi

if [[ "$MODEL_ONLY" == "true" ]]; then
  # ``val_only`` still builds a train dataloader, but it must not attempt to
  # restore FSDP shards from an HF model directory.
  EVAL_STEP=0
  LOAD_DESCRIPTION="HF model directory: $MODEL_PATH"
  RESUME_ARGS=("trainer.resume_mode=disable")
elif [[ "$MODEL_ONLY" == "false" ]]; then
  : "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to an actor checkpoint global_step_* directory, or set MODEL_ONLY=true.}"
  if [[ ! -d "$CHECKPOINT_PATH" || "$(basename "$CHECKPOINT_PATH")" != global_step_* ]]; then
    echo "ERROR: CHECKPOINT_PATH must be an existing global_step_* directory: $CHECKPOINT_PATH" >&2
    exit 2
  fi
  EVAL_STEP="${CHECKPOINT_PATH##*global_step_}"
  if ! [[ "$EVAL_STEP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse checkpoint step from $CHECKPOINT_PATH" >&2
    exit 2
  fi
  LOAD_DESCRIPTION="FSDP checkpoint: $CHECKPOINT_PATH"
  RESUME_ARGS=("trainer.resume_mode=resume_path" "trainer.resume_from_path=$CHECKPOINT_PATH")
else
  echo "ERROR: MODEL_ONLY must be true or false, got: $MODEL_ONLY" >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$INTERACTIVECHAT_CONDA_ENV"
export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0 VLLM_ATTENTION_BACKEND=XFORMERS
export IGPO_MAX_SEARCH_QUERIES="$MAX_SEARCH_QUERIES" IGPO_SEARCH_TOP_K="$SEARCH_TOP_K"

python - <<'PY'
import requests
response = requests.post(
    "http://127.0.0.1:8002/retrieve",
    json={"queries": ["simulated-user retriever readiness probe"], "topk": 1, "return_scores": True},
    timeout=60,
)
response.raise_for_status()
print("Local retriever API: ready")
PY

OUTPUT_DIR="$PROJECT_ROOT/outputs/$DATASET/$EXPERIMENT_NAME"
EVAL_DIR="$PROJECT_ROOT/eval_log/$DATASET/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR" "$EVAL_DIR" "$PROJECT_ROOT/cache/task_queue"

echo "[SimUser Val] dataset=$DATASET $LOAD_DESCRIPTION val_batch=256 gpus=$N_GPUS ulysses=$ULYSSES_SEQUENCE_PARALLEL_SIZE"
echo "[SimUser Val] nonanswer=$ALLOW_NONANSWER_ACTION clarify=$SIMULATED_USER_ALLOW_CLARIFY queries/tool=$MAX_SEARCH_QUERIES topk=$SEARCH_TOP_K"
echo "[SimUser Val] feedback=$SIMULATED_USER_ENABLE_FEEDBACK static_gold_context=$SIMULATED_USER_STATIC_GOLD_CONTEXT"

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
  "algorithm.query_group_advantage=disabled" \
  "algorithm.simulated_user_enabled=true" \
  "algorithm.simulated_user_enable_feedback=$SIMULATED_USER_ENABLE_FEEDBACK" \
  "algorithm.simulated_user_static_gold_context=$SIMULATED_USER_STATIC_GOLD_CONTEXT" \
  "algorithm.simulated_user_mode=openai" \
  "algorithm.allow_nonanswer_action=$ALLOW_NONANSWER_ACTION" \
  "algorithm.simulated_user_allow_clarify=$SIMULATED_USER_ALLOW_CLARIFY" \
  "algorithm.simulated_user_max_tool_calls=$MAX_TOOL_CALLS" \
  "algorithm.simulated_user_max_search_queries=$MAX_SEARCH_QUERIES" \
  "algorithm.simulated_user_search_top_k=$SEARCH_TOP_K" \
  "algorithm.simulated_user_max_answer_depth=$MAX_ANSWER_DEPTH" \
  "algorithm.simulated_user_tool_observation_token_cap=$TOOL_OBSERVATION_TOKEN_CAP" \
  "algorithm.simulated_user_exact_context_batch=true" \
  "algorithm.simulated_user_validation_context_batching=true" \
  "trainer.logger=['console']" \
  "trainer.project_name=$DATASET" \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  "trainer.default_hdfs_dir=null" \
  "trainer.default_local_dir=$OUTPUT_DIR" \
  "trainer.validation_data_dir=$EVAL_DIR" \
  "trainer.val_before_train=true" \
  "+trainer.val_only=true" \
  "${RESUME_ARGS[@]}" \
  "trainer.n_gpus_per_node=$N_GPUS" \
  "trainer.nnodes=1" \
  "trainer.total_training_steps=$((EVAL_STEP + 1))" \
  "trainer.total_epochs=3" \
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
  --bert-score-device "${BERT_SCORE_DEVICE:-cuda}" \
  --bert-score-batch-size "${BERT_SCORE_BATCH_SIZE:-16}"

echo "Completed evaluation: $EXPERIMENT_NAME"
echo "Validation JSONL: $VALIDATION_JSON"
echo "Metric summary: $EVAL_DIR/metrics_summary.json"
