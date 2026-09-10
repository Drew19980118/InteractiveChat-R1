#!/usr/bin/env bash
# Evaluate either a simulated-user-trained FSDP checkpoint or an exported
# actor-only Hugging Face policy on a static ConvAgent test parquet. This
# deliberately uses the legacy one-trajectory search loop: no user simulator,
# feedback, retry, or online fallback is enabled.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATASET:?Set DATASET to inscit or topiocqa.}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES, e.g. 2,3.}"

case "$DATASET" in
  inscit|topiocqa) ;;
  *)
    echo "ERROR: static ConvAgent evaluation supports only inscit or topiocqa, got: $DATASET" >&2
    exit 2
    ;;
esac

# The checkpoint restores an FSDP state sharded over two ranks.  The training
# parquet is retained only to restore the trainer's dataloader state; metrics
# are computed exclusively on STATIC_VAL_FILE.
TRAIN_STATE_FILE="${TRAIN_STATE_FILE:-$PROJECT_ROOT/data/sim_user_inscit_train.parquet}"
STATIC_VAL_FILE="${STATIC_VAL_FILE:-$PROJECT_ROOT/data/static_convagent_raw/ConvAgent/$DATASET/${DATASET}_test.parquet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?Set an experiment name.}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
ACTOR_ONLY_MODEL_PATH="${ACTOR_ONLY_MODEL_PATH:-}"
MODEL_PATH="${MODEL_PATH:-}"
N_GPUS="${N_GPUS:-2}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.04}"
MAX_SEARCH_QUERIES="${MAX_SEARCH_QUERIES:-1}"
SEARCH_TOP_K="${SEARCH_TOP_K:-3}"
BERT_SCORE_DEVICE="${BERT_SCORE_DEVICE:-cuda:0}"
BERT_SCORE_BATCH_SIZE="${BERT_SCORE_BATCH_SIZE:-64}"
# A policy still emits one answer; report its maximum match over all released
# answer alternatives. This is harmless for single-reference datasets.
MULTI_REFERENCE="${MULTI_REFERENCE:-true}"
STATIC_CONVAGENT_MODE="${STATIC_CONVAGENT_MODE:-false}"
# When enabled, static generation remains unchanged.  After it finishes, a
# frozen simulator passively judges every generated answer without feedback or
# retries and writes an annotated copy of the validation JSONL.
STATIC_USER_SATISFACTION_EVALUATION="${STATIC_USER_SATISFACTION_EVALUATION:-false}"
USER_SIMULATOR_TIMEOUT_SECONDS="${USER_SIMULATOR_TIMEOUT_SECONDS:-120}"
USER_SIMULATOR_MAX_OUTPUT_TOKENS="${USER_SIMULATOR_MAX_OUTPUT_TOKENS:-128}"

if [[ "$N_GPUS" != "2" || "$ULYSSES_SEQUENCE_PARALLEL_SIZE" != "2" ]]; then
  echo "ERROR: this static ConvAgent evaluation requires N_GPUS=2 and ULYSSES_SEQUENCE_PARALLEL_SIZE=2." >&2
  exit 2
fi
if [[ "$VAL_BATCH_SIZE" != "256" ]]; then
  echo "ERROR: this requested static evaluation uses VAL_BATCH_SIZE=256." >&2
  exit 2
fi
if [[ "$STATIC_USER_SATISFACTION_EVALUATION" != "true" && "$STATIC_USER_SATISFACTION_EVALUATION" != "false" ]]; then
  echo "ERROR: STATIC_USER_SATISFACTION_EVALUATION must be true or false." >&2
  exit 2
fi
if [[ "$STATIC_CONVAGENT_MODE" != "true" && "$STATIC_CONVAGENT_MODE" != "false" ]]; then
  echo "ERROR: STATIC_CONVAGENT_MODE must be true or false." >&2
  exit 2
fi
if [[ "$MULTI_REFERENCE" != "true" && "$MULTI_REFERENCE" != "false" ]]; then
  echo "ERROR: MULTI_REFERENCE must be true or false." >&2
  exit 2
fi
if [[ "$STATIC_USER_SATISFACTION_EVALUATION" == "true" ]]; then
  : "${USER_SIMULATOR_BASE_URL:?Set USER_SIMULATOR_BASE_URL, e.g. http://127.0.0.1:8010}"
  : "${USER_SIMULATOR_MODEL:?Set USER_SIMULATOR_MODEL to the served frozen simulator.}"
fi
if [[ ! -f "$TRAIN_STATE_FILE" || ! -f "$STATIC_VAL_FILE" ]]; then
  echo "ERROR: missing parquet: TRAIN_STATE_FILE=$TRAIN_STATE_FILE STATIC_VAL_FILE=$STATIC_VAL_FILE" >&2
  exit 2
fi

if [[ -n "$ACTOR_ONLY_MODEL_PATH" ]]; then
  if [[ ! -d "$ACTOR_ONLY_MODEL_PATH" || ! -f "$ACTOR_ONLY_MODEL_PATH/config.json" ]]; then
    echo "ERROR: ACTOR_ONLY_MODEL_PATH must be an exported Hugging Face model directory." >&2
    exit 2
  fi
  MODEL_PATH="$ACTOR_ONLY_MODEL_PATH"
  EVAL_STEP="${ACTOR_ONLY_EVAL_STEP:-0}"
  if ! [[ "$EVAL_STEP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ACTOR_ONLY_EVAL_STEP must be a non-negative integer." >&2
    exit 2
  fi
  LOAD_DESCRIPTION="actor-only HF model=$ACTOR_ONLY_MODEL_PATH"
  RESUME_ARGS=("trainer.resume_mode=disable")
else
  if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH must be the base Qwen2.5-3B Instruct directory when using CHECKPOINT_PATH." >&2
    exit 2
  fi
  if [[ ! -d "$CHECKPOINT_PATH" || "$(basename "$CHECKPOINT_PATH")" != global_step_* ]]; then
    echo "ERROR: set CHECKPOINT_PATH to an existing global_step_* directory, or set ACTOR_ONLY_MODEL_PATH." >&2
    exit 2
  fi
  EVAL_STEP="${CHECKPOINT_PATH##*global_step_}"
  if ! [[ "$EVAL_STEP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse checkpoint step from $CHECKPOINT_PATH" >&2
    exit 2
  fi
  LOAD_DESCRIPTION="checkpoint=$CHECKPOINT_PATH"
  RESUME_ARGS=("trainer.resume_mode=resume_path" "trainer.resume_from_path=$CHECKPOINT_PATH")
fi

# The original InteractiveChat-R1 static ConvAgent evaluator uses a different
# label grammar from the generic IGPO static evaluator: it permits native
# <search>/<clarify> turns and a set of valid terminal actions per row.
if [[ "$STATIC_CONVAGENT_MODE" == "true" ]]; then
  STATIC_CONVAGENT_ARGS=(
    "+algorithm.info_gain_norm_mode=separate"
    "algorithm.use_action_reward=true"
    # These fields are now part of the base Hydra schema, so they must be
    # overridden directly rather than added with a `+` prefix.
    "algorithm.action_incorrect_reward=-0.5"
    "algorithm.static_convagent_mode=true"
    "algorithm.static_convagent_direct_evidence_reward=false"
    "algorithm.static_convagent_info_gain_weight=0.5"
    "algorithm.static_convagent_action_weight=0.5"
  )
else
  STATIC_CONVAGENT_ARGS=(
    "algorithm.use_action_reward=false"
  )
fi

INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-${IGPO_CONDA_ENV:-interactivechat-r1}}"
IGPO_CONDA_ENV="$INTERACTIVECHAT_CONDA_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$IGPO_CONDA_ENV"

export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PET_NODE_RANK="${PET_NODE_RANK:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAY_memory_monitor_refresh_ms=0 VLLM_ATTENTION_BACKEND=XFORMERS
# The tool server reads these once at startup. They reproduce the trained
# policy's one-query / top-3 retrieval setting for the static comparison.
export IGPO_MAX_SEARCH_QUERIES="$MAX_SEARCH_QUERIES" IGPO_SEARCH_TOP_K="$SEARCH_TOP_K"

python - <<'PY'
import requests

response = requests.post(
    "http://127.0.0.1:8002/retrieve",
    json={"queries": ["static ConvAgent evaluation readiness probe"], "topk": 1, "return_scores": True},
    timeout=60,
)
response.raise_for_status()
print("Local retriever API: ready")
PY

OUTPUT_DIR="$PROJECT_ROOT/outputs/$DATASET/$EXPERIMENT_NAME"
EVAL_DIR="$PROJECT_ROOT/eval_log/$DATASET/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR" "$EVAL_DIR" "$PROJECT_ROOT/cache/task_queue"

echo "[Static ConvAgent Val] dataset=$DATASET $LOAD_DESCRIPTION"
echo "[Static ConvAgent Val] val_batch=256 gpus=$N_GPUS; queries/tool=$MAX_SEARCH_QUERIES topk=$SEARCH_TOP_K"
echo "[Static ConvAgent Val] static policy rollout: no feedback, no retry, no online fallback"
echo "[Static ConvAgent Val] InteractiveChat-R1 native static mode=$STATIC_CONVAGENT_MODE"
echo "[Static ConvAgent Val] post-hoc passive simulator satisfaction=$STATIC_USER_SATISFACTION_EVALUATION"

python -u -m verl.trainer.main_ppo \
  "data.train_files=$TRAIN_STATE_FILE" \
  "data.val_files=$STATIC_VAL_FILE" \
  "data.train_batch_size=128" \
  "data.val_batch_size=$VAL_BATCH_SIZE" \
  "data.max_prompt_length=4096" \
  "data.truncation=left" \
  "data.max_response_length=500" \
  "+data.max_model_len=8192" \
  "+data.data_writing_path=$PROJECT_ROOT/cache/task_queue/" \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  "actor_rollout_ref.model.use_remove_padding=true" \
  "actor_rollout_ref.actor.optim.lr=0" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=64" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192" \
  "actor_rollout_ref.actor.use_kl_loss=true" \
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
  "algorithm.gamma=1.0" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.query_group_advantage=disabled" \
  "algorithm.simulated_user_enabled=false" \
  "algorithm.max_search_queries=$MAX_SEARCH_QUERIES" \
  "algorithm.allow_nonanswer_action=true" \
  "${STATIC_CONVAGENT_ARGS[@]}" \
  "algorithm.kl_ctrl.kl_coef=0.001" \
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
  echo "ERROR: static validation JSONL was not created: $VALIDATION_JSON" >&2
  exit 3
fi

METRICS_INPUT_JSON="$VALIDATION_JSON"
if [[ "$STATIC_USER_SATISFACTION_EVALUATION" == "true" ]]; then
  SATISFACTION_JSON="$EVAL_DIR/${EVAL_STEP}.with_user_satisfaction.jsonl"
  python -u scripts/annotate_static_user_satisfaction.py \
    --input "$VALIDATION_JSON" \
    --output "$SATISFACTION_JSON" \
    --base-url "$USER_SIMULATOR_BASE_URL" \
    --model "$USER_SIMULATOR_MODEL" \
    --timeout-seconds "$USER_SIMULATOR_TIMEOUT_SECONDS" \
    --max-output-tokens "$USER_SIMULATOR_MAX_OUTPUT_TOKENS"
  METRICS_INPUT_JSON="$SATISFACTION_JSON"
fi

METRIC_ARGS=()
if [[ "$MULTI_REFERENCE" == "true" ]]; then
  METRIC_ARGS=(--multi-reference)
fi
python -u scripts/compute_convagent_eval_metrics.py \
  --input "$METRICS_INPUT_JSON" \
  --output-dir "$EVAL_DIR" \
  --bert-score-device "$BERT_SCORE_DEVICE" \
  --bert-score-batch-size "$BERT_SCORE_BATCH_SIZE" \
  "${METRIC_ARGS[@]}"

echo "Completed static ConvAgent-style evaluation: $EXPERIMENT_NAME"
echo "Validation JSONL: $VALIDATION_JSON"
echo "Metric summary: $EVAL_DIR/metrics_summary.json"
