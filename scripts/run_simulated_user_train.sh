#!/usr/bin/env bash
# Generic full-dialogue simulated-user Sparse-GRPO trainer.
# Dataset-specific launchers set DATASET, MODEL_PATH, and the experiment name.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATASET:?Set DATASET to inscit or qrecc.}"
: "${MODEL_PATH:?Set MODEL_PATH to a Qwen2.5 Instruct model directory.}"
: "${CUDA_VISIBLE_DEVICES:?Set one or more training GPUs not used by the simulator/retriever.}"

case "$DATASET" in
  inscit)
    DEFAULT_ALLOW_NONANSWER=true
    DEFAULT_ALLOW_CLARIFY=true
    DEFAULT_STEPS=15
    ;;
  qrecc)
    DEFAULT_ALLOW_NONANSWER=false
    DEFAULT_ALLOW_CLARIFY=false
    DEFAULT_STEPS=30
    ;;
  *)
    echo "ERROR: training DATASET must be inscit or qrecc, got: $DATASET" >&2
    exit 2
    ;;
esac

INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-interactivechat-r1}"
N_GPUS="${N_GPUS:-4}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_${DATASET}_train.parquet}"
VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/sim_user_${DATASET}_test.parquet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME.}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
TOTAL_STEPS="${TOTAL_STEPS:-$DEFAULT_STEPS}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-5}"
GRPO_N="${GRPO_N:-8}"
EXACT_CONTEXT_BATCH="${EXACT_CONTEXT_BATCH:-true}"
ALLOW_NONANSWER_ACTION="${ALLOW_NONANSWER_ACTION:-$DEFAULT_ALLOW_NONANSWER}"
SIMULATED_USER_ALLOW_CLARIFY="${SIMULATED_USER_ALLOW_CLARIFY:-$DEFAULT_ALLOW_CLARIFY}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-4}"
MAX_SEARCH_QUERIES="${MAX_SEARCH_QUERIES:-1}"
SEARCH_TOP_K="${SEARCH_TOP_K:-3}"
MAX_ANSWER_DEPTH="${MAX_ANSWER_DEPTH:-3}"
SIMULATED_USER_REWARD_MODE="${SIMULATED_USER_REWARD_MODE:-full}"
SIMULATED_USER_ENABLE_FEEDBACK="${SIMULATED_USER_ENABLE_FEEDBACK:-true}"
SIMULATED_USER_STATIC_GOLD_CONTEXT="${SIMULATED_USER_STATIC_GOLD_CONTEXT:-false}"
SIMULATED_USER_ENABLE_EVIDENCE_UTILITY="${SIMULATED_USER_ENABLE_EVIDENCE_UTILITY:-false}"
SIMULATED_USER_ENABLE_SEARCH_EFFICIENCY="${SIMULATED_USER_ENABLE_SEARCH_EFFICIENCY:-false}"
# 0 is the normal dynamic-evidence mode: retain each fresh top-3 in full and
# compact only old retrievals if the live policy context becomes too long.
TOOL_OBSERVATION_TOKEN_CAP="${TOOL_OBSERVATION_TOKEN_CAP:-0}"
# All sparse reward channels are independently normalized before their weights
# are applied.  Expose them here so dedicated experiment launchers can state
# their reward definition explicitly without editing Hydra YAML.
SIMULATED_USER_ACTION_WEIGHT="${SIMULATED_USER_ACTION_WEIGHT:-1.0}"
SIMULATED_USER_ANSWER_F1_WEIGHT="${SIMULATED_USER_ANSWER_F1_WEIGHT:-1.0}"
SIMULATED_USER_EVIDENCE_UTILITY_WEIGHT="${SIMULATED_USER_EVIDENCE_UTILITY_WEIGHT:-0.0}"
SIMULATED_USER_SEARCH_EFFICIENCY_WEIGHT="${SIMULATED_USER_SEARCH_EFFICIENCY_WEIGHT:-0.0}"
SIMULATED_USER_CLARITY_WEIGHT="${SIMULATED_USER_CLARITY_WEIGHT:-1.0}"
SIMULATED_USER_PATIENCE_WEIGHT="${SIMULATED_USER_PATIENCE_WEIGHT:-1.0}"
SIMULATED_USER_FORMAT_WEIGHT="${SIMULATED_USER_FORMAT_WEIGHT:-1.0}"
SIMULATED_USER_CLARIFY_F1_WEIGHT="${SIMULATED_USER_CLARIFY_F1_WEIGHT:-1.0}"
SIMULATED_USER_VALIDATION_MAX_BATCHES="${SIMULATED_USER_VALIDATION_MAX_BATCHES:-}"
SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ="${SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ:-0}"
SIMULATED_USER_INTERMEDIATE_VALIDATION_MAX_BATCHES="${SIMULATED_USER_INTERMEDIATE_VALIDATION_MAX_BATCHES:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.15}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

if (( TRAIN_BATCH_SIZE != 128 || PPO_MINI_BATCH_SIZE != 64 || GRPO_N != 8 )); then
  echo "ERROR: formal configuration requires TRAIN_BATCH_SIZE=128, PPO_MINI_BATCH_SIZE=64, GRPO_N=8." >&2
  exit 2
fi
if (( VAL_BATCH_SIZE != 256 )); then
  echo "ERROR: formal configuration requires VAL_BATCH_SIZE=256." >&2
  exit 2
fi
if [[ "$EXACT_CONTEXT_BATCH" != "true" ]]; then
  echo "ERROR: EXACT_CONTEXT_BATCH must be true." >&2
  exit 2
fi
if (( N_GPUS < 1 )); then
  echo "ERROR: N_GPUS must be >= 1." >&2
  exit 2
fi
if (( ULYSSES_SEQUENCE_PARALLEL_SIZE < 1 || ULYSSES_SEQUENCE_PARALLEL_SIZE > N_GPUS || N_GPUS % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "ERROR: ULYSSES_SEQUENCE_PARALLEL_SIZE must be a positive divisor of N_GPUS." >&2
  exit 2
fi
if [[ "$RESUME_MODE" != "disable" && "$RESUME_MODE" != "auto" && "$RESUME_MODE" != "resume_path" ]]; then
  echo "ERROR: RESUME_MODE must be disable, auto, or resume_path." >&2
  exit 2
fi
if [[ "$RESUME_MODE" == "resume_path" && -z "$RESUME_FROM_PATH" ]]; then
  echo "ERROR: Set RESUME_FROM_PATH when RESUME_MODE=resume_path." >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "ERROR: missing simulated-user parquet: TRAIN_FILE=$TRAIN_FILE VAL_FILE=$VAL_FILE" >&2
  exit 2
fi
if ! [[ "$MAX_SEARCH_QUERIES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_SEARCH_QUERIES must be a positive integer." >&2
  exit 2
fi
if ! [[ "$SEARCH_TOP_K" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SEARCH_TOP_K must be a positive integer." >&2
  exit 2
fi
if [[ "$SIMULATED_USER_REWARD_MODE" != "full" \
   && "$SIMULATED_USER_REWARD_MODE" != "uci_replaced_by_answer_f1" \
   && "$SIMULATED_USER_REWARD_MODE" != "answer_f1_only" ]]; then
  echo "ERROR: SIMULATED_USER_REWARD_MODE must be full, uci_replaced_by_answer_f1, or answer_f1_only." >&2
  exit 2
fi
for boolean_name in \
  SIMULATED_USER_ENABLE_FEEDBACK \
  SIMULATED_USER_STATIC_GOLD_CONTEXT \
  SIMULATED_USER_ENABLE_EVIDENCE_UTILITY \
  SIMULATED_USER_ENABLE_SEARCH_EFFICIENCY; do
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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$INTERACTIVECHAT_CONDA_ENV"
export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# The handler reads these at process startup.  They make every tool call use
# one canonical query and expose only the top-k passages to later reasoning.
export IGPO_MAX_SEARCH_QUERIES="$MAX_SEARCH_QUERIES" IGPO_SEARCH_TOP_K="$SEARCH_TOP_K"
export RAY_memory_monitor_refresh_ms=0 VLLM_ATTENTION_BACKEND=XFORMERS

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
ROLLOUT_DIR="$OUTPUT_DIR/rollout"
mkdir -p "$OUTPUT_DIR" "$EVAL_DIR" "$ROLLOUT_DIR" "$PROJECT_ROOT/cache/task_queue"

RESUME_ARGS=("trainer.resume_mode=$RESUME_MODE")
if [[ "$RESUME_MODE" == "resume_path" ]]; then
  RESUME_ARGS+=("trainer.resume_from_path=$RESUME_FROM_PATH")
fi

VALIDATION_LIMIT_ARGS=()
if [[ -n "$SIMULATED_USER_VALIDATION_MAX_BATCHES" ]]; then
  if ! [[ "$SIMULATED_USER_VALIDATION_MAX_BATCHES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SIMULATED_USER_VALIDATION_MAX_BATCHES must be a positive integer or empty." >&2
    exit 2
  fi
  VALIDATION_LIMIT_ARGS+=("algorithm.simulated_user_validation_max_batches=$SIMULATED_USER_VALIDATION_MAX_BATCHES")
fi
if ! [[ "$SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "$SIMULATED_USER_INTERMEDIATE_VALIDATION_MAX_BATCHES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SIMULATED_USER_INTERMEDIATE_VALIDATION_MAX_BATCHES must be a positive integer." >&2
  exit 2
fi

echo "[SimUser Train] dataset=$DATASET experiment=$EXPERIMENT_NAME reward_mode=$SIMULATED_USER_REWARD_MODE steps=$TOTAL_STEPS batch=128 n=8 val_batch=256 gpus=$N_GPUS ulysses=$ULYSSES_SEQUENCE_PARALLEL_SIZE"
echo "[SimUser Train] nonanswer=$ALLOW_NONANSWER_ACTION clarify=$SIMULATED_USER_ALLOW_CLARIFY queries/tool=$MAX_SEARCH_QUERIES topk=$SEARCH_TOP_K"
echo "[SimUser Train] feedback=$SIMULATED_USER_ENABLE_FEEDBACK static_gold_context=$SIMULATED_USER_STATIC_GOLD_CONTEXT evidence_utility=$SIMULATED_USER_ENABLE_EVIDENCE_UTILITY search_efficiency=$SIMULATED_USER_ENABLE_SEARCH_EFFICIENCY"
if [[ -n "$SIMULATED_USER_VALIDATION_MAX_BATCHES" ]]; then
  echo "[SimUser Train] validation smoke cap: $SIMULATED_USER_VALIDATION_MAX_BATCHES packed batches"
fi
if (( SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ > 0 )); then
  echo "[SimUser Train] intermediate validation: every $SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ steps, cap=$SIMULATED_USER_INTERMEDIATE_VALIDATION_MAX_BATCHES packed batches"
fi

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
  "actor_rollout_ref.actor.optim.lr=1e-6" \
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
  "algorithm.simulated_user_reward_mode=$SIMULATED_USER_REWARD_MODE" \
  "algorithm.simulated_user_enable_feedback=$SIMULATED_USER_ENABLE_FEEDBACK" \
  "algorithm.simulated_user_static_gold_context=$SIMULATED_USER_STATIC_GOLD_CONTEXT" \
  "algorithm.simulated_user_enable_evidence_utility=$SIMULATED_USER_ENABLE_EVIDENCE_UTILITY" \
  "algorithm.simulated_user_enable_search_efficiency=$SIMULATED_USER_ENABLE_SEARCH_EFFICIENCY" \
  "algorithm.simulated_user_mode=openai" \
  "algorithm.allow_nonanswer_action=$ALLOW_NONANSWER_ACTION" \
  "algorithm.simulated_user_allow_clarify=$SIMULATED_USER_ALLOW_CLARIFY" \
  "algorithm.simulated_user_max_tool_calls=$MAX_TOOL_CALLS" \
  "algorithm.simulated_user_max_search_queries=$MAX_SEARCH_QUERIES" \
  "algorithm.simulated_user_search_top_k=$SEARCH_TOP_K" \
  "algorithm.simulated_user_max_answer_depth=$MAX_ANSWER_DEPTH" \
  "algorithm.simulated_user_tool_observation_token_cap=$TOOL_OBSERVATION_TOKEN_CAP" \
  "algorithm.simulated_user_action_weight=$SIMULATED_USER_ACTION_WEIGHT" \
  "algorithm.simulated_user_answer_f1_weight=$SIMULATED_USER_ANSWER_F1_WEIGHT" \
  "algorithm.simulated_user_evidence_utility_weight=$SIMULATED_USER_EVIDENCE_UTILITY_WEIGHT" \
  "algorithm.simulated_user_search_efficiency_weight=$SIMULATED_USER_SEARCH_EFFICIENCY_WEIGHT" \
  "algorithm.simulated_user_clarity_weight=$SIMULATED_USER_CLARITY_WEIGHT" \
  "algorithm.simulated_user_patience_weight=$SIMULATED_USER_PATIENCE_WEIGHT" \
  "algorithm.simulated_user_format_weight=$SIMULATED_USER_FORMAT_WEIGHT" \
  "algorithm.simulated_user_clarify_f1_weight=$SIMULATED_USER_CLARIFY_F1_WEIGHT" \
  "algorithm.simulated_user_exact_context_batch=$EXACT_CONTEXT_BATCH" \
  "algorithm.simulated_user_validation_context_batching=true" \
  "${VALIDATION_LIMIT_ARGS[@]}" \
  "algorithm.simulated_user_intermediate_validation_freq=$SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ" \
  "algorithm.simulated_user_intermediate_validation_max_batches=$SIMULATED_USER_INTERMEDIATE_VALIDATION_MAX_BATCHES" \
  "trainer.logger=['console']" \
  "trainer.project_name=$DATASET" \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  "trainer.default_hdfs_dir=null" \
  "trainer.default_local_dir=$OUTPUT_DIR" \
  "trainer.rollout_data_dir=$ROLLOUT_DIR" \
  "trainer.validation_data_dir=$EVAL_DIR" \
  "trainer.val_before_train=false" \
  "${RESUME_ARGS[@]}" \
  "trainer.n_gpus_per_node=$N_GPUS" \
  "trainer.nnodes=1" \
  "trainer.total_training_steps=$TOTAL_STEPS" \
  "trainer.total_epochs=$TOTAL_EPOCHS" \
  "trainer.save_freq=$SAVE_FREQ" \
  "trainer.test_freq=-1" \
  "agent_grpo.n=$GRPO_N" \
  "max_turns=4" \
  "search_engine=local_retriever" \
  "codeact_env_disabled=true"

FINAL_STEP=$((TOTAL_STEPS - 1))
FINAL_JSON="$EVAL_DIR/$FINAL_STEP.jsonl"
if [[ ! -f "$FINAL_JSON" ]]; then
  echo "ERROR: final validation JSONL was not created: $FINAL_JSON" >&2
  exit 3
fi
python -u scripts/compute_convagent_eval_metrics.py \
  --input "$FINAL_JSON" \
  --output-dir "$EVAL_DIR" \
  --bert-score-device "${BERT_SCORE_DEVICE:-cuda}" \
  --bert-score-batch-size "${BERT_SCORE_BATCH_SIZE:-16}"

# Intermediate validations already print online metrics when their checkpoints
# are reached.  Once training has completed, additionally compute BERTScore
# and write a self-contained summary for every capped intermediate JSONL.
# These files are intentionally separate from the final full-test summary.
if (( SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ > 0 )); then
  for ((completed_step=SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ; completed_step<TOTAL_STEPS; completed_step+=SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ)); do
    intermediate_step=$((completed_step - 1))
    intermediate_json="$EVAL_DIR/$intermediate_step.jsonl"
    intermediate_metrics_dir="$EVAL_DIR/intermediate_step_$intermediate_step"
    if [[ ! -f "$intermediate_json" ]]; then
      echo "WARNING: expected intermediate validation JSONL is missing: $intermediate_json" >&2
      continue
    fi
    echo "[SimUser Train] computing complete metrics for capped validation at step $intermediate_step"
    python -u scripts/compute_convagent_eval_metrics.py \
      --input "$intermediate_json" \
      --output-dir "$intermediate_metrics_dir" \
      --bert-score-device "${BERT_SCORE_DEVICE:-cuda}" \
      --bert-score-batch-size "${BERT_SCORE_BATCH_SIZE:-16}"
  done
fi

echo "Completed: $EXPERIMENT_NAME"
echo "Final checkpoint: $OUTPUT_DIR/global_step_$FINAL_STEP"
echo "Validation JSONL: $FINAL_JSON"
echo "Metric summary: $EVAL_DIR/metrics_summary.json"
