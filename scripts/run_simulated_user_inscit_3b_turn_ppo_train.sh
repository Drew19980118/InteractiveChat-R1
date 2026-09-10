#!/usr/bin/env bash
# Train the online InsCIT 3B policy with one-trajectory, action-level Turn-PPO.
# The Qwen32B simulator supplies public feedback only; it is not a reward model.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-3B-Instruct.}"
: "${USER_SIMULATOR_BASE_URL:?Example: http://127.0.0.1:8010}"
: "${USER_SIMULATOR_MODEL:?Set the served Qwen32B user-simulator name.}"
: "${CUDA_VISIBLE_DEVICES:?Set the training GPUs (not those used by the retriever/simulator).}"

INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-${IGPO_CONDA_ENV:-interactivechat-r1}}"
IGPO_CONDA_ENV="$INTERACTIVECHAT_CONDA_ENV"
N_GPUS="${N_GPUS:-2}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-1000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10000}"
HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"
MONITOR_FREQUENCY="${MONITOR_FREQUENCY:-5}"
MONITOR_PATIENCE="${MONITOR_PATIENCE:-3}"
# The exact monitor evaluator runs BERTScore on CPU by default so it cannot
# OOM alongside the resident actor/reference/critic workers.
SELECTION_BERTSCORE_MODEL="${SELECTION_BERTSCORE_MODEL:-roberta-large}"
SELECTION_BERTSCORE_BATCH_SIZE="${SELECTION_BERTSCORE_BATCH_SIZE:-8}"
SELECTION_BERTSCORE_DEVICE="${SELECTION_BERTSCORE_DEVICE:-cpu}"
TURN_PPO_GAMMA="${TURN_PPO_GAMMA:-0.99}"
TURN_PPO_LAMBDA="${TURN_PPO_LAMBDA:-0.95}"
TURN_PPO_ACTION_CORRECT_REWARD="${TURN_PPO_ACTION_CORRECT_REWARD:-0.0}"
TURN_PPO_ACTION_INCORRECT_REWARD="${TURN_PPO_ACTION_INCORRECT_REWARD:--1.0}"
TURN_PPO_NONANSWER_CORRECT_REWARD="${TURN_PPO_NONANSWER_CORRECT_REWARD:-0.2}"
TURN_PPO_FORMAT_VALID_REWARD="${TURN_PPO_FORMAT_VALID_REWARD:-0.0}"
TURN_PPO_FORMAT_INVALID_REWARD="${TURN_PPO_FORMAT_INVALID_REWARD:--1.0}"
TURN_PPO_ANSWER_F1_WEIGHT="${TURN_PPO_ANSWER_F1_WEIGHT:-1.0}"
TURN_PPO_CLARIFY_F1_WEIGHT="${TURN_PPO_CLARIFY_F1_WEIGHT:-1.0}"
CRITIC_LR="${CRITIC_LR:-1e-5}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.15}"
TOOL_OBSERVATION_TOKEN_CAP="${TOOL_OBSERVATION_TOKEN_CAP:-0}"
BERT_SCORE_DEVICE="${BERT_SCORE_DEVICE:-cuda}"
BERT_SCORE_BATCH_SIZE="${BERT_SCORE_BATCH_SIZE:-16}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-turn_ppo_inscit_qwen25_3b_stable_v1}"
SOURCE_TRAIN_FILE="${SOURCE_TRAIN_FILE:-$PROJECT_ROOT/data/sim_user_inscit_train.parquet}"
TEST_FILE="${TEST_FILE:-$PROJECT_ROOT/data/sim_user_inscit_test.parquet}"
# Match the static ConvAgent/ChatR1 selector: first source user question,
# 10% deterministic holdout, seed 42.  A new directory prevents accidental
# reuse of the old dialogue-id-based split.
SPLIT_DIR="${SPLIT_DIR:-$PROJECT_ROOT/data/sim_user_turn_ppo_splits/inscit_static_matched}"
# This is deliberately a separate, static ConvAgent-style protocol.  It does
# not start the user simulator or replay feedback/retries. Override it with
# the downloaded DrewZhang/conv test parquet if that is the chosen static
# source for a particular comparison.
STATIC_VAL_FILE="${STATIC_VAL_FILE:-$PROJECT_ROOT/data/static_convagent_raw/ConvAgent/inscit/inscit_test.parquet}"
STATIC_EXPERIMENT_NAME="${STATIC_EXPERIMENT_NAME:-${EXPERIMENT_NAME}_static_inscit}"
TRAIN_FILE="$SPLIT_DIR/inscit_train.parquet"
MONITOR_FILE="$SPLIT_DIR/inscit_monitor.parquet"

if (( TRAIN_BATCH_SIZE != 128 || PPO_MINI_BATCH_SIZE != 64 || VAL_BATCH_SIZE != 256 )); then
  echo "ERROR: Turn-PPO requires TRAIN_BATCH_SIZE=128, PPO_MINI_BATCH_SIZE=64, VAL_BATCH_SIZE=256." >&2
  exit 2
fi
if (( N_GPUS < 1 || ULYSSES_SEQUENCE_PARALLEL_SIZE < 1 || N_GPUS % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "ERROR: ULYSSES_SEQUENCE_PARALLEL_SIZE must be a positive divisor of N_GPUS." >&2
  exit 2
fi
if (( MAX_TRAINING_STEPS < 1 || MONITOR_FREQUENCY < 1 || MONITOR_PATIENCE < 1 )); then
  echo "ERROR: training/monitor step counts must be positive." >&2
  exit 2
fi
if [[ ! -f "$SOURCE_TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
  echo "ERROR: missing InsCIT simulated-user data: train=$SOURCE_TRAIN_FILE test=$TEST_FILE" >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$IGPO_CONDA_ENV"
export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0 VLLM_ATTENTION_BACKEND=XFORMERS
export IGPO_MAX_SEARCH_QUERIES=1 IGPO_SEARCH_TOP_K=3
source "$PROJECT_ROOT/scripts/configure_wandb.sh"
configure_wandb

if [[ ! -f "$TRAIN_FILE" || ! -f "$MONITOR_FILE" ]]; then
  python -u scripts/prepare_simulated_user_monitor_split.py \
    --input "$SOURCE_TRAIN_FILE" \
    --output-dir "$SPLIT_DIR" \
    --dataset inscit \
    --holdout-fraction "$HOLDOUT_FRACTION" \
    --seed "$SPLIT_SEED" \
    --split-key first_question
fi

# Never silently reuse the old dialogue-id split: the holdout must match the
# static baseline's first-source-question assignment, not merely its seed.
python - "$SPLIT_DIR/inscit_split_manifest.json" "$HOLDOUT_FRACTION" "$SPLIT_SEED" <<'PY'
import json
import math
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.is_file():
    raise SystemExit(f"ERROR: missing Turn-PPO split manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (
    manifest.get("split_key") != "first_question"
    or manifest.get("split_unit") != "first_source_user_question"
    or int(manifest.get("seed", -1)) != int(sys.argv[3])
    or not math.isclose(float(manifest.get("holdout_fraction", -1)), float(sys.argv[2]))
):
    raise SystemExit(
        "ERROR: existing Turn-PPO holdout is not baseline-matched. "
        "Use a fresh SPLIT_DIR (the default is inscit_static_matched) or intentionally rebuild it."
    )
print(
    "[TurnPPO Train] verified baseline-matched holdout: "
    f"key={manifest['split_key']} seed={manifest['seed']} "
    f"fraction={manifest['holdout_fraction']}"
)
PY

python - <<'PY'
import requests
response = requests.post(
    "http://127.0.0.1:8002/retrieve",
    json={"queries": ["Turn-PPO InsCIT retriever readiness probe"], "topk": 3, "return_scores": True},
    timeout=60,
)
response.raise_for_status()
print("Local retriever API: ready")
PY

OUTPUT_DIR="$PROJECT_ROOT/outputs/inscit/$EXPERIMENT_NAME"
MONITOR_EVAL_DIR="$PROJECT_ROOT/eval_log/inscit/$EXPERIMENT_NAME"
ROLLOUT_DIR="$OUTPUT_DIR/rollout"
mkdir -p "$OUTPUT_DIR" "$MONITOR_EVAL_DIR" "$ROLLOUT_DIR" "$PROJECT_ROOT/cache/task_queue"

echo "[TurnPPO Train] dataset=inscit experiment=$EXPERIMENT_NAME train_batch=128 ppo_mini_batch=64 n=1 monitor_batch=256"
echo "[TurnPPO Train] split_seed=$SPLIT_SEED data_seed=$DATA_SEED"
echo "[TurnPPO Train] monitor every $MONITOR_FREQUENCY steps; exact composite=(normalized F1 + BERTScore-F1 + NDCG@3)/3; stop after $MONITOR_PATIENCE consecutive non-improvements; ceiling=$MAX_TRAINING_STEPS"
echo "[TurnPPO Train] selection BERTScore: model=$SELECTION_BERTSCORE_MODEL device=$SELECTION_BERTSCORE_DEVICE batch_size=$SELECTION_BERTSCORE_BATCH_SIZE"
echo "[TurnPPO Train] terminal reward=wrong-action($TURN_PPO_ACTION_INCORRECT_REWARD), malformed-format($TURN_PPO_FORMAT_INVALID_REWARD), answer/clarify benchmark F1, correct-nonanswer($TURN_PPO_NONANSWER_CORRECT_REWARD); no fixed positive answer reward; feedback=context only"

python -u -m verl.trainer.main_ppo \
  "data.train_files=$TRAIN_FILE" \
  "data.val_files=$MONITOR_FILE" \
  "data.train_batch_size=$TRAIN_BATCH_SIZE" \
  "data.val_batch_size=$VAL_BATCH_SIZE" \
  "+data.seed=$DATA_SEED" \
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
  "critic.model.use_remove_padding=true" \
  "critic.model.fsdp_config.param_offload=true" \
  "critic.model.fsdp_config.optimizer_offload=true" \
  "critic.optim.lr=$CRITIC_LR" \
  "critic.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
  "critic.ppo_micro_batch_size_per_gpu=1" \
  "critic.ppo_max_token_len_per_gpu=8192" \
  "critic.use_dynamic_bsz=true" \
  "critic.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  "algorithm.adv_estimator=gae" \
  "algorithm.gamma=$TURN_PPO_GAMMA" \
  "algorithm.lam=$TURN_PPO_LAMBDA" \
  "algorithm.query_group_advantage=disabled" \
  "algorithm.simulated_user_enabled=true" \
  "algorithm.simulated_user_reward_mode=turn_ppo" \
  "algorithm.simulated_user_turn_ppo=true" \
  "algorithm.simulated_user_turn_ppo_action_correct_reward=$TURN_PPO_ACTION_CORRECT_REWARD" \
  "algorithm.simulated_user_turn_ppo_action_incorrect_reward=$TURN_PPO_ACTION_INCORRECT_REWARD" \
  "algorithm.simulated_user_turn_ppo_nonanswer_correct_reward=$TURN_PPO_NONANSWER_CORRECT_REWARD" \
  "algorithm.simulated_user_turn_ppo_format_valid_reward=$TURN_PPO_FORMAT_VALID_REWARD" \
  "algorithm.simulated_user_turn_ppo_format_invalid_reward=$TURN_PPO_FORMAT_INVALID_REWARD" \
  "algorithm.simulated_user_turn_ppo_answer_f1_weight=$TURN_PPO_ANSWER_F1_WEIGHT" \
  "algorithm.simulated_user_turn_ppo_clarify_f1_weight=$TURN_PPO_CLARIFY_F1_WEIGHT" \
  "algorithm.simulated_user_mode=openai" \
  "algorithm.simulated_user_enable_feedback=true" \
  "algorithm.simulated_user_passive_satisfaction_evaluation=false" \
  "algorithm.simulated_user_static_gold_context=false" \
  "algorithm.simulated_user_enable_evidence_utility=false" \
  "algorithm.simulated_user_enable_search_efficiency=false" \
  "algorithm.simulated_user_action_weight=0.0" \
  "algorithm.simulated_user_answer_f1_weight=0.0" \
  "algorithm.simulated_user_evidence_utility_weight=0.0" \
  "algorithm.simulated_user_search_efficiency_weight=0.0" \
  "algorithm.simulated_user_uci_weight=0.0" \
  "algorithm.simulated_user_clarity_weight=0.0" \
  "algorithm.simulated_user_patience_weight=0.0" \
  "algorithm.simulated_user_format_weight=0.0" \
  "algorithm.simulated_user_clarify_f1_weight=0.0" \
  "algorithm.allow_nonanswer_action=true" \
  "algorithm.simulated_user_allow_clarify=true" \
  "algorithm.simulated_user_max_tool_calls=4" \
  "algorithm.simulated_user_max_search_queries=1" \
  "algorithm.simulated_user_search_top_k=3" \
  "algorithm.simulated_user_max_answer_depth=3" \
  "algorithm.simulated_user_tool_observation_token_cap=$TOOL_OBSERVATION_TOKEN_CAP" \
  "algorithm.simulated_user_exact_context_batch=true" \
  "algorithm.simulated_user_validation_context_batching=true" \
  "algorithm.simulated_user_intermediate_validation_freq=0" \
  "trainer.logger=$TRAINER_LOGGERS" \
  "trainer.log_val_generations=$WANDB_LOG_VAL_GENERATIONS" \
  "trainer.project_name=inscit" \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  "trainer.default_hdfs_dir=null" \
  "trainer.default_local_dir=$OUTPUT_DIR" \
  "trainer.rollout_data_dir=$ROLLOUT_DIR" \
  "trainer.validation_data_dir=$MONITOR_EVAL_DIR" \
  "trainer.val_before_train=false" \
  "trainer.resume_mode=disable" \
  "trainer.n_gpus_per_node=$N_GPUS" \
  "trainer.nnodes=1" \
  "trainer.total_training_steps=$MAX_TRAINING_STEPS" \
  "trainer.total_epochs=$TOTAL_EPOCHS" \
  "trainer.save_freq=-1" \
  "trainer.test_freq=-1" \
  "trainer.static_convagent_monitor_enabled=false" \
  "trainer.simulated_user_turn_ppo_monitor_enabled=true" \
  "trainer.static_convagent_monitor_frequency=$MONITOR_FREQUENCY" \
  "trainer.static_convagent_monitor_metric=val/selection/composite_score" \
  "trainer.static_convagent_monitor_patience=$MONITOR_PATIENCE" \
  "trainer.selection_monitor_composite_enabled=true" \
  "trainer.selection_monitor_bertscore_model=$SELECTION_BERTSCORE_MODEL" \
  "trainer.selection_monitor_bertscore_batch_size=$SELECTION_BERTSCORE_BATCH_SIZE" \
  "trainer.selection_monitor_bertscore_device=$SELECTION_BERTSCORE_DEVICE" \
  "agent_grpo.n=1" \
  "max_turns=4" \
  "search_engine=local_retriever" \
  "codeact_env_disabled=true"

FINAL_STEP="$(< "$OUTPUT_DIR/latest_checkpointed_iteration.txt")"
FINAL_CHECKPOINT="$OUTPUT_DIR/global_step_$FINAL_STEP"
MONITOR_JSON="$MONITOR_EVAL_DIR/$FINAL_STEP.jsonl"
if [[ ! -d "$FINAL_CHECKPOINT" || ! -f "$MONITOR_JSON" ]]; then
  echo "ERROR: selected Turn-PPO checkpoint or its monitor JSONL is missing: $FINAL_CHECKPOINT" >&2
  exit 3
fi
printf '%s\n' "$FINAL_CHECKPOINT" > "$OUTPUT_DIR/final_checkpoint.txt"

DATASET=inscit \
MODEL_PATH="$MODEL_PATH" \
TRAIN_FILE="$TRAIN_FILE" \
VAL_FILE="$TEST_FILE" \
CHECKPOINT_PATH="$FINAL_CHECKPOINT" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}_test" \
N_GPUS="$N_GPUS" \
ULYSSES_SEQUENCE_PARALLEL_SIZE="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
SIMULATED_USER_REWARD_MODE=turn_ppo \
SIMULATED_USER_TURN_PPO=true \
TURN_PPO_GAMMA="$TURN_PPO_GAMMA" \
TURN_PPO_LAMBDA="$TURN_PPO_LAMBDA" \
TURN_PPO_ACTION_CORRECT_REWARD="$TURN_PPO_ACTION_CORRECT_REWARD" \
TURN_PPO_ACTION_INCORRECT_REWARD="$TURN_PPO_ACTION_INCORRECT_REWARD" \
TURN_PPO_NONANSWER_CORRECT_REWARD="$TURN_PPO_NONANSWER_CORRECT_REWARD" \
TURN_PPO_FORMAT_VALID_REWARD="$TURN_PPO_FORMAT_VALID_REWARD" \
TURN_PPO_FORMAT_INVALID_REWARD="$TURN_PPO_FORMAT_INVALID_REWARD" \
TURN_PPO_ANSWER_F1_WEIGHT="$TURN_PPO_ANSWER_F1_WEIGHT" \
TURN_PPO_CLARIFY_F1_WEIGHT="$TURN_PPO_CLARIFY_F1_WEIGHT" \
SIMULATED_USER_ENABLE_FEEDBACK=true \
SIMULATED_USER_PASSIVE_SATISFACTION_EVALUATION=false \
SIMULATED_USER_STATIC_GOLD_CONTEXT=false \
TOOL_OBSERVATION_TOKEN_CAP="$TOOL_OBSERVATION_TOKEN_CAP" \
ROLLOUT_GPU_MEMORY_UTILIZATION="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
IGPO_CONDA_ENV="$IGPO_CONDA_ENV" \
BERT_SCORE_DEVICE="$BERT_SCORE_DEVICE" \
BERT_SCORE_BATCH_SIZE="$BERT_SCORE_BATCH_SIZE" \
MULTI_REFERENCE=true \
bash scripts/run_simulated_user_val.sh

# Evaluate the identical selected actor a second time under the prior static
# InsCIT ConvAgent-style protocol. The static runner restores only the actor
# shards; it deliberately disables the simulator, feedback, retry, and online
# fallback. Keep its directory distinct from the dynamic test directory.
if [[ ! -f "$STATIC_VAL_FILE" ]]; then
  echo "ERROR: static InsCIT validation parquet is missing: $STATIC_VAL_FILE" >&2
  exit 3
fi
DATASET=inscit \
MODEL_PATH="$MODEL_PATH" \
TRAIN_STATE_FILE="$TRAIN_FILE" \
STATIC_VAL_FILE="$STATIC_VAL_FILE" \
CHECKPOINT_PATH="$FINAL_CHECKPOINT" \
EXPERIMENT_NAME="$STATIC_EXPERIMENT_NAME" \
STATIC_CONVAGENT_MODE=false \
STATIC_USER_SATISFACTION_EVALUATION=false \
N_GPUS="$N_GPUS" \
ULYSSES_SEQUENCE_PARALLEL_SIZE="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
MAX_SEARCH_QUERIES=1 \
SEARCH_TOP_K=3 \
ROLLOUT_GPU_MEMORY_UTILIZATION="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
IGPO_CONDA_ENV="$IGPO_CONDA_ENV" \
BERT_SCORE_DEVICE="$BERT_SCORE_DEVICE" \
BERT_SCORE_BATCH_SIZE="$BERT_SCORE_BATCH_SIZE" \
MULTI_REFERENCE=true \
bash scripts/run_static_convagent_val.sh

echo "Completed Turn-PPO InsCIT training: $EXPERIMENT_NAME"
echo "Final checkpoint: $FINAL_CHECKPOINT"
echo "Monitor metrics: $MONITOR_EVAL_DIR/metrics_summary.json"
echo "Full test metrics: $PROJECT_ROOT/eval_log/inscit/${EXPERIMENT_NAME}_test/metrics_summary.json"
echo "Static test metrics: $PROJECT_ROOT/eval_log/inscit/$STATIC_EXPERIMENT_NAME/metrics_summary.json"
