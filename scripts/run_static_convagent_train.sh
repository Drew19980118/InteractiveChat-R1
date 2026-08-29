#!/usr/bin/env bash
# Train a reproducible, static ConvAgent-style GRPO baseline.
#
# This launcher deliberately uses no user simulator and no dynamic dialogue
# environment. It creates a conversation-disjoint monitor split from the
# static training parquet, evaluates it periodically, and saves exactly one
# selected checkpoint when the monitor F1 reaches a stable plateau.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATASET:?Set DATASET to inscit or qrecc.}"
: "${MODEL_PATH:?Set MODEL_PATH to a Qwen2.5 Instruct model directory.}"
: "${CUDA_VISIBLE_DEVICES:?Reserve the training GPUs through CUDA_VISIBLE_DEVICES.}"

case "$DATASET" in
  inscit) ALLOW_NONANSWER_ACTION=true ;;
  qrecc) ALLOW_NONANSWER_ACTION=false ;;
  *) echo "ERROR: DATASET must be inscit or qrecc." >&2; exit 2 ;;
esac

INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-interactivechat-r1}"
N_GPUS="${N_GPUS:-4}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
GRPO_N="${GRPO_N:-8}"
# A safety ceiling only; it is not a final-training-step choice. The selected
# checkpoint is decided by the holdout plateau criterion below.
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-1000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10000}"
STATIC_HOLDOUT_FRACTION="${STATIC_HOLDOUT_FRACTION:-0.10}"
STATIC_SPLIT_SEED="${STATIC_SPLIT_SEED:-42}"
STATIC_MONITOR_FREQUENCY="${STATIC_MONITOR_FREQUENCY:-5}"
STATIC_MONITOR_PATIENCE="${STATIC_MONITOR_PATIENCE:-3}"
STATIC_MONITOR_MIN_DELTA="${STATIC_MONITOR_MIN_DELTA:-0.002}"
STATIC_MONITOR_STABILITY_WINDOW="${STATIC_MONITOR_STABILITY_WINDOW:-3}"
STATIC_MONITOR_STABILITY_TOLERANCE="${STATIC_MONITOR_STABILITY_TOLERANCE:-0.005}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.08}"
BERT_SCORE_DEVICE="${BERT_SCORE_DEVICE:-cuda}"
BERT_SCORE_BATCH_SIZE="${BERT_SCORE_BATCH_SIZE:-64}"
 # This is the unmodified `ConvAgent/` directory downloaded from
 # DrewZhang/conv.  Do not route static baseline training through the dynamic
 # benchmark converter.
STATIC_SOURCE_ROOT="${STATIC_SOURCE_ROOT:-$PROJECT_ROOT/data/static_convagent_raw/ConvAgent}"
STATIC_TRAIN_SOURCE="${STATIC_TRAIN_SOURCE:-$STATIC_SOURCE_ROOT/$DATASET/${DATASET}_train.parquet}"
STATIC_SPLIT_DIR="${STATIC_SPLIT_DIR:-$PROJECT_ROOT/data/static_convagent_splits/$DATASET}"
STATIC_TRAIN_FILE="$STATIC_SPLIT_DIR/${DATASET}_train.parquet"
STATIC_MONITOR_FILE="$STATIC_SPLIT_DIR/${DATASET}_monitor.parquet"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME (for example static_convagent_inscit_qwen25_3b).}"

if (( TRAIN_BATCH_SIZE != 128 || PPO_MINI_BATCH_SIZE != 64 || GRPO_N != 8 || VAL_BATCH_SIZE != 256 )); then
  echo "ERROR: static baseline requires TRAIN_BATCH_SIZE=128, PPO_MINI_BATCH_SIZE=64, GRPO_N=8, VAL_BATCH_SIZE=256." >&2
  exit 2
fi
if (( N_GPUS < 1 || ULYSSES_SEQUENCE_PARALLEL_SIZE < 1 || N_GPUS % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "ERROR: invalid N_GPUS / ULYSSES_SEQUENCE_PARALLEL_SIZE." >&2
  exit 2
fi
if [[ ! -f "$STATIC_TRAIN_SOURCE" ]]; then
  echo "ERROR: static source parquet is missing: $STATIC_TRAIN_SOURCE" >&2
  echo "Set STATIC_SOURCE_ROOT or STATIC_TRAIN_SOURCE to the downloaded ConvAgent static data." >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$INTERACTIVECHAT_CONDA_ENV"
export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0 VLLM_ATTENTION_BACKEND=XFORMERS
export IGPO_MAX_SEARCH_QUERIES=1 IGPO_SEARCH_TOP_K=3

if [[ ! -f "$STATIC_TRAIN_FILE" || ! -f "$STATIC_MONITOR_FILE" ]]; then
  python -u scripts/prepare_static_convagent_monitor_split.py \
    --input "$STATIC_TRAIN_SOURCE" \
    --output-dir "$STATIC_SPLIT_DIR" \
    --dataset "$DATASET" \
    --holdout-fraction "$STATIC_HOLDOUT_FRACTION" \
    --seed "$STATIC_SPLIT_SEED"
fi

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
ROLLOUT_DIR="$OUTPUT_DIR/rollout"
mkdir -p "$OUTPUT_DIR" "$EVAL_DIR" "$ROLLOUT_DIR" "$PROJECT_ROOT/cache/task_queue"

echo "[StaticConvAgent] dataset=$DATASET experiment=$EXPERIMENT_NAME train_batch=128 ppo_mini_batch=64 n=8 val_batch=256"
echo "[StaticConvAgent] monitor=$STATIC_MONITOR_FREQUENCY steps, patience=$STATIC_MONITOR_PATIENCE, window=$STATIC_MONITOR_STABILITY_WINDOW, ceiling=$MAX_TRAINING_STEPS"
echo "[StaticConvAgent] reward=answer-F1 + 0.5*direct-evidence + 0.5*action; simulator=disabled"

python -u -m verl.trainer.main_ppo \
  "data.train_files=$STATIC_TRAIN_FILE" \
  "data.val_files=$STATIC_MONITOR_FILE" \
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
  "+algorithm.info_gain_norm_mode=separate" \
  "algorithm.query_group_advantage=disabled" \
  "algorithm.max_search_queries=1" \
  "algorithm.allow_nonanswer_action=$ALLOW_NONANSWER_ACTION" \
  "algorithm.use_action_reward=true" \
  "algorithm.action_incorrect_reward=-0.5" \
  "algorithm.static_convagent_mode=true" \
  "algorithm.static_convagent_direct_evidence_reward=true" \
  "algorithm.static_convagent_short_answer_tokens=4" \
  "algorithm.static_convagent_info_gain_weight=0.5" \
  "algorithm.static_convagent_action_weight=0.5" \
  "algorithm.simulated_user_enabled=false" \
  "trainer.logger=['console']" \
  "trainer.project_name=static_convagent" \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  "trainer.default_hdfs_dir=null" \
  "trainer.default_local_dir=$OUTPUT_DIR" \
  "trainer.rollout_data_dir=$ROLLOUT_DIR" \
  "trainer.validation_data_dir=$EVAL_DIR" \
  "trainer.val_before_train=false" \
  "trainer.resume_mode=disable" \
  "trainer.n_gpus_per_node=$N_GPUS" \
  "trainer.nnodes=1" \
  "trainer.total_training_steps=$MAX_TRAINING_STEPS" \
  "trainer.total_epochs=$TOTAL_EPOCHS" \
  "trainer.save_freq=-1" \
  "trainer.test_freq=-1" \
  "trainer.static_convagent_monitor_enabled=true" \
  "trainer.static_convagent_monitor_frequency=$STATIC_MONITOR_FREQUENCY" \
  "trainer.static_convagent_monitor_metric=val/test_score/${DATASET}_f1" \
  "trainer.static_convagent_monitor_patience=$STATIC_MONITOR_PATIENCE" \
  "trainer.static_convagent_monitor_min_delta=$STATIC_MONITOR_MIN_DELTA" \
  "trainer.static_convagent_monitor_stability_window=$STATIC_MONITOR_STABILITY_WINDOW" \
  "trainer.static_convagent_monitor_stability_tolerance=$STATIC_MONITOR_STABILITY_TOLERANCE" \
  "agent_grpo.n=$GRPO_N" \
  "max_turns=4" \
  "search_engine=local_retriever" \
  "codeact_env_disabled=true"

FINAL_STEP="$(< "$OUTPUT_DIR/latest_checkpointed_iteration.txt")"
FINAL_CHECKPOINT="$OUTPUT_DIR/global_step_$FINAL_STEP"
FINAL_JSON="$EVAL_DIR/$FINAL_STEP.jsonl"
if [[ ! -f "$FINAL_JSON" ]]; then
  echo "ERROR: selected monitor validation JSONL is missing: $FINAL_JSON" >&2
  exit 3
fi
printf '%s\n' "$FINAL_CHECKPOINT" > "$OUTPUT_DIR/final_checkpoint.txt"
python -u scripts/compute_convagent_eval_metrics.py \
  --input "$FINAL_JSON" \
  --output-dir "$EVAL_DIR" \
  --bert-score-device "$BERT_SCORE_DEVICE" \
  --bert-score-batch-size "$BERT_SCORE_BATCH_SIZE"

echo "Completed static ConvAgent training: $EXPERIMENT_NAME"
echo "Final checkpoint: $FINAL_CHECKPOINT"
echo "Selection record: $OUTPUT_DIR/static_convagent_selection.json"
echo "Monitor metrics: $EVAL_DIR/metrics_summary.json"
