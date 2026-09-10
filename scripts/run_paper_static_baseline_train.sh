#!/usr/bin/env bash
# Train one static ConvAgent (GRPO) or ChatR1 (PPO/GAE) baseline under the
# protocol-matched budget used by Turn-PPO. Algorithm-specific choices remain
# intact, while shared length, batching, sampling, and interaction limits are
# fixed for a controlled comparison.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${STATIC_BASELINE:?Set STATIC_BASELINE to convagent or chatr1.}"
: "${DATASET:?Set DATASET to inscit or qrecc.}"
: "${MODEL_PATH:?Set MODEL_PATH to a Qwen2.5 3B/7B Instruct directory.}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"

case "$DATASET" in
  inscit|qrecc) ;;
  *) echo "ERROR: DATASET must be inscit or qrecc." >&2; exit 2 ;;
esac
case "$STATIC_BASELINE" in
  convagent|chatr1) ;;
  *) echo "ERROR: STATIC_BASELINE must be convagent or chatr1." >&2; exit 2 ;;
esac

# Keep the InteractiveChat-R1 launch surface while accepting IGPO_CONDA_ENV as
# a backwards-compatible alias for shared scripts.
INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-${IGPO_CONDA_ENV:-interactivechat-r1}}"
IGPO_CONDA_ENV="$INTERACTIVECHAT_CONDA_ENV"
N_GPUS="${N_GPUS:-2}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-1000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10000}"
HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
MONITOR_FREQUENCY="${MONITOR_FREQUENCY:-5}"
MONITOR_PATIENCE="${MONITOR_PATIENCE:-3}"
# The selection evaluator reuses the final-report metric definitions. Keep it
# off the training GPUs by default: actor/reference/(ChatR1) critic are live.
SELECTION_BERTSCORE_MODEL="${SELECTION_BERTSCORE_MODEL:-roberta-large}"
SELECTION_BERTSCORE_BATCH_SIZE="${SELECTION_BERTSCORE_BATCH_SIZE:-8}"
SELECTION_BERTSCORE_DEVICE="${SELECTION_BERTSCORE_DEVICE:-cpu}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.15}"
PPO_MICRO_BATCH_PER_GPU="${PPO_MICRO_BATCH_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
REF_LOG_PROB_MICRO_BATCH_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_PER_GPU:-1}"
ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-true}"
MAX_TURNS="${MAX_TURNS:-4}"
MAX_SEARCH_QUERIES="${MAX_SEARCH_QUERIES:-1}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
STATIC_SOURCE_ROOT="${STATIC_SOURCE_ROOT:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"

if (( TRAIN_BATCH_SIZE != 128 || VAL_BATCH_SIZE != 256 || ROLLOUT_N != 8 )); then
  echo "ERROR: requested protocol fixes TRAIN_BATCH_SIZE=128, VAL_BATCH_SIZE=256, and ROLLOUT_N=8." >&2
  exit 2
fi
if (( MAX_PROMPT_LENGTH != 4096 || MAX_RESPONSE_LENGTH != 500 || MAX_MODEL_LEN != 8192 || PPO_MINI_BATCH_SIZE != 64 || PPO_MICRO_BATCH_PER_GPU != 1 || REF_LOG_PROB_MICRO_BATCH_PER_GPU != 1 || PPO_MAX_TOKEN_LEN_PER_GPU != 8192 || MAX_TURNS != 4 || MAX_SEARCH_QUERIES != 1 )); then
  echo "ERROR: protocol-matched baselines require prompt=4096, response=500, model_len=8192, ppo_mini=64, ppo/ref_micro=1, ppo_token_len=8192, max_turns=4, and max_search_queries=1." >&2
  exit 2
fi
if [[ "$ACTOR_USE_DYNAMIC_BSZ" != "true" || "$ROLLOUT_TOP_P" != "1.0" ]]; then
  echo "ERROR: protocol-matched baselines require ACTOR_USE_DYNAMIC_BSZ=true and ROLLOUT_TOP_P=1.0." >&2
  exit 2
fi
if (( N_GPUS < 1 || ULYSSES_SEQUENCE_PARALLEL_SIZE < 1 || N_GPUS % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "ERROR: ULYSSES_SEQUENCE_PARALLEL_SIZE must be a positive divisor of N_GPUS." >&2
  exit 2
fi

if [[ "$STATIC_BASELINE" == "convagent" ]]; then
  PROJECT_NAME="paper_static_convagent"
  STATIC_SOURCE_ROOT="${STATIC_SOURCE_ROOT:-$PROJECT_ROOT/data/static_convagent_raw/ConvAgent}"
  SPLIT_DIR="${STATIC_SPLIT_DIR:-$PROJECT_ROOT/data/paper_static_convagent_splits/$DATASET}"
  TRAIN_SOURCE="$STATIC_SOURCE_ROOT/$DATASET/${DATASET}_train.parquet"
  DEFAULT_EXPERIMENT_NAME="turnppo_matched_static_convagent_${DATASET}_$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]' | tr '.' '_')"

  # Preserve ConvAgent's GRPO objective and paper reward, but run it with the
  # shared Turn-PPO resource/context budget declared above. The 1,000-update
  # ceiling is only a safeguard: the holdout monitor selects the checkpoint.
  ACTOR_ENTROPY_COEFF="${CONVAGENT_ENTROPY_COEFF:-0.0}"
  ACTOR_USE_KL_LOSS="${CONVAGENT_USE_KL_LOSS:-true}"
  ACTOR_LR_WARMUP_RATIO="${CONVAGENT_ACTOR_LR_WARMUP_RATIO:-0.0}"
  CRITIC_LR_WARMUP_RATIO="0.0"
  # Match Turn-PPO's activation-memory policy. This changes neither the GRPO
  # objective nor the paper reward; it recomputes activations during backward
  # so a 3B run with an 8k context has usable headroom on 80GB GPUs.
  MODEL_ENABLE_GRADIENT_CHECKPOINTING="${CONVAGENT_GRADIENT_CHECKPOINTING:-true}"
  ROLLOUT_MAX_NUM_SEQS="${CONVAGENT_ROLLOUT_MAX_NUM_SEQS:-256}"
  BASELINE_ARGS=(
    "algorithm.adv_estimator=grpo"
    "algorithm.gamma=1.0"
    "algorithm.query_group_advantage=disabled"
    "algorithm.allow_nonanswer_action=true"
    "algorithm.use_action_reward=false"
    "algorithm.static_convagent_mode=true"
    "algorithm.static_convagent_direct_evidence_reward=true"
    "algorithm.static_convagent_short_answer_tokens=4"
    "algorithm.static_convagent_paper_reward=true"
    "algorithm.static_chatr1_mode=false"
    "algorithm.static_chatr1_intent_reward=false"
    "algorithm.static_chatr1_paper_reward=false"
    "critic.optim.lr=0"
  )
  echo "[Paper ConvAgent] GRPO reward=answer-only max-F1 + 0.5*(top-3-concatenated evidence + MIA); MIA is enabled only on InsCiT."
else
  PROJECT_NAME="paper_static_chatr1"
  STATIC_SOURCE_ROOT="${STATIC_SOURCE_ROOT:-$PROJECT_ROOT/data/static_chatr1_raw/ChatR1}"
  RAW_TRAIN_SOURCE="$STATIC_SOURCE_ROOT/$DATASET/${DATASET}_train.parquet"
  # One answer-only row retains all released answer references. A distinct
  # directory prevents accidental reuse of the older expanded-reference split.
  SPLIT_DIR="${STATIC_SPLIT_DIR:-$PROJECT_ROOT/data/paper_static_chatr1_splits_max_reference/$DATASET}"
  DEFAULT_EXPERIMENT_NAME="turnppo_matched_static_chatr1_${DATASET}_$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]' | tr '.' '_')"

  # Preserve ChatR1's separate actor/critic PPO, gamma=lambda=1, reward, and
  # learning-rate schedule. Its shared runtime/context budget is declared
  # above; only paper-specific optimization choices remain below.
  CHATR1_CRITIC_MODEL_PATH="${CHATR1_CRITIC_MODEL_PATH:-$MODEL_PATH}"
  CHATR1_CRITIC_LR="${CHATR1_CRITIC_LR:-1e-5}"
  CHATR1_INTENT_WEIGHT="${CHATR1_INTENT_WEIGHT:-1.0}"
  ACTOR_ENTROPY_COEFF="${CHATR1_ENTROPY_COEFF:-0.001}"
  ACTOR_USE_KL_LOSS="${CHATR1_USE_KL_LOSS:-false}"
  ACTOR_LR_WARMUP_RATIO="${CHATR1_ACTOR_LR_WARMUP_RATIO:-0.285}"
  CRITIC_LR_WARMUP_RATIO="${CHATR1_CRITIC_LR_WARMUP_RATIO:-0.015}"
  MODEL_ENABLE_GRADIENT_CHECKPOINTING="${CHATR1_GRADIENT_CHECKPOINTING:-true}"
  ROLLOUT_MAX_NUM_SEQS="${CHATR1_ROLLOUT_MAX_NUM_SEQS:-256}"
  BASELINE_ARGS=(
    "algorithm.adv_estimator=gae"
    "algorithm.gamma=1.0"
    "algorithm.lam=1.0"
    "algorithm.query_group_advantage=disabled"
    "algorithm.allow_nonanswer_action=false"
    "algorithm.use_action_reward=false"
    "algorithm.static_convagent_mode=false"
    "algorithm.static_convagent_direct_evidence_reward=false"
    "algorithm.static_convagent_paper_reward=false"
    "algorithm.static_chatr1_mode=true"
    "algorithm.static_chatr1_intent_reward=true"
    "algorithm.static_chatr1_intent_weight=$CHATR1_INTENT_WEIGHT"
    "algorithm.static_chatr1_paper_reward=true"
    "algorithm.kl_ctrl.kl_coef=0.001"
    "critic.optim.lr=$CHATR1_CRITIC_LR"
  )
  echo "[Paper ChatR1] PPO/GAE with independent actor and critic; terminal reward=answer-F1 + ${CHATR1_INTENT_WEIGHT}*max(query-F1,rewrite)."
fi

# A strict improvement in this equal-weight mean resets patience. The three
# source metrics are ratios, so the trainer explicitly clips each to [0, 1]
# before averaging. Each method emits one answer, while F1/BERTScore retain the
# maximum over that sample's released answer references. TopiOCQA retains
# passage-text NDCG when this launcher is reused there.
MONITOR_METRIC="val/selection/composite_score"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}"
TRAIN_FILE="$SPLIT_DIR/${DATASET}_train.parquet"
MONITOR_FILE="$SPLIT_DIR/${DATASET}_monitor.parquet"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$IGPO_CONDA_ENV"
export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PET_NODE_RANK="${PET_NODE_RANK:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAY_memory_monitor_refresh_ms=0 VLLM_ATTENTION_BACKEND=XFORMERS
export IGPO_MAX_SEARCH_QUERIES="$MAX_SEARCH_QUERIES" IGPO_SEARCH_TOP_K=3
source "$PROJECT_ROOT/scripts/configure_wandb.sh"
configure_wandb

if [[ "$STATIC_BASELINE" == "chatr1" ]]; then
  if [[ ! -f "$RAW_TRAIN_SOURCE" ]]; then
    echo "ERROR: ChatR1 static source is missing: $RAW_TRAIN_SOURCE" >&2
    exit 2
  fi
  if [[ ! -f "$TRAIN_FILE" || ! -f "$MONITOR_FILE" ]]; then
    python -u scripts/prepare_chatr1_paper_split.py \
      --input "$RAW_TRAIN_SOURCE" \
      --output-dir "$SPLIT_DIR" \
      --dataset "$DATASET" \
      --holdout-fraction "$HOLDOUT_FRACTION" \
      --seed "$SPLIT_SEED"
  fi
elif [[ ! -f "$TRAIN_SOURCE" ]]; then
  echo "ERROR: ConvAgent static source is missing: $TRAIN_SOURCE" >&2
  exit 2
elif [[ ! -f "$TRAIN_FILE" || ! -f "$MONITOR_FILE" ]]; then
  python -u scripts/prepare_static_monitor_split.py \
    --input "$TRAIN_SOURCE" \
    --output-dir "$SPLIT_DIR" \
    --dataset "$DATASET" \
    --holdout-fraction "$HOLDOUT_FRACTION" \
    --seed "$SPLIT_SEED"
fi

# Keep checkpoint selection reproducible and explicitly compatible with the
# Turn-PPO first-source-question holdout assignment.
python - "$SPLIT_DIR/${DATASET}_split_manifest.json" "$HOLDOUT_FRACTION" "$SPLIT_SEED" "$STATIC_BASELINE" <<'PY'
import json
import math
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.is_file():
    raise SystemExit(f"ERROR: missing static baseline split manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (
    manifest.get("split_unit") != "source_conversation"
    or int(manifest.get("seed", -1)) != int(sys.argv[3])
    or not math.isclose(float(manifest.get("holdout_fraction", -1)), float(sys.argv[2]))
    or (
        sys.argv[4] == "chatr1"
        and manifest.get("monitor_view")
        != "one answer-only row with all released answer references"
    )
):
    raise SystemExit(
        "ERROR: existing baseline holdout does not match the configured seed/fraction/reference protocol. "
        "Use a fresh split directory or intentionally rebuild it before training."
    )
print(
    "[Static Baseline] verified protocol-matched holdout: "
    f"key=first_source_user_question seed={manifest['seed']} "
    f"fraction={manifest['holdout_fraction']}"
)
PY

python - <<'PY'
import requests
response = requests.post(
    "http://127.0.0.1:8002/retrieve",
    json={"queries": ["paper static baseline retriever readiness probe"], "topk": 3, "return_scores": True},
    timeout=60,
)
response.raise_for_status()
print("Local retriever API: ready")
PY

OUTPUT_DIR="$PROJECT_ROOT/outputs/$PROJECT_NAME/$EXPERIMENT_NAME"
EVAL_DIR="$PROJECT_ROOT/eval_log/$PROJECT_NAME/$EXPERIMENT_NAME"
ROLLOUT_DIR="$OUTPUT_DIR/rollout"
mkdir -p "$OUTPUT_DIR" "$EVAL_DIR" "$ROLLOUT_DIR" "$PROJECT_ROOT/cache/task_queue"

echo "[$PROJECT_NAME] protocol-matched: prompt=4096 response=500 model_len=8192 ppo_mini=64 ppo/ref_micro=1 dynamic_bsz=true turns=4 searches=1 topk=3"
echo "[$PROJECT_NAME] dataset=$DATASET experiment=$EXPERIMENT_NAME train_batch=128 validation_batch=256 n=8"
echo "[$PROJECT_NAME] validate every $MONITOR_FREQUENCY updates; exact composite=(normalized F1 + BERTScore-F1 + NDCG@3)/3; stop after $MONITOR_PATIENCE consecutive non-improvements"
echo "[$PROJECT_NAME] selection BERTScore: model=$SELECTION_BERTSCORE_MODEL device=$SELECTION_BERTSCORE_DEVICE batch_size=$SELECTION_BERTSCORE_BATCH_SIZE"

python -u -m verl.trainer.main_ppo \
  "data.train_files=$TRAIN_FILE" \
  "data.val_files=$MONITOR_FILE" \
  "data.train_batch_size=$TRAIN_BATCH_SIZE" \
  "data.val_batch_size=$VAL_BATCH_SIZE" \
  "data.max_prompt_length=$MAX_PROMPT_LENGTH" \
  "data.truncation=left" \
  "data.max_response_length=$MAX_RESPONSE_LENGTH" \
  "+data.max_model_len=$MAX_MODEL_LEN" \
  "+data.data_writing_path=$PROJECT_ROOT/cache/task_queue/" \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  "actor_rollout_ref.model.enable_gradient_checkpointing=$MODEL_ENABLE_GRADIENT_CHECKPOINTING" \
  "actor_rollout_ref.model.use_remove_padding=true" \
  "actor_rollout_ref.actor.optim.lr=1e-6" \
  "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$ACTOR_LR_WARMUP_RATIO" \
  "actor_rollout_ref.actor.clip_ratio=0.2" \
  "actor_rollout_ref.actor.entropy_coeff=$ACTOR_ENTROPY_COEFF" \
  "actor_rollout_ref.actor.use_kl_loss=$ACTOR_USE_KL_LOSS" \
  "actor_rollout_ref.actor.kl_loss_coef=0.001" \
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
  "actor_rollout_ref.actor.ppo_epochs=1" \
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
  "critic.model.enable_gradient_checkpointing=$MODEL_ENABLE_GRADIENT_CHECKPOINTING" \
  "critic.model.use_remove_padding=true" \
  "critic.model.fsdp_config.param_offload=true" \
  "critic.model.fsdp_config.optimizer_offload=true" \
  "critic.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
  "critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU" \
  "critic.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU" \
  "critic.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ" \
  "critic.optim.lr_warmup_steps_ratio=$CRITIC_LR_WARMUP_RATIO" \
  "critic.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  "${BASELINE_ARGS[@]}" \
  "algorithm.max_search_queries=$MAX_SEARCH_QUERIES" \
  "algorithm.simulated_user_enabled=false" \
  "trainer.logger=$TRAINER_LOGGERS" \
  "trainer.log_val_generations=$WANDB_LOG_VAL_GENERATIONS" \
  "trainer.project_name=$PROJECT_NAME" \
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
  "trainer.static_convagent_monitor_frequency=$MONITOR_FREQUENCY" \
  "trainer.static_convagent_monitor_metric=$MONITOR_METRIC" \
  "trainer.static_convagent_monitor_patience=$MONITOR_PATIENCE" \
  "trainer.selection_monitor_composite_enabled=true" \
  "trainer.selection_monitor_bertscore_model=$SELECTION_BERTSCORE_MODEL" \
  "trainer.selection_monitor_bertscore_batch_size=$SELECTION_BERTSCORE_BATCH_SIZE" \
  "trainer.selection_monitor_bertscore_device=$SELECTION_BERTSCORE_DEVICE" \
  "agent_grpo.n=$ROLLOUT_N" \
  "max_turns=$MAX_TURNS" \
  "search_engine=local_retriever" \
  "codeact_env_disabled=true"

FINAL_STEP="$(< "$OUTPUT_DIR/latest_checkpointed_iteration.txt")"
FINAL_CHECKPOINT="$OUTPUT_DIR/global_step_$FINAL_STEP"
FINAL_JSON="$EVAL_DIR/$FINAL_STEP.jsonl"
if [[ ! -d "$FINAL_CHECKPOINT" || ! -f "$FINAL_JSON" ]]; then
  echo "ERROR: selected checkpoint or its holdout validation JSONL is missing: $FINAL_CHECKPOINT" >&2
  exit 3
fi
printf '%s\n' "$FINAL_CHECKPOINT" > "$OUTPUT_DIR/final_checkpoint.txt"

python -u scripts/compute_convagent_eval_metrics.py \
  --input "$FINAL_JSON" \
  --output-dir "$EVAL_DIR" \
  --bert-score-device "${BERT_SCORE_DEVICE:-cuda}" \
  --bert-score-batch-size "${BERT_SCORE_BATCH_SIZE:-64}" \
  --multi-reference

echo "Completed protocol-matched $STATIC_BASELINE training: $EXPERIMENT_NAME"
echo "Final checkpoint: $FINAL_CHECKPOINT"
echo "Holdout metrics: $EVAL_DIR/metrics_summary.json"
