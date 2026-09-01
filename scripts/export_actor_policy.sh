#!/usr/bin/env bash
# Convert a full FSDP PPO checkpoint into a standalone actor-only Hugging Face
# model directory.  This is for inference/validation transfer only; it cannot
# resume RL training because optimizer, critic, scheduler, and RNG state are
# intentionally omitted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the full global_step_* directory.}"
: "${MODEL_PATH:?Set MODEL_PATH to the original Qwen base-model directory.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to any compatible static training Parquet; it is only used to initialize the runtime.}"
: "${ACTOR_EXPORT_DIR:?Set ACTOR_EXPORT_DIR to a new, empty destination directory.}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the checkpoint FSDP GPU count.}"

N_GPUS="${N_GPUS:-2}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
ACTOR_EXPORT_MAX_SHARD_SIZE="${ACTOR_EXPORT_MAX_SHARD_SIZE:-2GB}"
ACTOR_EXPORT_ROLLOUT_GPU_MEMORY_UTILIZATION="${ACTOR_EXPORT_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.15}"
RUNTIME_DIR="${ACTOR_EXPORT_RUNTIME_DIR:-$PROJECT_ROOT/outputs/actor_export_runtime}"

if [[ ! -d "$CHECKPOINT_PATH" || "$(basename "$CHECKPOINT_PATH")" != global_step_* ]]; then
  echo "ERROR: CHECKPOINT_PATH must be an existing global_step_* directory." >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" || ! -f "$TRAIN_FILE" ]]; then
  echo "ERROR: MODEL_PATH or TRAIN_FILE does not exist." >&2
  exit 2
fi
if [[ -e "$ACTOR_EXPORT_DIR" && -n "$(find "$ACTOR_EXPORT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: ACTOR_EXPORT_DIR already exists and is not empty: $ACTOR_EXPORT_DIR" >&2
  exit 2
fi
if (( N_GPUS < 1 || ULYSSES_SEQUENCE_PARALLEL_SIZE < 1 || N_GPUS % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "ERROR: invalid N_GPUS / ULYSSES_SEQUENCE_PARALLEL_SIZE." >&2
  exit 2
fi

EXPORT_STEP="${CHECKPOINT_PATH##*global_step_}"
if ! [[ "$EXPORT_STEP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not parse checkpoint step from $CHECKPOINT_PATH" >&2
  exit 2
fi

mkdir -p "$RUNTIME_DIR"
echo "[Actor export] source=$CHECKPOINT_PATH"
echo "[Actor export] destination=$ACTOR_EXPORT_DIR shards=$ACTOR_EXPORT_MAX_SHARD_SIZE gpus=$N_GPUS"

python -u -m verl.trainer.main_ppo \
  "data.train_files=$TRAIN_FILE" \
  "data.val_files=$TRAIN_FILE" \
  "data.train_batch_size=128" \
  "data.val_batch_size=256" \
  "data.max_prompt_length=4096" \
  "data.truncation=left" \
  "data.max_response_length=500" \
  "+data.max_model_len=8192" \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  "actor_rollout_ref.model.use_remove_padding=true" \
  "actor_rollout_ref.actor.optim.lr=0" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=64" \
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
  "actor_rollout_ref.rollout.gpu_memory_utilization=$ACTOR_EXPORT_ROLLOUT_GPU_MEMORY_UTILIZATION" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=8192" \
  "actor_rollout_ref.rollout.max_model_len=8192" \
  "critic.model.path=$MODEL_PATH" \
  "critic.optim.lr=0" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.gamma=1.0" \
  "+algorithm.info_gain_norm_mode=separate" \
  "algorithm.query_group_advantage=disabled" \
  "algorithm.max_search_queries=1" \
  "algorithm.use_action_reward=false" \
  "algorithm.simulated_user_enabled=false" \
  "trainer.logger=['console']" \
  "trainer.project_name=actor_export" \
  "trainer.experiment_name=global_step_$EXPORT_STEP" \
  "trainer.default_hdfs_dir=null" \
  "trainer.default_local_dir=$RUNTIME_DIR" \
  "trainer.val_before_train=false" \
  "trainer.resume_mode=resume_path" \
  "trainer.resume_from_path=$CHECKPOINT_PATH" \
  "trainer.export_actor_hf_path=$ACTOR_EXPORT_DIR" \
  "trainer.export_actor_hf_max_shard_size=$ACTOR_EXPORT_MAX_SHARD_SIZE" \
  "trainer.n_gpus_per_node=$N_GPUS" \
  "trainer.nnodes=1" \
  "trainer.total_training_steps=$((EXPORT_STEP + 1))" \
  "trainer.total_epochs=1" \
  "trainer.save_freq=-1" \
  "trainer.test_freq=-1" \
  "agent_grpo.n=1" \
  "max_turns=4" \
  "search_engine=local_retriever" \
  "codeact_env_disabled=true"

if [[ ! -f "$ACTOR_EXPORT_DIR/config.json" ]]; then
  echo "ERROR: export did not create config.json in $ACTOR_EXPORT_DIR" >&2
  exit 3
fi
if ! find "$ACTOR_EXPORT_DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .; then
  echo "ERROR: export did not create model weight shards in $ACTOR_EXPORT_DIR" >&2
  exit 3
fi
echo "Actor-only export ready: $ACTOR_EXPORT_DIR"
