#!/usr/bin/env bash
# Shared formal setting for InteractiveChat-R1 experiments.
#
# Source this file after setting DATASET, MODEL_PATH, EXPERIMENT_NAME, and the
# experiment-specific TOTAL_STEPS.  CUDA_VISIBLE_DEVICES is deliberately not
# assigned here: the caller must reserve GPUs that are not used by the local
# retriever or the Qwen32B user simulator.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source scripts/experiment_common.sh from an experiment launcher." >&2
  exit 2
fi

: "${DATASET:?Set DATASET before sourcing experiment_common.sh.}"
: "${MODEL_PATH:?Set MODEL_PATH before sourcing experiment_common.sh.}"
: "${EXPERIMENT_NAME:?Set EXPERIMENT_NAME before sourcing experiment_common.sh.}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export GRPO_N="${GRPO_N:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export EXACT_CONTEXT_BATCH="${EXACT_CONTEXT_BATCH:-true}"

# One query is executed at a time; its top-3 passages become the fresh private
# evidence.  Older evidence is compacted only if the policy context reaches
# 8192 tokens, so a newly retrieved top-3 is not globally clipped to 512.
export MAX_SEARCH_QUERIES="${MAX_SEARCH_QUERIES:-1}"
export SEARCH_TOP_K="${SEARCH_TOP_K:-3}"
export TOOL_OBSERVATION_TOKEN_CAP="${TOOL_OBSERVATION_TOKEN_CAP:-0}"
export MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-4}"
export MAX_ANSWER_DEPTH="${MAX_ANSWER_DEPTH:-3}"

# Main proposed reward: answer F1 for correctness plus simulator-derived
# clarity/patience for user satisfaction. Frozen-likelihood, evidence-only,
# and efficiency shaping are deliberately off in the canonical paper recipe.
export SIMULATED_USER_REWARD_MODE="${SIMULATED_USER_REWARD_MODE:-full}"
export SIMULATED_USER_ENABLE_FEEDBACK="${SIMULATED_USER_ENABLE_FEEDBACK:-true}"
export SIMULATED_USER_STATIC_GOLD_CONTEXT="${SIMULATED_USER_STATIC_GOLD_CONTEXT:-false}"
export SIMULATED_USER_ENABLE_EVIDENCE_UTILITY="${SIMULATED_USER_ENABLE_EVIDENCE_UTILITY:-false}"
export SIMULATED_USER_ENABLE_SEARCH_EFFICIENCY="${SIMULATED_USER_ENABLE_SEARCH_EFFICIENCY:-false}"
export SIMULATED_USER_ACTION_WEIGHT="${SIMULATED_USER_ACTION_WEIGHT:-1.0}"
export SIMULATED_USER_ANSWER_F1_WEIGHT="${SIMULATED_USER_ANSWER_F1_WEIGHT:-1.0}"
export SIMULATED_USER_EVIDENCE_UTILITY_WEIGHT="${SIMULATED_USER_EVIDENCE_UTILITY_WEIGHT:-0.0}"
export SIMULATED_USER_SEARCH_EFFICIENCY_WEIGHT="${SIMULATED_USER_SEARCH_EFFICIENCY_WEIGHT:-0.0}"
export SIMULATED_USER_CLARITY_WEIGHT="${SIMULATED_USER_CLARITY_WEIGHT:-1.0}"
export SIMULATED_USER_PATIENCE_WEIGHT="${SIMULATED_USER_PATIENCE_WEIGHT:-1.0}"
export SIMULATED_USER_FORMAT_WEIGHT="${SIMULATED_USER_FORMAT_WEIGHT:-1.0}"
export SIMULATED_USER_CLARIFY_F1_WEIGHT="${SIMULATED_USER_CLARIFY_F1_WEIGHT:-1.0}"

# Full final validation follows the final update. Set the intermediate knobs
# explicitly only for a smoke test; they are off in every formal run.
export SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ="${SIMULATED_USER_INTERMEDIATE_VALIDATION_FREQ:-0}"
export SIMULATED_USER_VALIDATION_MAX_BATCHES="${SIMULATED_USER_VALIDATION_MAX_BATCHES:-}"
export SIMULATED_USER_INTERMEDIATE_VALIDATION_MAX_BATCHES="${SIMULATED_USER_INTERMEDIATE_VALIDATION_MAX_BATCHES:-2}"

# Conservative H100 defaults.  These affect only vLLM KV-cache capacity, not
# the model context length (which remains 8192) nor the optimization recipe.
export N_GPUS="${N_GPUS:-4}"
export ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.08}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export BERT_SCORE_DEVICE="${BERT_SCORE_DEVICE:-cuda}"
export BERT_SCORE_BATCH_SIZE="${BERT_SCORE_BATCH_SIZE:-64}"
