#!/usr/bin/env bash
# QReCC latest 7B static-baseline lifecycle on GPUs 0,1:
#   ConvAgent 7B -> QReCC static test -> actor-only export/upload
#   ChatR1    7B -> QReCC static test -> actor-only export/upload
#
# The shared trainer validates every five completed updates, selects the best
# normalized F1/BERTScore/NDCG@3 composite, and immediately prunes superseded
# global_step_* checkpoints after a replacement monitor-best is durable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS="${N_GPUS:-2}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"
INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-interactivechat-r1}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-qrecc_convagent_chatr1_7b_cuda01_v1}"

if [[ "$CUDA_VISIBLE_DEVICES" != "0,1" ]]; then
  echo "ERROR: this CUDA-0/1 launcher requires CUDA_VISIBLE_DEVICES=0,1." >&2
  exit 2
fi
if (( N_GPUS != 2 || ULYSSES_SEQUENCE_PARALLEL_SIZE != 2 )); then
  echo "ERROR: this launcher requires N_GPUS=2 and ULYSSES_SEQUENCE_PARALLEL_SIZE=2." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES N_GPUS ULYSSES_SEQUENCE_PARALLEL_SIZE
export INTERACTIVECHAT_CONDA_ENV WANDB_RUN_GROUP
export DATASET=qrecc
export RUN_CONVAGENT_3B=false
export RUN_CHATR1_3B=false
export RUN_CONVAGENT_7B=true
export RUN_CHATR1_7B=true

# Fresh, overrideable names avoid colliding with a prior full 3B/7B suite.
export CONVAGENT_7B_EXPERIMENT_NAME="${CONVAGENT_7B_EXPERIMENT_NAME:-convagent_qrecc_qwen25_7b_latest_cuda01_v3}"
export CHATR1_7B_EXPERIMENT_NAME="${CHATR1_7B_EXPERIMENT_NAME:-chatr1_qrecc_qwen25_7b_latest_cuda01_v3}"

echo "[QReCC CUDA01 7B suite] ConvAgent 7B -> ChatR1 7B."
echo "[QReCC CUDA01 7B suite] Automatic stale-checkpoint pruning is enabled."
exec bash "$SCRIPT_DIR/run_latest_static_baselines_suite.sh"
