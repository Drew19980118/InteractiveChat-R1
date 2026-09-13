#!/usr/bin/env bash
# QReCC latest static-baseline lifecycle on GPUs 0 and 1:
# ConvAgent 3B -> ConvAgent 7B -> ChatR1 3B -> ChatR1 7B.
# The shared trainer retains the monitor-best checkpoint only and prunes each
# superseded global_step_* directory immediately after a new best is saved.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS="${N_GPUS:-2}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"
INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-interactivechat-r1}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-qrecc_baselines_latest_cuda01_v1}"

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
export CONVAGENT_3B_EXPERIMENT_NAME="${CONVAGENT_3B_EXPERIMENT_NAME:-convagent_qrecc_qwen25_3b_latest_cuda01_v1}"
export CONVAGENT_7B_EXPERIMENT_NAME="${CONVAGENT_7B_EXPERIMENT_NAME:-convagent_qrecc_qwen25_7b_latest_cuda01_v1}"
export CHATR1_3B_EXPERIMENT_NAME="${CHATR1_3B_EXPERIMENT_NAME:-chatr1_qrecc_qwen25_3b_latest_cuda01_v1}"
export CHATR1_7B_EXPERIMENT_NAME="${CHATR1_7B_EXPERIMENT_NAME:-chatr1_qrecc_qwen25_7b_latest_cuda01_v1}"

echo "[QReCC CUDA01 suite] GPUs=0,1; automatic stale-checkpoint pruning is enabled."
exec bash "$SCRIPT_DIR/run_latest_qrecc_static_baselines_3b_7b_export_upload.sh"
