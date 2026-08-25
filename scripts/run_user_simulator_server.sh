#!/usr/bin/env bash
# Start the frozen Qwen32B user simulator through an OpenAI-compatible vLLM API.
# This is intentionally a separate process/GPU allocation from policy training.
set -euo pipefail

: "${SIMULATOR_MODEL_PATH:?Set SIMULATOR_MODEL_PATH to the downloaded Qwen32B model.}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMULATOR_PORT="${SIMULATOR_PORT:-8010}"
SIMULATOR_GPU_MEMORY_UTILIZATION="${SIMULATOR_GPU_MEMORY_UTILIZATION:-0.85}"
SIMULATOR_TP_SIZE="${SIMULATOR_TP_SIZE:-1}"
SIMULATOR_MAX_MODEL_LEN="${SIMULATOR_MAX_MODEL_LEN:-8192}"
SIMULATOR_MAX_NUM_SEQS="${SIMULATOR_MAX_NUM_SEQS:-1}"
SIMULATOR_MAX_NUM_BATCHED_TOKENS="${SIMULATOR_MAX_NUM_BATCHED_TOKENS:-8192}"

# vLLM 0.6.3 pins outlines 0.0.46, whose eager optional airport import points
# at a broken PyPI dependency. Prepend the project-local compatibility shim.
export PYTHONPATH="${PROJECT_ROOT}/third_party${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$SIMULATOR_MODEL_PATH" \
  --served-model-name "${SIMULATOR_MODEL_NAME:-qwen32b-user-simulator}" \
  --port "$SIMULATOR_PORT" \
  --tensor-parallel-size "$SIMULATOR_TP_SIZE" \
  --gpu-memory-utilization "$SIMULATOR_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$SIMULATOR_MAX_MODEL_LEN" \
  --max-num-seqs "$SIMULATOR_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$SIMULATOR_MAX_NUM_BATCHED_TOKENS" \
  --dtype bfloat16
