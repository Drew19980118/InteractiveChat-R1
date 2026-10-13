#!/usr/bin/env bash
# Sequentially train the static-history and dynamic-history policies.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-3B-Instruct.}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-convagent_inscit}"

for variant in static dynamic; do
  echo "[InSCIT context ablation] starting $variant training"
  CONTEXT_VARIANT="$variant" \
  EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${variant}_context_seed42" \
    bash scripts/run_inscit_context_ablation_convagent_3b_train.sh
done

