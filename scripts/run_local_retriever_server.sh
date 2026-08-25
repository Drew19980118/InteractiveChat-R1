#!/usr/bin/env bash
# Launch the local FAISS retriever used by all online rollouts.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${CUDA_VISIBLE_DEVICES:?Set the GPU(s) reserved for retrieval.}"
: "${RETRIEVER_INDEX_PATH:?Set the merged FAISS index path.}"
: "${RETRIEVER_CORPUS_PATH:?Set the JSONL corpus path aligned to the FAISS ids.}"

INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-${IGPO_CONDA_ENV:-interactivechat-r1}}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$INTERACTIVECHAT_CONDA_ENV"

exec python -u scripts/local_retriever.py \
  --index_path "$RETRIEVER_INDEX_PATH" \
  --corpus_path "$RETRIEVER_CORPUS_PATH" \
  --retriever_model "${RETRIEVER_MODEL_PATH:-intfloat/e5-base-v2}" \
  --topk "${RETRIEVER_TOP_K:-3}" \
  --batch_size "${RETRIEVER_BATCH_SIZE:-512}" \
  --max_length "${RETRIEVER_MAX_LENGTH:-256}" \
  --host "${RETRIEVER_HOST:-127.0.0.1}" \
  --port "${RETRIEVER_PORT:-8002}" \
  --faiss_gpu
