#!/usr/bin/env bash
# Build frozen, paired static/dynamic InSCIT context views.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Keep the project-native environment name while accepting the shared IGPO
# variable as a backwards-compatible alias.
INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-${IGPO_CONDA_ENV:-interactivechat-r1}}"
IGPO_CONDA_ENV="$INTERACTIVECHAT_CONDA_ENV"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/inscit_context_ablation/v1}"
BUILDER_PROVIDER="${BUILDER_PROVIDER:-openai}"
if [[ "$BUILDER_PROVIDER" == "openai" ]]; then
  : "${OPENAI_API_KEY:?Set a newly rotated OPENAI_API_KEY in the environment.}"
  MODEL="${BUILDER_MODEL:-${GPT_MODEL:-gpt-4.1-2025-04-14}}"
elif [[ "$BUILDER_PROVIDER" == "local-qwen" ]]; then
  MODEL="${BUILDER_MODEL:-qwen25-32b-inscit-context-builder}"
else
  echo "Unsupported BUILDER_PROVIDER=$BUILDER_PROVIDER (expected openai or local-qwen)" >&2
  exit 2
fi
LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:8011/v1}"
LOCAL_MAX_OUTPUT_TOKENS="${LOCAL_MAX_OUTPUT_TOKENS:-768}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-180}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8002/retrieve}"
RETRIEVER_ID="${RETRIEVER_ID:-inscit-e5-flat-top3}"
RETRIEVER_CONFIG_HASH="${RETRIEVER_CONFIG_HASH:-}"
WORKERS="${WORKERS:-4}"
MAX_TRANSIENT_RETRIES="${MAX_TRANSIENT_RETRIES:-0}"
LIMIT_CONVERSATIONS="${LIMIT_CONVERSATIONS:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/inscit_dynamic_builder.log}"
RESUME="${RESUME:-true}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$IGPO_CONDA_ENV"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
mkdir -p "$(dirname "$LOG_FILE")"

arguments=(
  --train-input "$PROJECT_ROOT/data/inscit_train.json"
  --test-input "$PROJECT_ROOT/data/inscit_test.json"
  --output-dir "$OUTPUT_DIR"
  --provider "$BUILDER_PROVIDER"
  --model "$MODEL"
  --local-base-url "$LOCAL_LLM_BASE_URL"
  --local-max-output-tokens "$LOCAL_MAX_OUTPUT_TOKENS"
  --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS"
  --retriever-url "$RETRIEVER_URL"
  --retriever-id "$RETRIEVER_ID"
  --topk 3
  --max-feedback-rounds 2
  --max-transient-retries "$MAX_TRANSIENT_RETRIES"
  --format-retries 3
  --workers "$WORKERS"
  --log-file "$LOG_FILE"
)
[[ -n "$RETRIEVER_CONFIG_HASH" ]] && arguments+=(--retriever-config-hash "$RETRIEVER_CONFIG_HASH")
[[ -d "$TOKENIZER_PATH" ]] && arguments+=(--tokenizer-path "$TOKENIZER_PATH")
[[ "$RESUME" == true ]] && arguments+=(--resume)
[[ -n "$LIMIT_CONVERSATIONS" ]] && arguments+=(--limit-conversations "$LIMIT_CONVERSATIONS")

python -u scripts/build_inscit_context_ablation.py "${arguments[@]}"

