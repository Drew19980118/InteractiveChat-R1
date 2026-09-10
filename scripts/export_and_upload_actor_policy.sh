#!/usr/bin/env bash
# Export an FSDP actor and optionally upload the resulting inference model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH.}"
: "${MODEL_PATH:?Set MODEL_PATH.}"
: "${TRAIN_FILE:?Set TRAIN_FILE.}"
: "${ACTOR_EXPORT_DIR:?Set ACTOR_EXPORT_DIR.}"

HF_UPLOAD_ACTOR_ONLY="${HF_UPLOAD_ACTOR_ONLY:-false}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_UPLOAD_NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-8}"

bash "$SCRIPT_DIR/export_actor_policy.sh"

if [[ "$HF_UPLOAD_ACTOR_ONLY" != "true" ]]; then
  echo "Actor-only export completed locally; upload skipped."
  exit 0
fi
if [[ -z "$HF_REPO_ID" ]]; then
  echo "ERROR: Set HF_REPO_ID=namespace/repository when HF_UPLOAD_ACTOR_ONLY=true." >&2
  exit 2
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: Hugging Face CLI 'hf' is not installed." >&2
  exit 2
fi
if ! [[ "$HF_UPLOAD_NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: HF_UPLOAD_NUM_WORKERS must be a positive integer." >&2
  exit 2
fi
if ! hf auth whoami >/dev/null 2>&1 && ! hf whoami >/dev/null 2>&1; then
  echo "ERROR: No valid Hugging Face login. Run hf auth login (or hf login) first." >&2
  exit 2
fi

hf upload-large-folder "$HF_REPO_ID" "$ACTOR_EXPORT_DIR" \
  --repo-type=model \
  --num-workers "$HF_UPLOAD_NUM_WORKERS"
echo "Actor-only model uploaded: https://huggingface.co/$HF_REPO_ID"
