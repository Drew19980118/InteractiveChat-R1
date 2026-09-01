#!/usr/bin/env bash
# Export a selected full FSDP checkpoint as an actor-only Hugging Face model
# and, when explicitly requested, upload that smaller inference artifact.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the selected global_step_* directory.}"
: "${MODEL_PATH:?Set MODEL_PATH to the original Qwen base-model directory.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the matching prepared static train Parquet.}"
: "${ACTOR_EXPORT_DIR:?Set ACTOR_EXPORT_DIR to a new, empty actor-only export directory.}"

HF_UPLOAD_ACTOR_ONLY="${HF_UPLOAD_ACTOR_ONLY:-false}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_UPLOAD_NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-8}"

bash "$SCRIPT_DIR/export_actor_policy.sh"

if [[ "$HF_UPLOAD_ACTOR_ONLY" != "true" ]]; then
  echo "Actor-only export completed locally; upload skipped (HF_UPLOAD_ACTOR_ONLY=false)."
  exit 0
fi

if [[ -z "$HF_REPO_ID" ]]; then
  echo "ERROR: Set HF_REPO_ID=namespace/repository when HF_UPLOAD_ACTOR_ONLY=true." >&2
  exit 2
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: Hugging Face CLI 'hf' is not installed in this environment." >&2
  exit 2
fi
if ! [[ "$HF_UPLOAD_NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: HF_UPLOAD_NUM_WORKERS must be a positive integer." >&2
  exit 2
fi

# This checks the active token without printing it.  Support both the current
# `hf auth whoami` form and the older CLI form pinned by some evaluation
# environments.  The caller should use their own write-scoped token, so the
# Hub records their upload contribution.
if ! hf auth whoami >/dev/null 2>&1 && ! hf whoami >/dev/null 2>&1; then
  echo "ERROR: No valid Hugging Face login. Run 'hf auth login' (or the older 'hf login') first." >&2
  exit 2
fi
hf upload-large-folder "$HF_REPO_ID" "$ACTOR_EXPORT_DIR" \
  --repo-type model \
  --num-workers "$HF_UPLOAD_NUM_WORKERS"

echo "Actor-only model uploaded: https://huggingface.co/$HF_REPO_ID"
