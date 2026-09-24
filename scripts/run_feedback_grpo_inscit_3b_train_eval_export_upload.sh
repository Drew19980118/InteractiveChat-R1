#!/usr/bin/env bash
# Alternate two independent GRPO policies for ONE declared experimental setting
# per run. The selected System/User pair is exported and uploaded together.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

INTERACTIVECHAT_CONDA_ENV="${INTERACTIVECHAT_CONDA_ENV:-interactivechat-r1}"
if [[ -n "$INTERACTIVECHAT_CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$INTERACTIVECHAT_CONDA_ENV" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is unavailable. Activate the training environment, or set INTERACTIVECHAT_CONDA_ENV='' to use the current Python." >&2
    exit 2
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$INTERACTIVECHAT_CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
N_GPUS="${N_GPUS:-2}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
SYSTEM_MODEL_PATH="${SYSTEM_MODEL_PATH:-${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}}"
USER_MODEL_PATH="${USER_MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/static_convagent_raw/ConvAgent/inscit/inscit_train.parquet}"
TEST_FILE="${TEST_FILE:-$PROJECT_ROOT/data/static_convagent_raw/ConvAgent/inscit/inscit_test.parquet}"
SELECTION_SETTING="${SELECTION_SETTING:-direct-response}"
case "$SELECTION_SETTING" in
  direct-response)
    DEFAULT_EXPERIMENT_NAME="feedback_grpo_inscit_qwen25_3b_direct_second_only_v1"
    DEFAULT_SYSTEM_REPO_ID="DrewZhang/interactivechat-r1-feedback-grpo-inscit-qwen25-3b-direct-second-only-system"
    DEFAULT_USER_REPO_ID="DrewZhang/interactivechat-r1-feedback-grpo-inscit-qwen25-3b-direct-second-only-user"
    ;;
  feedback-refinement)
    DEFAULT_EXPERIMENT_NAME="feedback_grpo_inscit_qwen25_3b_feedback_second_only_v1"
    DEFAULT_SYSTEM_REPO_ID="DrewZhang/interactivechat-r1-feedback-grpo-inscit-qwen25-3b-feedback-second-only-system"
    DEFAULT_USER_REPO_ID="DrewZhang/interactivechat-r1-feedback-grpo-inscit-qwen25-3b-feedback-second-only-user"
    ;;
  *)
    echo "ERROR: SELECTION_SETTING must be direct-response or feedback-refinement." >&2
    exit 2
    ;;
esac
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/feedback_grpo}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/eval_log/feedback_grpo}"
EXPORT_ROOT="${EXPORT_ROOT:-$PROJECT_ROOT/exports/feedback_grpo}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-interactivechat-r1}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-inscit_feedback_grpo_3b}"
HF_UPLOAD="${HF_UPLOAD:-true}"
HF_UPLOAD_NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-4}"
HF_SYSTEM_REPO_ID="${HF_SYSTEM_REPO_ID:-$DEFAULT_SYSTEM_REPO_ID}"
HF_USER_REPO_ID="${HF_USER_REPO_ID:-$DEFAULT_USER_REPO_ID}"
RESUME="${RESUME:-false}"

for boolean in "$HF_UPLOAD" "$WANDB_ENABLED" "$RESUME"; do
  if [[ "$boolean" != true && "$boolean" != false ]]; then
    echo "ERROR: HF_UPLOAD, WANDB_ENABLED and RESUME must be true or false." >&2
    exit 2
  fi
done
if [[ "$EXPERIMENT_NAME" == .* || "$EXPERIMENT_NAME" == *[!a-zA-Z0-9_.-]* ]]; then
  echo "ERROR: EXPERIMENT_NAME must be a simple non-hidden directory name." >&2
  exit 2
fi
for model_path in "$SYSTEM_MODEL_PATH" "$USER_MODEL_PATH"; do
  if [[ ! -f "$model_path/config.json" ]]; then
    echo "ERROR: model config.json missing: $model_path" >&2
    exit 2
  fi
done
for data_file in "$TRAIN_FILE" "$TEST_FILE"; do
  if [[ ! -f "$data_file" ]]; then
    echo "ERROR: missing static ConvAgent source data: $data_file" >&2
    exit 2
  fi
done
if [[ "$HF_SYSTEM_REPO_ID" == "$HF_USER_REPO_ID" ]]; then
  echo "ERROR: system and user need different Hugging Face repositories." >&2
  exit 2
fi
if [[ "$ULYSSES_SEQUENCE_PARALLEL_SIZE" != "1" ]]; then
  echo "ERROR: Feedback GRPO uses one whole GPU per independent policy; set ULYSSES_SEQUENCE_PARALLEL_SIZE=1." >&2
  exit 2
fi

source "$SCRIPT_DIR/configure_wandb.sh"
configure_wandb

# The rollout continues using the native local search endpoint. The learned
# user is a second training policy; no external simulator server is required.
python - <<'PY'
import requests
response = requests.post(
    "http://127.0.0.1:8002/retrieve",
    json={"queries": ["feedback GRPO InsCiT retriever readiness probe"], "topk": 3, "return_scores": True},
    timeout=60,
)
response.raise_for_status()
print("Local retriever API: ready")
PY

HF_CLI=()
HF_TYPE_FLAG=""
if [[ "$HF_UPLOAD" == true ]]; then
  # New and historical environments expose different CLI layouts. Inspect
  # the installed help rather than assuming --type or --repo-type exists.
  if command -v hf >/dev/null 2>&1; then
    HF_CLI=(hf)
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI=(huggingface-cli)
  else
    echo "ERROR: HF_UPLOAD=true requires the Hugging Face CLI in the active environment." >&2
    exit 2
  fi
  upload_help="$("${HF_CLI[@]}" upload-large-folder --help)"
  if grep -q -- '--repo-type' <<< "$upload_help"; then
    HF_TYPE_FLAG="--repo-type"
  elif grep -q -- '--type' <<< "$upload_help"; then
    HF_TYPE_FLAG="--type"
  else
    echo "ERROR: installed upload-large-folder command has no supported repository type flag." >&2
    exit 2
  fi
  python - <<'PY'
from huggingface_hub import HfApi
identity = HfApi().whoami()
print("Hugging Face authenticated account:", identity.get("name", "unknown"))
PY
fi

arguments=(
  --system-model "$SYSTEM_MODEL_PATH"
  --user-model "$USER_MODEL_PATH"
  --train-file "$TRAIN_FILE"
  --test-file "$TEST_FILE"
  --experiment "$EXPERIMENT_NAME"
  --selection-setting "$SELECTION_SETTING"
  --n-gpus "$N_GPUS"
  --sequence-parallel "$ULYSSES_SEQUENCE_PARALLEL_SIZE"
  --n "${ROLLOUT_N:-8}"
  --train-batch-size "${TRAIN_BATCH_SIZE:-128}"
  --val-batch-size "${VAL_BATCH_SIZE:-256}"
  --user-updates-per-phase "${USER_UPDATES_PER_PHASE:-5}"
  --system-updates-per-phase "${SYSTEM_UPDATES_PER_PHASE:-5}"
  --max-system-updates "${MAX_SYSTEM_UPDATES:-1000}"
  --validate-every "${VALIDATE_EVERY:-5}"
  --patience "${EARLY_STOP_PATIENCE:-3}"
  --holdout-fraction "${HOLDOUT_FRACTION:-0.10}"
  --seed "${SPLIT_SEED:-42}"
  --output-root "$OUTPUT_ROOT"
  --eval-root "$EVAL_ROOT"
  --export-root "$EXPORT_ROOT"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
  --logprob-batch-size "${LOGPROB_BATCH_SIZE:-8}"
  --max-empty-system-batches "${MAX_EMPTY_SYSTEM_BATCHES:-20}"
  --rollout-memory "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.15}"
  --bert-score-device "${BERT_SCORE_DEVICE:-cpu}"
  --bert-score-batch-size "${BERT_SCORE_BATCH_SIZE:-8}"
  --wandb-project "$WANDB_PROJECT"
  --wandb-group "$WANDB_RUN_GROUP"
  --system-repo-id "$HF_SYSTEM_REPO_ID"
  --user-repo-id "$HF_USER_REPO_ID"
)
if [[ "$WANDB_ENABLED" == true ]]; then arguments+=(--wandb); fi
if [[ "$RESUME" == true ]]; then arguments+=(--resume); fi

echo "[Feedback GRPO] CUDA=$CUDA_VISIBLE_DEVICES; system/user each own one full GPU; n=${ROLLOUT_N:-8}."
echo "[Feedback GRPO] system phase: one retried first response + one learned-user feedback + n revised responses; only revised responses update the system policy."
echo "[Feedback GRPO] setting=$SELECTION_SETTING; train -> select one paired best -> test both views -> export/upload that pair."
python -u -m verl.recipe.feedback_grpo.main "${arguments[@]}" "$@"

if [[ "$HF_UPLOAD" == true ]]; then
  python - "$EXPORT_ROOT/$EXPERIMENT_NAME" "$HF_SYSTEM_REPO_ID" "$HF_USER_REPO_ID" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import HfApi
root = Path(sys.argv[1])
for role in ("system", "user"):
    path = root / role
    if not (path / "config.json").is_file():
        raise SystemExit(f"ERROR: missing exported {role} config: {path}")
    if not list(path.glob("*.safetensors")) and not list(path.glob("pytorch_model*.bin")):
        raise SystemExit(f"ERROR: missing exported {role} weights: {path}")
api = HfApi()
for repo_id in sys.argv[2:]:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
PY
  "${HF_CLI[@]}" upload-large-folder "$HF_SYSTEM_REPO_ID" "$EXPORT_ROOT/$EXPERIMENT_NAME/system" \
    "$HF_TYPE_FLAG" model --num-workers "$HF_UPLOAD_NUM_WORKERS"
  "${HF_CLI[@]}" upload-large-folder "$HF_USER_REPO_ID" "$EXPORT_ROOT/$EXPERIMENT_NAME/user" \
    "$HF_TYPE_FLAG" model --num-workers "$HF_UPLOAD_NUM_WORKERS"
  echo "Paired system: https://huggingface.co/$HF_SYSTEM_REPO_ID"
  echo "Paired user:   https://huggingface.co/$HF_USER_REPO_ID"
fi
echo "Selected $SELECTION_SETTING pair: $OUTPUT_ROOT/$EXPERIMENT_NAME/final_checkpoint.txt"
echo "Evaluation output root: $EVAL_ROOT/$EXPERIMENT_NAME"
echo "Paired policy exports: $EXPORT_ROOT/$EXPERIMENT_NAME/{system,user}"
