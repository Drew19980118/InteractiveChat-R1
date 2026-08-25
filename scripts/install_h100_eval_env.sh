#!/usr/bin/env bash

# Create a clean, reproducible InteractiveChat-R1 environment for a CUDA-capable
# Linux x86_64 host. This script deliberately never installs requirements.txt:
# its unconstrained vLLM and PyTorch entries can overwrite the locked runtime.

set -euo pipefail

ENV_NAME="${1:-interactivechat-r1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${REPO_ROOT}/requirements-eval-h100-cu121.txt"
MAX_JOBS="${MAX_JOBS:-8}"
export MAX_JOBS

# Do not impose a machine-specific GPU list.  A caller may set
# CUDA_VISIBLE_DEVICES before invoking the installer, otherwise the PyTorch
# check sees every GPU visible to the current shell.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES
fi

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    fail "This environment is supported only on Linux x86_64."
fi

command -v conda >/dev/null 2>&1 || fail "conda was not found in PATH."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi was not found; install a working NVIDIA driver first."
[[ -f "${LOCK_FILE}" ]] || fail "Missing lock file: ${LOCK_FILE}"

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
minimum_driver="525.60.13"
if [[ "$(printf '%s\n' "${minimum_driver}" "${driver_version}" | sort -V | head -n 1)" != "${minimum_driver}" ]]; then
    fail "NVIDIA driver ${driver_version} is too old for CUDA 12.1 wheels; need >= ${minimum_driver}."
fi

echo "=== GPU inventory ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
echo "=== Existing GPU compute processes ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true

if conda run --name "${ENV_NAME}" python --version >/dev/null 2>&1; then
    echo "Reusing existing conda environment '${ENV_NAME}'."
else
    echo "Creating conda environment '${ENV_NAME}' with Python 3.10..."
    conda create --yes --name "${ENV_NAME}" python=3.10 pip
fi

# Activating inside the script configures installation only. Run
# `conda activate ${ENV_NAME}` in the caller's shell afterwards.
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

python -m pip install --upgrade "pip<25" setuptools wheel

core_requirements="$(mktemp)"
trap 'rm -f "${core_requirements}"' EXIT
grep -v '^flash-attn==' "${LOCK_FILE}" > "${core_requirements}"

echo "Installing the locked PyTorch/vLLM runtime..."
python -m pip install --requirement "${core_requirements}"

# Transformers 4.49 constrains huggingface_hub to <1.0.  Hub 0.28.1 therefore
# supplies `huggingface-cli`, not the newer `hf` executable.  Keep the locked
# library version and make the current command name available inside this
# Conda environment, rather than installing a second global CLI with a
# conflicting Hub dependency.
hf_legacy_cli="${CONDA_PREFIX}/bin/huggingface-cli"
hf_cli="${CONDA_PREFIX}/bin/hf"
[[ -x "${hf_legacy_cli}" ]] || fail "huggingface-cli was not installed by huggingface_hub."
if [[ ! -e "${hf_cli}" ]]; then
    ln -s "huggingface-cli" "${hf_cli}"
fi
[[ -x "${hf_cli}" ]] || fail "Could not create the hf command in ${CONDA_PREFIX}/bin."
"${hf_cli}" download --help >/dev/null
hf_hub_version="$(python -c 'from importlib.metadata import version; print(version("huggingface_hub"))')"
echo "Hugging Face CLI: hf compatibility launcher ready (huggingface_hub ${hf_hub_version})"

python - <<'PY'
from importlib.metadata import version

import torch
import vllm
import xformers
import pyarrow

assert torch.__version__.startswith("2.4.0+cu121"), torch.__version__
vllm_distribution_version = version("vllm")
assert vllm_distribution_version == "0.6.3", vllm_distribution_version
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
visible_gpus = torch.cuda.device_count()
assert visible_gpus >= 1, "PyTorch cannot see any GPU"
print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; GPUs: {visible_gpus}")
print(f"vLLM distribution: {vllm_distribution_version}; xFormers: {xformers.__version__}")
print(f"PyArrow: {pyarrow.__version__} (Parquet support: OK)")
PY

echo "Installing Ninja before building FlashAttention (MAX_JOBS=${MAX_JOBS})..."
# FlashAttention 2.5.8 does not publish a wheel for CPython 3.10 + PyTorch 2.4
# + CUDA 12.1.  Install Ninja first: otherwise pip only installs it after the
# build has started and FlashAttention falls back to slow serial distutils.
python -m pip install 'ninja==1.13.0'
command -v ninja >/dev/null 2>&1 || fail "Ninja was installed but is not on PATH."

echo "Building FlashAttention from source..."
python -m pip install --verbose --no-cache-dir --no-build-isolation 'flash-attn==2.5.8'
python - <<'PY'
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

print("flash-attn import: OK")
PY

echo "Checking the locked runtime before installing the repository..."
python -m pip check

# Do NOT run `pip install -r requirements.txt` in this environment. That file
# is intentionally broad and may upgrade torch/vLLM. The lock file above
# includes every dependency used by evaluate.sh; optional packages such as
# wandb, modelscope, and smolagents are not needed for console-only evaluation.
echo "Installing the repository without resolving its broad package metadata..."
python -m pip install --editable "${REPO_ROOT}" --no-deps
python - <<'PY'
from codetiming import Timer
from tools_server.util import MessageClient
from verl.trainer.main_ppo import main

print("InteractiveChat-R1 imports: OK")
PY

echo
echo "Environment '${ENV_NAME}' is ready. Activate it with: conda activate ${ENV_NAME}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "The environment check used CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
else
    echo "The environment check used all GPUs visible to this shell."
fi
