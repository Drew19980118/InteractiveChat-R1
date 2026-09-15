#!/usr/bin/env bash
# Install the verified GPU FAISS binding for the locked Python-3.10 / CUDA-12.1
# InteractiveChat-R1 runtime.  Do not run while a retriever or training process
# is active in this environment.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: GPU FAISS installation is supported here only on Linux." >&2
  exit 2
fi
command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda was not found in PATH." >&2
  exit 2
}
[[ -n "${CONDA_PREFIX:-}" ]] || {
  echo "ERROR: activate the interactivechat-r1 Conda environment first." >&2
  exit 2
}

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"ERROR: expected Python 3.10, found {sys.version.split()[0]}. "
        "Use the locked interactivechat-r1 environment."
    )
PY

echo "Using Conda environment: $CONDA_PREFIX"
echo "Removing existing Conda FAISS packages, if present..."
mapfile -t conda_faiss_packages < <(
  conda list -p "$CONDA_PREFIX" | awk '$1 == "faiss" || $1 == "faiss-gpu" || $1 == "libfaiss" {print $1}'
)
if (( ${#conda_faiss_packages[@]} )); then
  conda remove -y -p "$CONDA_PREFIX" "${conda_faiss_packages[@]}"
fi

# A pip wheel at the same import path can shadow Conda's GPU bindings.  It is
# intentionally removed only after the Conda package removal above.
echo "Removing shadowing pip FAISS wheels, if present..."
python -m pip uninstall -y faiss-cpu faiss-gpu faiss faiss-gpu-cu12 || true

echo "Installing GPU FAISS 1.10.0 for CUDA 12.1 from the PyTorch channel..."
echo "Review the package plan: it must not remove torch, vllm, or flash-attn."
conda install -y \
  -p "$CONDA_PREFIX" \
  --override-channels \
  -c pytorch -c nvidia -c defaults \
  "faiss-gpu=1.10.0=py3.10_h4818125_0_cuda12.1.1"

FAISS_VISIBLE_GPUS="${FAISS_VISIBLE_GPUS:-0,1}"
FAISS_EXPECTED_GPUS="${FAISS_EXPECTED_GPUS:-2}"
CUDA_VISIBLE_DEVICES="$FAISS_VISIBLE_GPUS" \
FAISS_EXPECTED_GPUS="$FAISS_EXPECTED_GPUS" \
python - <<'PY'
import os
import faiss

expected = int(os.environ["FAISS_EXPECTED_GPUS"])
actual = faiss.get_num_gpus()
print("Faiss:", getattr(faiss, "__version__", "unknown"))
print("GPU API:", hasattr(faiss, "GpuMultipleClonerOptions"))
print("Visible GPUs:", actual)
assert hasattr(faiss, "GpuMultipleClonerOptions"), "GPU FAISS symbols are unavailable"
assert actual == expected, f"expected {expected} visible GPU(s), found {actual}"
print("GPU FAISS verification: OK")
PY
