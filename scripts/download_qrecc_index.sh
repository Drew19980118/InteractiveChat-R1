#!/usr/bin/env bash

# Download the complete QReCC passage collection and E5 FAISS index without
# modifying InteractiveChat-R1's locked training/evaluation Conda environment. The download
# is resumable: retain ${QRECC_LOCAL_DIR}/.cache/huggingface between runs.

set -euo pipefail

readonly DATASET_REPO="DrewZhang/qrecc-passages-index"
readonly EXPECTED_PARQUET_SHARDS=50
readonly EXPECTED_INDEX_SHARDS=4
readonly MIN_FREE_GB=500

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DOWNLOAD_ENV="${QRECC_DOWNLOAD_ENV:-hf-download}"
QRECC_LOCAL_DIR="${QRECC_LOCAL_DIR:-${REPO_ROOT}/collection/qrecc}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "$(uname -s)" == "Linux" ]] || fail "This downloader is intended for the remote Linux server."
command -v conda >/dev/null 2>&1 || fail "conda was not found in PATH."
command -v df >/dev/null 2>&1 || fail "df was not found in PATH."

mkdir -p "${QRECC_LOCAL_DIR}"

# Keep Hub/Xet metadata off a potentially small home-directory filesystem. Xet
# chunk caching remains disabled by default, so this does not duplicate 210 GB
# of downloaded files.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-download-cache}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
mkdir -p "${HF_HOME}"

# Set QRECC_SEQUENTIAL_WRITES=1 only when collection/qrecc is on an HDD.
if [[ "${QRECC_SEQUENTIAL_WRITES:-0}" == "1" ]]; then
    export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1
fi

available_kb="$(df -Pk "${QRECC_LOCAL_DIR}" | awk 'NR == 2 {print $4}')"
[[ "${available_kb}" =~ ^[0-9]+$ ]] || fail "Could not determine free disk space for ${QRECC_LOCAL_DIR}."
required_kb="$((MIN_FREE_GB * 1024 * 1024))"
if (( available_kb < required_kb )); then
    fail "${QRECC_LOCAL_DIR} has only $((available_kb / 1024 / 1024)) GB free; need at least ${MIN_FREE_GB} GB."
fi

if [[ -z "${TMUX:-}" ]]; then
    echo "WARNING: Not running inside tmux. For a long remote download, use:"
    echo "  tmux new -s qrecc-download"
fi

if ! conda run --name "${DOWNLOAD_ENV}" python --version >/dev/null 2>&1; then
    echo "Creating isolated Conda environment '${DOWNLOAD_ENV}'..."
    conda create --yes --name "${DOWNLOAD_ENV}" python=3.10 pip
fi

echo "Installing the current Hugging Face downloader in '${DOWNLOAD_ENV}'..."
conda run --no-capture-output --name "${DOWNLOAD_ENV}" \
    python -m pip install --upgrade "huggingface_hub>=0.32"

echo "Checking hf_xet availability..."
conda run --no-capture-output --name "${DOWNLOAD_ENV}" python - <<'PY'
from importlib.metadata import version

import hf_xet  # noqa: F401

print(f"huggingface_hub={version('huggingface_hub')}; hf_xet=available")
PY

echo "Downloading ${DATASET_REPO} to ${QRECC_LOCAL_DIR}..."
echo "This is 55 files / approximately 209.7 GB. Existing partial files will be resumed."
conda run --no-capture-output --name "${DOWNLOAD_ENV}" \
    hf download "${DATASET_REPO}" \
    --repo-type dataset \
    --local-dir "${QRECC_LOCAL_DIR}" \
    --max-workers "${HF_MAX_WORKERS}"

echo "Verifying that no files remain to download..."
dry_run_output="$(conda run --no-capture-output --name "${DOWNLOAD_ENV}" \
    hf download "${DATASET_REPO}" \
    --repo-type dataset \
    --local-dir "${QRECC_LOCAL_DIR}" \
    --dry-run \
    --format human)"
printf '%s\n' "${dry_run_output}"
grep -q "Will download 0 files" <<<"${dry_run_output}" || fail "The Hub reports unfinished files; rerun this script to resume."

parquet_count="$(find "${QRECC_LOCAL_DIR}" -maxdepth 1 -type f -name 'part_*.parquet' | wc -l | tr -d '[:space:]')"
index_count="$(find "${QRECC_LOCAL_DIR}" -maxdepth 1 -type f -name 'e5_Flat.index.part_*' | wc -l | tr -d '[:space:]')"
[[ "${parquet_count}" == "${EXPECTED_PARQUET_SHARDS}" ]] || fail "Expected ${EXPECTED_PARQUET_SHARDS} parquet shards, found ${parquet_count}."
[[ "${index_count}" == "${EXPECTED_INDEX_SHARDS}" ]] || fail "Expected ${EXPECTED_INDEX_SHARDS} index shards, found ${index_count}."

echo "QReCC download complete: ${parquet_count} parquet shards and ${index_count} index shards."
echo "Do not run scripts/download.py unless you explicitly need qrecc_index.jsonl."
echo "Do not concatenate e5_Flat.index.part_* unless your retrieval program requires a single FAISS index."
