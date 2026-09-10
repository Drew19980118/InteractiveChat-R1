#!/usr/bin/env bash
# Run the latest ConvAgent 3B/7B then ChatR1 3B/7B QReCC lifecycles.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET=qrecc bash "$SCRIPT_DIR/run_latest_static_baselines_suite.sh"
