#!/usr/bin/env bash
# Evaluate the static-context / no-feedback answer-F1 control online.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_inscit_3b_static_uci_no_feedback_val.sh"
