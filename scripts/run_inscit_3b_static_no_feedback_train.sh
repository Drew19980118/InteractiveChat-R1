#!/usr/bin/env bash
# Static-context / no-feedback control: answer F1 + action + format only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_inscit_3b_static_uci_no_feedback_train.sh"
