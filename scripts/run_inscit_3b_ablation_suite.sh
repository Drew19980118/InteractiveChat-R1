#!/usr/bin/env bash
# Run the two requested InsCiT-3B ablations serially.  Each child launcher
# performs its own final full online validation and metric computation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "===== [1/2] Controlled UCI -> answer-F1 ablation ====="
bash "$SCRIPT_DIR/run_inscit_3b_uci_to_f1_train.sh"

echo "===== [2/2] Static-context / no-feedback UCI ablation ====="
bash "$SCRIPT_DIR/run_inscit_3b_static_uci_no_feedback_train.sh"

echo "===== Both InsCiT-3B ablations completed ====="
