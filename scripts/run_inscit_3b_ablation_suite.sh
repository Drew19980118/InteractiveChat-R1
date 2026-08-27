#!/usr/bin/env bash
# Run the three requested InsCiT-3B user-centric ablations serially.  Every
# child keeps the formal 128-context / 8-rollout update configuration and
# performs final full online validation plus its own metric summary.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "===== [1/3] w/o satisfaction reward ====="
bash "$SCRIPT_DIR/run_inscit_3b_wo_satisfaction_reward_train.sh"

echo "===== [2/3] w/o user feedback (dynamic context; validation satisfaction probe) ====="
bash "$SCRIPT_DIR/run_inscit_3b_wo_user_feedback_train.sh"

echo "===== [3/3] w/o patience penalty ====="
bash "$SCRIPT_DIR/run_inscit_3b_wo_patience_penalty_train.sh"

echo "===== All three InsCiT-3B user-centric ablations completed ====="
