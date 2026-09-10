#!/usr/bin/env bash
# Optional W&B setup shared by the baseline and Turn-PPO training launchers.
# The caller must set PROJECT_ROOT and activate its Python environment first.

configure_wandb() {
  WANDB_ENABLED="${WANDB_ENABLED:-false}"
  WANDB_LOG_VAL_GENERATIONS="${WANDB_LOG_VAL_GENERATIONS:-0}"
  case "$WANDB_ENABLED" in
    true)
      if ! [[ "$WANDB_LOG_VAL_GENERATIONS" =~ ^[0-9]+$ ]]; then
        echo "ERROR: WANDB_LOG_VAL_GENERATIONS must be a non-negative integer." >&2
        exit 2
      fi
      if ! python -c 'import wandb; print("W&B SDK", wandb.__version__)'; then
        echo "ERROR: W&B is enabled but is not installed in this environment." >&2
        echo "Install it with: python -m pip install wandb" >&2
        exit 2
      fi
      export WANDB_PROJECT="${WANDB_PROJECT:-interactivechat-r1}"
      export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-inscit_static_comparison}"
      export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/wandb}"
      mkdir -p "$WANDB_DIR"
      TRAINER_LOGGERS="['console','wandb']"
      echo "[W&B] enabled: project=$WANDB_PROJECT group=$WANDB_RUN_GROUP dir=$WANDB_DIR"
      echo "[W&B] validation generations/table rows per check: $WANDB_LOG_VAL_GENERATIONS"
      ;;
    false)
      TRAINER_LOGGERS="['console']"
      ;;
    *)
      echo "ERROR: WANDB_ENABLED must be true or false." >&2
      exit 2
      ;;
  esac
}
