#!/usr/bin/env bash
# SwanLab setup for the Feedback-GRPO launcher.
# The caller must define PROJECT_ROOT and activate the training environment.

configure_swanlab() {
  SWANLAB_ENABLED="${SWANLAB_ENABLED:-false}"
  case "$SWANLAB_ENABLED" in
    true)
      if ! python -c 'import swanlab; print("SwanLab SDK", getattr(swanlab, "__version__", "available"))'; then
        echo "ERROR: SwanLab is enabled but is not installed in this environment." >&2
        echo "Install it with: python -m pip install swanlab" >&2
        exit 2
      fi

      # Some SwanLab SDK releases treat SWANLAB_PROJECT as a structured
      # settings value and fail while parsing a plain project-name string.
      # Keep our launcher input in SWANLAB_PROJ_NAME instead.
      export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-${SWANLAB_PROJECT:-interactivechat-r1}}"
      unset SWANLAB_PROJECT
      export SWANLAB_RUN_GROUP="${SWANLAB_RUN_GROUP:-inscit_feedback_grpo_3b}"
      export SWANLAB_MODE="${SWANLAB_MODE:-online}"
      export SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-${SWANLAB_LOG_DIR:-$PROJECT_ROOT/swanlog}}"
      mkdir -p "$SWANLAB_LOGDIR"
      echo "[SwanLab] enabled: project=$SWANLAB_PROJ_NAME group=$SWANLAB_RUN_GROUP mode=$SWANLAB_MODE logdir=$SWANLAB_LOGDIR"
      ;;
    false)
      ;;
    *)
      echo "ERROR: SWANLAB_ENABLED must be true or false." >&2
      exit 2
      ;;
  esac
}
