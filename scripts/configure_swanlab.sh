#!/usr/bin/env bash
# Optional SwanLab setup shared by the project training launchers.
# The caller must set PROJECT_ROOT and activate its Python environment first.

configure_swanlab() {
  SWANLAB_ENABLED="${SWANLAB_ENABLED:-true}"
  SWANLAB_LOG_VAL_GENERATIONS="${SWANLAB_LOG_VAL_GENERATIONS:-0}"
  case "$SWANLAB_ENABLED" in
    true)
      if ! [[ "$SWANLAB_LOG_VAL_GENERATIONS" =~ ^[0-9]+$ ]]; then
        echo "ERROR: SWANLAB_LOG_VAL_GENERATIONS must be a non-negative integer." >&2
        exit 2
      fi
      if ! python -c 'import swanlab; print("SwanLab SDK", swanlab.__version__)'; then
        echo "ERROR: SwanLab is enabled but is not installed in this environment." >&2
        echo "Install it with: python -m pip install -U swanlab" >&2
        exit 2
      fi
      export SWANLAB_PROJ_NAME="${SWANLAB_PROJECT:-interactivechat-r1}"
      export SWANLAB_GROUP="${SWANLAB_RUN_GROUP:-inscit_static_comparison}"
      export SWANLAB_MODE="${SWANLAB_MODE:-online}"
      export SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-$PROJECT_ROOT/swanlog}"
      mkdir -p "$SWANLAB_LOGDIR"
      TRAINER_LOGGERS="['console','swanlab']"
      echo "[SwanLab] enabled: project=$SWANLAB_PROJ_NAME group=$SWANLAB_GROUP mode=$SWANLAB_MODE logdir=$SWANLAB_LOGDIR"
      echo "[SwanLab] validation generations/text samples per check: $SWANLAB_LOG_VAL_GENERATIONS"
      ;;
    false)
      TRAINER_LOGGERS="['console']"
      ;;
    *)
      echo "ERROR: SWANLAB_ENABLED must be true or false." >&2
      exit 2
      ;;
  esac
}
