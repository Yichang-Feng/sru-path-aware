#!/usr/bin/env bash
# Path-aware B2W training launcher for the isolated workspace /home/yichangfeng/sru-path-aware.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="$SCRIPT_DIR/IsaacLab"
# Python environment. Resolution order:
#   1. $CONDA_ENV_DIR if the caller set it
#   2. the currently activated conda env ($CONDA_PREFIX)
#   3. the path on the machine this workspace was developed on
# Hardcoding only (3) breaks every other clone (different user, different env name).
if [[ -z "${CONDA_ENV_DIR:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    CONDA_ENV_DIR="$CONDA_PREFIX"
  else
    CONDA_ENV_DIR="/home/yichangfeng/miniforge3/envs/sru-pathaware"
  fi
fi
if [[ ! -x "$CONDA_ENV_DIR/bin/python" ]]; then
  echo "[ERROR] No Python at '$CONDA_ENV_DIR/bin/python'." >&2
  echo "        Activate the conda env first (conda activate <env>), or run:" >&2
  echo "        CONDA_ENV_DIR=/path/to/env $0 ..." >&2
  exit 1
fi
TRAIN_SCRIPT="$SCRIPT_DIR/sru-navigation-sim/scripts/train.py"

TASK="${TASK:-Isaac-Nav-PPO-B2W-v0}"
NUM_ENVS=""
MAX_ITER=""
SEED=""
RUN_NAME=""
HEADLESS="${HEADLESS:-0}"
VIDEO=0
EXTRA_ARGS=()

# No display means headless training.
if [[ -z "${DISPLAY:-}" ]]; then
  HEADLESS=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)             TASK="$2"; shift 2 ;;
    --num_envs|--envs)  NUM_ENVS="$2"; shift 2 ;;
    --max_iterations|--iter) MAX_ITER="$2"; shift 2 ;;
    --seed)             SEED="$2"; shift 2 ;;
    --run_name|--name)  RUN_NAME="$2"; shift 2 ;;
    --headless)         HEADLESS=1; shift ;;
    --gui)              HEADLESS=0; shift ;;
    --video)            VIDEO=1; HEADLESS=1; shift ;;
    --*)                EXTRA_ARGS+=("$1"); shift ;;
    *)                  EXTRA_ARGS+=("$1"); shift ;;
  esac
done

export TERM=xterm
export CONDA_PREFIX="$CONDA_ENV_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
# The conda env has `rsl_rl` installed editable from another checkout
# (~/sru-path-aware-official). Put this workspace's copy first on the path so edits made here - the
# MDPO log_std schedule, the extra logging - are the ones that actually run. Before this line,
# `import rsl_rl` silently resolved to the other checkout.
export PYTHONPATH="$SCRIPT_DIR/sru-navigation-learning:$SCRIPT_DIR/sru-navigation-sim${PYTHONPATH:+:$PYTHONPATH}"

cd "$ISAACLAB_DIR"

CMD=(./isaaclab.sh -p "$TRAIN_SCRIPT" --task "$TASK")
[[ -n "$NUM_ENVS" ]] && CMD+=(--num_envs "$NUM_ENVS")
[[ -n "$MAX_ITER" ]] && CMD+=(--max_iterations "$MAX_ITER")
[[ -n "$SEED" ]] && CMD+=(--seed "$SEED")
[[ -n "$RUN_NAME" ]] && CMD+=(--run_name "$RUN_NAME")
[[ "$HEADLESS" == "1" ]] && CMD+=(--headless)
[[ "$VIDEO" == "1" ]] && CMD+=(--video)
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && CMD+=("${EXTRA_ARGS[@]}")

echo "============================================================"
echo " Workspace : $SCRIPT_DIR"
echo " Conda env : $CONDA_ENV_DIR"
echo " Task      : $TASK"
echo " Headless  : $HEADLESS"
echo " Command   : ${CMD[*]}"
echo "============================================================"

exec "${CMD[@]}"
