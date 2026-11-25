#!/usr/bin/env bash
# Path-aware Go2W training launcher for the isolated workspace /home/yichangfeng/sru-path-aware.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="$SCRIPT_DIR/IsaacLab"
# Python environment. `isaaclab.sh` itself resolves the interpreter as
#   $CONDA_PREFIX/bin/python -> $VIRTUAL_ENV/bin/python -> IsaacLab/_isaac_sim/python.sh
# so the one thing we must not do is clobber CONDA_PREFIX with a wrong path while a perfectly good
# environment is already active - that is what broke on a plain virtualenv, where exporting a stale
# CONDA_PREFIX shadowed the valid VIRTUAL_ENV.
#
# Precedence: $CONDA_ENV_DIR (explicit) > activated venv > activated conda env > the path of
# the machine this workspace was developed on. A `python` merely present on PATH is NOT used:
# it is usually a base env without Isaac Sim, which fails much later and less clearly.
ACTIVE_ENV=""
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  ACTIVE_ENV="$VIRTUAL_ENV"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  ACTIVE_ENV="$CONDA_PREFIX"
fi

if [[ -n "${CONDA_ENV_DIR:-}" ]]; then
  ENV_DIR="$CONDA_ENV_DIR"
elif [[ -n "$ACTIVE_ENV" ]]; then
  ENV_DIR="$ACTIVE_ENV"
else
  ENV_DIR="/home/yichangfeng/miniforge3/envs/sru-pathaware"
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "[ERROR] No Python at '$ENV_DIR/bin/python'." >&2
  echo "        Activate your environment first (conda activate / source venv/bin/activate), or run:" >&2
  echo "        CONDA_ENV_DIR=/path/to/env $0 ..." >&2
  exit 1
fi
CONDA_ENV_DIR="$ENV_DIR"
TRAIN_SCRIPT="$SCRIPT_DIR/sru-navigation-sim/scripts/train.py"

TASK="${TASK:-Isaac-Nav-PPO-Go2W-PathAware-v0}"
NUM_ENVS=""
MAX_ITER=""
SEED=""
RUN_NAME=""
HEADLESS="${HEADLESS:-0}"
VIDEO=0
EXTRA_ARGS=()

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
export PATH="$ENV_DIR/bin:$PATH"
# Point CONDA_PREFIX at the chosen interpreter only when it is not already a working one;
# a venv keeps its VIRTUAL_ENV and must not be shadowed.
if [[ -z "$ACTIVE_ENV" || "$ENV_DIR" != "$ACTIVE_ENV" ]]; then
  export CONDA_PREFIX="$ENV_DIR"
fi
# The conda env has `rsl_rl` installed editable from another checkout
# (~/sru-path-aware-official). Put this workspace's copy first on the path so edits made here
# are the ones that actually run (see the same note in train_b2w_pathaware.sh).
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
