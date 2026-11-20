#!/usr/bin/env bash
# Path-aware B2W playback launcher for the isolated workspace /home/yichangfeng/sru-path-aware.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="$SCRIPT_DIR/IsaacLab"
CONDA_ENV_DIR="/home/yichangfeng/miniforge3/envs/sru-pathaware"
PLAY_SCRIPT="$SCRIPT_DIR/sru-navigation-sim/scripts/play.py"

TASK="${TASK:-Isaac-Nav-PPO-B2W-Play-v0}"
CHECKPOINT=""
NUM_ENVS=1
MAX_STEPS=""
HEADLESS="${HEADLESS:-0}"
VIDEO=0
VIDEO_LENGTH=200
EXTRA_ARGS=()

# No display means headless playback.
if [[ -z "${DISPLAY:-}" ]]; then
  HEADLESS=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)        CHECKPOINT="$2"; shift 2 ;;
    --task)              TASK="$2"; shift 2 ;;
    --num_envs|--envs)   NUM_ENVS="$2"; shift 2 ;;
    --max_steps|--steps) MAX_STEPS="$2"; shift 2 ;;
    --headless)          HEADLESS=1; shift ;;
    --gui)               HEADLESS=0; shift ;;
    --video)             VIDEO=1; HEADLESS=1; shift ;;
    --video_length)      VIDEO_LENGTH="$2"; shift 2 ;;
    --*)                 EXTRA_ARGS+=("$1"); shift ;;
    *)                   EXTRA_ARGS+=("$1"); shift ;;
  esac
done

export TERM=xterm
export CONDA_PREFIX="$CONDA_ENV_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"

cd "$ISAACLAB_DIR"

CMD=(./isaaclab.sh -p "$PLAY_SCRIPT" --task "$TASK" --num_envs "$NUM_ENVS")
[[ -n "$CHECKPOINT" ]] && CMD+=(--checkpoint "$CHECKPOINT")
[[ -n "$MAX_STEPS" ]] && CMD+=(--max_steps "$MAX_STEPS")
[[ "$HEADLESS" == "1" ]] && CMD+=(--headless)
[[ "$VIDEO" == "1" ]] && CMD+=(--video --video_length "$VIDEO_LENGTH")
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && CMD+=("${EXTRA_ARGS[@]}")

echo "============================================================"
echo " Workspace : $SCRIPT_DIR"
echo " Conda env : $CONDA_ENV_DIR"
echo " Task      : $TASK"
echo " Checkpoint: ${CHECKPOINT:-(none)}"
echo " Headless  : $HEADLESS"
echo " Command   : ${CMD[*]}"
echo "============================================================"

exec "${CMD[@]}"
