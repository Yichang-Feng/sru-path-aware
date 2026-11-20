#!/usr/bin/env bash
# One-shot editable install for the path-aware workspace.
# Prerequisites: conda env `sru-pathaware` with Isaac Sim 4.5 / IsaacLab 2.3.2 /
# PyTorch 2.7+cu126 already installed (see README).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-sru-pathaware}"
NAV_LINK="../../sru-navigation-sim"
NAV_TASK_PATH="$SCRIPT_DIR/IsaacLab/source/isaaclab_nav_task"

if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  CONDA_BIN="$(command -v conda)"
else
  echo "[ERROR] conda not found; install Miniforge first." >&2
  exit 1
fi

if ! "$CONDA_BIN" env list | grep -Eq "(^|[[:space:]])${ENV_NAME}([[:space:]]|$)"; then
  echo "[ERROR] conda env '$ENV_NAME' not found. Create it with Isaac Sim 4.5 / Python 3.10 / PyTorch 2.7+cu126 first." >&2
  exit 1
fi

run_py() {
  "$CONDA_BIN" run -n "$ENV_NAME" python "$@"
}

# Recreate the extension symlink if it is missing or broken (git may not preserve it on all platforms).
if [ -L "$NAV_TASK_PATH" ]; then
  if [ "$(readlink "$NAV_TASK_PATH")" != "$NAV_LINK" ]; then
    unlink "$NAV_TASK_PATH"
    ln -s "$NAV_LINK" "$NAV_TASK_PATH"
  fi
elif [ -e "$NAV_TASK_PATH" ]; then
  echo "[ERROR] $NAV_TASK_PATH exists but is not a symlink; inspect it manually." >&2
  exit 1
else
  ln -s "$NAV_LINK" "$NAV_TASK_PATH"
fi

echo "[INFO] Installing editable packages into conda env '$ENV_NAME'"
run_py -m pip install --no-deps -e "$SCRIPT_DIR/sru-navigation-learning"
run_py -m pip install --no-deps -e "$SCRIPT_DIR/sru-navigation-sim"
for m in isaaclab isaaclab_assets isaaclab_contrib isaaclab_mimic isaaclab_tasks isaaclab_rl; do
  echo "[INFO]   $m"
  run_py -m pip install --no-deps -e "$SCRIPT_DIR/IsaacLab/source/$m"
done

echo "[INFO] Verifying import paths"
run_py - <<'PY'
import isaaclab
import rsl_rl

print("rsl_rl  :", rsl_rl.__file__)
print("isaaclab:", isaaclab.__file__)
PY

echo "[INFO] Done. Example training commands:"
echo "  cd $SCRIPT_DIR && ./train_b2w_pathaware.sh --num_envs 32 --max_iterations 10 --headless"
echo "  cd $SCRIPT_DIR && ./train_go2w_pathaware.sh --num_envs 32 --max_iterations 10 --headless"
