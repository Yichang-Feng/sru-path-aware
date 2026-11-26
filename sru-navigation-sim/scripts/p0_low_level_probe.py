#!/usr/bin/env python3
"""P0 diagnostics D1 / D1b: can the *low-level* locomotion policy turn in place and climb stairs?

This bypasses the high-level navigation policy entirely: the environment's action is the 3-dim
velocity command `[vx, vy, omega]` that the low-level policy tracks (`PerceptiveNavigationSE2Action`,
`policy_scaling=[1.5, 1.0, 1.0]`, tanh -> scaling -> low-pass), so writing that vector directly is
exactly "the policy asked for this motion".

D1  - command omega in {0.3, 0.6, 1.0} rad/s with vx=vy=0 for 5 s; report the achieved yaw rate and
      the position drift. If 1.0 rad/s yields < 0.3 rad/s, the high-level can never learn to turn.
D1b - place the robot in front of a staircase found in the terrain, command forward at several
      speeds / approach angles, and report how much height it actually gained.

Terminations are disabled for the probe (no goal resets, no falls ending the episode), so a trial is
never silently cut short; the run duration is bounded by --steps.

Usage:
    python scripts/p0_low_level_probe.py --task Isaac-Nav-MDPO-B2W-Play-v0 --out docs/v1p1_p0_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Low-level turn / stair probes (P0 D1, D1b).")
parser.add_argument("--task", type=str, default="Isaac-Nav-MDPO-B2W-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--terrain_rows", type=int, default=3)
parser.add_argument("--terrain_cols", type=int, default=3)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--skip_stairs", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import isaaclab_nav_task  # noqa: F401,E402

STEP_DT = 0.2  # manager-based env step (policy dt), the low-level runs at 50 Hz inside it


def yaw_of(quat_w: torch.Tensor) -> torch.Tensor:
    """Yaw (rad) from a (w, x, y, z) quaternion batch."""
    w, x, y, z = quat_w.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def make_env():
    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs.get("env_cfg_entry_point")()
    env_cfg.scene.num_envs = args_cli.num_envs
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = args_cli.terrain_rows
        env_cfg.scene.terrain.terrain_generator.num_cols = args_cli.terrain_cols
    # No terminations: a goal reset or a fall ending the episode would silently truncate a trial.
    for name in ("time_out", "base_contact", "large_pitch_angle", "early_termination", "terrain_fall"):
        if getattr(env_cfg.terminations, name, None) is not None:
            setattr(env_cfg.terminations, name, None)
    for term_cfg in env_cfg.commands.__dict__.values():
        if getattr(term_cfg, "debug_vis", False):
            term_cfg.debug_vis = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    return env.unwrapped


def command_to_action(env, vx: float, vy: float, omega: float, scale=(1.5, 1.0, 1.0)) -> torch.Tensor:
    """Invert tanh(x) * scale -> x, so the *commanded* velocity is what we ask for."""
    raw = [max(-0.999, min(0.999, v / s)) for v, s in zip((vx, vy, omega), scale)]
    one = torch.tensor([[math.atanh(r) for r in raw]], dtype=torch.float32, device=env.device)
    return one.repeat(env.num_envs, 1)


def settle(env, steps: int = 10) -> None:
    zero = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(steps):
        env.step(zero)


def run_d1(env) -> list[dict]:
    robot = env.scene["robot"]
    results = []
    for omega in (0.3, 0.6, 1.0):
        env.reset()
        settle(env)
        yaw0 = yaw_of(robot.data.root_quat_w).clone()
        pos0 = robot.data.root_pos_w[:, :2].clone()
        act = command_to_action(env, 0.0, 0.0, omega)
        steps = int(round(5.0 / STEP_DT))
        prev = yaw0.clone()
        turned = torch.zeros_like(yaw0)
        for _ in range(steps):
            env.step(act)
            cur = yaw_of(robot.data.root_quat_w)
            turned += wrap(cur - prev).abs()
            prev = cur
        pos1 = robot.data.root_pos_w[:, :2]
        dyaw = turned / (steps * STEP_DT)
        drift = torch.norm(pos1 - pos0, dim=1)
        results.append(
            {
                "omega_cmd": omega,
                "yaw_rate_mean": float(dyaw.mean()),
                "yaw_rate_per_env": [round(float(v), 3) for v in dyaw],
                "drift_mean_m": float(drift.mean()),
                "drift_max_m": float(drift.max()),
            }
        )
        print(f"[D1] omega={omega:.1f} -> yaw_rate={float(dyaw.mean()):.3f} rad/s  drift={float(drift.mean()):.3f} m", flush=True)
    return results


def find_stairs_tile(terrain):
    """Template-match the StairConfig profile (5 risers of 40 units every 4 cells) in every tile.

    Returns (tile_index, axis, foot_index, row_index, height_map) where `foot_index` is the cell
    where the first riser starts, or None. Matching the profile directly is far more robust than
    median profiles: a tile also contains walls (300+ units) and pits (negative), and a median along
    the wrong axis just reports the wall.
    """
    heights = getattr(terrain, "_height_field_visual", None)
    if heights is None:
        print("[D1b] terrain exposes no _height_field_visual", flush=True)
        return None
    template = np.array([0, 40, 80, 120, 160, 200], dtype=np.int32)
    for idx, tile in enumerate(heights):
        h = tile.cpu().numpy().astype(np.int32) if hasattr(tile, "cpu") else np.asarray(tile, dtype=np.int32)
        for axis in (0, 1):
            arr = h if axis == 0 else h.T
            for row in range(arr.shape[0]):
                line = arr[row]
                for foot in range(0, len(line) - 26):
                    v = line[foot : foot + 21 : 4]
                    if not np.all(np.abs(v - template) <= 12):
                        continue
                    # the run must end on the 1 m platform, not just look like a ramp
                    if abs(int(line[foot + 24]) - 200) > 25:
                        continue
                    print(f"[D1b] stairs found: tile {idx} axis={axis} row={row} foot={foot}", flush=True)
                    return idx, axis, foot, row, h
    return None


def run_d1b(env) -> list[dict]:
    terrain = env.scene.terrain
    found = find_stairs_tile(terrain)
    if found is None:
        print("[D1b] no staircase tile found in this terrain - skipping", flush=True)
        return []
    tile_idx, axis, foot, perp_cell, h = found
    origins = terrain.terrain_origins  # (rows, cols, 3)
    n_cols = origins.shape[1]
    origin = origins[tile_idx // n_cols, tile_idx % n_cols].cpu().numpy().astype(np.float64)
    h_scale = 0.1
    print(
        f"[D1b] tile {tile_idx} origin={np.round(origin, 2)} axis={axis} foot={foot} perp_cell={perp_cell} "
        f"h_before={int(h[perp_cell, foot - 1]) if axis == 0 else int(h[foot - 1, perp_cell])} "
        f"h_top={int(h[perp_cell, foot + 24]) if axis == 0 else int(h[foot + 24, perp_cell])}",
        flush=True,
    )

    robot = env.scene["robot"]
    results = []
    for speed in (0.2, 0.5, 0.8):
        for angle_deg in (0.0, 15.0, 30.0):
            env.reset()
            settle(env, steps=15)
            # 1.5 m before the first riser, centred across the staircase, facing the steps.
            along = (foot - 15) * h_scale
            perp = perp_cell * h_scale
            # `arr = h` (axis 0) varies the SECOND index, `arr = h.T` (axis 1) the FIRST: `row` is
            # the fixed index, `foot` the one the staircase runs along.
            if axis == 0:
                x, y = origin[0] + perp, origin[1] + along
                yaw = math.pi / 2.0
            else:
                x, y = origin[0] + along, origin[1] + perp
                yaw = 0.0
            z = float(origin[2]) + 0.5
            yaw += math.radians(angle_deg)
            q = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
            pose = torch.tensor([[x, y, z, *q]] * env.num_envs, dtype=torch.float32, device=env.device)
            ids = torch.arange(env.num_envs, device=env.device)
            robot.write_root_pose_to_sim(pose, ids)
            robot.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device), ids)
            settle(env, steps=5)
            cell = (
                int(round((x - origin[0]) / h_scale)),
                int(round((y - origin[1]) / h_scale)),
            )
            cell = (min(max(cell[0], 0), h.shape[0] - 1), min(max(cell[1], 0), h.shape[1] - 1))
            ahead = (perp_cell, foot + 25) if axis == 0 else (foot + 25, perp_cell)
            print(f"[D1b]   placed at cell {cell} h={int(h[cell])}; cell 4 m ahead {ahead} h={int(h[ahead])}", flush=True)
            z0 = robot.data.root_pos_w[:, 2].clone()
            p0_xy = robot.data.root_pos_w[:, :2].clone()
            act = command_to_action(env, speed, 0.0, 0.0)
            steps = int(round(8.0 / STEP_DT))
            for _ in range(steps):
                env.step(act)
            z1 = robot.data.root_pos_w[:, 2]
            gain = (z1 - z0).mean()
            moved = torch.norm(robot.data.root_pos_w[:, :2] - p0_xy, dim=1).mean()
            results.append(
                {
                    "speed": speed,
                    "angle_deg": angle_deg,
                    "height_gain_mean_m": float(gain),
                    "height_gain_per_env": [round(float(v), 3) for v in (z1 - z0)],
                    "climbed": bool(gain > 0.6),
                    "moved_m": float(moved),
                }
            )
            print(
                f"[D1b] v={speed:.1f} angle={angle_deg:.0f}deg -> dz={float(gain):+.3f} m "
                f"moved={float(moved):.2f} m (per env {[round(float(v), 2) for v in (z1 - z0)]})",
                flush=True,
            )
    return results


def main():
    env = make_env()
    out = {}
    try:
        out["d1_turn_in_place"] = run_d1(env)
        if not args_cli.skip_stairs:
            out["d1b_stairs"] = run_d1b(env)
    finally:
        if args_cli.out:
            os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
            with open(args_cli.out, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[p0] written to {args_cli.out}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
