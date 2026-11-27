#!/usr/bin/env python3
"""Evaluate a navigation checkpoint episode-by-episode and report SR / SPL by terrain type.

Stage 0.5 of `docs/specs/agent_doing.md`: the acceptance baseline. The training logs report a rolling
success rate produced by a *different* termination rule (latched "ever touched the goal"), so any
comparison against the old `Metrics/robot_goal/success_rate` is meaningless after the switch to
continuous dwell. This script measures the policy under the rules the environment is configured
with now, bucketed by terrain type, so before/after comparisons are apples to apples.

Usage:
    python scripts/eval_policy.py --task Isaac-Nav-MDPO-B2W-Play-v0 \
        --checkpoint <path/to/model_6000.pt> --episodes 200 --out docs/evals/phase0_baseline_60s_dwell.md \
        [--episode_length_s 90] [--policy_idx 1]

Outputs a markdown table plus a JSON blob with the raw per-episode records.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Episode-level evaluation of a navigation policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--policy_idx", type=int, default=1, choices=(1, 2))
parser.add_argument("--max_steps", type=int, default=20000, help="Safety cap on total steps.")
parser.add_argument("--terrain_rows", type=int, default=4, help="Terrain rows for the evaluation scene.")
parser.add_argument("--terrain_cols", type=int, default=12, help="Terrain cols for the evaluation scene.")
parser.add_argument("--out", type=str, default=None, help="Markdown report path.")
parser.add_argument(
    "--no_debug_vis",
    action="store_true",
    default=True,
    help="Disable command debug markers (default on: they need Nucleus assets and add nothing to an offline eval).",
)
parser.add_argument(
    "--debug_vis", dest="no_debug_vis", action="store_false", help="Keep command debug markers enabled."
)
parser.add_argument(
    "--legacy_env",
    action="store_true",
    default=False,
    help=(
        "Restore the pre-redesign environment (60-dim global path only, legacy path pipeline, "
        "[0,0,0.5,0.5] category mix) so a checkpoint trained before the redesign can be evaluated. "
        "The termination rule is NOT reverted - measuring it under the current rule is the point."
    ),
)
parser.add_argument("--dump_json", type=str, default=None, help="Per-episode records (JSON) path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import isaaclab_nav_task  # noqa: F401,E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

TILE_NAMES = ["maze", "non_maze", "stairs", "pits"]


def yaw_of_quat(quat_w: torch.Tensor) -> torch.Tensor:
    """Yaw (rad) from a (w, x, y, z) quaternion batch."""
    import math  # noqa: PLC0415

    w, x, y, z = quat_w.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    import math  # noqa: PLC0415

    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_checkpoint(runner: OnPolicyRunner, checkpoint_path: str):
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if runner.is_mdpo:
        runner.alg.actor_critic_1.load_state_dict(loaded["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(loaded["model_state_dict"], strict=True)
    else:
        runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"], strict=True)
    runner.current_learning_iteration = loaded["iter"]


def classify_tiles(env) -> list:
    """Approximate sub-terrain label per tile, read off its height field.

    Isaac Lab assigns sub-terrains *randomly* per tile when `curriculum=False` and does not record
    which one a tile got, so `terrain_types` (the column index) says nothing about the terrain. The
    height field does: platforms are a 1.0 m plateau (stairs), pits are below ground, and the maze is
    wall-dense. Labels are therefore: 'stairs' > 'pits' > 'maze'/'non_maze' by wall density.
    """
    heights = getattr(env.unwrapped.scene.terrain, "_height_field_visual", None)
    if heights is None:
        return ["all"]
    labels = []
    for tile in heights:
        h = tile.cpu().numpy()
        platform_frac = float(((h >= 195) & (h <= 205)).mean())
        pit_frac = float((h < 0).mean())
        wall_frac = float((h > 10).mean())
        if platform_frac >= 0.03:
            labels.append("stairs")
        elif pit_frac >= 0.01:
            labels.append("pits")
        elif wall_frac >= 0.12:
            labels.append("maze")
        else:
            labels.append("non_maze")
    return labels


class EpisodeTracker:
    """Records one record per finished episode, without touching the environment's own bookkeeping."""

    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped
        cmd = self.unwrapped.command_manager.get_term("robot_goal")
        self.cmd = cmd
        self.tile_labels = classify_tiles(env)
        n = self.unwrapped.num_envs
        self.records = []
        self._prev_len = torch.zeros(n, dtype=torch.long, device=self.unwrapped.device)
        self._min_dist = torch.full((n,), float("inf"), device=self.unwrapped.device)
        self._d_path_samples = [[] for _ in range(n)]
        self._path_len = torch.zeros(n, device=self.unwrapped.device)
        self._start_dist = torch.zeros(n, device=self.unwrapped.device)
        self._tile = ["all"] * n
        self._prev_steps_at_goal = torch.zeros(n, device=self.unwrapped.device)
        # In-goal behaviour diagnostics (Q1 of the 2026-11-26 analysis): does the policy *leave* the
        # radius because it cannot brake (high speed, short runs) or because staying near the goal
        # pays better than completing (low speed, streaks that stop just short of the dwell)?
        self._in_goal_speeds = [[] for _ in range(n)]
        self._in_goal_steps = torch.zeros(n, device=self.unwrapped.device)
        self._run = torch.zeros(n, device=self.unwrapped.device)
        self._run_max = torch.zeros(n, device=self.unwrapped.device)
        # P0 diagnostics (D4-D8): heading offset at spawn, freeze / reverse time, path snapshot at
        # spawn (for invalid-segment bucketing and stair dz), paid metres, pitch on stairs.
        self._freeze_steps = torch.zeros(n, device=self.unwrapped.device)
        self._reverse_steps = torch.zeros(n, device=self.unwrapped.device)
        self._freeze_run = torch.zeros(n, device=self.unwrapped.device)
        self._d_path_at_freeze = torch.zeros(n, device=self.unwrapped.device)
        self._paid_m = torch.zeros(n, device=self.unwrapped.device)
        self._pitch = [[] for _ in range(n)]
        self._head_err0 = torch.full((n,), float("nan"), device=self.unwrapped.device)
        self._path_pts = None
        self._seg_pass = None
        self._pinch = None
        self._seg_gate = None

    @property
    def done(self):
        return len(self.records) >= args_cli.episodes

    def observe(self):
        cmd = self.cmd
        device = self.unwrapped.device
        xy_err = torch.norm(cmd.goal_position_world[:, :2] - self.unwrapped.scene["robot"].data.root_pos_w[:, :2], dim=1)
        self._min_dist = torch.minimum(self._min_dist, xy_err)

        dist_to_path, path_progress = cmd.get_path_metrics()
        for i in range(self.unwrapped.num_envs):
            if len(self._d_path_samples[i]) < 4000:
                self._d_path_samples[i].append(float(dist_to_path[i]))

        robot = self.unwrapped.scene["robot"]
        # P0 D5/D6/D7/D8 accumulators (reset-steps excluded, see below).
        speed_w = torch.norm(robot.data.root_state_w[:, 7:9], dim=1)
        vx_b = robot.data.root_lin_vel_b[:, 0]
        wz = torch.abs(robot.data.root_ang_vel_b[:, 2])
        qw, qx, qy, qz = robot.data.root_quat_w.unbind(-1)
        pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qx * qz), -1.0, 1.0))

        # In-goal speed / dwell-streak accumulation. Envs that just reset are excluded: for them the
        # state read here is already the new episode's spawn.
        fresh = self.unwrapped.episode_length_buf <= 1
        inside = (xy_err < 0.5) & ~fresh
        speed_2d = speed_w
        self._in_goal_steps += inside.float()
        self._run = torch.where(inside, self._run + 1.0, torch.zeros_like(self._run))
        self._run_max = torch.maximum(self._run_max, self._run)
        for i in torch.where(inside)[0].tolist():
            self._in_goal_speeds[i].append(float(speed_2d[i]))

        live = ~fresh
        frozen = (speed_w < 0.1) & (wz < 0.15) & live
        self._freeze_steps += frozen.float()
        self._reverse_steps += ((vx_b < -0.1) & live).float()
        self._freeze_run = torch.where(frozen, self._freeze_run + 1.0, torch.zeros_like(self._freeze_run))
        # d_path at the onset of the first sustained freeze (D6): was the payment gate already shut?
        onset = (self._freeze_run == 3.0) & (self._d_path_at_freeze == 0.0)
        self._d_path_at_freeze = torch.where(onset, dist_to_path, self._d_path_at_freeze)
        self._paid_m += torch.where(live, path_progress.clamp(min=0.0), torch.zeros_like(path_progress))
        for i in torch.where(live & (torch.abs(pitch) > 0.0))[0].tolist():
            if len(self._pitch[i]) < 4000:
                self._pitch[i].append(float(pitch[i]))

        types = self.unwrapped.scene.terrain.terrain_types
        levels = self.unwrapped.scene.terrain.terrain_levels
        num_cols = self.unwrapped.cfg.scene.terrain.terrain_generator.num_cols
        tile_ids = (levels * num_cols + types).tolist()
        new_episode = self.unwrapped.episode_length_buf < self._prev_len
        if new_episode.any():
            # Records use the state captured *before* this step's reset, which `observe` sees because
            # it is called at the end of every step (see the call site).
            for env_id in torch.where(new_episode)[0].tolist():
                self._record(env_id, device)
                self._reset_env_state(env_id, device)

        # refresh per-episode baselines for envs starting a new episode
        started = self.unwrapped.episode_length_buf <= 1
        if started.any():
            ids = torch.where(started)[0]
            self._min_dist[ids] = xy_err[ids]
            self._start_dist[ids] = torch.norm(
                cmd.goal_position_world[ids, :2] - self.unwrapped.scene["robot"].data.root_pos_w[ids, :2], dim=1
            )
            # Planned path length (sum of the resampled waypoint segments). Reported next to the
            # straight-line distance to show how much the planned route detours; the SPL denominator
            # is the distance actually travelled (see `_prev_traveled`).
            paths_w = cmd.paths_manager.paths_w
            self._path_len[ids] = torch.norm(paths_w[ids, 1:] - paths_w[ids, :-1], dim=-1).sum(dim=1)
            # P0 snapshots: the planned path + its segment masks (offline D3 bucketing / D3b stair
            # dz), and the heading offset between the robot and the path's first tangent (D4).
            n_pts = paths_w.shape[1]
            if self._path_pts is None or self._path_pts.shape[1] != n_pts:
                nx = self.unwrapped.num_envs
                self._path_pts = torch.zeros(nx, n_pts, 3, device=device)
                self._seg_pass = torch.zeros(nx, n_pts - 1, dtype=torch.bool, device=device)
                self._pinch = torch.zeros(nx, n_pts - 1, dtype=torch.bool, device=device)
                self._seg_gate = torch.zeros(nx, n_pts - 1, device=device)
            pm = cmd.paths_manager
            self._path_pts[ids] = paths_w[ids]
            self._seg_pass[ids] = pm.seg_passable[ids]
            self._pinch[ids] = pm.pinch_mask[ids]
            self._seg_gate[ids] = pm.seg_gate[ids]
            tangent = paths_w[ids, 1, :2] - paths_w[ids, 0, :2]
            self._head_err0[ids] = wrap_angle(
                torch.atan2(tangent[:, 1], tangent[:, 0]) - yaw_of_quat(robot.data.root_quat_w[ids])
            ).abs()
            for env_id in ids.tolist():
                tile = tile_ids[env_id]
                self._tile[env_id] = self.tile_labels[tile] if 0 <= tile < len(self.tile_labels) else "all"
            self._prev_steps_at_goal[ids] = 0.0

        self._prev_steps_at_goal = cmd.steps_at_goal.clone()
        # Distance actually driven during the episode - the SPL denominator. Captured here (and not
        # read at record time) because the command resets it inside the same `env.step` that ends the
        # episode.
        self._prev_traveled = cmd.total_distance_traveled.clone()
        self._prev_len = self.unwrapped.episode_length_buf.clone()

    def _record(self, env_id: int, device):
        samples = self._d_path_samples[env_id]
        steps = int(self._prev_len[env_id])
        # `_prev_len` holds the last value of `episode_length_buf` before the reset, i.e. the last step
        # index: a full-horizon episode records `max_episode_length - 1`, not `max_episode_length`.
        # Comparing against the maximum therefore marks *every* timeout as "ended early", which silently
        # turned this metric into the legacy latched rule (verified: sr == sr_old_rule on model_3999).
        full_horizon = int(self.unwrapped.max_episode_length) - 1
        ended_early = steps < full_horizon
        # Completion is only observable indirectly: the step that trips `at_goal_navigation` resets the
        # env inside the same `env.step`, so `steps_at_goal`/`goal_dwell_completed` are already zeroed
        # here, and the in-goal accumulation below skips it. An episode that ends early after
        # `required - 1` consecutive in-goal steps therefore did complete the dwell (the missing step is
        # the completing one).
        required = int(self.cmd.required_steps_at_goal)
        completed_dwell = ended_early and int(self._run_max[env_id]) >= required - 1
        success = completed_dwell
        self.records.append(
            {
                "success": bool(success),
                "completed_dwell": bool(completed_dwell),
                "ended_early": bool(ended_early),
                "tile": self._tile[env_id],
                "start_dist": float(self._start_dist[env_id]),
                "path_len": float(self._path_len[env_id]),
                "traveled": float(self._prev_traveled[env_id]),
                "min_dist": float(self._min_dist[env_id]),
                "d_path_median": float(np.median(samples)) if samples else float("nan"),
                "d_path_p90": float(np.percentile(samples, 90)) if samples else float("nan"),
                "steps": steps,
                "in_goal_steps": int(self._in_goal_steps[env_id]),
                "in_goal_speed_p50": float(np.median(self._in_goal_speeds[env_id]))
                if self._in_goal_speeds[env_id]
                else float("nan"),
                "max_dwell_run": int(self._run_max[env_id]),
                "freeze_frac": float(self._freeze_steps[env_id]) / max(steps, 1),
                "reverse_frac": float(self._reverse_steps[env_id]) / max(steps, 1),
                "d_path_at_freeze": float(self._d_path_at_freeze[env_id]) if self._d_path_at_freeze[env_id] > 0 else None,
                "paid_m": float(self._paid_m[env_id]),
                "pitch_p95_deg": float(np.percentile(self._pitch[env_id], 95) * 180.0 / np.pi)
                if self._pitch[env_id]
                else float("nan"),
                "head_err0_deg": float(self._head_err0[env_id]) * 180.0 / 3.141592653589793
                if self._head_err0[env_id] == self._head_err0[env_id]
                else float("nan"),
                **self._path_diagnostics(env_id),
            }
        )
        self._d_path_samples[env_id] = []
        self._in_goal_speeds[env_id] = []
        self._in_goal_steps[env_id] = 0.0
        self._run[env_id] = 0.0
        self._run_max[env_id] = 0.0

    def _path_diagnostics(self, env_id: int) -> dict:
        """Per-episode path-quality stats from the spawn-time snapshot (P0 D3 / D3b).

        `n_invalid_corner` counts invalid segments that touch a vertex turning more than 45 degrees,
        `n_invalid_straight` the rest. `dir_z_max` / `dz_max_m` describe the steepest segment, which
        is what a staircase looks like in the path - if it approaches 1.0 the path really does point
        up like a wall there.
        """
        if self._path_pts is None:
            return {}
        pts = self._path_pts[env_id].cpu().numpy()
        ok = self._seg_pass[env_id].cpu().numpy()
        pinch = self._pinch[env_id].cpu().numpy()
        gate = self._seg_gate[env_id].cpu().numpy()
        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        dz = np.abs(seg[:, 2])
        # Vertex i (1..n-2) turns: angle between segment i-1 and segment i.
        corner = np.zeros(len(seg), dtype=bool)
        for i in range(1, len(seg)):
            a = seg[i - 1, :2] / max(seg_len[i - 1], 1e-6)
            b = seg[i, :2] / max(seg_len[i], 1e-6)
            if np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)) > np.pi / 4:
                corner[i - 1] = True
                corner[i] = True
        invalid = ~ok
        steep = int(np.argmax(dz))
        # How far below the threshold the invalid segments really are: re-ask `segment_passable`
        # with progressively smaller body radii. An invalid segment that passes at 0.25 m is a
        # discretisation-level false alarm; one that fails even at 0.0 is blocked or leaves the map.
        n_height = 0
        clearance_min = float("nan")
        try:
            gp = self.cmd.paths_manager.paths[env_id]
            prm = gp.prm
            tile_pts = np.array(gp.path, dtype=np.float64)
            clears = []
            for i in np.where(invalid)[0]:
                best = None
                for thr in (0.31, 0.28, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0):
                    if prm.segment_passable(tile_pts[i], tile_pts[i + 1], thr):
                        best = thr
                        break
                if best is None:
                    n_height += 1
                else:
                    clears.append(best)
            clearance_min = float(min(clears)) if clears else float("nan")
        except Exception:  # diagnostics must never break the evaluation
            pass
        return {
            "invalid_frac": float(invalid.mean()),
            "pinch_frac": float(pinch.mean()),
            "n_invalid": int(invalid.sum()),
            "n_invalid_corner": int((invalid & corner).sum()),
            "n_invalid_straight": int((invalid & ~corner).sum()),
            "n_invalid_height": n_height,
            "invalid_clearance_min": clearance_min,
            "gate_mean_m": float(gate.mean()),
            "dz_max_m": float(dz[steep]),
            "dir_z_max": float(dz[steep] / max(seg_len[steep], 1e-6)),
        }

    def _reset_env_state(self, env_id: int, device):
        self._min_dist[env_id] = float("inf")
        self._start_dist[env_id] = 0.0
        self._path_len[env_id] = 0.0
        self._freeze_steps[env_id] = 0.0
        self._reverse_steps[env_id] = 0.0
        self._freeze_run[env_id] = 0.0
        self._d_path_at_freeze[env_id] = 0.0
        self._paid_m[env_id] = 0.0
        self._pitch[env_id] = []
        self._head_err0[env_id] = float("nan")


def summarise(records):
    by_tile = defaultdict(list)
    for r in records:
        by_tile[r["tile"]].append(r)

    def agg(rows):
        n = len(rows)
        if n == 0:
            return None
        sr = sum(r["success"] for r in rows) / n
        # Legacy ("latched") success: ever came within 0.5 m during the episode, regardless of whether
        # it stopped or whether the episode ended there. Reported so the old logs' numbers stay
        # interpretable next to the new rule - never use it as the acceptance metric.
        sr_old = sum(1 for r in rows if r["min_dist"] < 0.5) / n
        # SPL, standard definition: success * (shortest-path length / distance actually travelled),
        # clamped at 1. The straight-line spawn-goal distance is used as the shortest path.
        spl_num = sum(
            r["success"] * min(r["start_dist"] / max(r["traveled"], 1e-6), 1.0) for r in rows
        )
        speeds = [r["in_goal_speed_p50"] for r in rows if r["in_goal_speed_p50"] == r["in_goal_speed_p50"]]
        return {
            "n": n,
            "sr": sr,
            "sr_old_rule": sr_old,
            "frac_timeout": sum(1 for r in rows if not r["ended_early"]) / n,
            "spl": spl_num / n,
            "d_path_median": statistics.median(r["d_path_median"] for r in rows),
            "d_path_p90": statistics.median(r["d_path_p90"] for r in rows),
            "start_dist": statistics.median(r["start_dist"] for r in rows),
            "path_len": statistics.median(r["path_len"] for r in rows),
            "traveled": statistics.median(r["traveled"] for r in rows),
            "in_goal_steps": statistics.median(r["in_goal_steps"] for r in rows),
            "in_goal_speed_p50": statistics.median(speeds) if speeds else float("nan"),
            "max_dwell_run_p50": statistics.median(r["max_dwell_run"] for r in rows),
            "max_dwell_run_p90": float(np.percentile([r["max_dwell_run"] for r in rows], 90)),
        }

    return {"all": agg(records), **{tile: agg(rows) for tile, rows in by_tile.items()}}


def main():
    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs.get("env_cfg_entry_point")()
    agent_cfg = spec.kwargs.get("rsl_rl_cfg_entry_point")()

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    # The Play config shrinks the terrain to 2x2 tiles, which is far too few to bucket results by
    # terrain type. Use a wider grid (the sub-terrain mix stays the same proportions).
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = args_cli.terrain_rows
        env_cfg.scene.terrain.terrain_generator.num_cols = args_cli.terrain_cols
    if args_cli.no_debug_vis:
        for term_cfg in env_cfg.commands.__dict__.values():
            if getattr(term_cfg, "debug_vis", False):
                term_cfg.debug_vis = False

    if args_cli.legacy_env:
        # A checkpoint trained before the redesign expects 60 path floats and the old category mix;
        # feeding it the 120-float layout would simply fail to load. Everything except the
        # termination rule is rolled back, because the *rule* is what the baseline is re-measured for.
        for group in (env_cfg.observations.policy, env_cfg.observations.critic):
            group.__dict__.pop("distance_to_path", None)
            group.__dict__.pop("target_path_local", None)
        env_cfg.commands.robot_goal.path_cfg.path_pipeline = "legacy"
        env_cfg.commands.robot_goal.path_cfg.category_weights = [0.0, 0.0, 0.5, 0.5]
        agent_cfg.policy.path_length = 60
        print("[eval] legacy env: 60-dim path obs, legacy pipeline, [0,0,0.5,0.5] categories", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_checkpoint(runner, args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device, policy_index=args_cli.policy_idx)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    if hasattr(obs, "keys"):
        obs = obs["policy"]

    tracker = EpisodeTracker(env)
    steps = 0
    while simulation_app.is_running() and not tracker.done and steps < args_cli.max_steps:
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        if isinstance(obs, tuple):
            obs = obs[0]
        if hasattr(obs, "keys"):
            obs = obs["policy"]
        tracker.observe()
        steps += 1

    summary = summarise(tracker.records)
    print(json.dumps(summary, indent=2))

    if args_cli.dump_json:
        with open(args_cli.dump_json, "w") as f:
            json.dump(tracker.records, f, indent=1)

    if args_cli.out:
        lines = [
            "# Phase 0 基线重评",
            "",
            f"- checkpoint: `{os.path.basename(args_cli.checkpoint)}`",
            f"- horizon: {env_cfg.episode_length_s:.0f} s,  episodes: {len(tracker.records)},  actor: {args_cli.policy_idx}",
            "",
            "| bucket | n | SR | SR(old rule) | timeout share | SPL | median d_path | p90 d_path | median start dist | median planned len | median travelled | in-goal steps | in-goal speed p50 | longest dwell run p50 / p90 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for key in ["all"] + [t for t in TILE_NAMES if t in summary]:
            row = summary[key]
            if row is None:
                continue
            lines.append(
                f"| {key} | {row['n']} | {row['sr']:.3f} | {row['sr_old_rule']:.3f} | {row['frac_timeout']:.3f} | {row['spl']:.3f} | "
                f"{row['d_path_median']:.2f} | {row['d_path_p90']:.2f} | {row['start_dist']:.1f} | "
                f"{row['path_len']:.1f} | {row['traveled']:.1f} | {row['in_goal_steps']:.0f} | "
                f"{row['in_goal_speed_p50']:.2f} | {row['max_dwell_run_p50']:.0f} / {row['max_dwell_run_p90']:.0f} |"
            )
        os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
        with open(args_cli.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[eval] report written to {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
