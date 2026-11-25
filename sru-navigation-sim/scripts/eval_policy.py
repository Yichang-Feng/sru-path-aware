#!/usr/bin/env python3
"""Evaluate a navigation checkpoint episode-by-episode and report SR / SPL by terrain type.

Stage 0.5 of `docs/agent_doing.md`: the acceptance baseline. The training logs report a rolling
success rate produced by a *different* termination rule (latched "ever touched the goal"), so any
comparison against the old `Metrics/robot_goal/success_rate` is meaningless after the switch to
continuous dwell. This script measures the policy under the rules the environment is configured
with now, bucketed by terrain type, so before/after comparisons are apples to apples.

Usage:
    python scripts/eval_policy.py --task Isaac-Nav-MDPO-B2W-Play-v0 \
        --checkpoint <path/to/model_6000.pt> --episodes 200 --out docs/phase0_baseline.md \
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

    @property
    def done(self):
        return len(self.records) >= args_cli.episodes

    def observe(self):
        cmd = self.cmd
        device = self.unwrapped.device
        xy_err = torch.norm(cmd.goal_position_world[:, :2] - self.unwrapped.scene["robot"].data.root_pos_w[:, :2], dim=1)
        self._min_dist = torch.minimum(self._min_dist, xy_err)

        dist_to_path, _ = cmd.get_path_metrics()
        for i in range(self.unwrapped.num_envs):
            if len(self._d_path_samples[i]) < 4000:
                self._d_path_samples[i].append(float(dist_to_path[i]))

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
            self._path_len[ids] = torch.norm(cmd.goal_position_world[ids, :2] - cmd.spawn_position_world[ids, :2], dim=1)
            for env_id in ids.tolist():
                tile = tile_ids[env_id]
                self._tile[env_id] = self.tile_labels[tile] if 0 <= tile < len(self.tile_labels) else "all"
            self._prev_steps_at_goal[ids] = 0.0

        self._prev_steps_at_goal = cmd.steps_at_goal.clone()
        self._prev_len = self.unwrapped.episode_length_buf.clone()

    def _record(self, env_id: int, device):
        samples = self._d_path_samples[env_id]
        steps = int(self._prev_len[env_id])
        # A success is an episode that ended *before* the horizon while the robot was inside the goal
        # radius - i.e. `at_goal_navigation` fired after the 1.5 s dwell. (`steps_at_goal` itself is
        # already zeroed by the reset inside the same `env.step`, so it cannot be read here.)
        ended_early = steps < int(self.unwrapped.max_episode_length)
        success = ended_early and float(self._min_dist[env_id]) < 0.5
        self.records.append(
            {
                "success": bool(success),
                "ended_early": bool(ended_early),
                "tile": self._tile[env_id],
                "start_dist": float(self._start_dist[env_id]),
                "path_len": float(self._path_len[env_id]),
                "min_dist": float(self._min_dist[env_id]),
                "d_path_median": float(np.median(samples)) if samples else float("nan"),
                "d_path_p90": float(np.percentile(samples, 90)) if samples else float("nan"),
                "steps": steps,
            }
        )
        self._d_path_samples[env_id] = []

    def _reset_env_state(self, env_id: int, device):
        self._min_dist[env_id] = float("inf")
        self._start_dist[env_id] = 0.0
        self._path_len[env_id] = 0.0


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
        spl_num = sum(r["success"] * r["start_dist"] / max(r["path_len"], 1e-6) for r in rows)
        return {
            "n": n,
            "sr": sr,
            "sr_old_rule": sr_old,
            "spl": spl_num / n,
            "d_path_median": statistics.median(r["d_path_median"] for r in rows),
            "d_path_p90": statistics.median(r["d_path_p90"] for r in rows),
            "start_dist": statistics.median(r["start_dist"] for r in rows),
            "path_len": statistics.median(r["path_len"] for r in rows),
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

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_checkpoint(runner, args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device, policy_index=args_cli.policy_idx)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    tracker = EpisodeTracker(env)
    steps = 0
    while simulation_app.is_running() and not tracker.done and steps < args_cli.max_steps:
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        if isinstance(obs, tuple):
            obs = obs[0]
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
            "| bucket | n | SR | SR(old rule) | SPL | median d_path | p90 d_path | median start dist | median path len |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for key in ["all"] + [t for t in TILE_NAMES if t in summary]:
            row = summary[key]
            if row is None:
                continue
            lines.append(
                f"| {key} | {row['n']} | {row['sr']:.3f} | {row['sr_old_rule']:.3f} | {row['spl']:.3f} | "
                f"{row['d_path_median']:.2f} | {row['d_path_p90']:.2f} | {row['start_dist']:.1f} | {row['path_len']:.1f} |"
            )
        os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
        with open(args_cli.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[eval] report written to {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
