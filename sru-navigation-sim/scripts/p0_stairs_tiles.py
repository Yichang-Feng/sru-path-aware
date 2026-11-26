#!/usr/bin/env python3
"""Why are stairs tiles' paths 58% "invalid"? Inspect the free mask / clearance they are built on.

The 200-episode eval shows stairs tiles carry ~8 of 14 segments flagged `seg_passable = False`, while
maze/pits/non_maze carry 0.2-1.1. The synthetic staircase probe (`probe_path_geometry.py`) had zero,
so something about the *real* tile differs: either the corridor really is narrower than 0.31 m, or the
staircase / platform is missing from the mask the clearance map is computed from.

This dumps, per stairs tile: how much of the tile is free, whether the raised platform is free, and
the clearance distribution on and around the staircase.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Nav-MDPO-B2W-Play-v0")
parser.add_argument("--terrain_rows", type=int, default=4)
parser.add_argument("--terrain_cols", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import isaaclab_nav_task  # noqa: F401,E402


def main():
    spec = gym.spec(args_cli.task)
    cfg = spec.kwargs.get("env_cfg_entry_point")()
    cfg.scene.num_envs = 1
    cfg.scene.terrain.terrain_generator.num_rows = args_cli.terrain_rows
    cfg.scene.terrain.terrain_generator.num_cols = args_cli.terrain_cols
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()
    terrain = env.scene.terrain

    heights = terrain._height_field_visual
    valid = getattr(terrain, "_height_field_valid_mask", None)
    platform = getattr(terrain, "_height_field_platform_mask", None)
    print(f"tiles={len(heights)} valid_mask={'yes' if valid is not None else 'no'} "
          f"platform_mask={'yes' if platform is not None else 'no'}", flush=True)

    # The clearance map is built from the free mask the PRM uses; ask the PRM manager for it.
    pm = env.command_manager.get_term("robot_goal").paths_manager
    prm0 = getattr(pm, "prm_manager", None) if pm is not None else None
    print(f"prm_manager={'yes' if prm0 is not None else 'no'}", flush=True)

    for idx, tile in enumerate(heights):
        h = tile.cpu().numpy().astype(np.int32)
        raised = (h >= 150) & (h < 250)
        platform_cells = (h >= 195) & (h <= 205)
        tread_cells = (h >= 20) & (h <= 170)
        if platform_cells.mean() < 0.005 and tread_cells.mean() < 0.005:
            continue
        v = valid[idx].cpu().numpy().astype(bool) if valid is not None else None
        p = platform[idx].cpu().numpy().astype(bool) if platform is not None else None
        print(f"\ntile {idx}: max_h={h.max()} raised_frac={raised.mean():.2f} "
              f"valid_frac={(v.mean() if v is not None else float('nan')):.2f}", flush=True)
        if v is not None:
            print(f"  raised cells valid: {v[raised].mean():.3f}  "
                  f"floor cells valid: {v[h < 10].mean():.3f}", flush=True)
            zero_clear = (h < 10) & ~v
            print(f"  floor cells excluded from free mask: {zero_clear.mean():.3f}", flush=True)
        if p is not None:
            print(f"  platform mask frac={p.mean():.3f}  platform∩raised={p[raised].mean():.3f}", flush=True)
        if prm0 is not None:
            prm = prm0.prms[idx] if hasattr(prm0, "prms") else None
            if prm is not None and getattr(prm, "_clearance_np", None) is not None:
                c = np.asarray(prm._clearance_np)
                free = np.asarray(prm._free_np).astype(bool)
                print(f"  prm free_frac={free.mean():.2f} clearance(free) p50={np.median(c[free]):.2f} "
                      f"p10={np.percentile(c[free], 10):.2f}", flush=True)
                print(f"  clearance on raised cells: p50={np.median(c[raised]):.2f} "
                      f"p90={np.percentile(c[raised], 90):.2f}", flush=True)
                if platform_cells.any():
                    print(f"  PLATFORM(195-205) free: {free[platform_cells].mean():.3f} "
                          f"clearance p50={np.median(c[platform_cells]):.2f}", flush=True)
                if tread_cells.any():
                    print(f"  TREADS(20-170)    free: {free[tread_cells].mean():.3f} "
                          f"clearance p50={np.median(c[tread_cells]):.2f}", flush=True)
                print(f"  raised(150-250) free: {free[raised].mean():.3f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
