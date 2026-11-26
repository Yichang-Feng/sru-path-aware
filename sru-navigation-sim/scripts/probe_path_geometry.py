#!/usr/bin/env python3
"""Offline geometry probe for the v2 path pipeline: stairs, flat corridors, 90-degree corners.

Why this exists
---------------
The V1.1 review (`docs/specs/agent_do_qwen.md` §3.3) claimed that a resampled ~1.5 m chord across a
0.2 m staircase necessarily violates `max_height_diff` (0.67 m of rise over the chord vs a 0.25 m
threshold), and on that basis proposed rewriting `Prm.segment_passable`. Reading the code shows
`segment_passable` compares **consecutive cells** along the segment (`prm.py`), not the two
endpoints, so a staircase - one 0.2 m riser per cell - should pass. This script settles it with the
real `Prm` / `GlobalPath` code on synthetic height fields, and measures where invalid segments
actually come from.

It does not start Isaac: `isaaclab_nav_task/__init__.py` applies terrain patches that import `pxr`,
which only exists once an Isaac Sim app has bootstrapped. The packages are therefore pre-registered
as namespace stubs, and the one Isaac Lab symbol `path_generator` imports
(`isaaclab.utils.math.transform_points`, used only by `MultiPath`, never by `GlobalPath.find_path`)
is stubbed with the identity.

Usage:
    python sru-navigation-sim/scripts/probe_path_geometry.py
"""

from __future__ import annotations

import importlib
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "isaaclab_nav_task")

# --- import the modules without booting Isaac Sim -------------------------------------------------
for name, rel in [
    ("isaaclab_nav_task", ""),
    ("isaaclab_nav_task.terrains", "terrains"),
    ("isaaclab_nav_task.navigation", "navigation"),
    ("isaaclab_nav_task.navigation.mdp", "navigation/mdp"),
    ("isaaclab_nav_task.navigation.mdp.navigation", "navigation/mdp/navigation"),
]:
    module = types.ModuleType(name)
    module.__path__ = [os.path.join(PKG, rel)]
    sys.modules[name] = module

_isaaclab = types.ModuleType("isaaclab")
_isaaclab.__path__ = []
_isaaclab_utils = types.ModuleType("isaaclab.utils")
_isaaclab_utils.__path__ = []
_isaaclab_math = types.ModuleType("isaaclab.utils.math")
_isaaclab_math.transform_points = lambda points, pos=None, quat=None: points
sys.modules.update({"isaaclab": _isaaclab, "isaaclab.utils": _isaaclab_utils, "isaaclab.utils.math": _isaaclab_math})
sys.path.insert(0, ROOT)

_prm = importlib.import_module("isaaclab_nav_task.navigation.mdp.navigation.prm")
_pg = importlib.import_module("isaaclab_nav_task.navigation.mdp.navigation.path_generator")
Prm, PrmConfig = _prm.Prm, _prm.PrmConfig
GlobalPath, GlobalPathConfig = _pg.GlobalPath, _pg.GlobalPathConfig

H_SCALE, V_SCALE = 0.1, 0.005  # metres per cell, metres per height unit
WALL = 300
FREE_TOP = 210  # heights up to 1.05 m count as walkable (a 1 m staircase platform is ~200)


# --- synthetic terrains (height_map[x, y], as `Prm` indexes it) -----------------------------------
def flat_corridor(n: int = 200, y0: int = 80, y1: int = 120) -> np.ndarray:
    h = np.zeros((n, n), dtype=np.int16)
    h[:, :y0] = WALL
    h[:, y1:] = WALL
    return h


def stairs(n: int = 200, x0: int = 90, n_steps: int = 5, step_cells: int = 4, rise: int = 40) -> np.ndarray:
    """5 steps x 0.2 m, 0.4 m treads: 2 m of run, 1 m of climb, inside a 4 m wide corridor."""
    h = flat_corridor(n)
    for k in range(n_steps):
        h[x0 + k * step_cells : x0 + (k + 1) * step_cells, :] = rise * (k + 1)
    h[x0 + n_steps * step_cells :, :] = rise * n_steps
    return h


def l_corridor(n: int = 200) -> np.ndarray:
    """A 90-degree turn between two 4 m wide corridors."""
    h = np.full((n, n), WALL, dtype=np.int16)
    h[:, 80:120] = 0
    h[140:180, :] = 0
    return h


# --- harness -------------------------------------------------------------------------------------
def make_prm(h: np.ndarray) -> Prm:
    prm = Prm.__new__(Prm)
    prm.cfg = PrmConfig(fpr=0.0, fnr=0.0, padding=4, max_height_diff=50.0, N=10)
    prm._height_np = h
    prm.h_scale, prm.v_scale = H_SCALE, V_SCALE
    prm.size = (h.shape[0] * H_SCALE, h.shape[1] * H_SCALE)
    prm.terrain_origin = torch.zeros(3)
    free = (h >= 0) & (h <= FREE_TOP)
    prm.free_mask = torch.from_numpy(free)
    prm._free_np = free
    prm._padding_np = np.zeros_like(free)
    prm.compute_clearance_map()
    return prm


def grid_roadmap(prm: Prm, spacing: int = 10) -> dict:
    h = prm._height_np
    index, nodes = {}, []
    for x in range(0, h.shape[0], spacing):
        for y in range(0, h.shape[1], spacing):
            if prm._free_np[x, y]:
                index[(x, y)] = len(nodes)
                nodes.append((float(x), float(y), float(h[x, y])))
    prm.nodes = np.array(nodes)
    prm.free_nodes = list(range(len(nodes)))
    prm.roadmap = {i: [] for i in range(len(nodes))}
    for (x, y), i in index.items():
        for dx, dy in ((spacing, 0), (-spacing, 0), (0, spacing), (0, -spacing)):
            j = index.get((x + dx, y + dy))
            if j is None:
                continue
            if prm.is_traversable(prm.nodes[i], prm.nodes[j], check_padding=False):
                prm.roadmap[i].append(j)
                prm.roadmap[j].append(i)
    return index


def probe(name: str, h: np.ndarray, start: tuple, goal: tuple) -> None:
    prm = make_prm(h)
    index = grid_roadmap(prm)
    prm.cfg.start_idx = index[start]
    prm.cfg.goal_idx = index[goal]
    cfg = GlobalPathConfig(n_waypoints=0, category_weights=[1.0, 0.0, 0.0, 0.0], path_pipeline="v2")
    path = GlobalPath(cfg, env_id=0, prm=prm)
    path.find_path()
    pts = np.array(path.path)

    print(f"\n=== {name} ===")
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        cc, rr = prm._segment_cells(a, b)
        heights = prm._height_np[cc, rr]
        max_cell_dh = int(np.abs(np.diff(heights)).max()) if len(heights) > 1 else 0
        clearance = prm.segment_min_clearance(a, b)
        passable = prm.segment_passable(a, b, cfg.seg_padding_m)
        print(
            f"  seg{i:2d} chord={np.linalg.norm((b - a)[:2]) * H_SCALE:5.2f}m dz={abs(b[2] - a[2]) * V_SCALE:5.2f}m"
            f" max_cell_dh={max_cell_dh * V_SCALE:5.2f}m clear={clearance:5.2f}m"
            f" passable={passable!s:5s} pinch={clearance < cfg.pinch_clearance_m}"
        )
    print(
        f"  -> invalid_seg_frac={1 - np.mean(path.seg_passable):.3f}"
        f"  pinch_frac={np.mean(path.pinch_mask):.3f}"
    )


if __name__ == "__main__":
    probe("flat corridor", flat_corridor(), (10, 100), (190, 100))
    probe("stairs 5 x 0.2 m / 0.4 m", stairs(), (10, 100), (190, 100))
    probe("L corridor (90 deg)", l_corridor(), (10, 100), (160, 190))
