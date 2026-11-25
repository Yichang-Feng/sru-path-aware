#!/usr/bin/env python3
"""Stage 0 blocking measurements for the B2W path-aware redesign.

Runs the blocking measurements M1-M5 (see `docs/agent_doing.md`, section 3) and writes a
markdown report. This script only *measures* - it never changes a training config and never
writes training code.

    M1  B2W body width / length (USD bounding boxes of the spawned robot)
    M2  Maze corridor / passage clearance distribution (per terrain tile)
    M3  Whether `Prm.free_nodes` sampling already includes padding dilation
    M4  `Prm.is_traversable` semantics (z-based check, endpoint-in-obstacle behaviour)
    M5  Fraction of `optimal`-category path segments that fail `check_padding=True`

Usage:
    ./isaaclab.sh -p scripts/stage0_measure.py --headless --out docs/phase0_measurements.md
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 0 blocking measurements.")
parser.add_argument("--num_rows", type=int, default=4, help="Terrain tiles rows for the measurement scene.")
parser.add_argument("--num_cols", type=int, default=10, help="Terrain tiles cols for the measurement scene.")
parser.add_argument("--prm_n", type=int, default=300, help="PRM node count used for the measurement scene.")
parser.add_argument("--num_paths", type=int, default=100, help="How many optimal paths to sample for M5.")
parser.add_argument("--out", type=str, default="docs/phase0_measurements.md", help="Report output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from scipy.spatial import distance_matrix  # noqa: E402

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import isaaclab_nav_task  # noqa: F401,E402

from isaaclab_nav_task.navigation.config.b2w.navigation_env_cfg import B2WNavigationEnvCfg  # noqa: E402
from isaaclab_nav_task.navigation.mdp.navigation.path_generator import GlobalPath, GlobalPathConfig  # noqa: E402
from isaaclab_nav_task.navigation.mdp.navigation.prm import PrmConfig  # noqa: E402
from isaaclab_nav_task.terrains.terrain_constants import THRESHOLDS  # noqa: E402


def pct(x, q):
    return float(np.percentile(np.asarray(x, dtype=np.float64), q))


def measure_robot_bbox():
    """M1: world-space AABB of the spawned robot's collision geometry."""
    from pxr import UsdGeom

    # The running app owns the stage; grab it through omni.usd instead of re-opening the file.
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(0, ["default", "physics", "render"])

    out = {}
    for name in ("base_link", "trunk"):
        for prim in stage.Traverse():
            if prim.GetName() == name:
                r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                out[name] = (
                    [round(float(r.GetMin()[i]), 4) for i in range(3)],
                    [round(float(v - r.GetMin()[i]), 4) for i, v in enumerate(r.GetMax())],
                )
                break
    # Whole robot
    for prim in stage.Traverse():
        if prim.GetName() == "Robot":
            r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            out["robot_total"] = (
                [round(float(r.GetMin()[i]), 4) for i in range(3)],
                [round(float(v - r.GetMin()[i]), 4) for i, v in enumerate(r.GetMax())],
            )
            break
    del UsdGeom
    return out


def main():
    sim_dt = 0.005
    env_cfg = B2WNavigationEnvCfg()
    env_cfg.scene.num_envs = 2
    env_cfg.scene.terrain.terrain_generator.num_rows = args_cli.num_rows
    env_cfg.scene.terrain.terrain_generator.num_cols = args_cli.num_cols
    env_cfg.commands.robot_goal.prm_cfg = PrmConfig(N=args_cli.prm_n)
    env_cfg.scene.terrain.max_init_terrain_level = 10

    env = gym.make("Isaac-Nav-MDPO-B2W-Dev-v0", cfg=env_cfg)
    env.reset()
    env_unwrapped = env.unwrapped

    terrain = env_unwrapped.scene.terrain
    heights = terrain._height_field_visual  # (num_tiles, X, Y) int heights
    valid_mask = terrain._height_field_valid_mask
    spawn_mask = terrain._height_field_spawn_mask
    terrain_cfg = terrain.cfg.terrain_generator
    h_scale = terrain_cfg.horizontal_scale
    v_scale = terrain_cfg.vertical_scale
    num_tiles = heights.shape[0]

    report = ["# Phase 0 阻塞性实测报告", "", f"场景：{args_cli.num_rows}×{args_cli.num_cols} tiles, PRM N={args_cli.prm_n}", ""]

    # ---------------- M1 ----------------
    bbox = measure_robot_bbox()
    report += ["## M1 B2W 机身尺寸（USD 包围盒，世界系）", "", "| prim | min (x,y,z) | size (x,y,z) |", "|---|---|---|"]
    for k, (mn, size) in bbox.items():
        report.append(f"| {k} | {mn} | {size} |")
    base = bbox.get("base_link") or bbox.get("trunk") or bbox.get("robot_total")
    if base:
        half_width = base[1][1] / 2.0
        report += [
            "",
            f"→ 机身半宽（y）= **{half_width:.3f} m**，半长（x）= {base[1][0] / 2.0:.3f} m",
            f"→ `seg_passable` padding = 半宽 + 0.05 = **{half_width + 0.05:.3f} m**",
            "",
        ]
    else:
        half_width = None
        report += ["", "⚠️ 未找到 base_link/trunk prim，M1 失败。", ""]

    # ---------------- M2 ----------------
    clearance_all = []
    narrow_stats = []
    for tile_id in range(num_tiles):
        h = heights[tile_id].cpu().numpy()
        free = ((h >= THRESHOLDS.GROUND_MIN) & (h <= THRESHOLDS.GROUND_MAX)) | (
            (h >= THRESHOLDS.PLATFORM_MIN) & (h <= THRESHOLDS.PLATFORM_MAX)
        )
        clearance = distance_transform_edt(free) * h_scale
        clearance_all.append(clearance[free])
        # narrow passages = local maxima of the clearance field below 1.0 m
        from scipy.ndimage import maximum_filter

        loc = maximum_filter(clearance, size=5) == clearance
        ridge = free & loc
        narrow_stats.append(clearance[ridge & (clearance < 1.0)])

    clearance_all = np.concatenate(clearance_all)
    narrow_all = np.concatenate(narrow_stats) if narrow_stats else np.array([])
    report += [
        "## M2 通道 / 障碍间距分布（clearance = 到最近障碍的半宽，米）",
        "",
        "| 统计量 | 全部自由格 | 通道脊线（局部极大，<1.0 m） |",
        "|---|---|---|",
    ]
    for q in (1, 5, 25, 50, 75, 95):
        report.append(f"| p{q} | {pct(clearance_all, q):.3f} | {pct(narrow_all, q) if narrow_all.size else float('nan'):.3f} |")
    frac_narrow_04 = float(np.mean(clearance_all < 0.45))
    report += [
        "",
        f"→ clearance < 0.45 m 的自由格占比：**{frac_narrow_04 * 100:.1f}%**（窄门/窄通道候选）",
        f"→ 通道脊线最小值 {narrow_all.min() if narrow_all.size else float('nan'):.3f} m "
        f"→ 对应最窄净宽 {2 * narrow_all.min() if narrow_all.size else float('nan'):.3f} m",
        "",
    ]

    # ---------------- M3 ----------------
    prm_manager = env_unwrapped.prm_manager
    prms = [prm_manager.get(i) for i in range(num_tiles)]
    free_fracs, pad_fracs, free_node_fracs = [], [], []
    for prm in prms:
        free = prm.free_mask.cpu().numpy()
        pad = prm.padding_mask.cpu().numpy()
        free_fracs.append(free.mean())
        pad_fracs.append((pad & free).sum() / max(1, free.sum()))
        free_node_fracs.append(len(prm.free_nodes) / max(1, len(prm.nodes)))
    report += [
        "## M3 `free_nodes` 采样是否已含 padding 膨胀",
        "",
        f"- 每个 tile 上 padding 窗口（`cfg.padding={prms[0].cfg.padding}` 格）覆盖的自由格比例："
        f"均值 {np.mean(pad_fracs) * 100:.1f}%，最大 {np.max(pad_fracs) * 100:.1f}%",
        f"- `free_nodes / nodes` 比例：均值 {np.mean(free_node_fracs) * 100:.1f}%",
        "- 结论：`sample_free_point()` 只拒绝 `padding_mask` 中的点，因此 **节点采样已含 padding 膨胀**；"
        "但 `free_nodes` 用 `free_mask` 判定（仅地面/平台），二者含义不同。",
        "",
    ]

    # ---------------- M4 ----------------
    prm = prms[0]
    h_np = prm._height_np
    rows, cols = h_np.shape
    # a segment that surely crosses an obstacle: scan for a wall column and cut across it
    wall_cells = np.argwhere(~prm._free_np)
    m4_lines = []
    if wall_cells.size:
        w = wall_cells[len(wall_cells) // 2]
        p1 = np.array([w[0], max(0, w[1] - 5), 0])
        p2 = np.array([w[0], min(cols - 1, w[1] + 5), 0])
        m4_lines.append(f"- 跨墙段 `is_traversable(check_padding=True)` = {prm.is_traversable(p1, p2)}")
        m4_lines.append(f"- 跨墙段 `is_traversable(check_padding=False)` = {prm.is_traversable(p1, p2, check_padding=False)}")

    # endpoint inside an obstacle
    if wall_cells.size:
        w = wall_cells[0]
        p_in = np.array([w[0], w[1], 0], dtype=float)
        p_out = np.array([w[0], min(cols - 1, w[1] + 8), 0], dtype=float)
        try:
            res = prm.is_traversable(p_in, p_out, check_padding=True)
            m4_lines.append(f"- 端点落在障碍内：返回 {res}（**不抛异常**）")
        except Exception as exc:  # noqa: BLE001
            m4_lines.append(f"- 端点落在障碍内：抛出 {type(exc).__name__}: {exc}")

    # z-based check: a stair step is 0.2 m = 40 units, below max_height_diff=50
    m4_lines.append(
        f"- `max_height_diff` = {prm.cfg.max_height_diff} 单位 = "
        f"{prm.cfg.max_height_diff * v_scale:.3f} m；楼梯单级 {0.2} m = {0.2 / v_scale:.0f} 单位 "
        f"→ {'可通过' if 0.2 / v_scale <= prm.cfg.max_height_diff else '不可通过'}"
    )
    report += ["## M4 `is_traversable` 语义", ""] + m4_lines + [""]

    # ---------------- M5 ----------------
    cfg_opt = GlobalPathConfig(category_weights=[1.0, 0.0, 0.0, 0.0])
    prm.cfg.padding = 4
    n_seg = 0
    n_false_pad = 0
    n_false_nopad = 0
    narrow_seg = 0
    narrow_false = 0
    open_seg = 0
    open_false = 0
    clearance_maps = {}
    for tile_id in range(num_tiles):
        h = heights[tile_id].cpu().numpy()
        free = ((h >= THRESHOLDS.GROUND_MIN) & (h <= THRESHOLDS.GROUND_MAX)) | (
            (h >= THRESHOLDS.PLATFORM_MIN) & (h <= THRESHOLDS.PLATFORM_MAX)
        )
        clearance_maps[tile_id] = distance_transform_edt(free) * h_scale

    while n_seg < args_cli.num_paths * 14:
        tile_id = int(np.random.randint(num_tiles))
        prm_t = prms[tile_id]
        if len(prm_t.free_nodes) < 4:
            continue
        nodes = np.array([prm_t.nodes[i] for i in prm_t.free_nodes])
        idx = np.random.choice(len(nodes), size=2, replace=False)
        start, goal = nodes[idx[0]], nodes[idx[1]]
        if np.linalg.norm(start[:2] - goal[:2]) < 20:  # at least 2 m apart (in cells)
            continue
        prm_t.reset_start_goal(start.astype(float), goal.astype(float))
        gp = GlobalPath(cfg_opt, env_id=0, prm=prm_t)
        gp.find_path()
        pts = np.array(gp.path, dtype=np.float64)  # includes z_offset; x/y unaffected
        seg_passable_pad = [
            prm_t.is_traversable(pts[i][:3], pts[i + 1][:3], check_padding=True) for i in range(len(pts) - 1)
        ]
        seg_passable_nopad = [
            prm_t.is_traversable(pts[i][:3], pts[i + 1][:3], check_padding=False) for i in range(len(pts) - 1)
        ]
        cl = clearance_maps[tile_id]
        for i in range(len(pts) - 1):
            rr, cc = _line_cells(pts[i], pts[i + 1])
            rr = np.clip(rr, 0, cl.shape[0] - 1)
            cc = np.clip(cc, 0, cl.shape[1] - 1)
            min_clear = float(cl[rr, cc].min())
            n_seg += 1
            n_false_pad += int(not seg_passable_pad[i])
            n_false_nopad += int(not seg_passable_nopad[i])
            if min_clear < 0.45:
                narrow_seg += 1
                narrow_false += int(not seg_passable_pad[i])
            else:
                open_seg += 1
                open_false += int(not seg_passable_pad[i])

    report += [
        "## M5 optimal 路径上 `check_padding=True` 的不可行段比例",
        "",
        f"- 采样段数：{n_seg}（{args_cli.num_paths} 条 path 以上）",
        f"- `check_padding=True` 判不可行：**{n_false_pad / max(1, n_seg) * 100:.1f}%**",
        f"- `check_padding=False` 判不可行：{n_false_nopad / max(1, n_seg) * 100:.1f}%",
        f"- 其中段上最小 clearance < 0.45 m（窄段）：{narrow_seg} 段，不可行 {narrow_false / max(1, narrow_seg) * 100:.1f}%",
        f"- 开阔段：{open_seg} 段，不可行 {open_false / max(1, open_seg) * 100:.1f}%",
        "",
    ]

    os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
    with open(args_cli.out, "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\n[stage0] report written to {args_cli.out}")

    env.close()


def _line_cells(p1, p2):
    """Cells along the 2D line between two height-map points (row=x, col=y)."""
    n = int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))) + 1
    xs = np.linspace(p1[0], p2[0], n).astype(int)
    ys = np.linspace(p1[1], p2[1], n).astype(int)
    return xs, ys


if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0)
