#!/usr/bin/env python3
"""Aggregate the P0 diagnostics (D3-D8) out of an `eval_policy.py` episode dump.

Usage:
    python scripts/p0_eval_analysis.py docs/v1p1_p0_eval.json [--out docs/v1p1_p0_eval.md]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dump")
parser.add_argument("--out", default=None)
args = parser.parse_args()

records = json.load(open(args.dump))
lines: list[str] = []


def emit(text: str = "") -> None:
    print(text)
    lines.append(text)


def outcome(r: dict) -> str:
    if r.get("completed_dwell"):
        return "complete"
    return "fall" if r.get("ended_early") else "timeout"


def bucket(deg: float) -> str:
    if deg != deg:  # NaN
        return "unknown"
    for hi, name in ((30, "0-30"), (60, "30-60"), (120, "60-120"), (181, "120-180")):
        if deg < hi:
            return name
    return "120-180"


emit(f"# P0 诊断（eval dump: {args.dump}）\n")
emit(f"- 局数 {len(records)}；结局分布 {dict(Counter(outcome(r) for r in records))}\n")

# --- D4: success rate by initial heading error ------------------------------------------------
emit("## D4 按初始 |Δψ_path| 分层的 SR / 表现\n")
emit("| 区间 (deg) | n | SR | median d_path | median travelled | freeze_frac | reverse_frac |")
emit("|---|---:|---:|---:|---:|---:|---:|")
groups = defaultdict(list)
for r in records:
    groups[bucket(r.get("head_err0_deg", float("nan")))].append(r)
for name in ("0-30", "30-60", "60-120", "120-180", "unknown"):
    rows = groups.get(name)
    if not rows:
        continue
    emit(
        f"| {name} | {len(rows)} | {np.mean([bool(r['success']) for r in rows]):.3f} | "
        f"{st.median([r['d_path_median'] for r in rows]):.2f} | "
        f"{st.median([r['traveled'] for r in rows]):.1f} | "
        f"{st.median([r.get('freeze_frac', 0) for r in rows]):.3f} | "
        f"{st.median([r.get('reverse_frac', 0) for r in rows]):.3f} |"
    )

# --- D5: freeze / reverse overall --------------------------------------------------------------
fz = [r.get("freeze_frac", 0.0) for r in records]
rv = [r.get("reverse_frac", 0.0) for r in records]
emit("\n## D5 冻结 / 倒走\n")
emit(f"- freeze_frac: mean {np.mean(fz):.3f}, p50 {np.median(fz):.3f}, p90 {np.percentile(fz, 90):.3f}, "
     f"局数 >5% = {sum(f > 0.05 for f in fz)}/{len(fz)}")
emit(f"- reverse_frac: mean {np.mean(rv):.3f}, p50 {np.median(rv):.3f}, p90 {np.percentile(rv, 90):.3f}, "
     f"局数 >5% = {sum(v > 0.05 for v in rv)}/{len(rv)}")

# --- D6: d_path at the onset of the first sustained freeze -------------------------------------
onset = [r["d_path_at_freeze"] for r in records if r.get("d_path_at_freeze") is not None]
if onset:
    emit("\n## D6 首次持续冻结时的 d_path\n")
    emit(f"- n={len(onset)}；p50 {np.median(onset):.2f} m，p90 {np.percentile(onset, 90):.2f} m；"
         f"已 >0.8 m 的占 {np.mean([d > 0.8 for d in onset]):.2f}")
else:
    emit("\n## D6 首次持续冻结时的 d_path\n- 没有观测到持续冻结（或未记录）")

# --- D7: pitch on stairs ------------------------------------------------------------------------
emit("\n## D7 楼梯 tile 的 pitch 分布（阈值 40°）\n")
for tile in ("stairs", "maze", "pits", "non_maze", "all"):
    rows = [r for r in records if r["tile"] == tile and r.get("pitch_p95_deg") == r.get("pitch_p95_deg")]
    if not rows:
        continue
    p95 = [r["pitch_p95_deg"] for r in rows]
    emit(f"- {tile:9s} n={len(rows):3d}  pitch p95: p50 {np.median(p95):4.1f}°, p90 {np.percentile(p95, 90):4.1f}°, "
         f"max {max(p95):4.1f}°")

# --- D8: paid metres by outcome ------------------------------------------------------------------
emit("\n## D8 每回合被付费的路径米数（按结局）\n")
emit("| 结局 | n | paid_m p50 | paid_m mean | 计划路径 p50 | 实际行驶 p50 |")
emit("|---|---:|---:|---:|---:|---:|")
by = defaultdict(list)
for r in records:
    by[outcome(r)].append(r)
for name in ("complete", "timeout", "fall"):
    rows = by.get(name)
    if not rows:
        continue
    emit(f"| {name} | {len(rows)} | {st.median([r.get('paid_m', 0) for r in rows]):.1f} | "
         f"{np.mean([r.get('paid_m', 0) for r in rows]):.1f} | "
         f"{st.median([r['path_len'] for r in rows]):.1f} | {st.median([r['traveled'] for r in rows]):.1f} |")

# --- D3: invalid segment bucketing ----------------------------------------------------------------
emit("\n## D3 invalid 段分桶\n")
emit("| tile | n | invalid/tile(均) | 拐角处 | 直段 | 高度/出界 | 最小 clearance(中位) |")
emit("|---|---:|---:|---:|---:|---:|---:|")
for tile in ("stairs", "maze", "pits", "non_maze"):
    rows = [r for r in records if r["tile"] == tile and "n_invalid" in r]
    if not rows:
        continue
    clears = [r["invalid_clearance_min"] for r in rows if r.get("invalid_clearance_min") == r.get("invalid_clearance_min")]
    emit(f"| {tile} | {len(rows)} | {np.mean([r['n_invalid'] for r in rows]):.2f} | "
         f"{np.mean([r['n_invalid_corner'] for r in rows]):.2f} | "
         f"{np.mean([r['n_invalid_straight'] for r in rows]):.2f} | "
         f"{np.mean([r.get('n_invalid_height', 0) for r in rows]):.2f} | "
         f"{st.median(clears):.2f} |" if clears else
         f"| {tile} | {len(rows)} | {np.mean([r['n_invalid'] for r in rows]):.2f} | "
         f"{np.mean([r['n_invalid_corner'] for r in rows]):.2f} | "
         f"{np.mean([r['n_invalid_straight'] for r in rows]):.2f} | "
         f"{np.mean([r.get('n_invalid_height', 0) for r in rows]):.2f} | - |")

# --- D3b: stair-like path segments ---------------------------------------------------------------
emit("\n## D3b 路径在楼梯处的陡峭程度（dir_z）\n")
emit("| tile | n | dir_z_max p50 | p90 | >0.5 的占比 | dz_max p50 (m) |")
emit("|---|---:|---:|---:|---:|---:|")
for tile in ("stairs", "maze", "pits", "non_maze"):
    rows = [r for r in records if r["tile"] == tile and "dir_z_max" in r]
    if not rows:
        continue
    dz = [r["dir_z_max"] for r in rows]
    dzz = [r["dz_max_m"] for r in rows]
    emit(f"| {tile} | {len(rows)} | {np.median(dz):.3f} | {np.percentile(dz, 90):.3f} | "
         f"{np.mean([d > 0.5 for d in dz]):.2f} | {np.median(dzz):.2f} |")

if args.out:
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[analysis] written to {args.out}", flush=True)
