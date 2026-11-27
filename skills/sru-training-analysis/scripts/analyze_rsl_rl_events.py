#!/usr/bin/env python3
"""Parse and compare RSL-RL TensorBoard event logs, with the B2W path-aware reward arithmetic applied.

Plain `--tags` output is not enough to judge these runs: the numbers that decide whether a run is
healthy are *derived* from the reward table (per-episode income per term, a potential's bound, the
dt trap on one-shot penalties), and the config that defines those weights sits next to the event file
in `params/env.yaml`. This tool reads both and reports the derived quantities plus a set of checks
that encode the failure modes this project has actually hit.

Usage
-----
    # per-window table + health checks for one run
    python analyze_rsl_rl_events.py <run_dir>

    # compare several runs at the same windows
    python analyze_rsl_rl_events.py runA runB runC --windows 100-200,200-300

    # raw tags, as before
    python analyze_rsl_rl_events.py <run_dir> --tags Policy/mean_noise_std --steps 100,500

Derived quantities (all use the run's own `params/env.yaml`, not hard-coded weights):

    compl/ep      completions per episode = goal_reached_bonus * T / weight
    near$/ep      income from occupying the goal radius = (reach_goal + tight + soft) * T   [must be 0]
    head$/ep      heading_progress_to_path * T                    [a potential: |total| <= pi]
    toPaid        what one horizon timeout actually costs = time_out_penalty * T / timeout rate
                  (every term is multiplied by dt, so a -50 weight pays -10)
    paid_m/ep     metres of path actually paid = progress_along_path * T / weight
    d_path(lb)    mean lateral error back-solved from the deviation penalty (a lower bound)
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

DEFAULT_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Metrics/robot_goal/success_rate",
    "Metrics/robot_goal/path_progress",
    "Metrics/robot_goal/distance_to_path",
    "Metrics/robot_goal/velocity_magnitude",
    "Metrics/robot_goal/velocity_toward_goal",
    "Metrics/robot_goal/invalid_seg_frac",
    "Metrics/robot_goal/relax_frac",
    "Metrics/robot_goal/blocked_episode",
    "Metrics/robot_goal/a_star_failed_frac",
    "Episode_Termination/base_contact",
    "Episode_Termination/time_out",
    "Episode_Termination/early_termination",
    "Policy/mean_noise_std",
    "Policy/log_std_max",
    "Loss/explained_variance",
    "Loss/kl_divergence",
    "Loss/learning_rate",
]

# Terms whose *sum* is income for standing in the goal radius. Must stay exactly zero: any per-step
# payment there makes "orbit the goal and never finish" optimal once completing ends the episode.
GOAL_OCCUPANCY_TERMS = ("reach_goal", "reach_goal_xy_tight", "reach_goal_xy_soft")

# term -> config param used as a "per unit" divisor, so income can be reported in physical units
PER_UNIT = {
    "progress_along_path": ("per_metre", "m"),
    "goal_reached_bonus": (None, "completions"),
}


# ---------------------------------------------------------------------------
# TensorBoard event parsing (no tensorboard dependency)
# ---------------------------------------------------------------------------
def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while pos < len(buf):
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7
        if shift > 70:
            break
    raise ValueError("invalid varint")


def iter_fields(buf: bytes) -> Iterable[tuple[int, int, object]]:
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field = key >> 3
        wire = key & 7
        if wire == 0:
            value, pos = read_varint(buf, pos)
        elif wire == 1:
            value = struct.unpack_from("<d", buf, pos)[0]
            pos += 8
        elif wire == 2:
            length, pos = read_varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
        elif wire == 5:
            value = struct.unpack_from("<f", buf, pos)[0]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def parse_tensor(payload: bytes) -> float | None:
    values: list[float] = []
    for field, wire, value in iter_fields(payload):
        if field != 4:  # TensorProto.float_val
            continue
        if wire == 5:
            values.append(float(value))
        elif wire == 2:
            n = len(value) // 4
            if n:
                values.extend(struct.unpack_from("<" + "f" * n, value, 0))
    return values[0] if values else None


def parse_summary_value(payload: bytes) -> tuple[str | None, float | None]:
    tag = None
    simple = None
    tensor_value = None
    for field, wire, value in iter_fields(payload):
        if field == 1 and wire == 2:  # tag
            tag = value.decode("utf-8", errors="replace")
        elif field == 2 and wire in (1, 5):  # simple_value
            simple = float(value)
        elif field == 8 and wire == 2:  # tensor
            tensor_value = parse_tensor(value)
    return tag, simple if simple is not None else tensor_value


def parse_summary(payload: bytes) -> list[tuple[str | None, float | None]]:
    out = []
    for field, wire, value in iter_fields(payload):
        if field == 1 and wire == 2:  # Summary.value
            out.append(parse_summary_value(value))
    return out


def parse_event(payload: bytes) -> tuple[float, int, list[tuple[str | None, float | None]]]:
    wall_time = 0.0
    step = 0
    values: list[tuple[str | None, float | None]] = []
    for field, wire, value in iter_fields(payload):
        if field == 1 and wire == 1:  # wall_time
            wall_time = float(value)
        elif field == 2 and wire == 0:  # step
            step = int(value)
        elif field == 5 and wire == 2:  # summary
            values.extend(parse_summary(value))
    return wall_time, step, values


def read_events(path: Path) -> list[tuple[float, int, str, float]]:
    data = path.read_bytes()
    pos = 0
    rows = []
    while pos + 12 <= len(data):
        length = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        if pos + 4 + length + 4 > len(data):
            break
        pos += 4  # masked CRC
        payload = data[pos : pos + length]
        pos += length + 4  # payload + CRC
        try:
            wall, step, values = parse_event(payload)
        except Exception:
            continue
        for tag, value in values:
            if tag is not None and value is not None:
                rows.append((wall, step, tag, value))
    return rows


def resolve_event(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"no TensorBoard event file found under {path}")


# ---------------------------------------------------------------------------
# Run configuration (params/env.yaml written by the runner)
# ---------------------------------------------------------------------------
def _yaml_module():
    import yaml  # noqa: PLC0415

    return yaml


def _node_value(node) -> Any:
    """Scalar value of a composed node, honouring bool/int/float tags, or None."""
    yaml = _yaml_module()
    if not isinstance(node, yaml.ScalarNode):
        return None
    if node.tag.endswith(":bool"):
        return node.value.lower() in ("true", "yes", "on")
    if node.tag.endswith(":null"):
        return None
    try:
        return int(node.value)
    except ValueError:
        pass
    try:
        return float(node.value)
    except ValueError:
        return node.value


def _node_map(node) -> dict[str, Any]:
    yaml = _yaml_module()
    if not isinstance(node, yaml.MappingNode):
        return {}
    return {k.value: v for k, v in node.value if isinstance(k, yaml.ScalarNode)}


def load_run_config(run_dir: Path) -> dict[str, Any]:
    """Reward weights, per-term params and the horizon, straight from the run's own config.

    The YAML is *composed*, never constructed: `env.yaml` carries `!!python/object/apply:builtins.slice`
    and tuple tags (Isaac Lab writes them), which a SafeLoader rejects and an UnsafeLoader would
    execute. Walking the node tree reads exactly the scalars needed and nothing else.

    Falling back to defaults (rather than exiting) keeps this usable on archives without
    `params/env.yaml`; the three columns that need weights then print `-` instead of lying.
    """
    cfg: dict[str, Any] = {
        "weights": {},
        "params": {},
        "episode_length_s": 90.0,
        "step_dt": 0.2,
        "cfg_source": "defaults",
        "max_iterations": None,
    }
    env_yaml = run_dir / "params" / "env.yaml"
    agent_yaml = run_dir / "params" / "agent.yaml"
    if env_yaml.is_file():
        try:
            yaml = _yaml_module()
            top = _node_map(yaml.compose(env_yaml.read_text(encoding="utf-8")))
            rewards = _node_map(top.get("rewards"))
            for name, term_node in rewards.items():
                term = _node_map(term_node)
                weight = _node_value(term["weight"]) if "weight" in term else None
                if isinstance(weight, (int, float)):
                    cfg["weights"][name] = float(weight)
                    cfg["params"][name] = {k: _node_value(v) for k, v in _node_map(term.get("params")).items()}
            if "episode_length_s" in top:
                cfg["episode_length_s"] = float(_node_value(top["episode_length_s"]))
            sim = _node_map(top.get("sim"))
            sim_dt = float(_node_value(sim["dt"])) if "dt" in sim else 0.005
            decimation = float(_node_value(top["decimation"])) if "decimation" in top else 40.0
            cfg["step_dt"] = sim_dt * decimation
            cfg["cfg_source"] = str(env_yaml)
        except Exception as exc:  # never let a config oddity stop the analysis
            print(f"warning: could not read {env_yaml}: {exc}", file=sys.stderr)
    if agent_yaml.is_file():
        try:
            yaml = _yaml_module()
            top = _node_map(yaml.compose(agent_yaml.read_text(encoding="utf-8")))
            cfg["max_iterations"] = _node_value(top["max_iterations"]) if "max_iterations" in top else None
        except Exception:
            pass
    return cfg


# ---------------------------------------------------------------------------
# Window statistics and derived reward arithmetic
# ---------------------------------------------------------------------------
def window_count(scalars: dict[str, list[tuple[float, int, str, float]]], tag: str, a: int, b: int) -> int:
    return sum(1 for _wall, s, _tag, _value in scalars.get(tag, []) if a <= s <= b)


def window_mean(scalars: dict[str, list[tuple[float, int, str, float]]], tag: str, a: int, b: int) -> float | None:
    series = scalars.get(tag, [])
    vals = [value for _wall, s, _tag, value in series if a <= s <= b]
    return sum(vals) / len(vals) if vals else None


def nearest(scalars: dict[str, list[tuple[float, int, str, float]]], tag: str, step: int, tol: int = 1) -> float | None:
    series = scalars.get(tag, [])
    best = None
    for _wall, s, _tag, value in series:
        dist = abs(s - step)
        if best is None or dist < best[0]:
            best = (dist, value)
    return best[1] if best is not None and best[0] <= tol else None


def derive(scalars, cfg, a: int, b: int) -> dict[str, float | None]:
    """Everything a reviewer needs about one window, in physical units where possible."""
    T = cfg["episode_length_s"]
    dt = cfg["step_dt"]
    w = cfg["weights"]

    def wm(tag: str) -> float | None:
        return window_mean(scalars, tag, a, b)

    def income(term: str) -> float | None:
        """Per-episode income of a reward term (logs are the episode sum divided by T)."""
        v = wm(f"Episode_Reward/{term}")
        return None if v is None else v * T

    ep_len = wm("Train/mean_episode_length")
    bonus = income("goal_reached_bonus")
    w_bonus = w.get("goal_reached_bonus")
    completions = bonus / w_bonus if (bonus is not None and w_bonus) else None

    near = 0.0
    near_terms = 0
    for term in GOAL_OCCUPANCY_TERMS:
        v = income(term)
        if v is not None:
            near += v
            near_terms += 1
    near_income = near if near_terms else None

    heading = income("heading_progress_to_path")
    timeout_rate = wm("Episode_Termination/time_out")
    timeout_cost = income("time_out_penalty")
    timeout_paid = timeout_cost / timeout_rate if (timeout_cost is not None and timeout_rate and timeout_rate > 0.02) else None

    paid_m = None
    v = income("progress_along_path")
    w_path = w.get("progress_along_path")
    per_metre = bool((cfg["params"].get("progress_along_path") or {}).get("per_metre", False))
    if v is not None and w_path and per_metre:
        paid_m = v / w_path

    # How far beyond the dead band the robot actually strays, back-solved from the deviation penalty:
    # excess = |dev| * T / (ep_len * |w| * dt), capped in the reward itself, so this is a *lower*
    # bound. Reported as the excess over `corridor_radius` rather than an absolute distance, because
    # the corridor is deliberately wide (1.5 m) and adding it back would just print the corridor.
    dev_excess = None
    dev = income("path_deviation_penalty")
    w_dev = w.get("path_deviation_penalty")
    corridor = float((cfg["params"].get("path_deviation_penalty") or {}).get("corridor_radius", 0.5))
    if dev is not None and w_dev and ep_len:
        dev_excess = abs(dev) / (ep_len * abs(w_dev) * dt)

    return {
        "sr": wm("Metrics/robot_goal/success_rate"),
        "reward": wm("Train/mean_reward"),
        "ep_len": ep_len,
        "completions": completions,
        "near_income": near_income,
        "heading_income": heading,
        "timeout_rate": timeout_rate,
        "timeout_paid": timeout_paid,
        "paid_m": paid_m,
        "dev_excess": dev_excess,
        "corridor": corridor,
        "fall": wm("Episode_Termination/base_contact"),
        "early": wm("Episode_Termination/early_termination"),
        "velocity": wm("Metrics/robot_goal/velocity_magnitude"),
        "v_toward_goal": wm("Metrics/robot_goal/velocity_toward_goal"),
        "invalid": wm("Metrics/robot_goal/invalid_seg_frac"),
        "relax": wm("Metrics/robot_goal/relax_frac"),
        "blocked": wm("Metrics/robot_goal/blocked_episode"),
        "a_star_failed": wm("Metrics/robot_goal/a_star_failed_frac"),
        "ev": wm("Loss/explained_variance"),
        "noise": wm("Policy/mean_noise_std"),
        "log_std_max": wm("Policy/log_std_max"),
    }


def fmt(value: float | None, width: int = 7, digits: int = 2) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}.{digits}f}"


# ---------------------------------------------------------------------------
# Health checks: the failures this project has actually hit
# ---------------------------------------------------------------------------
def run_checks(stats_by_window: dict[tuple[int, int], dict[str, float | None]], cfg) -> list[tuple[str, str, str]]:
    """Return [(level, name, message)] with level in PASS / WARN / FAIL."""
    out: list[tuple[str, str, str]] = []
    if not stats_by_window:
        return [("WARN", "no-data", "no windows to check")]

    windows = sorted(stats_by_window)
    last = stats_by_window[windows[-1]]
    best_completion = max((s["completions"] or 0.0) for s in stats_by_window.values())
    best_len = min((s["ep_len"] or float("inf")) for s in stats_by_window.values())

    near = last["near_income"]
    if near is None:
        out.append(("WARN", "goal-occupancy", "term missing from this run (weights unknown)"))
    elif abs(near) > 1e-6:
        out.append(("FAIL", "goal-occupancy", f"goal-radius income is {near:+.2f}/episode, must be 0 - the hotel is back"))
    else:
        out.append(("PASS", "goal-occupancy", "0/episode in the goal radius"))

    heading = last["heading_income"]
    if heading is None:
        out.append(("WARN", "heading-potential", "heading_progress_to_path absent"))
    elif heading > math.pi + 0.05:
        out.append(("FAIL", "heading-potential", f"{heading:+.2f}/episode exceeds the pi bound - the clamp is leaking"))
    elif heading > math.pi * 0.9:
        out.append(("WARN", "heading-potential", f"{heading:.2f}/episode is at the pi bound (saturated)"))
    else:
        out.append(("PASS", "heading-potential", f"{heading:+.2f}/episode within the pi bound"))

    paid = last["timeout_paid"]
    if paid is None:
        out.append(("WARN", "timeout-cost", "too few horizon timeouts in the last window to measure"))
    elif paid > -35:
        out.append(
            ("FAIL", "timeout-cost", f"{paid:.1f} per timeout - every term is multiplied by dt, so a -50 weight pays -10")
        )
    else:
        out.append(("PASS", "timeout-cost", f"{paid:.1f} per horizon timeout"))

    if best_completion > 0.02 and last["completions"] is not None:
        ratio = last["completions"] / best_completion
        msg = f"{last['completions']:.2f}/episode now vs a best window of {best_completion:.2f} ({ratio:.0%})"
        out.append(("PASS" if ratio > 0.6 else "WARN", "completion-trend", msg))

    if last["ep_len"] and math.isfinite(best_len) and last["ep_len"] > 2.0 * best_len:
        out.append(("WARN", "episode-length", f"{last['ep_len']:.0f} steps now vs a best window of {best_len:.0f}"))
    elif last["ep_len"]:
        out.append(("PASS", "episode-length", f"{last['ep_len']:.0f} steps"))

    fall = last["fall"]
    if fall is not None:
        out.append(("WARN" if fall > 0.45 else "PASS", "falls", f"base_contact {fall:.2f}"))

    invalid = last["invalid"]
    if invalid is not None:
        out.append(("WARN" if invalid > 0.15 else "PASS", "path-quality", f"invalid segments {invalid:.2f} of the path"))

    excess = last["dev_excess"]
    if excess is not None:
        out.append(
            ("WARN" if excess > 0.3 else "PASS", "path-tracking",
             f"strays {excess:.2f} m beyond the {last['corridor']:.1f} m corridor (lower bound)")
        )

    ev = last["ev"]
    if ev is not None:
        out.append(("WARN" if ev < 0.5 else "PASS", "critic", f"explained variance {ev:.2f}"))

    to_rate = last["timeout_rate"]
    if to_rate is not None:
        out.append(("WARN" if to_rate > 0.25 else "PASS", "timeout-share", f"{to_rate:.2f} of episodes"))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_windows(text: str | None, last_step: int) -> list[tuple[int, int]]:
    if text:
        out = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            a, b = part.split("-", 1)
            out.append((int(a), int(b)))
        return out
    # auto: 250-iteration buckets over the completed part of the run, at most 12 rows
    end = max(last_step + 1, 1)
    width = max(50, min(250, end // 12 or 50))
    return [(a, min(a + width - 1, last_step)) for a in range(0, end, width)]


def print_table(label: str, cfg, windows, stats) -> None:
    print(f"\n--- {label}  (horizon {cfg['episode_length_s']:.0f} s, step_dt {cfg['step_dt']:.2f} s, config: {Path(cfg['cfg_source']).name if cfg['cfg_source'] != 'defaults' else 'defaults'})")
    print(
        f"{'window':>12} {'SR':>6} {'comp':>6} {'ep_len':>7} {'timeout':>8} {'fall':>6} "
        f"{'devX':>6} {'paid_m':>7} {'head':>7} {'near':>6} {'to$':>7} {'EV':>6}"
    )
    for key in windows:
        s = stats[key]
        print(
            f"{key[0]:>5}-{key[1]:<6} {fmt(s['sr'], 6, 3)} {fmt(s['completions'], 6)} {fmt(s['ep_len'], 7, 0)} "
            f"{fmt(s['timeout_rate'], 8)} {fmt(s['fall'], 6)} {fmt(s['dev_excess'], 6)} {fmt(s['paid_m'], 7, 1)} "
            f"{fmt(s['heading_income'], 7)} {fmt(s['near_income'], 6)} {fmt(s['timeout_paid'], 7, 1)} {fmt(s['ev'], 6)}"
        )
    print("  (comp = completions/episode, head = heading income/episode vs pi, near = goal-radius income/episode,")
    print("   to$ = what one horizon timeout actually costs, devX = metres strayed beyond the deviation corridor)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories or event files")
    ap.add_argument("--windows", default="", help="comma-separated windows, e.g. 0-250,250-500 (default: auto buckets)")
    ap.add_argument("--steps", default="", help="comma-separated exact iterations for raw tag lookup")
    ap.add_argument("--tags", default="", help="comma-separated tags for --steps/--tags-only mode")
    ap.add_argument("--tags-only", action="store_true", help="only the raw tag table (old behaviour)")
    ap.add_argument("--list-tags", action="store_true", help="list scalar tags for each run and exit")
    ap.add_argument("--no-checks", action="store_true", help="skip the health checks")
    ap.add_argument("--json", dest="json_path", default="", help="write structured results to JSON")
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        event_path = resolve_event(spec)
        run_dir = Path(spec).expanduser()
        cfg = load_run_config(run_dir if run_dir.is_dir() else run_dir.parent)
        rows = read_events(event_path)
        if not rows:
            print(f"warning: no scalars parsed from {event_path}", file=sys.stderr)
        scalars: dict[str, list[tuple[float, int, str, float]]] = {}
        for row in rows:
            scalars.setdefault(row[2], []).append(row)
        walls = [r[0] for r in rows if r[0] > 0]
        runs.append(
            {
                "label": run_dir.resolve().name if run_dir.is_dir() else event_path.name,
                "event": str(event_path),
                "last_step": max((r[1] for r in rows), default=0),
                "wall_hours": (max(walls) - min(walls)) / 3600.0 if walls else 0.0,
                "scalars": scalars,
                "cfg": cfg,
            }
        )

    if args.list_tags:
        for run in runs:
            print(f"=== {run['label']} ===")
            for tag in sorted(run["scalars"]):
                print(tag)
        return 0

    result: dict[str, Any] = {"runs": [], "windows": {}, "checks": {}, "raw": {}, "summary": []}
    summary: list[dict[str, Any]] = []

    for run in runs:
        print(f"\n=== {run['label']} ===")
        print(f"event: {run['event']}")
        print(f"last_step: {run['last_step']}  wall_hours: {run['wall_hours']:.2f}  config: {run['cfg']['cfg_source']}")
        result["runs"].append(
            {
                "label": run["label"],
                "event": run["event"],
                "last_step": run["last_step"],
                "wall_hours": run["wall_hours"],
                "cfg_source": run["cfg"]["cfg_source"],
            }
        )

        windows = parse_windows(args.windows or None, run["last_step"])
        stats = {w: derive(run["scalars"], run["cfg"], w[0], w[1]) for w in windows}
        # Trend checks always look at the whole run, not at the windows the user asked to display:
        # "did it peak and fall back" is a property of the run, not of the reported slice.
        whole_run = {w: derive(run["scalars"], run["cfg"], w[0], w[1]) for w in parse_windows(None, run["last_step"])}
        if not args.tags_only:
            print_table(run["label"], run["cfg"], windows, stats)
            result["windows"][run["label"]] = {f"{a}-{b}": stats[(a, b)] for a, b in windows}
            if not args.no_checks:
                checks = run_checks(whole_run, run["cfg"])
                print("\n  checks:")
                for level, name, message in checks:
                    print(f"    [{level:<4}] {name:<18} {message}")
                result["checks"][run["label"]] = [{"level": lv, "name": n, "message": m} for lv, n, m in checks]
            # The final bucket of a running job can hold one or two iterations and is pure noise;
            # use the last bucket that actually has samples.
            keys = sorted(whole_run)
            last_key = None
            for key in reversed(keys):
                if window_count(run["scalars"], "Train/mean_reward", key[0], key[1]) >= 10:
                    last_key = key
                    break
            last_key = last_key or (keys[-1] if keys else None)
            summary.append(
                {
                    "label": run["label"],
                    "last_step": run["last_step"],
                    "stats": whole_run.get(last_key) if last_key else None,
                    "fails": sum(1 for lv, _n, _m in checks if lv == "FAIL") if not args.no_checks else 0,
                    "warns": sum(1 for lv, _n, _m in checks if lv == "WARN") if not args.no_checks else 0,
                }
            )

        if args.steps:
            print("\n  raw tags at steps:")
            tags = [x.strip() for x in args.tags.split(",") if x.strip()] or DEFAULT_TAGS
            for step in [int(s) for s in args.steps.split(",") if s.strip()]:
                print(f"    -- step {step}")
                for tag in tags:
                    value = nearest(run["scalars"], tag, step)
                    print(f"      {tag:<46} {fmt(value, 10, 4)}")
                    result["raw"].setdefault(run["label"], {}).setdefault(str(step), {})[tag] = value

    if len(summary) > 1:
        print("\n===== cross-run summary (last bucket of each run, whole-run checks) =====")
        print(
            f"{'run':<44} {'iter':>5} {'SR':>6} {'comp':>6} {'ep_len':>7} {'head':>6} "
            f"{'near':>7} {'to$':>7} {'FAIL':>5} {'WARN':>5}"
        )
        for row in summary:
            s_ = row["stats"] or {}
            print(
                f"{row['label'][:44]:<44} {row['last_step']:>5} {fmt(s_.get('sr'), 6, 3)} {fmt(s_.get('completions'), 6)} "
                f"{fmt(s_.get('ep_len'), 7, 0)} {fmt(s_.get('heading_income'), 6)} {fmt(s_.get('near_income'), 7)} "
                f"{fmt(s_.get('timeout_paid'), 7, 1)} {row['fails']:>5} {row['warns']:>5}"
            )
    result["summary"] = [
        {
            "label": row["label"],
            "last_step": row["last_step"],
            "fails": row["fails"],
            "warns": row["warns"],
            "stats": row["stats"],
        }
        for row in summary
    ]

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\nJSON written to {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
