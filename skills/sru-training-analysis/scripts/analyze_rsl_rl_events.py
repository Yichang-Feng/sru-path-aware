#!/usr/bin/env python3
"""Parse and compare RSL-RL TensorBoard event logs without tensorboard installed.

Supports run directories (uses newest events.out.tfevents.*) or direct event files.
The parser handles the subset of protobuf used by RSL-RL scalar summaries:
wall_time, step, Summary.Value.tag, simple_value, and tensor.float_val.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_TAGS = [
    "Train/mean_reward",
    "Metrics/robot_goal/success_rate",
    "Metrics/robot_goal/velocity_toward_goal",
    "Episode_Reward/action_rate_l1",
    "Episode_Reward/joint_acc_l2_joint",
    "Episode_Reward/progress_along_path",
    "Episode_Reward/reach_goal_xy_soft",
    "Episode_Reward/reach_goal_xy_tight",
    "Episode_Termination/base_contact",
    "Episode_Termination/time_out",
    "Policy/mean_noise_std",
    "Loss/learning_rate",
    "Loss/kl_divergence",
]


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
        elif field == 2 and wire in (1, 5):  # simple_value (float/double depending on schema)
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


def parse_steps(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_windows(text: str | None) -> list[tuple[int, int]]:
    if not text:
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-", 1)
        out.append((int(a), int(b)))
    return out


def nearest(series: list[tuple[float, int, str, float]], step: int, tol: int = 1) -> float | None:
    best = None
    for wall, s, tag, value in series:
        dist = abs(s - step)
        if best is None or dist < best[0]:
            best = (dist, value)
    return best[1] if best is not None and best[0] <= tol else None


def window_mean(series: list[tuple[float, int, str, float]], a: int, b: int) -> float | None:
    vals = [value for wall, s, tag, value in series if a <= s <= b]
    return sum(vals) / len(vals) if vals else None


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories or event files")
    ap.add_argument("--steps", default="", help="comma-separated exact/near iterations, e.g. 25,50,100")
    ap.add_argument("--windows", default="", help="comma-separated inclusive windows, e.g. 1-50,100-200")
    ap.add_argument("--tags", default="", help="comma-separated tags; default is the SRU navigation set")
    ap.add_argument("--list-tags", action="store_true", help="list scalar tags for each run and exit")
    ap.add_argument("--json", dest="json_path", default="", help="write structured results to JSON")
    args = ap.parse_args()

    tags = [x.strip() for x in args.tags.split(",") if x.strip()] or DEFAULT_TAGS
    steps = parse_steps(args.steps)
    windows = parse_windows(args.windows)

    runs = []
    for spec in args.runs:
        event_path = resolve_event(spec)
        rows = read_events(event_path)
        if not rows:
            print(f"warning: no scalars parsed from {event_path}", file=sys.stderr)
        scalars: dict[str, list[tuple[float, int, str, float]]] = {}
        for row in rows:
            scalars.setdefault(row[2], []).append(row)
        walls = [r[0] for r in rows if r[0] > 0]
        label = Path(spec).expanduser().resolve().name
        runs.append(
            {
                "label": label,
                "spec": str(spec),
                "event": str(event_path),
                "last_step": max((r[1] for r in rows), default=0),
                "wall_hours": (max(walls) - min(walls)) / 3600.0 if walls else 0.0,
                "scalars": scalars,
            }
        )

    if args.list_tags:
        for run in runs:
            print(f"=== {run['label']} ({run['event']}) ===")
            for tag in sorted(run["scalars"]):
                print(tag)
        return 0

    result: dict = {
        "runs": [],
        "at_steps": {},
        "windows": {},
    }

    for run in runs:
        print(f"=== {run['label']} ===")
        print(f"event: {run['event']}")
        print(f"last_step: {run['last_step']}  wall_hours: {run['wall_hours']:.2f}")
        result["runs"].append(
            {
                "label": run["label"],
                "event": run["event"],
                "last_step": run["last_step"],
                "wall_hours": run["wall_hours"],
            }
        )

    if steps:
        print("\n===== values at selected steps =====")
        for step in steps:
            print(f"\n-- step {step}")
            header = f"{'tag':<48}" + "".join(f"{r['label'][:20]:>22}" for r in runs)
            print(header)
            for tag in tags:
                line = f"{tag:<48}"
                for run in runs:
                    line += f"{fmt(nearest(run['scalars'].get(tag, []), step)):>22}"
                print(line)
                for run in runs:
                    result["at_steps"].setdefault(str(step), {}).setdefault(run["label"], {})[tag] = nearest(
                        run["scalars"].get(tag, []), step
                    )

    if windows:
        print("\n===== window means =====")
        for a, b in windows:
            print(f"\n-- window {a}-{b}")
            header = f"{'tag':<48}" + "".join(f"{r['label'][:20]:>22}" for r in runs)
            print(header)
            for tag in tags:
                line = f"{tag:<48}"
                for run in runs:
                    line += f"{fmt(window_mean(run['scalars'].get(tag, []), a, b)):>22}"
                print(line)
                for run in runs:
                    result["windows"].setdefault(f"{a}-{b}", {}).setdefault(run["label"], {})[tag] = window_mean(
                        run["scalars"].get(tag, []), a, b
                    )

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON written to {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
