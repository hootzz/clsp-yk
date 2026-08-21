"""
Merge watch PPG/ACC data with phone/PC digital events into datastream.jsonl.

Each 10-second window produces one record:
  { timestamp, window, user_state{physical,social,digital}, text_context, ppg, measures }
  - text_context: physical + social + digital context rendered by text_context.py
  - ppg: 1250-sample model input
  - measures: cognitive_load, valence, and arousal initialized to null

Usage:
  python merge.py \
      --processed Data_files/ppg/processed_XXXX.csv \
      --raw       Data_files/ppg/raw_XXXX.csv \
      --digital   Data_files/digital/events_pc.jsonl Data_files/digital/events_android.jsonl \
      --out       datastream.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Local context modules.
sys.path.append(str(Path(__file__).parent / "state_estimation_mvp"))
sys.path.append(str(Path(__file__).parent / "context"))

from text_context import build_text
from digital import _extract_valid_events, summarize_events

# Configuration.
WINDOW_MS      = 10_000      # 10-second window
TARGET_SAMPLES = 1250        # 125 Hz × 10 s

# Treat timezone-naive Android timestamps as UTC+9.
ANDROID_TZ_OFFSET_HOURS = 9

# Map ACC motion level (m/s²) to movement labels.
MOVE_SEDENTARY_MAX = 0.3     # sedentary threshold
MOVE_LIGHT_MAX     = 1.5     # light threshold
MOVE_MODERATE_MAX  = 4.0     # moderate threshold; above this is vigorous

DEFAULT_POSTURE = None       # Posture is not inferred from watch ACC.

SOCIAL_ENGAGEMENT        = "low"
SOCIAL_DENSITY           = "0"
GRAVITY = 9.81
UTC = timezone.utc


def to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


# Load processed PPG windows.
def load_ppg_windows(path: Path) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("flatline_skipped") == "1":
                continue
            try:
                end_ms = int(row["window_end_ts"])
            except (KeyError, ValueError):
                continue
            ppg, ok = [], True
            for i in range(TARGET_SAMPLES):
                v = row.get(f"ppg_{i}", "")
                if v in ("", None):
                    ok = False
                    break
                ppg.append(float(v))
            if ok:
                out.append({"end_ms": end_ms, "start_ms": end_ms - WINDOW_MS, "ppg": ppg})
    return out


# Load watch ACC and derive movement labels.
def load_acc(path: Path) -> list[tuple[int, float]]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ts = int(row["ts"])
                ax, ay, az = float(row["acc_x"]), float(row["acc_y"]), float(row["acc_z"])
            except (KeyError, ValueError, TypeError):
                continue
            mag = (ax * ax + ay * ay + az * az) ** 0.5
            out.append((ts, abs(mag - GRAVITY)))
    out.sort(key=lambda x: x[0])
    return out


def physical_state(start_ms: int, end_ms: int, acc: list[tuple[int, float]]) -> dict:
    vals = [m for ts, m in acc if start_ms <= ts <= end_ms]
    if not vals:
        return {"posture": DEFAULT_POSTURE, "movement": None, "motion_level": None}
    motion = sum(vals) / len(vals)
    if motion <= MOVE_SEDENTARY_MAX:
        mv = "sedentary"
    elif motion <= MOVE_LIGHT_MAX:
        mv = "light"
    elif motion <= MOVE_MODERATE_MAX:
        mv = "moderate"
    else:
        mv = "vigorous"
    return {"posture": DEFAULT_POSTURE, "movement": mv, "motion_level": round(motion, 3)}


# Load phone/PC digital events.
def load_raw_events(paths: list[Path]) -> list[dict]:
    events = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def digital_state(start_ms: int, end_ms: int, valid_events: list[dict]) -> dict:
    matched = [
        e for e in valid_events
        if int(e["start_time"].timestamp() * 1000) <= end_ms
        and int(e["end_time"].timestamp() * 1000) >= start_ms
    ]

    if not matched:
        return {"device_type": None, "usage": None, "active_devices": []}

    usage = summarize_events(matched)
    devices = sorted({e.get("source", "unknown") for e in matched})
    device_type = matched[0].get("source", "unknown")

    return {
        "device_type":    device_type,
        "usage":          usage,
        "active_devices": devices,
    }


# Build text context.
def build_text_context(phys: dict, soc: dict, digi: dict) -> str:
    context = {
        "posture":                     phys.get("posture") or "sitting",
        "movement":                    phys.get("movement") or "sedentary",
        "social_engagement":           soc.get("engagement") or "low",
        "interpersonal_density":       str(soc.get("interpersonal_density") or "0"),
        "device_interaction_behavior": "passive_viewing",
        "environment":                 "indoor",
        "temporal":                    "continuous",
        "digital_summary":             digi.get("usage") or "",
    }
    return build_text(context)


# Merge streams.
def merge(windows, acc, valid_events):
    rows = []
    for w in windows:
        phys = physical_state(w["start_ms"], w["end_ms"], acc)
        soc  = {
            "engagement":            SOCIAL_ENGAGEMENT,
            "interpersonal_density": SOCIAL_DENSITY,
        }
        digi = digital_state(w["start_ms"], w["end_ms"], valid_events)
        rows.append({
            "timestamp":    to_iso(w["end_ms"]),
            "window":       {"start": to_iso(w["start_ms"]), "end": to_iso(w["end_ms"]), "len_s": WINDOW_MS // 1000},
            "user_state":   {"physical": phys, "social": soc, "digital": digi},
            "text_context": build_text_context(phys, soc, digi),
            "ppg":          w["ppg"],
            "measures":     {"cognitive_load": None, "valence": None, "arousal": None},
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, type=Path, help="watch processed_*.csv")
    ap.add_argument("--raw",       type=Path, default=None,  help="watch raw_*.csv for movement labels")
    ap.add_argument("--digital",   nargs="+", type=Path, default=[], help="phone/PC events_*.jsonl")
    ap.add_argument("--out",       type=Path, default=Path("datastream.jsonl"))
    args = ap.parse_args()

    windows = load_ppg_windows(args.processed)
    acc     = load_acc(args.raw) if args.raw else []

    raw_events   = load_raw_events(args.digital)
    valid_events = _extract_valid_events(raw_events) if raw_events else []

    if not windows:
        sys.exit("[merge] No PPG windows found. Check the processed file.")

    rows = merge(windows, acc, valid_events)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n      = len(rows)
    n_move = sum(1 for r in rows if r["user_state"]["physical"]["movement"] is not None)
    n_digi = sum(1 for r in rows if r["user_state"]["digital"]["usage"] is not None)
    print(f"[merge] Merge complete → {args.out}")
    print(f"  Windows (rows)    : {n}")
    print(f"  Watch time range  : {to_iso(windows[0]['start_ms'])} ~ {to_iso(windows[-1]['end_ms'])}")
    print(f"  Movement available: {n_move}/{n}")
    print(f"  Digital available : {n_digi}/{n}")
    if acc and n_move == 0:
        print("  Warning: raw ACC does not overlap the watch window time range.")
    if raw_events and n_digi == 0:
        print("  Warning: digital events do not overlap the watch window time range.")


if __name__ == "__main__":
    main()
