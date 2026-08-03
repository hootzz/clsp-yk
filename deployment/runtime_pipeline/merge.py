"""Merge processed Galaxy PPG, motion, and digital logs into model-ready JSONL."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .context_text import build_context_text, context_signature
    from .digital import extract_intervals, load_events, overlapping, summarize_intervals
except ImportError:
    from context_text import build_context_text, context_signature
    from digital import extract_intervals, load_events, overlapping, summarize_intervals

WINDOW_MS = 10_000
TARGET_SAMPLES = 1_250
GRAVITY = 9.81
MOVE_SEDENTARY_MAX = 0.3
MOVE_LIGHT_MAX = 1.5
MOVE_MODERATE_MAX = 4.0


def to_iso(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_ppg_windows(path: Path) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            if str(row.get("flatline_skipped", "0")) == "1":
                continue
            try:
                end_ms = int(row["window_end_ts"])
                ppg = [float(row[f"ppg_{index}"]) for index in range(TARGET_SAMPLES)]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid processed PPG row") from exc
            windows.append(
                {
                    "start_ms": end_ms - WINDOW_MS,
                    "end_ms": end_ms,
                    "ppg": ppg,
                    "window_idx": int(row.get("window_idx") or len(windows)),
                }
            )
    return windows


def load_acc(path: Path | None) -> list[tuple[int, float]]:
    if path is None:
        return []
    values: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                timestamp = int(row["ts"])
                x = float(row["acc_x"])
                y = float(row["acc_y"])
                z = float(row["acc_z"])
            except (KeyError, TypeError, ValueError):
                continue
            magnitude = (x * x + y * y + z * z) ** 0.5
            values.append((timestamp, abs(magnitude - GRAVITY)))
    return sorted(values)


def movement_for_window(
    start_ms: int,
    end_ms: int,
    acc: list[tuple[int, float]],
) -> tuple[str, float | None]:
    motion = [value for stamp, value in acc if start_ms <= stamp <= end_ms]
    if not motion:
        return "not_recorded", None
    mean_motion = sum(motion) / len(motion)
    if mean_motion <= MOVE_SEDENTARY_MAX:
        label = "sedentary"
    elif mean_motion <= MOVE_LIGHT_MAX:
        label = "light"
    elif mean_motion <= MOVE_MODERATE_MAX:
        label = "moderate"
    else:
        label = "vigorous"
    return label, round(mean_motion, 4)


def merge_records(
    windows: list[dict[str, Any]],
    acc: list[tuple[int, float]],
    digital_intervals: list[dict[str, Any]],
    *,
    posture: str,
    social_engagement: str,
    interpersonal_density: str,
    environment: str,
    temporal: str,
    context_backend: str,
    context_model: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for window in windows:
        movement, motion_level = movement_for_window(
            window["start_ms"], window["end_ms"], acc
        )
        matched = overlapping(
            digital_intervals, window["start_ms"], window["end_ms"]
        )
        digital_summary = summarize_intervals(matched)
        device_behavior = "active_input" if matched else "no_interaction"
        slots = {
            "posture": posture,
            "movement": movement,
            "social_engagement": social_engagement,
            "interpersonal_density": interpersonal_density,
            "device_interaction_behavior": device_behavior,
            "environment": environment,
            "temporal": temporal,
        }
        text = build_context_text(
            slots,
            observable_detail=digital_summary,
            backend=context_backend,
            model=context_model,
        )
        records.append(
            {
                "sample_id": f"watch_{window['end_ms']}",
                "timestamp": to_iso(window["end_ms"]),
                "window": {
                    "start": to_iso(window["start_ms"]),
                    "end": to_iso(window["end_ms"]),
                    "length_seconds": WINDOW_MS // 1000,
                    "index": window["window_idx"],
                },
                "context_slots": slots,
                "context_signature": context_signature(slots),
                "context_source": {
                    "posture": "cli_protocol_value" if posture != "not_recorded" else "not_recorded",
                    "movement": "watch_acc" if motion_level is not None else "not_recorded",
                    "social_engagement": "cli_or_sensor",
                    "interpersonal_density": "cli_or_sensor",
                    "device_interaction_behavior": "digital_event_overlap",
                    "environment": "cli_protocol_value",
                    "temporal": "cli_protocol_value",
                },
                "text_context": text,
                "user_state": {
                    "physical": {
                        "posture": posture,
                        "movement": movement,
                        "motion_level": motion_level,
                    },
                    "social": {
                        "engagement": social_engagement,
                        "interpersonal_density": interpersonal_density,
                    },
                    "digital": {
                        "active": bool(matched),
                        "summary": digital_summary,
                        "event_count": len(matched),
                    },
                },
                "ppg": window["ppg"],
                "measures": {
                    "cognitive_load": None,
                    "valence": None,
                    "arousal": None,
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--digital", nargs="*", type=Path, default=[])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--posture", default="not_recorded")
    parser.add_argument("--social-engagement", default="not_recorded")
    parser.add_argument("--interpersonal-density", default="not_recorded")
    parser.add_argument("--environment", default="indoor")
    parser.add_argument("--temporal", default="continuous")
    parser.add_argument(
        "--android-naive-timezone-offset",
        type=int,
        default=9,
        help="UTC offset used only for Android timestamps without timezone",
    )
    parser.add_argument(
        "--context-backend",
        choices=("deterministic", "openai"),
        default="deterministic",
    )
    parser.add_argument("--context-model", default="gpt-4o-mini")
    args = parser.parse_args()

    windows = load_ppg_windows(args.processed)
    if not windows:
        raise SystemExit(f"[merge] no usable PPG windows: {args.processed}")
    acc = load_acc(args.raw)
    events = load_events(args.digital)
    intervals = extract_intervals(
        events, naive_offset_hours=args.android_naive_timezone_offset
    )
    records = merge_records(
        windows,
        acc,
        intervals,
        posture=args.posture,
        social_engagement=args.social_engagement,
        interpersonal_density=args.interpersonal_density,
        environment=args.environment,
        temporal=args.temporal,
        context_backend=args.context_backend,
        context_model=args.context_model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[merge] wrote {len(records)} records -> {args.out}")
    print(f"  motion observed: {sum(r['user_state']['physical']['motion_level'] is not None for r in records)}")
    print(f"  digital overlap: {sum(r['user_state']['digital']['active'] for r in records)}")
    if args.posture == "not_recorded":
        print("  note: posture is explicit not_recorded; pass --posture when protocol-observed")


if __name__ == "__main__":
    main()
