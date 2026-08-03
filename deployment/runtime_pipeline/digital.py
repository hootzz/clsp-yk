"""Normalize PC/Android app-switch JSONL into time intervals and summaries."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PRIVATE_TITLE_TERMS = (
    "stress",
    "anxiety",
    "arousal",
    "valence",
    "cognitive load",
)


def load_events(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                event["_source_file"] = str(path)
                events.append(event)
    return events


def _parse_datetime(value: str, naive_offset_hours: int) -> datetime:
    clean = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(clean)
    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone(timedelta(hours=int(naive_offset_hours)))
        )
    return parsed.astimezone(timezone.utc)


def extract_intervals(
    events: list[dict[str, Any]],
    naive_offset_hours: int = 9,
) -> list[dict[str, Any]]:
    """Pair the event formats emitted by the imported PC and Android collectors."""
    intervals: list[dict[str, Any]] = []

    pc_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        if event.get("device_type") == "pc" or event.get("event_type") == "app_switch":
            key = (str(event.get("app", "")), str(event.get("timestamp", "")))
            pc_groups.setdefault(key, []).append(event)
    for (app, timestamp), group in pc_groups.items():
        ended = [
            event for event in group
            if float(event.get("duration_seconds") or 0.0) > 0.0
        ]
        if not ended:
            continue
        event = max(ended, key=lambda item: float(item.get("duration_seconds") or 0.0))
        duration = float(event["duration_seconds"])
        start = _parse_datetime(timestamp, naive_offset_hours)
        intervals.append(
            {
                "app": app,
                "title": str(event.get("title") or ""),
                "source": "pc",
                "start_time": start,
                "end_time": start + timedelta(seconds=duration),
                "duration": duration,
            }
        )

    pending: dict[str, tuple[datetime, dict[str, Any]]] = {}
    mobile = [
        event for event in events
        if event.get("event_type") in {"app_start", "app_close"}
    ]
    mobile.sort(key=lambda item: str(item.get("timestamp", "")))
    for event in mobile:
        app = str(event.get("app", ""))
        stamp = _parse_datetime(str(event.get("timestamp", "")), naive_offset_hours)
        if event.get("event_type") == "app_start":
            pending[app] = (stamp, event)
            continue
        start_pack = pending.pop(app, None)
        if start_pack is None:
            duration = float(event.get("duration_seconds") or 0.0)
            if duration <= 0:
                continue
            start = stamp - timedelta(seconds=duration)
        else:
            start = start_pack[0]
            duration = max(0.0, (stamp - start).total_seconds())
        if duration <= 0:
            continue
        intervals.append(
            {
                "app": app,
                "title": str(event.get("title") or ""),
                "source": "android",
                "start_time": start,
                "end_time": stamp,
                "duration": duration,
            }
        )
    return sorted(intervals, key=lambda item: item["start_time"])


def overlapping(
    intervals: list[dict[str, Any]],
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    start = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
    end = datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc)
    return [
        interval for interval in intervals
        if interval["start_time"] <= end and interval["end_time"] >= start
    ]


def _safe_title(title: str) -> str:
    clean = " ".join(str(title).split())
    low = clean.lower()
    if any(term in low for term in PRIVATE_TITLE_TERMS):
        return ""
    # Avoid putting an arbitrarily long/private document title into model input.
    return clean[:80]


def summarize_intervals(intervals: list[dict[str, Any]], limit: int = 3) -> str:
    if not intervals:
        return ""
    totals: dict[str, float] = {}
    for interval in intervals:
        app = str(interval.get("app") or "unknown application")
        totals[app] = totals.get(app, 0.0) + float(interval.get("duration") or 0.0)
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]
    chunks = [f"{app} for about {round(seconds)} seconds" for app, seconds in ranked]
    titles = [
        value for value in dict.fromkeys(
            _safe_title(str(interval.get("title") or "")) for interval in intervals
        ) if value
    ][:2]
    summary = "using " + ", ".join(chunks)
    if titles:
        summary += "; visible window titles included " + " and ".join(titles)
    return summary
