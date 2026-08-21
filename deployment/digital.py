"""
Digital activity summarization for PC and mobile JSONL event logs.

PC uses app_switch events. Mobile uses paired app_start/app_close events.
GPT-4o-mini generates a concise digital activity summary.

Usage:
    from context.digital import summarize_events, match_events_to_window, process_jsonl

    process_jsonl("data/events_20260424.jsonl", "data/events_20260424_summarized.jsonl")

    summary = match_events_to_window(
        events=events,
        window_start="2026-04-13T04:47:57+00:00",
        window_end="2026-04-13T04:48:07+00:00",
    )
"""

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

for _env in [
    Path(__file__).parent.parent / ".env",
    Path(".env"),
]:
    if _env.exists():
        load_dotenv(_env)
        break

import json
from datetime import datetime, timezone, timedelta
from typing import Any

from openai import OpenAI


# GPT prompt.
_SYSTEM_PROMPT = (
    "You summarize a user's recent digital activity for an XR personalization system. "
    "Given a list of app package names or executable names, their durations, and page titles, "
    "write ONE natural English sentence (15-25 words). "
    "Rules: "
    "1. Do NOT mention emotions, stress, cognitive load, or mental state. "
    "2. Translate app package names to common app names (e.g. com.google.android.youtube → YouTube). "
    "3. Ignore system apps like settings, file explorer, or device trackers. "
    "4. Include the duration spent on each app in the summary. "
    "5. Use the page title to describe what the user was specifically doing when relevant. "
    "6. Output the sentence only, no extra text."
)

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# Timestamp parsing.
def _parse_dt(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# Valid event extraction.
def _extract_valid_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract valid usage intervals from PC and mobile events.

    PC app_switch events are paired by app and timestamp.
    Mobile app_start/app_close events are paired by app.
    """
    valid: list[dict[str, Any]] = []

    # PC app_switch events.
    pc_events = [e for e in events if e.get("event_type") == "app_switch"]

    pc_groups: dict[tuple[str, str], list[dict]] = {}
    for e in pc_events:
        key = (e.get("app", ""), e.get("timestamp", ""))
        pc_groups.setdefault(key, []).append(e)

    for (app, ts), group in pc_groups.items():
        has_start = any(float(e.get("duration_seconds", 0)) == 0.0 for e in group)
        end_events = [e for e in group if float(e.get("duration_seconds", 0)) > 0]

        if has_start and end_events:
            e = end_events[0]
            dur = float(e["duration_seconds"])
            end_dt = _parse_dt(ts)
            start_dt = end_dt - timedelta(seconds=dur)
            valid.append({
                "app":        app,
                "start_time": start_dt,
                "end_time":   end_dt,
                "duration":   dur,
                "title":      e.get("title", ""),
                "source":     "pc",
            })

    # Mobile app_start/app_close events.
    pending_starts: dict[str, datetime] = {}

    for e in sorted(events, key=lambda x: x.get("timestamp", "")):
        etype = e.get("event_type", "")
        app   = e.get("app", "")

        if etype == "app_start":
            pending_starts[app] = _parse_dt(e["timestamp"])

        elif etype == "app_close":
            dur = float(e.get("duration_seconds", 0))
            if dur > 0 and app in pending_starts:
                start_dt = pending_starts.pop(app)
                end_dt   = _parse_dt(e["timestamp"])
                valid.append({
                    "app":        app,
                    "start_time": start_dt,
                    "end_time":   end_dt,
                    "duration":   dur,
                    "title":      e.get("title", ""),
                    "source":     "mobile",
                })

    return valid


# Event list to LLM summary.
def summarize_events(valid_events: list[dict[str, Any]]) -> str:
    """Generate an LLM summary from valid events."""
    if not valid_events:
        return ""

    # Aggregate usage duration by app.
    app_durations: dict[str, float] = {}
    for e in valid_events:
        app = e["app"]
        app_durations[app] = app_durations.get(app, 0) + e["duration"]

    if not app_durations:
        return ""

    lines = [
        f"- {app}: {dur:.0f}s"
        for app, dur in sorted(app_durations.items(), key=lambda x: -x[1])
    ]

    # Add unique page titles from PC events.
    pc_events = [e for e in valid_events if e.get("source") == "pc"]
    if pc_events:
        titles = list({e["title"] for e in pc_events if e.get("title")})
        if titles:
            lines.append(f"- Page titles: {', '.join(titles)}")

    user_prompt = (
        "Apps used:\n"
        + "\n".join(lines)
        + "\n\nWrite one sentence summarizing this digital activity."
    )

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=60,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# Match events to a PPG window.
def match_events_to_window(
    events: list[dict[str, Any]],
    window_start: str,
    window_end: str,
) -> str:
    """Summarize events overlapping the specified PPG window."""
    w_start = _parse_dt(window_start)
    w_end   = _parse_dt(window_end)

    valid = _extract_valid_events(events)

    matched = [
        e for e in valid
        if e["start_time"] <= w_end and e["end_time"] >= w_start
    ]

    if not matched:
        return ""

    return summarize_events(matched)


# Batch processing.
def process_jsonl(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """Summarize an entire JSONL session into a single llm_summary."""
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        raw_events = [json.loads(l) for l in f if l.strip()]

    if not raw_events:
        print("No events found.")
        return

    valid_events = _extract_valid_events(raw_events)

    if not valid_events:
        print("No valid events after pairing.")
        return

    summary = summarize_events(valid_events)
    print(f"Summary: {summary}")

    record = {
        "device_type": raw_events[0].get("device_type", "unknown"),
        "start_time":  min(e["start_time"] for e in valid_events).isoformat(),
        "end_time":    max(e["end_time"] for e in valid_events).isoformat(),
        "llm_summary": summary,
    }

    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved: {output_path}")


# CLI.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--out",   type=str, required=True)
    args = parser.parse_args()

    process_jsonl(args.jsonl, args.out)
