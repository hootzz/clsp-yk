"""
context/digital.py

PC(app_switch) / Mobile(app_start+app_close) JSONL 이벤트 파일을 읽어서
GPT-4o-mini로 digital_summary 문장을 생성하는 모듈.

사용법:
    from context.digital import summarize_events, match_events_to_window, process_jsonl

    # 파일 전체를 하나의 요약으로 (짧고 통제된 실험용)
    process_jsonl("data/events_20260424.jsonl", "data/events_20260424_summarized.jsonl")

    # PPG window에 해당하는 이벤트 찾아서 요약 (필요시)
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


# ── GPT 프롬프트 ──────────────────────────────────────────
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


# ── timestamp 파싱 ────────────────────────────────────────
def _parse_dt(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ── 유효한 이벤트 추출 ────────────────────────────────────
def _extract_valid_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    PC / Mobile 이벤트에서 유효한 사용 구간만 추출.

    PC (app_switch):
      - 같은 app + 같은 timestamp로 dur=0.0 / dur>0 쌍 확인
      - 사용 구간 = timestamp - duration ~ timestamp

    Mobile (app_start / app_close):
      - 같은 app의 app_start → app_close 쌍 확인
      - 사용 구간 = app_start.timestamp ~ app_close.timestamp
    """
    valid: list[dict[str, Any]] = []

    # ── PC: app_switch ────────────────────────────────────
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

    # ── Mobile: app_start / app_close ────────────────────
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


# ── 이벤트 리스트 → llm_summary ──────────────────────────
def summarize_events(valid_events: list[dict[str, Any]]) -> str:
    """
    유효한 이벤트 리스트를 받아서 llm_summary 문장을 반환.
    앱 이름은 raw 이름 그대로 GPT에 넘기고 GPT가 직접 변환.
    """
    if not valid_events:
        return ""

    # 앱별 총 사용 시간 집계
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

    # PC 이벤트의 title 추가 (중복 제거, 최대 3개)
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


# ── PPG window 기준 이벤트 매칭 ───────────────────────────
def match_events_to_window(
    events: list[dict[str, Any]],
    window_start: str,
    window_end: str,
) -> str:
    """
    PPG window (window_start ~ window_end) 구간과 겹치는 이벤트를 찾아서
    llm_summary를 반환. 겹치는 이벤트 없으면 빈 문자열 반환.
    """
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


# ── 배치 처리: 전체 파일을 하나의 요약으로 ───────────────
def process_jsonl(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    JSONL 파일 전체를 읽어서 하나의 llm_summary로 저장.
    짧고 통제된 실험 환경에서 전체 세션이 하나의 맥락이라는 가정.
    """
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
    print(f"요약: {summary}")

    record = {
        "device_type": raw_events[0].get("device_type", "unknown"),
        "start_time":  min(e["start_time"] for e in valid_events).isoformat(),
        "end_time":    max(e["end_time"] for e in valid_events).isoformat(),
        "llm_summary": summary,
    }

    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--out",   type=str, required=True)
    args = parser.parse_args()

    process_jsonl(args.jsonl, args.out)