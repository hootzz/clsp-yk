"""
merge.py — 워치(PPG/ACC) + 폰/PC(digital) → 통합 datastream.jsonl  (SCHEMA.md [A-2])

각 10초 윈도우마다 한 줄(= memory object):
  { timestamp, window, user_state{physical,social,digital}, text_context, ppg, measures }
  - text_context : physical+social+digital → GPT-4o-mini 영어 문장 (text_context.py)
  - ppg          : 모델 신호 입력 (1250샘플)
  - measures     : cognitive_load·valence·arousal = null (모델이 채움)

사용:
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

# ── text_context.py, digital.py 임포트 ─────────────────────
sys.path.append(str(Path(__file__).parent / "state_estimation_mvp"))
sys.path.append(str(Path(__file__).parent / "context"))

from text_context import build_text
from digital import _extract_valid_events, summarize_events

# ──────────────────────────────────────────────
# 설정 (CONFIG)
# ──────────────────────────────────────────────
WINDOW_MS      = 10_000      # 윈도우 10초
TARGET_SAMPLES = 1250        # 125Hz × 10s

# 안드로이드 로그 timestamp에 타임존이 없음 → KST(+9)로 간주
ANDROID_TZ_OFFSET_HOURS = 9

# ACC motion_level(m/s²) → movement 라벨 (우리 state_schema.py 어휘 기준)
MOVE_SEDENTARY_MAX = 0.3     # 이하 → sedentary
MOVE_LIGHT_MAX     = 1.5     # 이하 → light
MOVE_MODERATE_MAX  = 4.0     # 이하 → moderate, 초과 → vigorous

DEFAULT_POSTURE = None       # 워치 ACC로 추정 안 함 → null

SOCIAL_ENGAGEMENT        = "low"   # 우리 state_schema.py 어휘 기준
SOCIAL_DENSITY           = "0"
GRAVITY = 9.81
UTC = timezone.utc


def to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


# ── 1. 워치 PPG 윈도우 (processed_*.csv) ────────
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


# ── 2. 워치 ACC (raw_*.csv) → movement 라벨 ─────
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


# ── 3. 폰/PC digital 이벤트 (events_*.jsonl) ────
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


# ── 4. text_context ──────────────────────────────
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


# ── 5. 합치기 ───────────────────────────────────
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
    ap.add_argument("--processed", required=True, type=Path, help="워치 processed_*.csv")
    ap.add_argument("--raw",       type=Path, default=None,  help="워치 raw_*.csv (movement 라벨용)")
    ap.add_argument("--digital",   nargs="+", type=Path, default=[], help="폰/PC events_*.jsonl")
    ap.add_argument("--out",       type=Path, default=Path("datastream.jsonl"))
    args = ap.parse_args()

    windows = load_ppg_windows(args.processed)
    acc     = load_acc(args.raw) if args.raw else []

    raw_events   = load_raw_events(args.digital)
    valid_events = _extract_valid_events(raw_events) if raw_events else []

    if not windows:
        sys.exit("[merge] PPG 윈도우 없음. processed 파일 확인.")

    rows = merge(windows, acc, valid_events)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n      = len(rows)
    n_move = sum(1 for r in rows if r["user_state"]["physical"]["movement"] is not None)
    n_digi = sum(1 for r in rows if r["user_state"]["digital"]["usage"] is not None)
    print(f"[merge] 통합 완료 → {args.out}")
    print(f"  윈도우(row) 수   : {n}")
    print(f"  워치 시간대       : {to_iso(windows[0]['start_ms'])} ~ {to_iso(windows[-1]['end_ms'])}")
    print(f"  movement 채워짐   : {n_move}/{n}")
    print(f"  digital  채워짐   : {n_digi}/{n}")
    if acc and n_move == 0:
        print("  ⚠ raw(ACC)가 워치 윈도우 시간대와 안 겹침")
    if raw_events and n_digi == 0:
        print("  ⚠ digital이 워치 윈도우 시간대와 안 겹침 → 같은 시간대 동시 기록 필요")


if __name__ == "__main__":
    main()
