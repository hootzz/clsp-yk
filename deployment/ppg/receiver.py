# 실시간 수신 + 로우 데이터 덤프 + 전처리 파이프라인
# 이 파일만 실행
from __future__ import annotations

import argparse
import csv
import re
import signal
import socket
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from pipeline import (
    WINDOW_SAMPLES,
    TARGET_SAMPLES,
    STEP_SAMPLES,
    run_pipeline,
)

_RE_TS  = re.compile(r"ts=\s*(\d+)")
_RE_PPG = re.compile(r"PPG_GREEN=\s*(-?\d+(?:\.\d+)?)")
_RE_ACC = re.compile(r"SACC_MS2=\s*\[([-\d.e+, ]+)\]")
_RE_GYR = re.compile(r"GYR=\s*\[([-\d.e+, ]+)\]")
_RE_HR  = re.compile(r"\bHR=\s*(-?\d+)")

_last_acc: list[float] | None = None
_last_gyr: list[float] | None = None

def parse_udp_line(raw: str) -> dict | None:
    global _last_acc, _last_gyr
    ts_m  = _RE_TS.search(raw)
    ppg_m = _RE_PPG.search(raw)

    if not (ts_m and ppg_m):
        return None

    acc_m = _RE_ACC.search(raw)
    gyr_m = _RE_GYR.search(raw)
    hr_m  = _RE_HR.search(raw)

    if acc_m:
        _last_acc = [float(v) for v in acc_m.group(1).split(",")]
    if gyr_m:
        _last_gyr = [float(v) for v in gyr_m.group(1).split(",")]

    if _last_acc is None:
        return None

    return {
        "ts":  int(ts_m.group(1)),
        "ppg": int(float(ppg_m.group(1))),
        "acc": _last_acc,
        "gyr": _last_gyr,
        "hr":  int(hr_m.group(1)) if hr_m else None,
    }

def run(host: str = "0.0.0.0", port: int = 5005) -> None:
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path  = Path(f"raw_{ts_str}.csv")
    proc_path = Path(f"processed_{ts_str}.csv")
    dump_path = Path(f"dump_{ts_str}.txt")

    raw_fields = ["ts", "ppg_raw", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "hr"]
    proc_fields = (["window_end_ts", "window_idx", "flatline_skipped"]
                   + [f"ppg_{i}" for i in range(TARGET_SAMPLES)])

    raw_f = open(raw_path, "w", newline="", encoding="utf-8")
    raw_w = csv.DictWriter(raw_f, fieldnames=raw_fields)
    raw_w.writeheader()

    proc_f = open(proc_path, "w", newline="", encoding="utf-8")
    proc_w = csv.DictWriter(proc_f, fieldnames=proc_fields)
    proc_w.writeheader()

    dump_f = open(dump_path, "w", encoding="utf-8")

    print(f"[receiver] Raw CSV  → {raw_path}")
    print(f"[receiver] Proc CSV → {proc_path}")
    print(f"[receiver] Raw Dump → {dump_path}")

    ppg_buf: deque[int]        = deque(maxlen=WINDOW_SAMPLES)
    acc_buf: deque[list[float]] = deque(maxlen=WINDOW_SAMPLES)
    sample_count = 0
    window_idx   = 0

    stop = {"flag": False}
    def _handler(sig, frame):
        print("\n[receiver] Stopping...")
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handler)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(1.0)
    print(f"[receiver] Listening on {host}:{port}... (Ctrl+C to stop)\n")

    try:
        while not stop["flag"]:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue

            decoded_text = data.decode("utf-8", errors="replace").strip()
            dump_f.write(f"{datetime.now().isoformat()} | {decoded_text}\n")
            dump_f.flush()

            print(f"\033[90m[RAW]\033[0m {decoded_text[:100]}...")

            for line in decoded_text.splitlines():
                parsed = parse_udp_line(line.strip())
                if parsed is None:
                    continue

                raw_w.writerow({
                    "ts":      parsed["ts"],
                    "ppg_raw": parsed["ppg"],
                    "acc_x":   parsed["acc"][0], "acc_y": parsed["acc"][1], "acc_z": parsed["acc"][2],
                    "gyr_x":   parsed["gyr"][0] if parsed["gyr"] else None,
                    "gyr_y":   parsed["gyr"][1] if parsed["gyr"] else None,
                    "gyr_z":   parsed["gyr"][2] if parsed["gyr"] else None,
                    "hr":      parsed["hr"],
                })
                raw_f.flush()

                ppg_buf.append(parsed["ppg"])
                acc_buf.append(parsed["acc"])
                sample_count += 1

                if len(ppg_buf) == WINDOW_SAMPLES and sample_count % STEP_SAMPLES == 0:
                    ppg_arr = np.array(list(ppg_buf), dtype=np.float64)
                    acc_arr = np.array(list(acc_buf), dtype=np.float64)

                    result = run_pipeline(ppg_arr, acc_arr)
                    window_idx += 1

                    if result is not None:
                        row = {
                            "window_end_ts": parsed["ts"],
                            "window_idx": window_idx,
                            "flatline_skipped": 0,
                        }
                        for i, v in enumerate(result):
                            row[f"ppg_{i}"] = f"{v:.6f}"
                        proc_w.writerow(row)
                        proc_f.flush()
                        print(f"\033[92m[W{window_idx:04d}]\033[0m TS: {parsed['ts']} | OK")
                    else:
                        proc_w.writerow({
                            "window_end_ts": parsed["ts"],
                            "window_idx": window_idx,
                            "flatline_skipped": 1,
                            **{f"ppg_{i}": "" for i in range(TARGET_SAMPLES)},
                        })
                        proc_f.flush()
                        print(f"\033[91m[W{window_idx:04d}]\033[0m TS: {parsed['ts']} | SKIP (Flatline)")

    finally:
        sock.close()
        raw_f.close()
        proc_f.close()
        dump_f.close()
        print(f"\n[receiver] Finished. Saved to {raw_path} and {proc_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()
    run(host=args.ip, port=args.port)
