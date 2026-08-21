"""
HQS PC Collector — ActivityWatch polling script
"""

import sys
import json
import time
import logging
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

AW_BASE_URL   = "http://localhost:5600/api/0"
POLL_INTERVAL = 5
OUTPUT_DIR    = Path("./data/pc")
BROWSER_APPS  = {"chrome.exe", "msedge.exe", "firefox.exe"}


class ActivityWatchClient:
    def __init__(self, base_url: str = AW_BASE_URL):
        self.base = base_url
        self._hostname = self._get_hostname()
        self._window_bucket  = f"aw-watcher-window_{self._hostname}"
        self._afk_bucket     = f"aw-watcher-afk_{self._hostname}"
        self._browser_bucket = f"aw-watcher-web-chrome_{self._hostname}"

    def _get_hostname(self) -> str:
        try:
            info = requests.get(f"{self.base}/info", timeout=3).json()
            return info.get("hostname", "unknown")
        except Exception:
            import socket
            return socket.gethostname()

    def get_buckets(self) -> dict:
        try:
            return requests.get(f"{self.base}/buckets", timeout=5).json()
        except Exception as e:
            log.error(f"Failed to retrieve bucket list: {e}")
            return {}

    def get_latest_window_event(self) -> Optional[dict]:
        try:
            r = requests.get(
                f"{self.base}/buckets/{self._window_bucket}/events",
                params={"limit": 1}, timeout=5
            )
            r.raise_for_status()
            events = r.json()
            return events[0] if events else None
        except requests.exceptions.ConnectionError:
            log.error("Unable to connect to the ActivityWatch server.")
            return None
        except Exception as e:
            log.error(f"Failed to retrieve event: {e}")
            return None

    def get_latest_browser_event(self) -> Optional[dict]:
        try:
            r = requests.get(
                f"{self.base}/buckets/{self._browser_bucket}/events",
                params={"limit": 1}, timeout=5
            )
            r.raise_for_status()
            events = r.json()
            return events[0] if events else None
        except Exception:
            return None

    def is_afk(self) -> bool:
        try:
            r = requests.get(
                f"{self.base}/buckets/{self._afk_bucket}/events",
                params={"limit": 1}, timeout=5
            )
            events = r.json()
            if not events:
                return False
            return events[0].get("data", {}).get("status") == "afk"
        except Exception:
            return False


def save_event(timestamp: str, app: str, title: str, duration_seconds: float, url: str = ""):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"events_{date_str}.jsonl"

    record = {
        "timestamp":        timestamp,
        "device_type":      "pc",
        "app":              app,
        "title":            title,
        "url":              url,
        "duration_seconds": duration_seconds,
        "event_type":       "app_switch",
    }

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class PCCollector:
    def __init__(self):
        self.aw = ActivityWatchClient()
        self._prev_app:        Optional[str]      = None
        self._prev_title:      Optional[str]      = None
        self._prev_url:        Optional[str]      = None
        self._prev_timestamp:  Optional[str]      = None
        self._prev_start_time: Optional[datetime] = None

        buckets = self.aw.get_buckets()
        if not buckets:
            log.warning("ActivityWatch buckets are empty or the server connection failed.")
        else:
            log.info(f"Connected buckets: {list(buckets.keys())}")

    def poll(self):
        if self.aw.is_afk():
            return

        raw = self.aw.get_latest_window_event()
        if not raw:
            return

        data          = raw.get("data", {})
        current_app   = data.get("app", "unknown")
        current_title = data.get("title", "")
        current_ts    = raw.get("timestamp", datetime.now(timezone.utc).isoformat())
        current_url   = ""

        if current_app in BROWSER_APPS:
            browser_raw = self.aw.get_latest_browser_event()
            if browser_raw:
                current_title = browser_raw.get("data", {}).get("title", current_title)
                current_url   = browser_raw.get("data", {}).get("url", "")

        if current_app != self._prev_app or current_url != self._prev_url:
            if self._prev_app is not None and self._prev_start_time is not None:
                duration = (datetime.now(timezone.utc) - self._prev_start_time).total_seconds()
                save_event(self._prev_timestamp, self._prev_app, self._prev_title, duration, self._prev_url or "")
                log.info(f"App closed: {self._prev_app} | {self._prev_title} | {duration:.1f}s")

            log.info(f"App switched: {self._prev_app} → {current_app} | {current_title}")
            save_event(current_ts, current_app, current_title, 0.0, current_url)
            self._prev_app        = current_app
            self._prev_title      = current_title
            self._prev_url        = current_url
            self._prev_timestamp  = current_ts
            self._prev_start_time = datetime.now(timezone.utc)

    def run(self):
        log.info(f"PC Collector started (poll interval: {POLL_INTERVAL}s)")
        if not SCHEDULE_AVAILABLE:
            log.error("Missing schedule package. Run: pip install schedule")
            return
        schedule.every(POLL_INTERVAL).seconds.do(self.poll)

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Collection stopped.")
            if self._prev_app is not None and self._prev_start_time is not None:
                duration = (datetime.now(timezone.utc) - self._prev_start_time).total_seconds()
                save_event(self._prev_timestamp, self._prev_app, self._prev_title, duration, self._prev_url or "")
                log.info(f"Final app saved: {self._prev_app} | {duration:.1f}s")


if __name__ == "__main__":
    collector = PCCollector()
    collector.run()
