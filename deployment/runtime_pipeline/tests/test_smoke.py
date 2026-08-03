from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from runtime_pipeline.context_text import build_context_text
from runtime_pipeline.digital import extract_intervals, overlapping
from runtime_pipeline.finalize import (
    MockEmbedder,
    MockStateEstimator,
    finalize_window_records,
)
from runtime_pipeline.merge import merge_records


class RuntimePipelineSmokeTest(unittest.TestCase):
    def test_context_is_seven_slot_and_label_free(self):
        text = build_context_text(
            {
                "posture": "sitting",
                "movement": "sedentary",
                "social_engagement": "not_recorded",
                "interpersonal_density": "not_recorded",
                "device_interaction_behavior": "active_input",
                "environment": "indoor",
                "temporal": "continuous",
            },
            "using a document editor",
        )
        self.assertIn("not recorded", text.lower())
        self.assertNotIn("cognitive load", text.lower())

    def test_android_naive_timestamp_uses_kst(self):
        events = [
            {
                "timestamp": "2026-07-26T12:00:00",
                "device_type": "android",
                "app": "example.app",
                "event_type": "app_start",
            },
            {
                "timestamp": "2026-07-26T12:00:10",
                "device_type": "android",
                "app": "example.app",
                "event_type": "app_close",
                "duration_seconds": 10,
            },
        ]
        intervals = extract_intervals(events, naive_offset_hours=9)
        start_ms = int(intervals[0]["start_time"].timestamp() * 1000)
        self.assertEqual(len(overlapping(intervals, start_ms, start_ms + 10_000)), 1)

    def test_merge_then_mock_finalize(self):
        windows = [
            {
                "start_ms": 1_700_000_000_000,
                "end_ms": 1_700_000_010_000,
                "window_idx": 1,
                "ppg": [0.2] * 1_250,
            }
        ]
        merged = merge_records(
            windows,
            [],
            [],
            posture="sitting",
            social_engagement="not_recorded",
            interpersonal_density="not_recorded",
            environment="indoor",
            temporal="continuous",
            context_backend="deterministic",
            context_model="gpt-4o-mini",
        )
        finalized = finalize_window_records(
            merged,
            MockEmbedder(),
            MockStateEstimator(),
            keep_ppg=False,
            keep_embedding=True,
        )
        self.assertEqual(finalized[0]["measures"]["arousal"], 0.2)
        self.assertEqual(len(finalized[0]["ppg_embedding"]), 512)
        self.assertNotIn("ppg", finalized[0])


if __name__ == "__main__":
    unittest.main()
