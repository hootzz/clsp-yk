"""Build synchronized 10-second CASE PPG/continuous-V/A window sources."""
from __future__ import annotations

import argparse
import gc
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

from case_window_common import (
    DEFAULT_PLAN,
    configure_console,
    json_dump,
    load_plan,
    participant_split,
    resolve,
    sha256_file,
    stable_hash,
    work_dir,
)

MODEL_DIR = Path(__file__).resolve().parents[2]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from build_context7_gpt import (  # noqa: E402
    LEGACY_SCHEMA_COLUMNS,
    SCHEMA_VERSION,
    SLOTS,
    deterministic_text,
    duration_bucket,
    validate_spec,
    validate_text,
)

VA_LOW = 0.5
VA_SPAN = 9.0
PPG_DIM = 512
LEGACY_VIDEO_ZSCORE_VERSION = (
    "case_window_v1_global_filter_resample_video_zscore"
)
PREPROCESS_VERSION_PREFIX = "case_window_rescue_v2"


def _video_metadata(case_root: Path) -> tuple[dict[int, str], dict[int, str]]:
    frame = pd.read_excel(case_root / "metadata" / "videos.xlsx")
    frame = frame.dropna(subset=["Video-ID"])
    titles: dict[int, str] = {}
    categories: dict[int, str] = {}
    for _, row in frame.iterrows():
        video_id = int(row["Video-ID"])
        titles[video_id] = str(row["Source (Year)"]).strip()
        categories[video_id] = str(row["Video-label"]).strip()
    return titles, categories


def _case_context_spec(
    title: str = "",
    duration_seconds: float = 10.0,
) -> dict[str, str]:
    spec = {
        "posture": "sitting",
        "energy_expenditure": "sedentary",
        "social_engagement": "not_recorded",
        "interpersonal_density": "not_recorded",
        "app_window": "video_media_player",
        "duration": duration_bucket(duration_seconds),
        "event_type": "passive_video_viewing",
        "observable_detail": str(title).strip(),
    }
    validate_spec(spec)
    return spec


def _zscore_scope(source_cfg: dict[str, Any]) -> str:
    scope = str(
        source_cfg.get("zscore_scope", "participant_video")
    ).strip().lower()
    if scope not in {
        "participant_window",
        "participant_video",
        "participant_recording",
        "resting_baseline",
    }:
        raise ValueError(
            "source.zscore_scope must be participant_window, "
            "participant_video, participant_recording, or "
            "resting_baseline"
        )
    return scope


def _preprocess_version(source_cfg: dict[str, Any]) -> str:
    scope = _zscore_scope(source_cfg)
    if scope == "participant_video":
        return LEGACY_VIDEO_ZSCORE_VERSION
    if scope == "resting_baseline":
        video_id = int(source_cfg.get("resting_video_id", 11))
        skip_sec = float(source_cfg.get("resting_skip_sec", 10.0))
        duration_sec = float(
            source_cfg.get("resting_duration_sec", 60.0)
        )
        return (
            f"{PREPROCESS_VERSION_PREFIX}_{scope}_zscore_"
            f"v{video_id}_skip{skip_sec:g}_dur{duration_sec:g}"
        )
    return f"{PREPROCESS_VERSION_PREFIX}_{scope}_zscore"


def _load_embedder(model_dir: Path):
    sys.path.insert(0, str(model_dir))
    from papagei_embed import PapageiP

    return PapageiP()


def _cache_matches(path: Path, expected_version: str) -> bool:
    if not path.is_file():
        return False
    try:
        cached = np.load(path, allow_pickle=False)
        return (
            str(cached["preprocess_version"].item())
            == expected_version
        )
    except Exception:
        return False


def _participant_cache(
    participant: int,
    physiological_path: Path,
    cache_path: Path,
    embedder: Any,
    source_cfg: dict[str, Any],
    *,
    force: bool,
) -> dict[str, np.ndarray]:
    preprocess_version = _preprocess_version(source_cfg)
    zscore_scope = _zscore_scope(source_cfg)
    if cache_path.is_file() and not force:
        cached = np.load(cache_path, allow_pickle=False)
        if (
            str(cached["preprocess_version"].item())
            == preprocess_version
        ):
            return {name: cached[name] for name in cached.files}

    fs = int(source_cfg["physiological_hz"])
    target_hz = int(source_cfg["target_hz"])
    window_sec = int(source_cfg["window_sec"])
    window_samples = target_hz * window_sec
    low, high = map(float, source_cfg["bandpass_hz"])
    order = int(source_cfg["filter_order"])
    content_videos = set(map(int, source_cfg["content_video_ids"]))

    frame = pd.read_csv(
        physiological_path,
        usecols=["daqtime", "bvp", "video"],
        dtype={"daqtime": np.int64, "bvp": np.float32, "video": np.int16},
    )
    times_ms = frame["daqtime"].to_numpy(dtype=np.float64)
    if len(times_ms) < fs * window_sec:
        raise ValueError(f"physiological file is too short: {physiological_path}")
    if np.any(np.diff(times_ms) <= 0):
        raise ValueError(f"daqtime is not strictly increasing: {physiological_path}")

    # Filtering and resampling occur on the entire participant recording.
    # Only afterwards are content videos sliced, avoiding per-video filter
    # edge transients. Z-scoring scope is configured explicitly:
    # participant-recording global or participant-by-video local.
    sos = butter(
        order,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    filtered = sosfiltfilt(
        sos,
        frame["bvp"].to_numpy(dtype=np.float64),
    )
    resampled = resample_poly(filtered, target_hz, fs)
    resampled_times_ms = (
        times_ms[0]
        + np.arange(len(resampled), dtype=np.float64)
        * (1000.0 / target_hz)
    )
    recording_mean = float(resampled.mean())
    recording_std = float(resampled.std())
    if recording_std <= 1e-8:
        recording_std = 1.0
    videos = frame["video"].to_numpy(dtype=np.int16)

    resting_mean = float("nan")
    resting_std = float("nan")
    resting_start_sec = float("nan")
    resting_end_sec = float("nan")
    if zscore_scope == "resting_baseline":
        resting_video_id = int(source_cfg.get("resting_video_id", 11))
        resting_skip_sec = float(source_cfg.get("resting_skip_sec", 10.0))
        resting_duration_sec = float(
            source_cfg.get("resting_duration_sec", 60.0)
        )
        resting_indices = np.flatnonzero(videos == resting_video_id)
        if len(resting_indices) == 0:
            raise ValueError(
                f"participant {participant} has no resting video "
                f"{resting_video_id}"
            )
        # CASE repeats video 11 between content clips. Only the first
        # contiguous occurrence precedes any content and is enrollment-safe.
        first_index = int(resting_indices[0])
        episode_end = first_index
        while (
            episode_end + 1 < len(videos)
            and int(videos[episode_end + 1]) == resting_video_id
        ):
            episode_end += 1
        episode_start_ms = float(times_ms[first_index])
        episode_end_ms = float(times_ms[episode_end] + 1.0)
        resting_start_ms = episode_start_ms + resting_skip_sec * 1000.0
        resting_end_ms = min(
            episode_end_ms,
            resting_start_ms + resting_duration_sec * 1000.0,
        )
        resting_selected = (
            (resampled_times_ms >= resting_start_ms)
            & (resampled_times_ms < resting_end_ms)
        )
        resting_signal = resampled[resting_selected]
        required_samples = int(
            round(target_hz * resting_duration_sec * 0.9)
        )
        if len(resting_signal) < required_samples:
            raise ValueError(
                f"participant {participant} resting baseline is too "
                f"short: {len(resting_signal)} < {required_samples}"
            )
        resting_mean = float(resting_signal.mean())
        resting_std = float(resting_signal.std())
        if resting_std <= 1e-8:
            resting_std = 1.0
        resting_start_sec = resting_start_ms / 1000.0
        resting_end_sec = resting_end_ms / 1000.0

    windows: list[np.ndarray] = []
    metadata: list[tuple[str, int, int, float, float, float, float, float, float]] = []
    for video_id in sorted(content_videos):
        original_indices = np.flatnonzero(videos == video_id)
        if len(original_indices) == 0:
            continue
        video_start_ms = float(times_ms[original_indices[0]])
        video_end_ms = float(times_ms[original_indices[-1]] + 1.0)
        selected = (
            (resampled_times_ms >= video_start_ms)
            & (resampled_times_ms < video_end_ms)
        )
        segment = resampled[selected]
        segment_times = resampled_times_ms[selected]
        if len(segment) < window_samples:
            continue
        if zscore_scope == "participant_video":
            standard_deviation = float(segment.std())
            segment = (
                segment - float(segment.mean())
            ) / (
                standard_deviation
                if standard_deviation > 1e-8
                else 1.0
            )
        elif zscore_scope == "participant_recording":
            segment = (segment - recording_mean) / recording_std
        elif zscore_scope == "resting_baseline":
            segment = (segment - resting_mean) / resting_std
        window_count = len(segment) // window_samples
        for window_index in range(window_count):
            start = window_index * window_samples
            end = start + window_samples
            window = segment[start:end]
            if zscore_scope == "participant_window":
                window_std = float(window.std())
                window = (window - float(window.mean())) / (
                    window_std if window_std > 1e-8 else 1.0
                )
            window = window.astype(np.float32, copy=False)
            if float(window.std()) <= 1e-6:
                continue
            global_start_ms = float(segment_times[start])
            global_end_ms = global_start_ms + window_sec * 1000.0
            if global_end_ms > video_end_ms + 1e-6:
                raise AssertionError("window crossed a video boundary")
            sample_id = (
                f"case_s{participant}_v{video_id}_w{window_index:03d}"
            )
            windows.append(window)
            metadata.append(
                (
                    sample_id,
                    video_id,
                    window_index,
                    window_index * window_sec,
                    (window_index + 1) * window_sec,
                    global_start_ms / 1000.0,
                    global_end_ms / 1000.0,
                    video_start_ms / 1000.0,
                    video_end_ms / 1000.0,
                )
            )

    if not windows:
        raise ValueError(f"no valid CASE windows for participant {participant}")
    features = embedder.embed(windows).astype(np.float32)
    if features.shape != (len(metadata), PPG_DIM):
        raise ValueError(
            f"PaPaGEI output mismatch: {features.shape} != "
            f"({len(metadata)}, {PPG_DIM})"
        )
    values = list(zip(*metadata))
    payload = {
        "preprocess_version": np.asarray(preprocess_version),
        "zscore_scope": np.asarray(zscore_scope),
        "normalization_passes": np.asarray(1, dtype=np.int8),
        "embedder_preprocess_called": np.asarray(False),
        "resting_mean": np.asarray(resting_mean, dtype=np.float64),
        "resting_std": np.asarray(resting_std, dtype=np.float64),
        "resting_start_sec": np.asarray(
            resting_start_sec, dtype=np.float64
        ),
        "resting_end_sec": np.asarray(resting_end_sec, dtype=np.float64),
        "sample_id": np.asarray(values[0], dtype="U40"),
        "video_id": np.asarray(values[1], dtype=np.int16),
        "window_index": np.asarray(values[2], dtype=np.int16),
        "window_start_sec": np.asarray(values[3], dtype=np.float32),
        "window_end_sec": np.asarray(values[4], dtype=np.float32),
        "ppg_global_start_sec": np.asarray(values[5], dtype=np.float64),
        "ppg_global_end_sec": np.asarray(values[6], dtype=np.float64),
        "video_global_start_sec": np.asarray(values[7], dtype=np.float64),
        "video_global_end_sec": np.asarray(values[8], dtype=np.float64),
        "features": features,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def _annotation_rows(
    participant: int,
    annotation_path: Path,
    cache: dict[str, np.ndarray],
    lag_sec: float,
    source_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    annotation = pd.read_csv(
        annotation_path,
        usecols=["jstime", "valence", "arousal", "video"],
        dtype={
            "jstime": np.int64,
            "valence": np.float32,
            "arousal": np.float32,
            "video": np.int16,
        },
    )
    annotation_hz = float(source_cfg["annotation_hz"])
    window_sec = float(source_cfg["window_sec"])
    minimum_coverage = float(source_cfg["minimum_annotation_coverage"])
    minimum_count = int(
        math.floor(annotation_hz * window_sec * minimum_coverage)
    )
    by_video: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for video_id, group in annotation.groupby("video", sort=False):
        by_video[int(video_id)] = (
            group["jstime"].to_numpy(dtype=np.float64),
            group["valence"].to_numpy(dtype=np.float64),
            group["arousal"].to_numpy(dtype=np.float64),
        )

    rows: list[dict[str, Any]] = []
    lag_ms = float(lag_sec) * 1000.0
    for index, sample_id in enumerate(cache["sample_id"].astype(str)):
        video_id = int(cache["video_id"][index])
        times, valence, arousal = by_video[video_id]
        ppg_start_sec = float(cache["ppg_global_start_sec"][index])
        ppg_end_sec = float(cache["ppg_global_end_sec"][index])
        label_start_ms = ppg_start_sec * 1000.0 + lag_ms
        label_end_ms = ppg_end_sec * 1000.0 + lag_ms
        left = int(np.searchsorted(times, label_start_ms, side="left"))
        right = int(np.searchsorted(times, label_end_ms, side="left"))
        if right - left < minimum_count:
            continue
        selected_v = np.clip((valence[left:right] - VA_LOW) / VA_SPAN, 0, 1)
        selected_a = np.clip((arousal[left:right] - VA_LOW) / VA_SPAN, 0, 1)
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": "case",
                "participant_id": f"case_{participant}",
                "video_id": video_id,
                "window_index": int(cache["window_index"][index]),
                "window_start_sec": float(cache["window_start_sec"][index]),
                "window_end_sec": float(cache["window_end_sec"][index]),
                "analysis_window_sec": float(window_sec),
                "ppg_global_start_sec": ppg_start_sec,
                "ppg_global_end_sec": ppg_end_sec,
                "annotation_window_start_sec": label_start_ms / 1000.0,
                "annotation_window_end_sec": label_end_ms / 1000.0,
                "annotation_lag_sec": float(lag_sec),
                "annotation_sample_count": int(right - left),
                "video_global_start_sec": float(
                    cache["video_global_start_sec"][index]
                ),
                "video_global_end_sec": float(
                    cache["video_global_end_sec"][index]
                ),
                "valence": float(selected_v.mean()),
                "arousal": float(selected_a.mean()),
                "cognitive_load": np.nan,
                "valence_mean": float(selected_v.mean()),
                "arousal_mean": float(selected_a.mean()),
                "valence_std": float(selected_v.std(ddof=0)),
                "arousal_std": float(selected_a.std(ddof=0)),
                "valence_min": float(selected_v.min()),
                "arousal_min": float(selected_a.min()),
                "valence_max": float(selected_v.max()),
                "arousal_max": float(selected_a.max()),
                "valence_range": float(selected_v.max() - selected_v.min()),
                "arousal_range": float(selected_a.max() - selected_a.min()),
                "mask_v": 1,
                "mask_a": 1,
                "mask_c": 0,
                "ppg_available": 1,
                "ppg_quality_reliable": 1,
                "ppg_confidence": 1.0,
            }
        )
    return rows


def _pool_case_windows(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    pool_windows: int,
    base_window_sec: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool consecutive 10-second PaPaGEI rows without crossing a video."""
    pool_windows = int(pool_windows)
    if pool_windows == 1:
        result = manifest.copy()
        result["ppg_window_count"] = 1
        result["ppg_pooling"] = "none_window_level"
        return result, features.copy()
    if pool_windows < 1:
        raise ValueError("source.temporal_pool_windows must be positive")

    feature_columns = [f"ppg_f{i}" for i in range(PPG_DIM)]
    feature_lookup = features.set_index("sample_id")[feature_columns]
    pooled_rows: list[dict[str, Any]] = []
    pooled_features: list[dict[str, Any]] = []
    for (participant_id, video_id), group in manifest.groupby(
        ["participant_id", "video_id"],
        sort=True,
    ):
        ordered = group.sort_values("window_index").reset_index(drop=True)
        pool_index = 0
        for offset in range(0, len(ordered), pool_windows):
            selected = ordered.iloc[offset : offset + pool_windows]
            if len(selected) != pool_windows:
                continue
            indices = selected["window_index"].to_numpy(dtype=int)
            if not np.array_equal(
                np.diff(indices),
                np.ones(pool_windows - 1, dtype=int),
            ):
                continue
            if not np.allclose(
                selected["window_start_sec"].to_numpy(dtype=float)[1:],
                selected["window_end_sec"].to_numpy(dtype=float)[:-1],
                atol=1e-5,
            ):
                continue
            row = selected.iloc[0].to_dict()
            pooled_duration = float(base_window_sec) * pool_windows
            sample_id = (
                f"{participant_id}_v{int(video_id)}_"
                f"p{int(pooled_duration)}_{pool_index:03d}"
            )
            counts = selected[
                "annotation_sample_count"
            ].to_numpy(dtype=float)
            total_count = float(counts.sum())
            if total_count <= 0:
                continue
            row.update(
                {
                    "sample_id": sample_id,
                    "window_index": int(pool_index),
                    "window_start_sec": float(
                        selected["window_start_sec"].iloc[0]
                    ),
                    "window_end_sec": float(
                        selected["window_end_sec"].iloc[-1]
                    ),
                    "analysis_window_sec": pooled_duration,
                    "ppg_global_start_sec": float(
                        selected["ppg_global_start_sec"].iloc[0]
                    ),
                    "ppg_global_end_sec": float(
                        selected["ppg_global_end_sec"].iloc[-1]
                    ),
                    "annotation_window_start_sec": float(
                        selected[
                            "annotation_window_start_sec"
                        ].iloc[0]
                    ),
                    "annotation_window_end_sec": float(
                        selected[
                            "annotation_window_end_sec"
                        ].iloc[-1]
                    ),
                    "annotation_sample_count": int(total_count),
                    "ppg_window_count": int(pool_windows),
                    "ppg_pooling": (
                        f"mean_of_{pool_windows}_consecutive_"
                        "papagei_10s_embeddings"
                    ),
                    "source_window_ids": "|".join(
                        selected["sample_id"].astype(str)
                    ),
                }
            )
            for target in ("valence", "arousal"):
                means = selected[f"{target}_mean"].to_numpy(dtype=float)
                variances = np.square(
                    selected[f"{target}_std"].to_numpy(dtype=float)
                )
                mean = float(np.average(means, weights=counts))
                second_moment = float(
                    np.average(
                        variances + np.square(means),
                        weights=counts,
                    )
                )
                minimum = float(selected[f"{target}_min"].min())
                maximum = float(selected[f"{target}_max"].max())
                row[target] = mean
                row[f"{target}_mean"] = mean
                row[f"{target}_std"] = float(
                    np.sqrt(max(0.0, second_moment - mean * mean))
                )
                row[f"{target}_min"] = minimum
                row[f"{target}_max"] = maximum
                row[f"{target}_range"] = maximum - minimum
            selected_features = feature_lookup.loc[
                selected["sample_id"].astype(str)
            ].to_numpy(dtype=np.float32)
            mean_feature = selected_features.mean(axis=0)
            pooled_features.append(
                {
                    "sample_id": sample_id,
                    **{
                        column: float(value)
                        for column, value in zip(
                            feature_columns, mean_feature
                        )
                    },
                }
            )
            pooled_rows.append(row)
            pool_index += 1
    if not pooled_rows:
        raise ValueError("temporal pooling produced no CASE samples")
    return pd.DataFrame(pooled_rows), pd.DataFrame(pooled_features)


def _decorate_manifest(
    frame: pd.DataFrame,
    titles: dict[int, str],
    split_map: dict[int, dict[str, list[str]]],
    primary_seed: int,
    *,
    include_title: bool,
) -> pd.DataFrame:
    result = frame.copy()
    result["content"] = result["video_id"].map(titles)
    specs = [
        _case_context_spec(
            str(title) if include_title else "",
            float(duration),
        )
        for title, duration in zip(
            result["content"],
            result["analysis_window_sec"],
        )
    ]
    result["context_text"] = [deterministic_text(spec) for spec in specs]
    for text, spec in zip(result["context_text"], specs):
        issues = validate_text(text, spec)
        if issues:
            raise ValueError(f"CASE original-7 text failed: {issues}")
    result["case_context_condition"] = (
        "CASE_with_title_ablation"
        if include_title
        else "CASE_no_title"
    )
    result["text"] = result["context_text"]
    for slot in SLOTS:
        result[f"context7_{slot}"] = [spec[slot] for spec in specs]
    result["context7_observable_detail"] = [
        spec["observable_detail"] for spec in specs
    ]
    result["context_schema_version"] = SCHEMA_VERSION
    result["context_input_source"] = (
        "case_protocol_mapping_then_existing_original7_renderer"
    )
    result["context_input_validation"] = "passed"
    result["context_sig"] = result["context_text"].map(stable_hash)
    result["teacher_text"] = ""
    result["clsp_teacher_available"] = 0
    result["clsp_temporal_match"] = 0
    result["clsp_primary_positive_eligible"] = 0
    result["clsp_pair_id"] = result["sample_id"]
    result["clsp_positive_sig"] = result["sample_id"]
    result["clsp_negative_mask_sig"] = (
        result["participant_id"].astype(str)
        + ":v"
        + result["video_id"].astype(str)
    )
    result["clsp_semantic_sig"] = ""
    result["clsp_loss_group"] = ""
    result["session_id"] = result["clsp_negative_mask_sig"]
    result["label_group_sig"] = result["clsp_negative_mask_sig"]
    result["negative_adjacent_radius"] = 1
    result["ppg_temporal_scope"] = result[
        "analysis_window_sec"
    ].map(lambda value: f"{float(value):g}_second_analysis_window")
    if "ppg_pooling" not in result:
        result["ppg_pooling"] = "none_window_level"
    result["label_temporal_scope"] = result[
        "analysis_window_sec"
    ].map(lambda value: f"same_{float(value):g}_second_window")
    result["ppg_sampling_rate_hz"] = 125
    if "ppg_window_count" not in result:
        result["ppg_window_count"] = 1
    result["ppg_duration_seconds"] = result["analysis_window_sec"]

    for seed, split in split_map.items():
        inverse = {
            participant: name
            for name, participants in split.items()
            for participant in participants
        }
        result[f"split_seed_{seed}"] = result["participant_id"].map(inverse)
    result["split"] = result[f"split_seed_{primary_seed}"]

    counts = result.groupby(["participant_id", "video_id"])[
        "sample_id"
    ].transform("count")
    result["case_window_weight"] = 1.0 / counts
    result["sample_weight"] = result["case_window_weight"]
    return result


def _audit(
    no_title: pd.DataFrame,
    with_title: pd.DataFrame,
    features: pd.DataFrame,
    categories: dict[int, str],
    titles: dict[int, str],
    split_map: dict[int, dict[str, list[str]]],
    source_cfg: dict[str, Any],
    lag_sec: float,
) -> dict[str, Any]:
    feature_columns = [f"ppg_f{i}" for i in range(PPG_DIM)]
    errors: list[str] = []
    required_context_columns = {
        "context_schema_version",
        *(f"context7_{slot}" for slot in SLOTS),
    }
    missing_context_columns = sorted(
        required_context_columns - set(no_title.columns)
    )
    legacy_context_columns = sorted(
        LEGACY_SCHEMA_COLUMNS & set(no_title.columns)
    )
    if missing_context_columns:
        errors.append(
            f"missing original-7 columns: {missing_context_columns}"
        )
    if legacy_context_columns:
        errors.append(
            f"legacy schema columns present: {legacy_context_columns}"
        )
    if (
        "context_schema_version" in no_title
        and set(no_title["context_schema_version"].astype(str))
        != {SCHEMA_VERSION}
    ):
        errors.append("CASE context schema version mismatch")
    if no_title["sample_id"].duplicated().any():
        errors.append("duplicate sample_id")
    if set(no_title["sample_id"]) != set(features["sample_id"]):
        errors.append("manifest/PPG sample_id mismatch")
    if any(column not in features for column in feature_columns):
        errors.append("PPG feature dimension is not 512")
    if not np.isfinite(features[feature_columns].to_numpy()).all():
        errors.append("non-finite PPG embedding")
    for target in ("valence", "arousal"):
        values = no_title[target].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            errors.append(f"{target} is not finite [0,1]")
    if not no_title["cognitive_load"].isna().all():
        errors.append("CASE cognitive_load must be NaN")
    if not (no_title["mask_c"] == 0).all():
        errors.append("CASE mask_c must be zero")
    if np.any(
        no_title["ppg_global_start_sec"]
        < no_title["video_global_start_sec"] - 1e-8
    ) or np.any(
        no_title["ppg_global_end_sec"]
        > no_title["video_global_end_sec"] + 1e-8
    ):
        errors.append("a PPG window crosses a video boundary")
    alignment_start = (
        no_title["annotation_window_start_sec"]
        - no_title["ppg_global_start_sec"]
    )
    alignment_end = (
        no_title["annotation_window_end_sec"]
        - no_title["ppg_global_end_sec"]
    )
    if not np.allclose(alignment_start, lag_sec) or not np.allclose(
        alignment_end, lag_sec
    ):
        errors.append("PPG/annotation time alignment mismatch")

    forbidden_categories = [
        value.lower()
        for video_id, value in categories.items()
        if video_id in set(map(int, source_cfg["content_video_ids"]))
    ]
    for text in with_title["context_text"].astype(str).str.lower():
        if any(category in text for category in forbidden_categories):
            errors.append("experimenter emotion category leaked into text")
            break
    lowered_no_title = no_title["context_text"].astype(str).str.lower()
    for title in titles.values():
        if str(title).lower() == "n.a. -- blue screen":
            continue
        if lowered_no_title.str.contains(
            re.escape(str(title).lower()), regex=True
        ).any():
            errors.append("clip title leaked into title-off text")
            break

    split_audit: dict[str, Any] = {}
    for seed, split in split_map.items():
        sets = {name: set(values) for name, values in split.items()}
        overlap = sorted(
            (sets["train"] & sets["validation"])
            | (sets["train"] & sets["test"])
            | (sets["validation"] & sets["test"])
        )
        if overlap:
            errors.append(f"participant leakage for seed {seed}: {overlap}")
        split_audit[str(seed)] = {
            name: {
                "participants": values,
                "participant_count": len(values),
                "window_count": int(
                    (
                        no_title[f"split_seed_{seed}"] == name
                    ).sum()
                ),
            }
            for name, values in split.items()
        }
        split_audit[str(seed)]["participant_overlap"] = overlap

    group_weight_sum = no_title.groupby(
        ["participant_id", "video_id"]
    )["sample_weight"].sum()
    if not np.allclose(group_weight_sum.to_numpy(), 1.0):
        errors.append("participant-by-video weights do not sum to one")
    pool_windows = int(source_cfg.get("temporal_pool_windows", 1))
    analysis_window_sec = (
        float(source_cfg["window_sec"]) * pool_windows
    )
    if not np.allclose(
        no_title["analysis_window_sec"].to_numpy(dtype=float),
        analysis_window_sec,
    ):
        errors.append("CASE analysis window duration mismatch")
    zscore_scope = _zscore_scope(source_cfg)
    normalization_stage = {
        "participant_window": "within each non-overlap 10-second window",
        "participant_video": "within each complete content video",
        "participant_recording": "within the complete participant recording",
        "resting_baseline": "using only the first pre-content resting episode",
    }[zscore_scope]

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "preprocessing": {
            "order": [
                "whole-participant 0.5-8 Hz zero-phase bandpass",
                "whole-participant 1000->125 Hz polyphase resample",
                "slice participant-by-content-video",
                "non-overlap 10-second windows",
                f"z-score {normalization_stage}",
                "frozen PaPaGEI-P 512D per window",
                (
                    "no temporal feature pooling"
                    if pool_windows == 1
                    else (
                        f"mean-pool {pool_windows} consecutive 10-second "
                        "PaPaGEI embeddings"
                    )
                ),
            ],
            "filter_scope": "entire continuous participant recording",
            "zscore_scope": zscore_scope,
            "normalization_scope": normalization_stage,
            "normalization_passes": 1,
            "embedder_preprocess_called": False,
            "case_window_double_normalization": False,
            "deployment_caveat": (
                "This experiment isolates normalization scope. The shared "
                "zero-phase bandpass still uses the complete recording and "
                "is not a proof of end-to-end causal filtering."
            ),
            "resting_baseline": (
                {
                    "video_id": int(source_cfg.get("resting_video_id", 11)),
                    "occurrence": "first_contiguous_pre_content_only",
                    "skip_sec": float(
                        source_cfg.get("resting_skip_sec", 10.0)
                    ),
                    "duration_sec": float(
                        source_cfg.get("resting_duration_sec", 60.0)
                    ),
                }
                if zscore_scope == "resting_baseline"
                else None
            ),
            "temporal_pool_windows": pool_windows,
            "analysis_window_sec": analysis_window_sec,
            "remainder": "discarded_if_shorter_than_10_seconds",
            "excluded_video_ids": source_cfg["excluded_video_ids"],
            "annotation_lag_sec": float(lag_sec),
            "positive_lag_definition": (
                "PPG [t0,t1) uses annotation [t0+lag,t1+lag)"
            ),
            "normalization": "(value - 0.5) / 9.0; no participant min-max",
        },
        "rows": int(len(no_title)),
        "participants": int(no_title["participant_id"].nunique()),
        "videos": int(no_title["video_id"].nunique()),
        "windows_by_video": {
            str(key): int(value)
            for key, value in no_title["video_id"].value_counts().sort_index().items()
        },
        "windows_by_participant": {
            str(key): int(value)
            for key, value in no_title["participant_id"].value_counts().sort_index().items()
        },
        "labels": {
            column: {
                "min": float(no_title[column].min()),
                "max": float(no_title[column].max()),
                "mean": float(no_title[column].mean()),
                "nan": int(no_title[column].isna().sum()),
            }
            for column in (
                "valence",
                "arousal",
                "valence_std",
                "arousal_std",
                "valence_range",
                "arousal_range",
            )
        },
        "split_audit": split_audit,
        "weight_audit": {
            "participant_video_weight_sum_min": float(group_weight_sum.min()),
            "participant_video_weight_sum_max": float(group_weight_sum.max()),
            "sampler_note": (
                "Source construction does not assume batch mass; rescue "
                "training enforces capped repeats separately."
            ),
        },
        "text_audit": {
            "schema_version": SCHEMA_VERSION,
            "legacy_columns_present": legacy_context_columns,
            "missing_original7_columns": missing_context_columns,
            "emotion_category_hits": 0,
            "title_off_title_hits": 0,
            "title_off_unique_contexts": int(
                no_title["context_text"].nunique()
            ),
            "title_on_unique_contexts": int(
                with_title["context_text"].nunique()
            ),
        },
    }


def build(
    plan: dict[str, Any],
    *,
    lag_sec: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    model_dir = resolve(plan, plan["model_dir"])
    case_root = resolve(plan, plan["case_root"])
    source_cfg = plan["source"]
    lag_sec = (
        float(source_cfg["annotation_lag_sec"])
        if lag_sec is None
        else float(lag_sec)
    )
    output_root = work_dir(plan) / "source_data"
    if lag_sec != float(source_cfg["annotation_lag_sec"]):
        output_root = output_root / f"lag_{lag_sec:g}s"
    output_root.mkdir(parents=True, exist_ok=True)
    zscore_scope = _zscore_scope(source_cfg)
    shared_cache = source_cfg.get("shared_cache_root")
    cache_root = (
        resolve(plan, shared_cache) / zscore_scope
        if shared_cache
        else work_dir(plan) / "source_data" / f"cache_{zscore_scope}"
    )
    fallback_cache = source_cfg.get("legacy_video_cache_root")
    fallback_cache_root = (
        resolve(plan, fallback_cache)
        if fallback_cache and zscore_scope == "participant_video"
        else None
    )

    physiological_dir = (
        case_root / "data" / "interpolated" / "physiological"
    )
    annotation_dir = (
        case_root / "data" / "interpolated" / "annotations"
    )
    files = sorted(
        physiological_dir.glob("sub_*.csv"),
        key=lambda path: int(re.search(r"\d+", path.stem).group()),
    )
    if len(files) != 30:
        raise ValueError(f"expected 30 CASE participants, found {len(files)}")

    participant_numbers = [
        int(re.search(r"\d+", path.stem).group()) for path in files
    ]
    writable_cache_paths = [
        cache_root / f"sub_{participant:02d}.npz"
        for participant in participant_numbers
    ]
    expected_version = _preprocess_version(source_cfg)
    cache_paths: list[Path] = []
    for participant, writable_path in zip(
        participant_numbers, writable_cache_paths
    ):
        fallback_path = (
            fallback_cache_root / f"sub_{participant:02d}.npz"
            if fallback_cache_root is not None
            else None
        )
        if (
            not force
            and not _cache_matches(writable_path, expected_version)
            and fallback_path is not None
            and _cache_matches(fallback_path, expected_version)
        ):
            cache_paths.append(fallback_path)
        else:
            cache_paths.append(writable_path)
    need_embedder = force or any(
        not _cache_matches(path, expected_version)
        for path in cache_paths
    )
    embedder = _load_embedder(model_dir) if need_embedder else None
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for physiological_path, cache_path in zip(files, cache_paths):
        participant = int(re.search(r"\d+", physiological_path.stem).group())
        if embedder is None and not _cache_matches(
            cache_path, expected_version
        ):
            embedder = _load_embedder(model_dir)
        cache = _participant_cache(
            participant,
            physiological_path,
            cache_path,
            embedder,
            source_cfg,
            force=force,
        )
        participant_rows = _annotation_rows(
            participant,
            annotation_dir / f"sub_{participant}.csv",
            cache,
            lag_sec,
            source_cfg,
        )
        retained = {row["sample_id"] for row in participant_rows}
        for index, sample_id in enumerate(cache["sample_id"].astype(str)):
            if sample_id not in retained:
                continue
            feature_rows.append(
                {
                    "sample_id": sample_id,
                    **{
                        f"ppg_f{dimension}": float(
                            cache["features"][index, dimension]
                        )
                        for dimension in range(PPG_DIM)
                    },
                }
            )
        rows.extend(participant_rows)
        print(
            f"[CASE] participant {participant:02d}: "
            f"{len(participant_rows)} aligned windows",
            flush=True,
        )
        gc.collect()

    base = pd.DataFrame(rows).sort_values(
        ["participant_id", "video_id", "window_index"]
    )
    features = pd.DataFrame(feature_rows).sort_values("sample_id")
    pool_windows = int(source_cfg.get("temporal_pool_windows", 1))
    base, features = _pool_case_windows(
        base,
        features,
        pool_windows,
        float(source_cfg["window_sec"]),
    )
    base = base.sort_values(
        ["participant_id", "video_id", "window_index"]
    )
    features = features.sort_values("sample_id")
    titles, categories = _video_metadata(case_root)
    participants = sorted(base["participant_id"].unique())
    split_map = {
        int(seed): participant_split(
            participants,
            int(seed),
            float(plan["split"]["legacy_holdout_fraction"]),
            float(plan["split"]["validation_fraction"]),
        )
        for seed in plan["split"]["seeds"]
    }
    primary_seed = int(plan["split"]["primary_seed"])
    no_title = _decorate_manifest(
        base,
        titles,
        split_map,
        primary_seed,
        include_title=False,
    )
    with_title = _decorate_manifest(
        base,
        titles,
        split_map,
        primary_seed,
        include_title=True,
    )
    audit = _audit(
        no_title,
        with_title,
        features,
        categories,
        titles,
        split_map,
        source_cfg,
        lag_sec,
    )
    if audit["status"] != "pass":
        raise ValueError(f"CASE window audit failed: {audit['errors']}")

    no_title_path = output_root / "case_window_manifest_no_title.csv"
    canonical_path = output_root / "case_window_manifest.csv"
    with_title_path = output_root / "case_window_manifest_with_title.csv"
    ppg_path = output_root / "case_window_ppg512.csv"
    no_title.to_csv(no_title_path, index=False)
    no_title.to_csv(canonical_path, index=False)
    with_title.to_csv(with_title_path, index=False)
    features.to_csv(ppg_path, index=False)

    split_payload = {
        "schema_version": 1,
        "policy": (
            "historical 15% CASE validation participants become test; "
            "next 15% become validation; remaining participants train"
        ),
        "splits": {str(seed): split for seed, split in split_map.items()},
    }
    split_path = output_root / "participant_splits.json"
    json_dump(split_path, split_payload)

    audit["files"] = {
        "manifest": str(canonical_path.resolve()),
        "manifest_no_title": str(no_title_path.resolve()),
        "manifest_with_title": str(with_title_path.resolve()),
        "ppg512": str(ppg_path.resolve()),
        "participant_splits": str(split_path.resolve()),
    }
    audit["hashes"] = {
        "manifest_no_title": sha256_file(no_title_path),
        "manifest_with_title": sha256_file(with_title_path),
        "ppg512": sha256_file(ppg_path),
        "participant_splits": sha256_file(split_path),
    }
    audit_path = output_root / "source_audit.json"
    json_dump(audit_path, audit)
    statistics_path = output_root / "case_window_statistics.json"
    json_dump(
        statistics_path,
        {
            key: audit[key]
            for key in (
                "rows",
                "participants",
                "videos",
                "windows_by_video",
                "windows_by_participant",
                "labels",
                "weight_audit",
                "text_audit",
            )
        },
    )
    print(f"[DONE] manifest -> {canonical_path}")
    print(f"[DONE] PPG 512D -> {ppg_path}")
    print(f"[DONE] audit -> {audit_path}")
    return audit


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--lag-sec", type=float)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(
        load_plan(args.plan),
        lag_sec=args.lag_sec,
        force=args.force,
    )


if __name__ == "__main__":
    main()
