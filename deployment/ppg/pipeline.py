from __future__ import annotations

import numpy as np
from fractions import Fraction
from math import gcd
from scipy.signal import butter, filtfilt, resample_poly

FS_RAW: int    = 25
FS_TARGET: int = 125
WINDOW_SEC: int = 10

WINDOW_SAMPLES: int = FS_RAW * WINDOW_SEC
TARGET_SAMPLES: int = FS_TARGET * WINDOW_SEC

STEP_SAMPLES: int = FS_RAW * 2


def invert_ppg(ppg: np.ndarray) -> np.ndarray:
    return ppg * -1.0


def kalman_motion_correction(
    ppg: np.ndarray,
    acc: np.ndarray,
    q: float = 1e-4,
    r_base: float = 0.1,
) -> np.ndarray:
    gravity = 9.81
    mag = np.linalg.norm(acc, axis=1)
    motion = np.abs(mag - gravity)
    mean_motion = motion.mean() + 1e-8

    ppg_clean = np.empty_like(ppg)
    x_est = float(ppg[0])
    p_cov = 1.0

    for i in range(len(ppg)):
        r_adaptive = r_base * (1.0 + motion[i] / mean_motion)

        p_cov = p_cov + q

        k_gain = p_cov / (p_cov + r_adaptive)
        x_est  = x_est + k_gain * (ppg[i] - x_est)
        p_cov  = (1.0 - k_gain) * p_cov

        ppg_clean[i] = x_est

    return ppg_clean


def bandpass_filter(
    signal: np.ndarray,
    fs: float = FS_RAW,
    low_hz: float = 0.5,
    high_hz: float = 8.0,
    order: int = 4,
) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, signal)


def detect_flatline(signal: np.ndarray, threshold: float = 0.25) -> bool:
    diff = np.abs(np.diff(signal))
    flat_ratio = np.sum(diff < 1e-6) / len(diff)
    return bool(flat_ratio > threshold)


def resample_to_target(
    signal: np.ndarray,
    fs_in: float = FS_RAW,
    fs_out: float = FS_TARGET,
) -> np.ndarray:
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    up   = frac.numerator
    down = frac.denominator
    g    = gcd(up, down)
    return resample_poly(signal, up // g, down // g)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean()
    std  = signal.std()
    if std < 1e-8:
        return np.zeros_like(signal)
    return (signal - mean) / std


def run_pipeline(
    ppg_raw: np.ndarray,
    acc_raw: np.ndarray,
) -> np.ndarray | None:
    ppg = ppg_raw.astype(np.float64).copy()
    acc = acc_raw.astype(np.float64).copy()

    ppg = invert_ppg(ppg)
    ppg = kalman_motion_correction(ppg, acc)
    ppg = bandpass_filter(ppg, fs=FS_RAW)

    if detect_flatline(ppg):
        return None

    ppg = zscore_normalize(ppg)
    ppg = resample_to_target(ppg)

    return ppg
