"""PaPaGEI-P (ResNet1D) embedder with PPG preprocessing.

Pipeline: raw PPG → bandpass(0.5-8) → 125Hz resample → z-score →
1250-sample (10s) windows → PaPaGEI-P 512-dimensional embeddings.
Uses the P variant (papagei_p.pt, ResNet1D).
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import numpy as np
import torch
from scipy.signal import butter, filtfilt, resample_poly

FS = 125
WIN = 1250
# Self-contained defaults: PaPaGEI repo + weights at the repo root. Override via env.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAP_ROOT = os.environ.get("PAPAGEI_ROOT", os.path.join(_REPO, "papagei-foundation-model"))
PAP_CKPT = os.environ.get("PAPAGEI_CKPT", os.path.join(_REPO, "weights", "papagei_p.pt"))


def preprocess(sig, fs):
    """Bandpass 0.5-8 → 125Hz resample → z-score. Returns a 125Hz 1D signal."""
    sig = np.asarray(sig, dtype=float).flatten()
    if len(sig) < fs * 2:
        return None
    b, a = butter(3, [0.5, 8.0], btype="band", fs=fs)
    sig = filtfilt(b, a, sig)
    if int(fs) != FS:
        fr = Fraction(FS, int(round(fs))).limit_denominator(1000)
        sig = resample_poly(sig, fr.numerator, fr.denominator)
    sd = sig.std()
    return (sig - sig.mean()) / (sd if sd > 1e-8 else 1.0)


def windows(sig, n=WIN, stride=None, max_win=None):
    """Create 1250-sample windows, remove flat segments, and optionally subsample evenly."""
    stride = stride or n
    ws = [sig[i:i + n] for i in range(0, len(sig) - n + 1, stride)]
    ws = [w for w in ws if np.std(w) > 1e-6]
    if max_win and len(ws) > max_win:
        idx = np.linspace(0, len(ws) - 1, max_win).astype(int)
        ws = [ws[i] for i in idx]
    return ws


class PapageiP:
    def __init__(self, root=PAP_ROOT, ckpt=PAP_CKPT, device=None):
        sys.path.insert(0, root)
        from linearprobing.utils import load_model_without_module_prefix
        from models.resnet import ResNet1D
        m = ResNet1D(in_channels=1, base_filters=32, kernel_size=3, stride=2,
                     groups=1, n_block=18, n_classes=512)
        self.m = load_model_without_module_prefix(m, ckpt)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m.to(self.device).eval()

    @torch.inference_mode()
    def embed(self, wins, batch=256):
        """list of 1250 arrays → (N,512)."""
        out = []
        for i in range(0, len(wins), batch):
            x = torch.tensor(np.stack(wins[i:i + batch])[:, None, :], dtype=torch.float32, device=self.device)
            o = self.m(x)
            emb = o[0] if isinstance(o, (tuple, list)) else o
            out.append(emb.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
