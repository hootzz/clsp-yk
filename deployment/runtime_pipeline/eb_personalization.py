"""Target-wise empirical-Bayes (EB) output calibration for deployment.

This is the §3.1.3.6 personalization as a runtime output-offset layer: it does
NOT touch the frozen target-routed model. For each user and target it accumulates
the residuals of the first K enrollment labels, forms a shrunk per-user intercept

    r_bar   = mean(label - base_prediction) over the K enrollment samples
    alpha   = K * tau2 / (K * tau2 + sigma2)          (grows with K and tau2)
    offset  = alpha * r_bar
    y_pers  = clamp01(y_base + offset)

and adds it to later predictions. tau2 (between-user) and sigma2 (within-user) come
from a calibration bundle. Behaviour by design:

- K = 0 (no enrollment / no calibration entry) -> offset 0 -> zero-shot population.
- Small between-user variance (e.g. valence, tau2 ~ 0) -> alpha small -> offset ~ 0,
  so the base valence (a context-only route) is left essentially unchanged.

Leakage-free: the target user's later (query) labels are never used; only the first
K enrollment labels contribute, and tau2/sigma2 are external (fit offline, never on the query user).
"""
from __future__ import annotations

from collections import defaultdict


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


class EBPersonalizer:
    def __init__(self, calibration: dict, enroll_k: int = 4):
        # calibration: {target: {"tau2": float, "sigma2": float}}
        self.cal = {t: c for t, c in calibration.items() if isinstance(c, dict)}
        self.enroll_k = int(enroll_k)
        self._residuals: dict[tuple, list[float]] = defaultdict(list)
        self._offset: dict[tuple, float] = {}

    def enroll(self, user: str, target: str, label: float, base_pred: float) -> None:
        """Feed one enrollment (label, base prediction) pair for a target."""
        key = (user, target)
        if len(self._residuals[key]) >= self.enroll_k:
            return
        self._residuals[key].append(float(label) - float(base_pred))
        self._recompute(key, target)

    def _recompute(self, key: tuple, target: str) -> None:
        res = self._residuals[key]
        K = len(res)
        c = self.cal.get(target)
        if not c or K == 0:
            self._offset[key] = 0.0
            return
        tau2 = float(c.get("tau2", 0.0))
        sigma2 = float(c.get("sigma2", 0.0))
        denom = K * tau2 + sigma2
        alpha = (K * tau2 / denom) if denom > 1e-12 else 0.0
        self._offset[key] = alpha * (sum(res) / K)

    def offset(self, user: str, target: str) -> float:
        return self._offset.get((user, target), 0.0)

    def support_count(self, user: str, target: str) -> int:
        return len(self._residuals.get((user, target), ()))

    def apply(self, user: str, base_vac: dict[str, float]) -> dict[str, float]:
        """Return calibrated V/A/C = clamp01(base + per-target offset)."""
        return {t: _clamp01(v + self.offset(user, t)) for t, v in base_vac.items()}
