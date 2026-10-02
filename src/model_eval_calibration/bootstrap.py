"""Nonparametric bootstrap uncertainty for calibration metrics.

Point estimates of ECE / Brier on a few hundred rows move a lot between
resamples. These helpers attach percentile confidence intervals so that
method comparisons in the README can be read with their noise.

Design notes
------------
* Resampling is over *rows* (i.i.d. bootstrap). It ignores any uncertainty
  from refitting the model or calibrator -- intervals are conditional on
  the fitted predictor and therefore optimistic about total variance.
* ``paired_bootstrap_difference`` reuses the same row indices for both
  methods, which is the right design when two calibrators are scored on
  the same test rows (their errors are correlated).
* For reliability-diagram bands, bin edges are fixed on the full sample
  and only the per-bin accuracy is resampled; bins that end up empty in a
  resample are skipped (NaN-aware percentiles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from model_eval_calibration.metrics import (
    BinningStrategy,
    _bin_ids_from_edges,
    _equal_width_edges,
    _quantile_edges,
    _validate_binary_probs,
)

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate plus percentile interval."""

    estimate: float
    lower: float
    upper: float
    alpha: float
    n_boot: int

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "alpha": self.alpha,
            "n_boot": self.n_boot,
        }


def _check_boot_args(n_boot: int, alpha: float) -> None:
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: MetricFn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for ``metric_fn(y_true, y_prob)``.

    The point estimate is the metric on the full sample, not the bootstrap
    mean (bootstrap means of ECE are biased upward by binning noise).
    """
    _check_boot_args(n_boot, alpha)
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape or y_true.size == 0:
        raise ValueError("y_true and y_prob must be non-empty and same shape")
    rng = np.random.default_rng(random_state)
    n = y_true.size
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = metric_fn(y_true[idx], y_prob[idx])
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        estimate=float(metric_fn(y_true, y_prob)),
        lower=float(lo),
        upper=float(hi),
        alpha=float(alpha),
        n_boot=int(n_boot),
    )


@dataclass(frozen=True)
class PairedDifference:
    """Bootstrap distribution summary for metric(a) - metric(b)."""

    estimate: float
    lower: float
    upper: float
    prob_a_better: float  # share of resamples where metric(a) < metric(b)
    alpha: float
    n_boot: int

    @property
    def excludes_zero(self) -> bool:
        return bool(self.lower > 0.0 or self.upper < 0.0)

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "prob_a_better": self.prob_a_better,
            "excludes_zero": self.excludes_zero,
            "alpha": self.alpha,
            "n_boot": self.n_boot,
        }


def paired_bootstrap_difference(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metric_fn: MetricFn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = 0,
) -> PairedDifference:
    """Paired bootstrap of ``metric(a) - metric(b)`` on shared rows.

    Lower-is-better metrics (Brier, ECE): a negative difference favours
    ``a``. ``prob_a_better`` is the bootstrap share with metric(a) < metric(b)
    -- a descriptive stability number, not a p-value.
    """
    _check_boot_args(n_boot, alpha)
    y_true = np.asarray(y_true, dtype=float).ravel()
    a = np.asarray(y_prob_a, dtype=float).ravel()
    b = np.asarray(y_prob_b, dtype=float).ravel()
    if not (y_true.shape == a.shape == b.shape) or y_true.size == 0:
        raise ValueError("inputs must be non-empty and share one shape")
    rng = np.random.default_rng(random_state)
    n = y_true.size
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        diffs[i] = metric_fn(yt, a[idx]) - metric_fn(yt, b[idx])
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return PairedDifference(
        estimate=float(metric_fn(y_true, a) - metric_fn(y_true, b)),
        lower=float(lo),
        upper=float(hi),
        prob_a_better=float(np.mean(diffs < 0.0)),
        alpha=float(alpha),
        n_boot=int(n_boot),
    )


@dataclass(frozen=True)
class ReliabilityBands:
    """Reliability diagram with per-bin bootstrap accuracy bands."""

    bin_edges: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    acc_lower: np.ndarray
    acc_upper: np.ndarray
    bin_counts: np.ndarray
    alpha: float

    def as_dict(self) -> dict:
        return {
            "bin_edges": self.bin_edges.tolist(),
            "bin_confidence": self.bin_confidence.tolist(),
            "bin_accuracy": self.bin_accuracy.tolist(),
            "acc_lower": self.acc_lower.tolist(),
            "acc_upper": self.acc_upper.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "alpha": self.alpha,
        }


def reliability_bootstrap_bands(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: BinningStrategy = "equal_width",
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: Optional[int] = 0,
) -> ReliabilityBands:
    """Per-bin percentile bands on observed frequency (fixed bin edges).

    A bin whose band excludes its own mean confidence is evidence of
    miscalibration *in that region* beyond resampling noise.
    """
    _check_boot_args(n_boot, alpha)
    y_true, y_prob = _validate_binary_probs(y_true, y_prob, n_bins)
    if strategy == "equal_width":
        edges = _equal_width_edges(n_bins)
    elif strategy in ("quantile", "adaptive"):
        edges = _quantile_edges(y_prob, n_bins)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")
    n_eff = int(edges.size - 1)
    bin_ids = _bin_ids_from_edges(y_prob, edges)

    counts = np.bincount(bin_ids, minlength=n_eff).astype(int)
    pos = np.bincount(bin_ids, weights=y_true, minlength=n_eff)
    psum = np.bincount(bin_ids, weights=y_prob, minlength=n_eff)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(counts > 0, pos / np.maximum(counts, 1), np.nan)
        conf = np.where(counts > 0, psum / np.maximum(counts, 1), np.nan)

    rng = np.random.default_rng(random_state)
    n = y_true.size
    boot_acc = np.full((n_boot, n_eff), np.nan, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b_ids = bin_ids[idx]
        c = np.bincount(b_ids, minlength=n_eff)
        p = np.bincount(b_ids, weights=y_true[idx], minlength=n_eff)
        ok = c > 0
        boot_acc[i, ok] = p[ok] / c[ok]

    lower = np.full(n_eff, np.nan)
    upper = np.full(n_eff, np.nan)
    for b in range(n_eff):
        col = boot_acc[:, b]
        col = col[~np.isnan(col)]
        if col.size:
            lower[b], upper[b] = np.quantile(col, [alpha / 2.0, 1.0 - alpha / 2.0])

    return ReliabilityBands(
        bin_edges=edges,
        bin_confidence=conf,
        bin_accuracy=acc,
        acc_lower=lower,
        acc_upper=upper,
        bin_counts=counts,
        alpha=float(alpha),
    )
