"""Calibration diagnostics: Brier score, ECE, and reliability curves.

Accuracy alone can hide over-/under-confident probabilities. These metrics
diagnose that gap. They are research diagnostics, not production SLAs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between binary labels and predicted probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels in {0, 1}.
    y_prob : array-like of shape (n,)
        Predicted P(y=1) in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.size == 0:
        raise ValueError("empty inputs")
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width Expected Calibration Error for binary probabilities.

    Bins predicted probabilities on [0, 1], then averages
    |bin accuracy - bin confidence| weighted by bin mass.

    ECE is binning-sensitive and noisy on small samples — use beside Brier,
    not instead of it.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.size == 0:
        raise ValueError("empty inputs")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize on interior edges; last bin includes 1.0
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)

    ece = 0.0
    n = float(y_true.size)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


@dataclass(frozen=True)
class ReliabilityCurve:
    """Per-bin reliability diagram data (equal-width bins)."""

    bin_centers: np.ndarray
    bin_accuracy: np.ndarray
    bin_confidence: np.ndarray
    bin_counts: np.ndarray
    n_bins: int

    def as_dict(self) -> dict:
        return {
            "n_bins": self.n_bins,
            "bin_centers": self.bin_centers.tolist(),
            "bin_accuracy": self.bin_accuracy.tolist(),
            "bin_confidence": self.bin_confidence.tolist(),
            "bin_counts": self.bin_counts.tolist(),
        }


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> ReliabilityCurve:
    """Compute equal-width reliability diagram coordinates.

    Empty bins get NaN accuracy/confidence so callers can skip them when
    plotting. Counts are always defined.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.size == 0:
        raise ValueError("empty inputs")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)

    acc = np.full(n_bins, np.nan, dtype=float)
    conf = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        counts[b] = int(mask.sum())
        if counts[b] == 0:
            continue
        acc[b] = float(y_true[mask].mean())
        conf[b] = float(y_prob[mask].mean())

    return ReliabilityCurve(
        bin_centers=centers,
        bin_accuracy=acc,
        bin_confidence=conf,
        bin_counts=counts,
        n_bins=n_bins,
    )


def plot_reliability_diagram(
    curve: ReliabilityCurve,
    ax: Optional[object] = None,
    title: str = "Reliability diagram",
):
    """Plot a reliability diagram; matplotlib is optional at import time.

    Returns the matplotlib Axes used. Skips empty bins.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    mask = curve.bin_counts > 0
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    ax.plot(
        curve.bin_confidence[mask],
        curve.bin_accuracy[mask],
        "o-",
        label="model",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    return ax
