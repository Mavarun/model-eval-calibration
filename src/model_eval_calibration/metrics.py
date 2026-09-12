"""Calibration diagnostics: Brier score, ECE, and reliability curves.

Accuracy alone can hide over-/under-confident probabilities. These metrics
diagnose that gap. They are research diagnostics, not production SLAs.

Equal-width ECE is sensitive to probability mass piled in a few bins
(common under class imbalance). Adaptive / quantile ECE bins by probability
quantiles so each bin has roughly equal mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


BinningStrategy = Literal["equal_width", "quantile", "adaptive"]


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


def _validate_binary_probs(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.size == 0:
        raise ValueError("empty inputs")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    return y_true, y_prob


def _equal_width_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def _quantile_edges(y_prob: np.ndarray, n_bins: int) -> np.ndarray:
    """Adaptive equal-mass bin edges from probability quantiles.

    Duplicate quantile values (e.g. many identical probs) collapse into
    fewer unique edges; callers must handle fewer bins gracefully.
    """
    # endpoints pinned to [0, 1]; interior from empirical quantiles
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y_prob, qs)
    edges = np.asarray(edges, dtype=float)
    edges[0] = 0.0
    edges[-1] = 1.0
    # enforce non-decreasing unique edges (duplicates remove empty mass splits)
    uniq = [float(edges[0])]
    for e in edges[1:]:
        if float(e) > uniq[-1]:
            uniq.append(float(e))
    if uniq[-1] < 1.0:
        uniq.append(1.0)
    if len(uniq) < 2:
        return np.array([0.0, 1.0], dtype=float)
    return np.asarray(uniq, dtype=float)


def _bin_ids_from_edges(y_prob: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each probability to a bin index in [0, n_bins_eff).

    Uses digitize on interior right edges so the last bin includes 1.0.
    """
    if edges.size < 2:
        return np.zeros(y_prob.shape[0], dtype=int)
    return np.digitize(y_prob, edges[1:-1], right=False)


def _ece_from_bins(y_true: np.ndarray, y_prob: np.ndarray, bin_ids: np.ndarray, n_bins: int) -> float:
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


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: BinningStrategy = "equal_width",
) -> float:
    """Expected Calibration Error for binary probabilities.

    Parameters
    ----------
    strategy :
        ``equal_width`` — classic linspace bins on [0, 1].
        ``quantile`` / ``adaptive`` — equal-mass bins from probability
        quantiles (more stable when probs are imbalanced / peaked).

    ECE remains binning- and sample-size-sensitive — use beside Brier.
    """
    y_true, y_prob = _validate_binary_probs(y_true, y_prob, n_bins)

    if strategy == "equal_width":
        edges = _equal_width_edges(n_bins)
    elif strategy in ("quantile", "adaptive"):
        edges = _quantile_edges(y_prob, n_bins)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    n_eff = int(edges.size - 1)
    bin_ids = _bin_ids_from_edges(y_prob, edges)
    return _ece_from_bins(y_true, y_prob, bin_ids, n_eff)


def adaptive_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Quantile / equal-mass ECE (alias for strategy='quantile')."""
    return expected_calibration_error(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )


@dataclass(frozen=True)
class ReliabilityCurve:
    """Per-bin reliability diagram data."""

    bin_centers: np.ndarray
    bin_accuracy: np.ndarray
    bin_confidence: np.ndarray
    bin_counts: np.ndarray
    n_bins: int
    strategy: str = "equal_width"
    bin_edges: Optional[np.ndarray] = None

    def as_dict(self) -> dict:
        out = {
            "n_bins": self.n_bins,
            "strategy": self.strategy,
            "bin_centers": self.bin_centers.tolist(),
            "bin_accuracy": self.bin_accuracy.tolist(),
            "bin_confidence": self.bin_confidence.tolist(),
            "bin_counts": self.bin_counts.tolist(),
        }
        if self.bin_edges is not None:
            out["bin_edges"] = self.bin_edges.tolist()
        return out


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: BinningStrategy = "equal_width",
) -> ReliabilityCurve:
    """Compute reliability diagram coordinates.

    Empty bins get NaN accuracy/confidence so callers can skip them when
    plotting. Counts are always defined.
    """
    y_true, y_prob = _validate_binary_probs(y_true, y_prob, n_bins)

    if strategy == "equal_width":
        edges = _equal_width_edges(n_bins)
    elif strategy in ("quantile", "adaptive"):
        edges = _quantile_edges(y_prob, n_bins)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    n_eff = int(edges.size - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_ids = _bin_ids_from_edges(y_prob, edges)

    acc = np.full(n_eff, np.nan, dtype=float)
    conf = np.full(n_eff, np.nan, dtype=float)
    counts = np.zeros(n_eff, dtype=int)

    for b in range(n_eff):
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
        n_bins=n_eff,
        strategy="quantile" if strategy in ("quantile", "adaptive") else "equal_width",
        bin_edges=edges,
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
