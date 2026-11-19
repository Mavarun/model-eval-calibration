"""Multiclass calibration metrics (slice G).

Binary tools elsewhere in the package look at one probability per row. With K classes there
are three common views of "calibrated", and they disagree:

* **top-label (confidence) ECE**: bin rows by max_k p_k, compare to accuracy (Guo et al., 2017).
  Says nothing about the non-predicted classes.
* **classwise ECE**: for every class k, the binary ECE of (1[y = k], p_k), averaged over k
  (Kull et al., 2019). Sees miscalibration in minority classes and in second choices.
* **multiclass Brier / log loss**: proper scores over the full vector.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _check(y, p):
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[0] != y.size or p.shape[1] < 2:
        raise ValueError("need probs of shape (n, K >= 2) and n labels")
    if y.min() < 0 or y.max() >= p.shape[1]:
        raise ValueError("labels must be in [0, K)")
    if np.any(p < -1e-9) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("rows of probs must be non-negative and sum to 1")
    return y, np.clip(p, 0.0, 1.0)


def _binned_gap(conf, hit, n_bins, adaptive):
    n = conf.size
    if adaptive:
        order = np.argsort(conf, kind="mergesort")
        bins = np.empty(n, dtype=int)
        bins[order] = np.minimum((np.arange(n) * n_bins) // n, n_bins - 1)
    else:
        bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    gap = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            gap += m.sum() / n * abs(hit[m].mean() - conf[m].mean())
    return float(gap)


def top_label_ece(y, p, n_bins: int = 15, adaptive: bool = False) -> float:
    """ECE of the confidence max_k p_k against top-label accuracy."""
    y, p = _check(y, p)
    return _binned_gap(p.max(axis=1), (p.argmax(axis=1) == y).astype(float), n_bins, adaptive)


def classwise_ece(y, p, n_bins: int = 15, adaptive: bool = True) -> float:
    """Mean over classes of the binary ECE of (1[y = k], p_k). Adaptive (equal-mass) bins by default."""
    y, p = _check(y, p)
    k = p.shape[1]
    return float(np.mean([_binned_gap(p[:, j], (y == j).astype(float), n_bins, adaptive) for j in range(k)]))


def multiclass_brier(y, p) -> float:
    """Mean over rows of sum_k (p_k - 1[y = k])^2 (range [0, 2])."""
    y, p = _check(y, p)
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def multiclass_nll(y, p) -> float:
    y, p = _check(y, p)
    return float(-np.mean(np.log(np.clip(p[np.arange(y.size), y], _EPS, 1.0))))
