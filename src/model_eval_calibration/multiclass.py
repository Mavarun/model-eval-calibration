"""Multiclass calibration metrics and post-hoc maps (slice G).

Binary tools elsewhere in the package look at one probability per row. With K classes there
are three common views of "calibrated", and they disagree:

* **top-label (confidence) ECE**: bin rows by max_k p_k, compare to accuracy (Guo et al., 2017).
  Says nothing about the non-predicted classes.
* **classwise ECE**: for every class k, the binary ECE of (1[y = k], p_k), averaged over k
  (Kull et al., 2019). Sees miscalibration in minority classes and in second choices.
* **multiclass Brier / log loss**: proper scores over the full vector.

Maps act on the frozen base model's log-probabilities ``z = log p`` (shift-invariant per row):

* ``MulticlassTemperature``: softmax(z / T), one parameter, keeps the argmax.
* ``VectorScaling``: softmax(w * z + b) with per-class w, b (2K parameters, L2 on w - 1 and b),
  can move the argmax.

Both are fitted by minimising mean multiclass NLL on calibration rows with L-BFGS-B.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

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


def _debiased_sq_gap(conf, hit, n_bins):
    """Debiased squared L2 calibration error over equal-mass bins (Kumar et al., 2019).

    The plug-in sum_b w_b (hit_b - conf_b)^2 is biased upward by each bin's sampling variance
    of the hit rate; subtracting hit_b (1 - hit_b) / (n_b - 1) per bin removes that bias to
    first order, so a calibrated model scores ~0 instead of a positive floor. Can be negative.
    """
    n = conf.size
    order = np.argsort(conf, kind="mergesort")
    bins = np.empty(n, dtype=int)
    bins[order] = np.minimum((np.arange(n) * n_bins) // n, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        nb = int(m.sum())
        if nb < 2:
            continue
        hb = hit[m].mean()
        total += nb / n * ((hb - conf[m].mean()) ** 2 - hb * (1.0 - hb) / (nb - 1))
    return float(total)


def classwise_ce_debiased(y, p, n_bins: int = 15) -> float:
    """Classwise calibration error with the debiased L2 estimator: sqrt(max(0, mean_k CE_k^2)).

    A second estimator for slice G2. Binned ECE has a positive floor that grows as bins get
    small, so a map that only changes how many rows sit in sparse bins can move ECE without
    changing calibration; the debiased estimate is far less sensitive to that.
    """
    y, p = _check(y, p)
    sq = np.mean([_debiased_sq_gap(p[:, j], (y == j).astype(float), n_bins) for j in range(p.shape[1])])
    return float(np.sqrt(max(0.0, sq)))


def multiclass_brier(y, p) -> float:
    """Mean over rows of sum_k (p_k - 1[y = k])^2 (range [0, 2])."""
    y, p = _check(y, p)
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def multiclass_nll(y, p) -> float:
    y, p = _check(y, p)
    return float(-np.mean(np.log(np.clip(p[np.arange(y.size), y], _EPS, 1.0))))


def _logp(p):
    return np.log(np.clip(np.asarray(p, dtype=float), _EPS, 1.0))


class MulticlassTemperature:
    """softmax(log p / T), T > 0 fitted by NLL on calibration rows."""

    def __init__(self) -> None:
        self.T_: float | None = None

    def fit(self, p, y):
        y, p = _check(y, p)
        z = _logp(p)

        def f(theta):
            t = np.exp(theta[0])
            ls = log_softmax(z / t, axis=1)
            nll = -ls[np.arange(y.size), y].mean()
            q = np.exp(ls)
            # theta = log T: d nll / d theta = mean_i (z_iy - sum_k q_ik z_ik) / T
            g = np.mean(z[np.arange(y.size), y] - np.sum(q * z, axis=1)) / t
            return nll, np.array([g])

        r = minimize(f, x0=np.zeros(1), jac=True, method="L-BFGS-B", bounds=[(-5.0, 5.0)])
        self.T_ = float(np.exp(r.x[0]))
        return self

    def transform(self, p) -> np.ndarray:
        if self.T_ is None:
            raise RuntimeError("fit first")
        return np.exp(log_softmax(_logp(p) / self.T_, axis=1))


class VectorScaling:
    """softmax(w * log p + b), per-class w and b, with an L2 penalty ``reg`` on (w - 1, b)."""

    def __init__(self, reg: float = 1e-3) -> None:
        if reg < 0:
            raise ValueError("reg must be >= 0")
        self.reg = reg
        self.w_: np.ndarray | None = None
        self.b_: np.ndarray | None = None

    def fit(self, p, y):
        y, p = _check(y, p)
        z = _logp(p)
        n, k = z.shape
        onehot = np.eye(k)[y]

        def f(theta):
            w, b = theta[:k], theta[k:]
            ls = log_softmax(z * w + b, axis=1)
            nll = -np.sum(ls * onehot) / n + self.reg * (np.sum((w - 1) ** 2) + np.sum(b ** 2))
            r = (np.exp(ls) - onehot) / n
            gw = np.sum(r * z, axis=0) + 2 * self.reg * (w - 1)
            gb = np.sum(r, axis=0) + 2 * self.reg * b
            return nll, np.concatenate([gw, gb])

        r = minimize(f, x0=np.concatenate([np.ones(k), np.zeros(k)]), jac=True, method="L-BFGS-B")
        self.w_, self.b_ = r.x[:k], r.x[k:]
        return self

    def transform(self, p) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("fit first")
        return np.exp(log_softmax(_logp(p) * self.w_ + self.b_, axis=1))


# Slice G, pre-registered H1 passed and H2 failed (results/slice_g_decisions.json): temperature
# lowered test classwise ECE on 10/10 seeds in 4 of 6 (dataset, model) cells without raising NLL
# by more than 0.01 anywhere; vector scaling beat it in only 2 cells (both naive Bayes).
RECOMMENDED_MULTICLASS_MAP = "temperature"


def fit_multiclass_map(p_cal, y_cal, method: str = RECOMMENDED_MULTICLASS_MAP):
    """Fit a post-hoc multiclass map on calibration rows; ``method`` in {'temperature', 'vector', 'none'}.

    The default is the slice-G recommendation. ``'none'`` returns an object whose ``transform`` is
    the identity, so callers can switch calibration off without branching.
    """
    if method == "temperature":
        return MulticlassTemperature().fit(p_cal, y_cal)
    if method == "vector":
        return VectorScaling().fit(p_cal, y_cal)
    if method == "none":
        _check(y_cal, p_cal)
        return _Identity()
    raise ValueError("method must be 'temperature', 'vector' or 'none'")


class _Identity:
    def transform(self, p) -> np.ndarray:
        return np.asarray(p, dtype=float)
