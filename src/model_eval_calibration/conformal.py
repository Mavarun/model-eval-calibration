"""Split conformal prediction sets for binary classifiers.

Score: LAC / "1 - softmax of the true class", s(x, y) = 1 - p_hat(y | x).
Prediction set: C(x) = {y in {0, 1} : 1 - p_hat(y | x) <= q_hat}.

Guarantees and their assumptions
--------------------------------
* ``split_conformal_threshold``: if calibration and test rows are
  exchangeable, P(Y in C(X)) >= 1 - alpha (marginally, averaged over the
  draw of the calibration set), and <= 1 - alpha + 1/(n + 1) when scores
  have no ties. Nothing is promised conditionally on x or on the class.
* ``weighted_conformal_thresholds``: under *covariate shift* with known
  density ratio w(x) = p_t(x)/p_s(x), the weighted quantile of Tibshirani,
  Barber, Candes & Ramdas (2019) restores marginal coverage on the target.
  With an *estimated* ratio the guarantee is only approximate, and heavy
  weights inflate variance (and can yield q_hat = inf -> the full set).

These are finite-sample marginal statements; the tests check them by
averaging coverage over many independent calibration draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _p_matrix(p1: np.ndarray) -> np.ndarray:
    p1 = np.clip(np.asarray(p1, dtype=float).ravel(), 0.0, 1.0)
    return np.column_stack([1.0 - p1, p1])


def nonconformity_scores(p1: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LAC score 1 - p_hat(y_i | x_i) for binary labels."""
    y = np.asarray(y).astype(int).ravel()
    P = _p_matrix(p1)
    if P.shape[0] != y.size:
        raise ValueError("p1 and y must have the same length")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must be binary 0/1")
    return 1.0 - P[np.arange(y.size), y]


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def split_conformal_threshold(cal_scores: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample corrected quantile: the ceil((n+1)(1-alpha))-th score.

    Returns ``inf`` when n is too small for the requested level
    (ceil((n+1)(1-alpha)) > n), which yields the trivial full set.
    """
    _check_alpha(alpha)
    s = np.sort(np.asarray(cal_scores, dtype=float).ravel())
    n = s.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def weighted_conformal_thresholds(
    cal_scores: np.ndarray,
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Per-test-point weighted conformal quantiles (Tibshirani et al., 2019).

    For a test point with weight w_t, the threshold is the (1 - alpha)
    quantile of sum_i p_i delta_{s_i} + p_t delta_{+inf}, where
    p_i = w_i / (sum_j w_j + w_t) and p_t = w_t / (sum_j w_j + w_t).
    With all weights equal this reproduces ``split_conformal_threshold``.
    """
    _check_alpha(alpha)
    s = np.asarray(cal_scores, dtype=float).ravel()
    w = np.asarray(cal_weights, dtype=float).ravel()
    wt = np.asarray(test_weights, dtype=float).ravel()
    if s.shape != w.shape or s.size == 0:
        raise ValueError("cal_scores and cal_weights must be non-empty, same shape")
    if np.any(w < 0) or np.any(wt < 0):
        raise ValueError("weights must be non-negative")
    order = np.argsort(s, kind="stable")
    s_sorted, cum_w = s[order], np.cumsum(w[order])
    total = cum_w[-1] + wt  # per test point
    # smallest k with cum_w[k] >= (1 - alpha) * total ; none -> +inf
    target = (1.0 - alpha) * total
    k = np.searchsorted(cum_w, target - 1e-12 * np.maximum(total, 1.0), side="left")
    out = np.full(wt.size, np.inf)
    ok = k < s_sorted.size
    out[ok] = s_sorted[k[ok]]
    return out


def prediction_sets(p1: np.ndarray, q_hat) -> np.ndarray:
    """Boolean (n, 2) membership matrix; column j means label j is in the set.

    ``q_hat`` may be a scalar or a per-row array (weighted conformal).
    """
    P = _p_matrix(p1)
    q = np.asarray(q_hat, dtype=float)
    if q.ndim == 0:
        q = np.full(P.shape[0], float(q))
    q = q.ravel()
    if q.size != P.shape[0]:
        raise ValueError("q_hat must be scalar or one value per row")
    return (1.0 - P) <= q[:, None]


@dataclass(frozen=True)
class SetDiagnostics:
    coverage: float
    mean_set_size: float
    singleton_rate: float
    empty_rate: float
    full_rate: float
    n: int

    def as_dict(self) -> dict:
        return {
            "coverage": self.coverage,
            "mean_set_size": self.mean_set_size,
            "singleton_rate": self.singleton_rate,
            "empty_rate": self.empty_rate,
            "full_rate": self.full_rate,
            "n": self.n,
        }


def set_diagnostics(sets: np.ndarray, y: np.ndarray) -> SetDiagnostics:
    """Empirical coverage and set-size profile on labelled test rows."""
    sets = np.asarray(sets, dtype=bool)
    y = np.asarray(y).astype(int).ravel()
    if sets.ndim != 2 or sets.shape != (y.size, 2):
        raise ValueError("sets must have shape (n, 2) matching y")
    size = sets.sum(axis=1)
    return SetDiagnostics(
        coverage=float(sets[np.arange(y.size), y].mean()),
        mean_set_size=float(size.mean()),
        singleton_rate=float(np.mean(size == 1)),
        empty_rate=float(np.mean(size == 0)),
        full_rate=float(np.mean(size == 2)),
        n=int(y.size),
    )


def class_conditional_coverage(sets: np.ndarray, y: np.ndarray) -> dict[int, float]:
    """Coverage within each true class (marginal guarantee does NOT imply these)."""
    sets = np.asarray(sets, dtype=bool)
    y = np.asarray(y).astype(int).ravel()
    out = {}
    for c in (0, 1):
        m = y == c
        out[c] = float(sets[m, c].mean()) if m.any() else float("nan")
    return out
