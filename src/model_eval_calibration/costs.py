"""Cost-aware binary decisions on probabilities.

With a false-positive cost ``c_fp`` and a false-negative cost ``c_fn``
(correct decisions cost 0), the Bayes decision on a *calibrated* posterior
p = P(y=1 | x) is "predict 1 iff p >= c_fp / (c_fp + c_fn)"
(``bayes_threshold``). Two ways to get a threshold in practice:

* **plug-in**: apply the Bayes threshold to (re-)calibrated probabilities;
  correct only as far as the probabilities are calibrated *on the rows being
  decided*.
* **tuned**: pick the threshold that minimises empirical cost on labelled
  calibration rows (``tune_threshold``); needs no calibration but is tied to
  the calibration rows' class prior.

Under label shift both break in the same way: the source posterior is off by
the prior log-odds, so the cost-optimal threshold *on source scores* moves to
``shifted_threshold`` -- e.g. with c_fn = 5 c_fp, pi_s = 0.3 -> pi_t = 0.1
moves it from 0.167 to 0.435.
"""

from __future__ import annotations

import numpy as np


def _check_costs(c_fp: float, c_fn: float) -> None:
    if c_fp <= 0 or c_fn <= 0:
        raise ValueError("costs must be positive")


def bayes_threshold(c_fp: float, c_fn: float) -> float:
    _check_costs(c_fp, c_fn)
    return c_fp / (c_fp + c_fn)


def decisions(p: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(p, dtype=float) >= threshold).astype(int)


def expected_cost(y: np.ndarray, yhat: np.ndarray, c_fp: float, c_fn: float) -> float:
    """Mean cost per decided row."""
    _check_costs(c_fp, c_fn)
    y = np.asarray(y).astype(int).ravel()
    yhat = np.asarray(yhat).astype(int).ravel()
    if y.size != yhat.size or y.size == 0:
        raise ValueError("y and yhat must be non-empty and the same length")
    fp = np.sum((yhat == 1) & (y == 0))
    fn = np.sum((yhat == 0) & (y == 1))
    return float((c_fp * fp + c_fn * fn) / y.size)


def tune_threshold(p: np.ndarray, y: np.ndarray, c_fp: float, c_fn: float) -> float:
    """Threshold on ``p`` minimising empirical cost on labelled rows (ties -> lowest cost, then nearest 0.5).

    Candidates are midpoints between sorted unique scores plus the two ends,
    evaluated in O(n log n) with cumulative counts.
    """
    _check_costs(c_fp, c_fn)
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y).astype(int).ravel()
    if p.size != y.size or p.size == 0:
        raise ValueError("p and y must be non-empty and the same length")
    u = np.unique(p)
    cands = np.r_[u[0] - 1e-9, (u[:-1] + u[1:]) / 2.0, u[-1] + 1e-9]
    order = np.argsort(p)
    ps, ys = p[order], y[order]
    # rows with score < t are predicted 0
    below = np.searchsorted(ps, cands, side="left")
    pos_cum = np.r_[0, np.cumsum(ys)]
    neg_cum = np.r_[0, np.cumsum(1 - ys)]
    fn = pos_cum[below]
    fp = neg_cum[-1] - neg_cum[below]
    cost = c_fp * fp + c_fn * fn
    best = np.flatnonzero(cost == cost.min())
    return float(cands[best[np.argmin(np.abs(cands[best] - 0.5))]])


def shifted_threshold(c_fp: float, c_fn: float, source_prior: float, target_prior: float) -> float:
    """Cost-optimal threshold on *source-calibrated* scores for rows drawn at ``target_prior``."""
    t = bayes_threshold(c_fp, c_fn)
    z = np.log(t / (1 - t)) - np.log(target_prior / (1 - target_prior)) + np.log(source_prior / (1 - source_prior))
    return float(1.0 / (1.0 + np.exp(-z)))
