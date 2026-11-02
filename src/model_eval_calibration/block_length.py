"""Data-driven block length for the circular block bootstrap.

Slice D fixed the bootstrap block at 4 batches. This module replaces the
hand pick with the Politis & White (2004) plug-in rule, using the
Patton, Politis & White (2009) correction of the circular-block constant:

    b_opt = (2 G^2 / D_CB)^(1/3) n^(1/3),
    G     = sum_{|k| <= M} lam(k / M) |k| R(k),
    D_CB  = (4/3) g(0)^2,   g(0) = sum_{|k| <= M} lam(k / M) R(k),

with R the sample autocovariance and lam the flat-top (trapezoid) kernel.
The bandwidth M = 2 m_hat, where m_hat is the smallest lag after which
K_N = max(5, sqrt(log10 n)) consecutive sample autocorrelations are all
inside +/- c sqrt(log10 n / n) (c = 2). The result is capped at
min(3 sqrt(n), n / 3) and floored at 1.

For several independent series (one per seed) ``pooled_block_length``
returns the median of the per-series rules, rounded up.

m_hat follows the paper (lags m+1 .. m+K_N quiet). ``arch``'s
implementation tests lags m .. m+K_N-1, so on dependent series it can pick
a slightly longer block; on white noise the two agree.
"""

from __future__ import annotations

import math

import numpy as np


def _flat_top(t: np.ndarray) -> np.ndarray:
    a = np.abs(t)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def _autocov(x: np.ndarray, max_lag: int) -> np.ndarray:
    n = x.size
    xc = x - x.mean()
    return np.array([np.dot(xc[: n - k], xc[k:]) / n for k in range(max_lag + 1)])


def optimal_block_length(x: np.ndarray, c: float = 2.0) -> float:
    """Politis-White / Patton-Politis-White circular-block bootstrap block length (real-valued, >= 1)."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 8:
        raise ValueError("need at least 8 observations")
    k_n = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    m_max = int(math.ceil(math.sqrt(n))) + k_n
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    m_max = min(m_max, n - 1)
    acv = _autocov(x, m_max)
    if acv[0] <= 0:  # constant series: no dependence to protect
        return 1.0
    rho = acv[1:] / acv[0]
    band = c * math.sqrt(math.log10(n) / n)
    inside = np.abs(rho) < band
    m_hat = None
    for m in range(0, m_max - k_n + 1):
        if inside[m : m + k_n].all():  # lags m+1 .. m+k_n are insignificant
            m_hat = m
            break
    M = m_max if m_hat is None else min(2 * max(m_hat, 1), m_max)  # no quiet run: widest bandwidth
    k = np.arange(1, M + 1)
    lam = _flat_top(k / M)
    g0 = acv[0] + 2.0 * np.sum(lam * acv[1 : M + 1])
    G = 2.0 * np.sum(lam * k * acv[1 : M + 1])
    d_cb = (4.0 / 3.0) * g0**2
    if d_cb <= 0 or G == 0:
        return 1.0
    b = (2.0 * G**2 / d_cb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 1.0), b_max))


def pooled_block_length(series: list[np.ndarray]) -> int:
    """Median of per-series rules, rounded up to an integer block (>= 1)."""
    if not series:
        raise ValueError("need at least one series")
    return max(1, int(math.ceil(float(np.median([optimal_block_length(s) for s in series])))))
