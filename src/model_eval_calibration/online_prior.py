"""Row-by-row (online) target-prior tracking from unlabelled rows (slice E).

Slice D tracked the prior per batch of 400 rows and its best policy ran EM
on the batch being decided, i.e. decisions waited for a whole batch. Here
every row is decided as it arrives, using a prior estimated from the
*previous* rows' unlabelled features only. Every tracker returns
``prior_used`` with ``prior_used[i]`` a function of ``p[:i]`` (never of
row i or later), which ``tests/test_online_prior.py`` checks by
perturbing the future.

Under label shift the target feature density is the source density times
``r1 p_s(1|x) + r0 p_s(0|x)`` (r1 = pi/pi_s, r0 = (1-pi)/(1-pi_s)), so the
per-row log-likelihood of a candidate prior needs only the source posterior
(``mixture_loglik``); Saerens EM maximises its sum.

* ``online_em_path`` -- stochastic-approximation online EM (Cappe &
  Moulines, 2009) with a constant step ``step``:
  pi <- pi + step (adjusted posterior of the new row - pi). A constant step
  is exponential forgetting with a nominal memory of 1/step rows. The
  *effective* reaction is slower: each EM step only moves pi by the
  observed-information share of the gap, which is small when the classes
  overlap and the prior is low (a 0.002 step takes ~870 rows to cover half
  of a 0.3 -> 0.05 jump, a 0.02 step ~100).
* ``window_em_path`` -- full Saerens EM on the last ``window`` rows,
  warm-started from the previous estimate.
* ``cusum_em_path`` -- EM on the rows since the last detected change,
  with a two-sided Page CUSUM on the mixture likelihood to detect jumps.
"""

from __future__ import annotations

import numpy as np

from model_eval_calibration.prior_estimation import _clip_prior, adjust_posterior

_EPS = 1e-12


def mixture_loglik(p_source: np.ndarray, source_prior: float, prior: float) -> np.ndarray:
    """Per-row log p_t(x) - log p_s(x) under target prior ``prior`` (source posterior ``p_source``)."""
    p = np.clip(np.asarray(p_source, dtype=float), _EPS, 1.0 - _EPS)
    r1 = prior / source_prior
    r0 = (1.0 - prior) / (1.0 - source_prior)
    return np.log(r1 * p + r0 * (1.0 - p))


def _check(p: np.ndarray, source_prior: float) -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("need at least one row")
    if not 0.0 < source_prior < 1.0:
        raise ValueError("source_prior must be in (0, 1)")
    return p


def online_em_path(p: np.ndarray, source_prior: float, step: float, start: float | None = None) -> np.ndarray:
    """Prior used for each row by constant-step online EM (row i sees rows < i only)."""
    p = _check(p, source_prior)
    if not 0.0 < step <= 1.0:
        raise ValueError("step must be in (0, 1]")
    pi = float(source_prior if start is None else start)
    out = np.empty(p.size)
    r_s = source_prior / (1.0 - source_prior)
    for i in range(p.size):
        out[i] = pi
        q = min(max(p[i], _EPS), 1.0 - _EPS)
        a = (pi / (1.0 - pi)) / r_s * q  # adjusted posterior of row i at the current prior, inlined
        post = a / (a + (1.0 - q))
        pi = _clip_prior(pi + step * (post - pi))
    return out


def em_on(p: np.ndarray, source_prior: float, start: float, tol: float = 1e-6, max_iter: int = 500) -> float:
    """Saerens EM fixed point on ``p`` warm-started at ``start``."""
    pi = float(start)
    for _ in range(max_iter):
        new = _clip_prior(float(adjust_posterior(p, source_prior, pi).mean()))
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


def window_em_path(p: np.ndarray, source_prior: float, window: int, refit_every: int = 1,
                   start: float | None = None) -> np.ndarray:
    """Prior used for each row by EM on the last ``window`` rows (refit every ``refit_every`` rows)."""
    p = _check(p, source_prior)
    if window < 1 or refit_every < 1:
        raise ValueError("window and refit_every must be >= 1")
    pi = float(source_prior if start is None else start)
    out = np.empty(p.size)
    for i in range(p.size):
        if i > 0 and i % refit_every == 0:
            pi = em_on(p[max(0, i - window) : i], source_prior, pi)
        out[i] = pi
    return out


def cusum_em_path(p: np.ndarray, source_prior: float, threshold: float = 6.0, delta_logit: float = 0.8,
                  min_rows: int = 100, max_rows: int = 2000, refit_every: int = 5,
                  start: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Change-point-restarted EM: EM on the current segment, two-sided Page CUSUM to end it.

    The prior used at row i is EM on rows [seg_start, i) (at most
    ``max_rows`` back; at least the last ``min_rows`` rows, so a young
    segment borrows a few pre-change rows). After row i is decided its
    *unlabelled* source posterior feeds two CUSUM statistics, one per
    direction, each the running log-likelihood ratio (``mixture_loglik``)
    of the prior moved by +/- ``delta_logit`` on the logit scale against the
    current estimate. When one exceeds ``threshold`` a change is declared;
    the new segment starts where that statistic last left zero (Page's
    change-point estimate) and EM refits there. Returns (prior used per
    row, alarm rows).
    """
    p = _check(p, source_prior)
    if threshold <= 0 or delta_logit <= 0 or not 1 <= min_rows <= max_rows or refit_every < 1:
        raise ValueError("bad CUSUM settings")
    pi = float(source_prior if start is None else start)
    seg = 0
    s_up = s_dn = 0.0
    z_up = z_dn = 0  # row after which each statistic last sat at zero
    out, alarms = np.empty(p.size), []
    refit_at = 0
    for i in range(p.size):
        if i > 0 and (i >= refit_at or i - seg <= min_rows):
            lo = max(min(seg, i - min_rows), i - max_rows, 0)
            pi = em_on(p[lo:i], source_prior, pi)
            refit_at = i + refit_every
        out[i] = pi
        lg = np.log(pi / (1.0 - pi))
        up = 1.0 / (1.0 + np.exp(-(lg + delta_logit)))
        dn = 1.0 / (1.0 + np.exp(-(lg - delta_logit)))
        base = mixture_loglik(p[i : i + 1], source_prior, pi)[0]
        s_up += mixture_loglik(p[i : i + 1], source_prior, up)[0] - base
        s_dn += mixture_loglik(p[i : i + 1], source_prior, dn)[0] - base
        if s_up <= 0:
            s_up, z_up = 0.0, i + 1
        if s_dn <= 0:
            s_dn, z_dn = 0.0, i + 1
        if s_up > threshold or s_dn > threshold:
            alarms.append(i)
            seg = z_up if s_up > threshold else z_dn
            s_up = s_dn = 0.0
            z_up = z_dn = i + 1
            refit_at = i + 1  # refit on the new segment at the next row
    return out, np.asarray(alarms, dtype=int)
