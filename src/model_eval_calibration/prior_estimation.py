"""Target class-prior estimation from unlabelled target rows, and posterior adjustment.

Under label shift (``label_shift.py``) the target posterior is the source
posterior with its logit moved by the prior log-odds difference
(``adjust_posterior``), so everything hinges on estimating pi_t without
target labels. Two standard estimators:

* ``em_prior`` -- Saerens, Latinne & Decaestecker (2002). Alternate
  E-step (re-weight source posteriors to the current prior guess) and M-step
  (new prior = mean adjusted posterior) until the prior stops moving. It is a
  maximum-likelihood estimate *if the source posteriors are calibrated*;
  with miscalibrated posteriors it converges to the wrong fixed point (the
  hypothesis slice D tests: calibration on source matters for EM).
* ``bbse_prior`` -- black-box shift estimation (Lipton, Wang & Smola,
  2018). Uses hard predictions only: with the source joint confusion matrix
  C[i, j] = P_s(yhat=i, y=j) on labelled source calibration rows and the
  target prediction rate mu_t[i] = P_t(yhat=i), solve C w = mu_t for the
  class importance weights w = pi_t / pi_s. Consistent for any fixed
  classifier with invertible C, calibrated or not, but noisier.

Both return a prior clipped to [``PRIOR_FLOOR``, 1 - ``PRIOR_FLOOR``].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRIOR_FLOOR = 1e-3
_EPS = 1e-12


def _clip_prior(p: float) -> float:
    return float(np.clip(p, PRIOR_FLOOR, 1.0 - PRIOR_FLOOR))


def adjust_posterior(p_source: np.ndarray, source_prior: float, target_prior: float) -> np.ndarray:
    """Move P_s(y=1|x) to the target prior: p' = r p / (r p + (1 - p) r0)."""
    p = np.clip(np.asarray(p_source, dtype=float), _EPS, 1.0 - _EPS)
    r1 = target_prior / source_prior
    r0 = (1.0 - target_prior) / (1.0 - source_prior)
    num = r1 * p
    return num / (num + r0 * (1.0 - p))


@dataclass(frozen=True)
class EMResult:
    prior: float
    n_iter: int
    converged: bool


def em_prior(p_source_target: np.ndarray, source_prior: float, tol: float = 1e-6,
             max_iter: int = 1000) -> EMResult:
    """Saerens et al. (2002) EM on source-model posteriors of unlabelled target rows."""
    p = np.asarray(p_source_target, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("need at least one target row")
    if not 0.0 < source_prior < 1.0:
        raise ValueError("source_prior must be in (0, 1)")
    pi = float(source_prior)
    for it in range(1, max_iter + 1):
        new = _clip_prior(float(adjust_posterior(p, source_prior, pi).mean()))
        if abs(new - pi) < tol:
            return EMResult(new, it, True)
        pi = new
    return EMResult(pi, max_iter, False)


def bbse_prior(yhat_cal: np.ndarray, y_cal: np.ndarray, yhat_target: np.ndarray) -> float:
    """Lipton et al. (2018) binary BBSE from hard predictions."""
    yhat_cal = np.asarray(yhat_cal).astype(int).ravel()
    y_cal = np.asarray(y_cal).astype(int).ravel()
    yhat_target = np.asarray(yhat_target).astype(int).ravel()
    if yhat_cal.size != y_cal.size:
        raise ValueError("yhat_cal and y_cal must have the same length")
    C = np.array([[np.mean((yhat_cal == i) & (y_cal == j)) for j in (0, 1)] for i in (0, 1)])
    if abs(np.linalg.det(C)) < 1e-9:
        raise ValueError("confusion matrix is singular; BBSE is not identified")
    mu = np.array([np.mean(yhat_target == 0), np.mean(yhat_target == 1)])
    w = np.linalg.solve(C, mu)
    w = np.clip(w, 0.0, None)
    pi_s = np.array([np.mean(y_cal == 0), np.mean(y_cal == 1)])
    pi_t = w * pi_s
    if pi_t.sum() <= 0:
        return _clip_prior(0.0)
    return _clip_prior(float(pi_t[1] / pi_t.sum()))
