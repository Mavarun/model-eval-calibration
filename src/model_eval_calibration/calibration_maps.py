"""Post-hoc probability calibration maps fitted on a held-out calibration set.

Unlike :mod:`calibration` (which wraps ``CalibratedClassifierCV`` and refits
the base model), these maps act on *already predicted* probabilities from a
frozen base model. That lets several calibrators be compared on exactly the
same base predictions and calibration rows, and lets each be fitted with
per-row ``sample_weight`` (used for importance-weighted calibration under
covariate shift).

Maps (binary, P(y=1)):

* ``TemperatureScaling`` -- ``sigmoid(logit(p) / T)``; one parameter, no
  bias, preserves the 0.5 decision boundary and the ranking (Guo et al., 2017).
* ``PlattScaling`` -- ``sigmoid(a * logit(p) + b)`` on the base logit.
* ``BetaCalibration`` -- ``sigmoid(a ln p - b ln(1 - p) + c)`` with
  ``a, b >= 0`` (Kull, Silva Filho & Flach, 2017). Contains the identity
  (a=b=1, c=0) and Platt-on-logit (a=b) as special cases.
* ``IsotonicMap`` -- weighted isotonic regression, non-parametric,
  monotone, can overfit small calibration sets and produces ties.

Parametric maps are fitted by minimising weighted binary log-loss with
L-BFGS-B (bounds enforce T > 0 and the beta-calibration sign constraints).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

_EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float).ravel(), _EPS, 1.0 - _EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _prep(p, y, sample_weight):
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probabilities and labels must be non-empty, same shape")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("labels must be binary 0/1")
    if sample_weight is None:
        w = np.ones_like(p)
    else:
        w = np.asarray(sample_weight, dtype=float).ravel()
        if w.shape != p.shape or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("sample_weight must be non-negative, same shape, sum > 0")
    return p, y, w / w.mean()


def _weighted_nll_and_grad(z: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted mean log-loss of sigmoid(z) and d/dz."""
    # log(1+exp(z)) - y z, numerically stable
    loss = np.logaddexp(0.0, z) - y * z
    grad_z = _sigmoid(z) - y
    return float(np.mean(w * loss)), w * grad_z / z.size


class _BaseMap:
    fitted_: bool = False

    def _check(self):
        if not self.fitted_:
            raise RuntimeError(f"{type(self).__name__} is not fitted")

    def fit_transform(self, p, y, sample_weight=None) -> np.ndarray:
        return self.fit(p, y, sample_weight=sample_weight).transform(p)


class TemperatureScaling(_BaseMap):
    """Single-temperature scaling of the base logit (bounded T in [0.05, 20])."""

    def __init__(self, t_bounds: tuple[float, float] = (0.05, 20.0)) -> None:
        self.t_bounds = t_bounds
        self.temperature_: Optional[float] = None

    def fit(self, p, y, sample_weight=None) -> "TemperatureScaling":
        p, y, w = _prep(p, y, sample_weight)
        z0 = _logit(p)

        # optimise log T for conditioning
        def obj(theta):
            inv_t = np.exp(-theta[0])
            loss, gz = _weighted_nll_and_grad(z0 * inv_t, y, w)
            return loss, np.array([float(np.sum(gz * z0) * -inv_t)])

        lo, hi = np.log(self.t_bounds[0]), np.log(self.t_bounds[1])
        res = minimize(obj, x0=np.array([0.0]), jac=True, method="L-BFGS-B", bounds=[(lo, hi)])
        self.temperature_ = float(np.exp(res.x[0]))
        self.fitted_ = True
        return self

    def transform(self, p) -> np.ndarray:
        self._check()
        return _sigmoid(_logit(p) / self.temperature_)


class PlattScaling(_BaseMap):
    """Logistic regression on the base logit: sigmoid(a * logit(p) + b)."""

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None

    def fit(self, p, y, sample_weight=None) -> "PlattScaling":
        p, y, w = _prep(p, y, sample_weight)
        z0 = _logit(p)

        def obj(theta):
            loss, gz = _weighted_nll_and_grad(theta[0] * z0 + theta[1], y, w)
            return loss, np.array([np.sum(gz * z0), np.sum(gz)])

        res = minimize(obj, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
        self.coef_ = np.asarray(res.x, dtype=float)
        self.fitted_ = True
        return self

    def transform(self, p) -> np.ndarray:
        self._check()
        a, b = self.coef_
        return _sigmoid(a * _logit(p) + b)


class BetaCalibration(_BaseMap):
    """Three-parameter beta calibration with a, b >= 0 (monotone map)."""

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None  # (a, b, c)

    @staticmethod
    def _features(p):
        p = _clip(p)
        return np.log(p), -np.log1p(-p)

    def fit(self, p, y, sample_weight=None) -> "BetaCalibration":
        p, y, w = _prep(p, y, sample_weight)
        f1, f2 = self._features(p)

        def obj(theta):
            z = theta[0] * f1 + theta[1] * f2 + theta[2]
            loss, gz = _weighted_nll_and_grad(z, y, w)
            return loss, np.array([np.sum(gz * f1), np.sum(gz * f2), np.sum(gz)])

        res = minimize(
            obj,
            x0=np.array([1.0, 1.0, 0.0]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None), (0.0, None), (None, None)],
        )
        self.coef_ = np.asarray(res.x, dtype=float)
        self.fitted_ = True
        return self

    def transform(self, p) -> np.ndarray:
        self._check()
        f1, f2 = self._features(p)
        a, b, c = self.coef_
        return _sigmoid(a * f1 + b * f2 + c)


class IsotonicMap(_BaseMap):
    """Weighted isotonic regression clipped to [eps, 1 - eps]."""

    def __init__(self) -> None:
        self.model_: Optional[IsotonicRegression] = None

    def fit(self, p, y, sample_weight=None) -> "IsotonicMap":
        p, y, w = _prep(p, y, sample_weight)
        self.model_ = IsotonicRegression(
            y_min=_EPS, y_max=1.0 - _EPS, out_of_bounds="clip", increasing=True
        )
        self.model_.fit(p, y, sample_weight=w)
        self.fitted_ = True
        return self

    def transform(self, p) -> np.ndarray:
        self._check()
        return np.asarray(self.model_.predict(np.asarray(p, dtype=float).ravel()), dtype=float)


class IdentityMap(_BaseMap):
    """No-op baseline so 'raw' can be handled like any other map."""

    def fit(self, p, y, sample_weight=None) -> "IdentityMap":
        _prep(p, y, sample_weight)
        self.fitted_ = True
        return self

    def transform(self, p) -> np.ndarray:
        self._check()
        return np.asarray(p, dtype=float).ravel()


def make_calibration_maps() -> dict[str, _BaseMap]:
    """Fresh instances of every map, keyed by the name used in reports."""
    return {
        "raw": IdentityMap(),
        "temperature": TemperatureScaling(),
        "platt": PlattScaling(),
        "beta": BetaCalibration(),
        "isotonic": IsotonicMap(),
    }
