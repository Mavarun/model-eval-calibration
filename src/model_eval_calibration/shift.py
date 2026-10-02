"""Synthetic covariate shift and density-ratio estimation.

Covariate shift means P(x) changes between source and target while the
labelling mechanism P(y | x) is held fixed. The generator below makes that
explicit so calibration and conformal coverage can be studied with known
ground truth:

* source  x ~ N(0, I_d)
* target  x ~ N(mu, I_d), mu = shift * u, u a fixed unit-ish direction on
  the first two coordinates
* P(y=1 | x) = sigmoid(f(x)) with
  f(x) = 1.2 x0 - 1.0 x1 + 0.7 (x0^2 - 1) - 0.5

The quadratic term makes a linear-logit model misspecified; under shift
the target mass moves to where that misspecification is largest, so a
model calibrated on source is expected to be miscalibrated on target.

Because both marginals are Gaussians with identity covariance the true
density ratio is analytic: w(x) = p_t(x) / p_s(x) = exp(mu.x - |mu|^2 / 2).
That lets tests check the domain-classifier estimate against the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def true_logit(X: np.ndarray) -> np.ndarray:
    """Ground-truth log-odds f(x) shared by source and target."""
    X = np.asarray(X, dtype=float)
    return 1.2 * X[:, 0] - 1.0 * X[:, 1] + 0.7 * (X[:, 0] ** 2 - 1.0) - 0.5


def true_probability(X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-true_logit(X)))


def shift_vector(n_features: int, shift: float) -> np.ndarray:
    """Mean offset of the target marginal: shift * (1, 0.5, 0, ...)/|.|."""
    if n_features < 2:
        raise ValueError("need at least 2 features")
    u = np.zeros(n_features)
    u[0], u[1] = 1.0, 0.5
    u /= np.linalg.norm(u)
    return float(shift) * u


@dataclass
class CovariateShiftData:
    X_source: np.ndarray
    y_source: np.ndarray
    X_target: np.ndarray
    y_target: np.ndarray
    mu: np.ndarray
    meta: dict = field(default_factory=dict)

    def true_density_ratio(self, X: np.ndarray) -> np.ndarray:
        """Analytic p_t(x) / p_s(x) for the Gaussian marginals."""
        X = np.asarray(X, dtype=float)
        return np.exp(X @ self.mu - 0.5 * float(self.mu @ self.mu))


def make_covariate_shift_data(
    n_source: int = 3000,
    n_target: int = 3000,
    n_features: int = 6,
    shift: float = 1.0,
    random_state: int = 0,
) -> CovariateShiftData:
    """Draw source/target samples with identical P(y|x) and shifted P(x).

    ``shift=0`` gives exchangeable source/target (a control condition).
    """
    if n_source < 1 or n_target < 1:
        raise ValueError("sample sizes must be positive")
    rng = np.random.default_rng(random_state)
    mu = shift_vector(n_features, shift)
    Xs = rng.normal(size=(n_source, n_features))
    Xt = rng.normal(size=(n_target, n_features)) + mu
    ys = (rng.uniform(size=n_source) < true_probability(Xs)).astype(int)
    yt = (rng.uniform(size=n_target) < true_probability(Xt)).astype(int)
    meta = {
        "name": "gaussian_covariate_shift",
        "n_source": int(n_source),
        "n_target": int(n_target),
        "n_features": int(n_features),
        "shift": float(shift),
        "positive_rate_source": float(ys.mean()),
        "positive_rate_target": float(yt.mean()),
        "random_state": int(random_state),
    }
    return CovariateShiftData(Xs, ys, Xt, yt, mu, meta)


class DomainClassifierRatio:
    """Estimate w(x) = p_t(x) / p_s(x) with a probabilistic domain classifier.

    Bayes: p_t(x)/p_s(x) = [P(t|x) / P(s|x)] * [n_s / n_t]. A (optionally
    quadratic) logistic regression separates source from target; weights are
    clipped at ``clip_quantile`` of the source-weight distribution because
    a few huge weights otherwise dominate weighted calibration and the
    weighted conformal quantile.

    Only *unlabelled* target features are used -- target labels never enter.
    """

    def __init__(
        self,
        degree: int = 1,
        C: float = 1.0,
        clip_quantile: Optional[float] = 0.99,
        random_state: int = 0,
    ) -> None:
        self.degree = degree
        self.C = C
        self.clip_quantile = clip_quantile
        self.random_state = random_state
        self.model_: Optional[Pipeline] = None
        self.prior_ratio_: Optional[float] = None
        self.clip_value_: Optional[float] = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "DomainClassifierRatio":
        Xs = np.asarray(X_source, dtype=float)
        Xt = np.asarray(X_target, dtype=float)
        X = np.vstack([Xs, Xt])
        d = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
        steps = []
        if self.degree > 1:
            steps.append(("poly", PolynomialFeatures(self.degree, include_bias=False)))
        steps += [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, max_iter=5000, random_state=self.random_state)),
        ]
        self.model_ = Pipeline(steps).fit(X, d)
        self.prior_ratio_ = len(Xs) / len(Xt)
        self.clip_value_ = None
        if self.clip_quantile is not None:
            raw = self._raw(Xs)
            self.clip_value_ = float(np.quantile(raw, self.clip_quantile))
        return self

    def _raw(self, X: np.ndarray) -> np.ndarray:
        pt = self.model_.predict_proba(np.asarray(X, dtype=float))[:, 1]
        pt = np.clip(pt, 1e-6, 1 - 1e-6)
        return pt / (1.0 - pt) * self.prior_ratio_

    def weights(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("DomainClassifierRatio is not fitted")
        w = self._raw(X)
        if self.clip_value_ is not None:
            w = np.minimum(w, self.clip_value_)
        return w


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish ESS = (sum w)^2 / sum w^2; equals n for uniform weights."""
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0 or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-empty, non-negative, sum > 0")
    return float(w.sum() ** 2 / np.sum(w**2))
