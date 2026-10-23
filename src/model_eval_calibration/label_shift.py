"""Synthetic label shift (prior shift) with an analytic Bayes posterior.

Label shift means the class prior P(y) changes between source and target
while the class-conditional densities P(x | y) are held fixed. It is the
mirror image of covariate shift (slice C): there P(y | x) was fixed and P(x)
moved; here P(y | x) itself moves, because by Bayes

    logit p_t(y=1 | x) = logit p_s(y=1 | x) + log(pi_t / (1 - pi_t)) - log(pi_s / (1 - pi_s)).

So a classifier that was perfectly calibrated on source is *miscalibrated*
on target by a known additive logit offset, and a cost-optimal threshold
chosen on source is no longer cost-optimal.

Generator (``make_label_shift_data``):

* y ~ Bernoulli(pi)
* x | y=0 ~ N(0, I_d)
* x | y=1 ~ N(m, diag(s^2)) with m = (1.4, -0.9, 0.5, 0, ...) and
  s = (1.6, 0.8, 1, 1, ...)

The unequal class covariances make the true log-likelihood ratio
quadratic in x, so a linear-logit model is misspecified (as in slice C).
``true_posterior`` evaluates the exact Bayes posterior for any prior, which
gives an oracle for prior estimation, posterior adjustment and costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def class_params(n_features: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Mean and per-coordinate std of the positive class (negatives are N(0, I))."""
    if n_features < 3:
        raise ValueError("need at least 3 features")
    m = np.zeros(n_features)
    m[:3] = (1.4, -0.9, 0.5)
    s = np.ones(n_features)
    s[:2] = (1.6, 0.8)
    return m, s


def log_likelihood_ratio(X: np.ndarray) -> np.ndarray:
    """log p(x | y=1) - log p(x | y=0) for the fixed class conditionals."""
    X = np.asarray(X, dtype=float)
    m, s = class_params(X.shape[1])
    z1 = (X - m) / s
    ll1 = -0.5 * np.sum(z1**2, axis=1) - np.sum(np.log(s))
    ll0 = -0.5 * np.sum(X**2, axis=1)
    return ll1 - ll0


def prior_logit(prior: float) -> float:
    if not 0.0 < prior < 1.0:
        raise ValueError("prior must be in (0, 1)")
    return float(np.log(prior / (1.0 - prior)))


def true_posterior(X: np.ndarray, prior: float) -> np.ndarray:
    """Exact Bayes P(y=1 | x) under class prior ``prior``."""
    z = log_likelihood_ratio(X) + prior_logit(prior)
    return 1.0 / (1.0 + np.exp(-z))


def sample_label_shift(n: int, prior: float, rng: np.random.Generator, n_features: int = 6):
    """n rows with P(y=1) = prior and the fixed class conditionals."""
    prior_logit(prior)  # validates
    y = (rng.random(n) < prior).astype(int)
    m, s = class_params(n_features)
    X = rng.standard_normal((n, n_features))
    pos = y == 1
    X[pos] = X[pos] * s + m
    return X, y


@dataclass
class LabelShiftData:
    X_source: np.ndarray
    y_source: np.ndarray
    X_target: np.ndarray
    y_target: np.ndarray
    source_prior: float
    target_prior: float
    meta: dict = field(default_factory=dict)

    def oracle_target_posterior(self, X: np.ndarray) -> np.ndarray:
        return true_posterior(X, self.target_prior)


def make_label_shift_data(
    n_source: int = 4500,
    n_target: int = 5000,
    source_prior: float = 0.3,
    target_prior: float = 0.1,
    n_features: int = 6,
    random_state: int = 0,
) -> LabelShiftData:
    """Source and target samples that differ only in the class prior."""
    rng = np.random.default_rng(random_state)
    Xs, ys = sample_label_shift(n_source, source_prior, rng, n_features)
    Xt, yt = sample_label_shift(n_target, target_prior, rng, n_features)
    return LabelShiftData(
        Xs, ys, Xt, yt, float(source_prior), float(target_prior),
        meta={"n_features": n_features, "random_state": random_state},
    )
