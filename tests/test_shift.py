"""Tests for the covariate-shift generator and density-ratio estimator."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from model_eval_calibration.shift import (
    DomainClassifierRatio,
    effective_sample_size,
    make_covariate_shift_data,
    true_probability,
)


def test_generator_shapes_and_reproducible():
    a = make_covariate_shift_data(n_source=200, n_target=150, shift=1.0, random_state=3)
    b = make_covariate_shift_data(n_source=200, n_target=150, shift=1.0, random_state=3)
    assert a.X_source.shape == (200, 6) and a.X_target.shape == (150, 6)
    np.testing.assert_array_equal(a.X_target, b.X_target)
    np.testing.assert_array_equal(a.y_source, b.y_source)


def test_marginal_shifts_but_labelling_mechanism_is_fixed():
    data = make_covariate_shift_data(n_source=20000, n_target=20000, shift=1.5, random_state=0)
    mean_gap = data.X_target.mean(axis=0) - data.X_source.mean(axis=0)
    np.testing.assert_allclose(mean_gap, data.mu, atol=0.05)
    # P(y|x) identical: within a narrow shared x-region empirical rates match
    def region(X):
        return (np.abs(X[:, 0] - 0.7) < 0.15) & (np.abs(X[:, 1] - 0.3) < 0.3)

    rs, rt = region(data.X_source), region(data.X_target)
    assert rs.sum() > 200 and rt.sum() > 200
    gap = abs(data.y_source[rs].mean() - data.y_target[rt].mean())
    assert gap < 0.08
    # and label rates follow the known true probability on target
    assert abs(data.y_target.mean() - true_probability(data.X_target).mean()) < 0.01


def test_zero_shift_true_ratio_is_one():
    data = make_covariate_shift_data(n_source=50, n_target=50, shift=0.0)
    np.testing.assert_allclose(data.true_density_ratio(data.X_source), 1.0)


def test_domain_classifier_ratio_tracks_analytic_ratio():
    data = make_covariate_shift_data(n_source=4000, n_target=4000, shift=1.0, random_state=1)
    est = DomainClassifierRatio(clip_quantile=None).fit(data.X_source, data.X_target)
    w_hat = est.weights(data.X_source)
    w_true = data.true_density_ratio(data.X_source)
    rho = spearmanr(w_hat, w_true).statistic
    assert rho > 0.95
    # importance-weighted source mean of x0 should approximate the target mean
    iw_mean = np.average(data.X_source[:, 0], weights=w_hat)
    assert iw_mean == pytest.approx(data.mu[0], abs=0.12)


def test_clipping_caps_weights():
    data = make_covariate_shift_data(n_source=2000, n_target=2000, shift=2.0, random_state=2)
    est = DomainClassifierRatio(clip_quantile=0.95).fit(data.X_source, data.X_target)
    w = est.weights(data.X_source)
    assert w.max() <= est.clip_value_ + 1e-12
    assert np.mean(w >= est.clip_value_ - 1e-12) == pytest.approx(0.05, abs=0.01)


def test_effective_sample_size():
    assert effective_sample_size(np.ones(100)) == pytest.approx(100.0)
    w = np.r_[np.ones(99), 1000.0]
    assert effective_sample_size(w) < 3.0
    with pytest.raises(ValueError):
        effective_sample_size(np.array([-1.0, 2.0]))


def test_ess_falls_as_shift_grows():
    ess = []
    for s in (0.25, 1.0, 2.0):
        data = make_covariate_shift_data(n_source=2000, n_target=2000, shift=s, random_state=0)
        w = data.true_density_ratio(data.X_source)
        ess.append(effective_sample_size(w))
    assert ess[0] > ess[1] > ess[2]
