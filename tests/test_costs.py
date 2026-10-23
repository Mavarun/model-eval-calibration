import numpy as np
import pytest

from model_eval_calibration.costs import (
    bayes_threshold,
    decisions,
    expected_cost,
    shifted_threshold,
    tune_threshold,
)
from model_eval_calibration.label_shift import sample_label_shift, true_posterior
from model_eval_calibration.prior_estimation import adjust_posterior


def test_expected_cost_by_hand():
    y = np.array([0, 0, 1, 1])
    assert expected_cost(y, np.array([1, 0, 0, 1]), 1.0, 5.0) == pytest.approx((1 + 5) / 4)
    assert bayes_threshold(1, 5) == pytest.approx(1 / 6)


def test_bayes_threshold_is_cost_optimal_on_the_oracle_posterior():
    rng = np.random.default_rng(0)
    X, y = sample_label_shift(40000, 0.2, rng)
    p = true_posterior(X, 0.2)
    t = bayes_threshold(1, 5)
    best = expected_cost(y, decisions(p, t), 1, 5)
    for other in (0.05, 0.1, 0.25, 0.4, 0.5):
        assert best <= expected_cost(y, decisions(p, other), 1, 5) + 1e-3


def test_tuned_threshold_is_the_empirical_minimiser_and_near_bayes_on_calibrated_scores():
    rng = np.random.default_rng(1)
    X, y = sample_label_shift(30000, 0.3, rng)
    p = true_posterior(X, 0.3)
    t = tune_threshold(p, y, 1, 5)
    grid = np.linspace(0.01, 0.99, 99)
    assert expected_cost(y, decisions(p, t), 1, 5) <= min(expected_cost(y, decisions(p, g), 1, 5) for g in grid) + 1e-12
    assert abs(t - 1 / 6) < 0.04


def test_shifted_threshold_equals_bayes_after_posterior_adjustment():
    t = shifted_threshold(1, 5, 0.3, 0.1)
    assert t == pytest.approx(0.4355, abs=1e-3)
    # thresholding source scores at t == thresholding target-adjusted scores at the Bayes threshold
    p = np.linspace(0.001, 0.999, 2001)
    a = decisions(p, t)
    b = decisions(adjust_posterior(p, 0.3, 0.1), bayes_threshold(1, 5))
    assert np.mean(a != b) <= 1 / 2001
    assert shifted_threshold(1, 5, 0.3, 0.3) == pytest.approx(bayes_threshold(1, 5))


def test_cost_checks():
    with pytest.raises(ValueError):
        bayes_threshold(0, 1)
    with pytest.raises(ValueError):
        expected_cost(np.array([]), np.array([]), 1, 1)
    with pytest.raises(ValueError):
        tune_threshold(np.array([0.1, 0.2]), np.array([1]), 1, 1)
