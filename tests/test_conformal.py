"""Coverage tests for split and weighted conformal prediction sets.

Test design: marginal coverage is a statement about the average over
calibration draws, so the coverage tests repeat the whole split (new
calibration + test sample per seed) and check the *mean* coverage against
the finite-sample band [1 - alpha, 1 - alpha + 1/(n+1)] with a tolerance
sized for Monte-Carlo error.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from model_eval_calibration.conformal import (
    class_conditional_coverage,
    nonconformity_scores,
    prediction_sets,
    set_diagnostics,
    split_conformal_threshold,
    weighted_conformal_thresholds,
)
from model_eval_calibration.shift import make_covariate_shift_data


def test_threshold_matches_hand_computed_rank():
    scores = np.linspace(0.1, 1.0, 10)
    # k = ceil(11 * 0.8) = 9 -> 9th smallest = 0.9
    assert split_conformal_threshold(scores, alpha=0.2) == pytest.approx(0.9)


def test_threshold_infinite_when_calibration_too_small():
    assert split_conformal_threshold(np.array([0.2, 0.4, 0.6]), alpha=0.1) == np.inf
    sets = prediction_sets(np.array([0.3, 0.8]), np.inf)
    assert sets.all()


def test_scores_definition():
    s = nonconformity_scores(np.array([0.9, 0.9, 0.2]), np.array([1, 0, 0]))
    np.testing.assert_allclose(s, [0.1, 0.9, 0.2])


def _draw(rng, n):
    """Imperfect but fixed model: reported p is a distorted version of truth."""
    q = rng.beta(2, 2, size=n)
    y = (rng.uniform(size=n) < q).astype(int)
    p = np.clip(q**1.5, 0, 1)
    return p, y


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_split_conformal_marginal_coverage_exchangeable(alpha):
    rng = np.random.default_rng(0)
    n_cal, n_test, reps = 199, 400, 300
    covs = []
    for _ in range(reps):
        p_c, y_c = _draw(rng, n_cal)
        p_t, y_t = _draw(rng, n_test)
        q = split_conformal_threshold(nonconformity_scores(p_c, y_c), alpha)
        covs.append(set_diagnostics(prediction_sets(p_t, q), y_t).coverage)
    mean_cov = float(np.mean(covs))
    assert mean_cov >= 1 - alpha - 0.006
    assert mean_cov <= 1 - alpha + 1 / (n_cal + 1) + 0.006


def test_weighted_with_uniform_weights_equals_split():
    rng = np.random.default_rng(1)
    p, y = _draw(rng, 250)
    s = nonconformity_scores(p, y)
    for alpha in (0.05, 0.1, 0.3):
        q = split_conformal_threshold(s, alpha)
        qw = weighted_conformal_thresholds(s, np.ones(s.size), np.ones(7), alpha)
        np.testing.assert_allclose(qw, q)


def test_weighted_heavy_test_weight_gives_full_set():
    s = np.linspace(0, 1, 50)
    qw = weighted_conformal_thresholds(s, np.ones(50), np.array([1.0, 1e6]), alpha=0.1)
    assert np.isfinite(qw[0]) and np.isinf(qw[1])


def test_weighted_conformal_restores_coverage_under_covariate_shift():
    alpha = 0.1
    cov_unw, cov_w = [], []
    for seed in range(20):
        d = make_covariate_shift_data(n_source=2000, n_target=2000, shift=1.5, random_state=seed)
        Xtr, ytr = d.X_source[:1000], d.y_source[:1000]
        Xc, yc = d.X_source[1000:], d.y_source[1000:]
        model = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        s = nonconformity_scores(model.predict_proba(Xc)[:, 1], yc)
        p_t = model.predict_proba(d.X_target)[:, 1]
        q = split_conformal_threshold(s, alpha)
        cov_unw.append(set_diagnostics(prediction_sets(p_t, q), d.y_target).coverage)
        qw = weighted_conformal_thresholds(
            s, d.true_density_ratio(Xc), d.true_density_ratio(d.X_target), alpha
        )
        cov_w.append(set_diagnostics(prediction_sets(p_t, qw), d.y_target).coverage)
    gap_unw = abs(np.mean(cov_unw) - (1 - alpha))
    gap_w = abs(np.mean(cov_w) - (1 - alpha))
    assert gap_w < 0.015
    assert gap_w < gap_unw


def test_set_diagnostics_and_class_coverage():
    sets = np.array([[True, False], [True, True], [False, False], [False, True]])
    y = np.array([0, 1, 1, 0])
    d = set_diagnostics(sets, y)
    assert d.coverage == pytest.approx(0.5)
    assert d.mean_set_size == pytest.approx(1.0)
    assert d.empty_rate == pytest.approx(0.25) and d.full_rate == pytest.approx(0.25)
    cc = class_conditional_coverage(sets, y)
    assert cc[0] == pytest.approx(0.5) and cc[1] == pytest.approx(0.5)


def test_input_validation():
    with pytest.raises(ValueError):
        split_conformal_threshold(np.array([0.1]), alpha=0.0)
    with pytest.raises(ValueError):
        nonconformity_scores(np.array([0.5]), np.array([2]))
    with pytest.raises(ValueError):
        prediction_sets(np.array([0.5, 0.6]), np.array([0.1, 0.2, 0.3]))
