import numpy as np
import pytest

from model_eval_calibration.label_shift import (
    log_likelihood_ratio,
    make_label_shift_data,
    prior_logit,
    sample_label_shift,
    true_posterior,
)


def test_priors_move_while_class_conditionals_stay_fixed():
    d = make_label_shift_data(n_source=20000, n_target=20000, source_prior=0.3, target_prior=0.1, random_state=1)
    assert abs(d.y_source.mean() - 0.3) < 0.01 and abs(d.y_target.mean() - 0.1) < 0.01
    for c in (0, 1):
        ms, mt = d.X_source[d.y_source == c].mean(0), d.X_target[d.y_target == c].mean(0)
        assert np.max(np.abs(ms - mt)) < 0.08  # P(x | y) shared


def test_posterior_logit_moves_by_the_prior_log_odds_only():
    X = np.random.default_rng(0).standard_normal((50, 6))
    ps, pt = true_posterior(X, 0.3), true_posterior(X, 0.1)
    lo = lambda p: np.log(p / (1 - p))  # noqa: E731
    assert np.allclose(lo(pt) - lo(ps), prior_logit(0.1) - prior_logit(0.3))


def test_oracle_posterior_is_calibrated_and_llr_is_not_linear():
    rng = np.random.default_rng(2)
    X, y = sample_label_shift(60000, 0.2, rng)
    p = true_posterior(X, 0.2)
    bins = np.digitize(p, np.linspace(0, 1, 11)[1:-1])
    for b in range(10):
        m = bins == b
        if m.sum() > 500:
            assert abs(y[m].mean() - p[m].mean()) < 0.03
    # quadratic term: the LLR is not affine in x0
    e = np.eye(6)[0]
    vals = [log_likelihood_ratio((t * e)[None])[0] for t in (-2.0, 0.0, 2.0)]
    assert abs((vals[0] + vals[2]) / 2 - vals[1]) > 0.5


def test_bad_prior_rejected():
    with pytest.raises(ValueError):
        prior_logit(0.0)
    with pytest.raises(ValueError):
        sample_label_shift(10, 1.0, np.random.default_rng(0))
