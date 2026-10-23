import numpy as np
import pytest

from model_eval_calibration.label_shift import sample_label_shift, true_posterior
from model_eval_calibration.prior_estimation import adjust_posterior, bbse_prior, em_prior


def test_adjust_posterior_matches_the_bayes_posterior_and_is_identity_at_equal_priors():
    X = np.random.default_rng(0).standard_normal((200, 6))
    ps = true_posterior(X, 0.3)
    assert np.allclose(adjust_posterior(ps, 0.3, 0.1), true_posterior(X, 0.1))
    assert np.allclose(adjust_posterior(ps, 0.3, 0.3), ps)


def test_em_recovers_the_target_prior_from_calibrated_posteriors():
    rng = np.random.default_rng(1)
    for pi_t in (0.05, 0.15, 0.5, 0.8):
        Xt, yt = sample_label_shift(20000, pi_t, rng)
        res = em_prior(true_posterior(Xt, 0.3), 0.3)
        assert res.converged and abs(res.prior - pi_t) < 0.015, (pi_t, res)


def test_em_is_biased_by_miscalibrated_posteriors_most_by_underconfidence():
    rng = np.random.default_rng(2)
    Xt, _ = sample_label_shift(20000, 0.1, rng)
    p = true_posterior(Xt, 0.3)
    z = np.log(p / (1 - p))
    good = em_prior(p, 0.3).prior
    under = em_prior(1 / (1 + np.exp(-0.4 * z)), 0.3).prior  # same ranking, logits shrunk
    biased = em_prior(1 / (1 + np.exp(-(z + 0.8))), 0.3).prior  # logit offset
    over = em_prior(1 / (1 + np.exp(-2.5 * z)), 0.3).prior  # logits stretched
    assert abs(good - 0.1) < 0.015
    assert under > 0.5 and biased > 0.3  # EM converges far from 0.1
    assert abs(over - 0.1) < 0.02  # overconfidence alone barely moves it here


def test_bbse_is_consistent_even_for_a_miscalibrated_hard_classifier():
    rng = np.random.default_rng(3)
    Xc, yc = sample_label_shift(20000, 0.3, rng)
    Xt, yt = sample_label_shift(20000, 0.1, rng)
    # any fixed rule works: threshold the overconfident score at 0.7
    rule = lambda X: (true_posterior(X, 0.3) > 0.7).astype(int)  # noqa: E731
    assert abs(bbse_prior(rule(Xc), yc, rule(Xt)) - 0.1) < 0.02


def test_input_checks():
    with pytest.raises(ValueError):
        em_prior(np.array([]), 0.3)
    with pytest.raises(ValueError):
        em_prior(np.array([0.2]), 1.0)
    with pytest.raises(ValueError):
        bbse_prior(np.zeros(10), np.r_[np.zeros(5), np.ones(5)], np.zeros(4))  # never predicts 1
    with pytest.raises(ValueError):
        bbse_prior(np.zeros(3), np.zeros(4), np.zeros(2))
