"""DirichletScaling(select="holdout"): lambda chosen on one stratified 30% holdout (slice J)."""

import numpy as np
import pytest

from model_eval_calibration.multiclass import DirichletScaling


def _data(seed=0, n=600, k=4):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k)) + 2.0 * np.eye(k)[y]
    p = np.exp(1.7 * logits)
    return p / p.sum(axis=1, keepdims=True), y


def test_cv_default_is_unchanged_by_the_new_option():
    p, y = _data(2)
    a, b = DirichletScaling(seed=3).fit(p, y), DirichletScaling(seed=3, select="cv").fit(p, y)
    assert a.lam_ == b.lam_ and np.allclose(a.W_, b.W_)


def test_holdout_choice_is_on_the_grid_seeded_and_refit_on_all_rows():
    p, y = _data(4)
    g = DirichletScaling.WIDE_LAMBDAS
    a = DirichletScaling(seed=5, lambdas=g, select="holdout").fit(p, y)
    b = DirichletScaling(seed=5, lambdas=g, select="holdout").fit(p, y)
    assert a.lam_ in g and a.lam_ == b.lam_ and np.allclose(a.W_, b.W_)
    full = DirichletScaling(lam=a.lam_).fit(p, y)
    assert np.allclose(a.W_, full.W_) and np.allclose(a.b_, full.b_)


def test_fixed_lambda_ignores_the_selector_and_bad_selector_raises():
    p, y = _data(6)
    a = DirichletScaling(lam=100.0, select="holdout").fit(p, y)
    b = DirichletScaling(lam=100.0).fit(p, y)
    assert a.lam_ == 100.0 and np.allclose(a.W_, b.W_)
    with pytest.raises(ValueError):
        DirichletScaling(select="loo")
