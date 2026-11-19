"""DirichletScaling(lambdas=...): the CV grid is a parameter (slice I)."""

import numpy as np
import pytest

from model_eval_calibration.multiclass import DirichletScaling


def _data(seed=0, n=600, k=4):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k)) + 2.0 * np.eye(k)[y]
    p = np.exp(1.7 * logits)
    return p / p.sum(axis=1, keepdims=True), y


def test_default_grid_is_unchanged():
    p, y = _data()
    a, b = DirichletScaling(seed=1).fit(p, y), DirichletScaling(seed=1, lambdas=DirichletScaling.LAMBDAS).fit(p, y)
    assert a.lam_ == b.lam_ and np.allclose(a.W_, b.W_)
    assert DirichletScaling().lambdas == (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def test_wide_grid_choice_is_on_the_grid_and_a_one_point_grid_is_forced():
    p, y = _data(1)
    d = DirichletScaling(seed=0, lambdas=DirichletScaling.WIDE_LAMBDAS).fit(p, y)
    assert d.lam_ in DirichletScaling.WIDE_LAMBDAS
    assert DirichletScaling(lambdas=(10.0,)).fit(p, y).lam_ == 10.0


def test_bad_grids_raise():
    for bad in ((), (-1.0, 1.0)):
        with pytest.raises(ValueError):
            DirichletScaling(lambdas=bad)
