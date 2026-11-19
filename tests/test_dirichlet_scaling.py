"""DirichletScaling: matrix calibration with ODIR regularisation (slice H)."""

import numpy as np
import pytest
from scipy.special import softmax

from model_eval_calibration.multiclass import (
    DirichletScaling, MulticlassTemperature, fit_multiclass_map, multiclass_nll,
)


def _draw(n, k, seed, mix=None):
    """Labels from true probabilities q; the model reports a distorted version of q."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2.0, (n, k))
    q = softmax(logits, axis=1)
    y = np.array([rng.choice(k, p=row) for row in q])
    m = np.eye(k) if mix is None else mix
    p = softmax(np.log(q) @ m.T, axis=1)
    return p, y, q


def test_recovers_a_class_mixing_distortion_that_temperature_cannot():
    k = 4
    mix = np.eye(k) * 1.0
    mix[0, 1], mix[1, 0] = 0.8, -0.6  # class 0 and 1 log-probs leak into each other
    p, y, _ = _draw(6000, k, 1, mix)
    pt, yt, qt = _draw(4000, k, 2, mix)
    d = DirichletScaling(lam=1e-3).fit(p, y)
    t = MulticlassTemperature().fit(p, y)
    assert multiclass_nll(yt, d.transform(pt)) < multiclass_nll(yt, t.transform(pt)) - 0.01
    assert multiclass_nll(yt, d.transform(pt)) < multiclass_nll(yt, qt) + 0.02


def test_identity_when_already_calibrated():
    p, y, _ = _draw(5000, 3, 3)
    d = DirichletScaling(lam=1.0).fit(p, y)
    assert np.allclose(d.W_, np.eye(3), atol=0.15) and np.allclose(d.b_, 0, atol=0.1)


def test_strong_odir_shrinks_off_diagonal():
    k = 4
    mix = np.eye(k)
    mix[0, 1] = 0.8
    p, y, _ = _draw(3000, k, 4, mix)
    weak, strong = DirichletScaling(lam=1e-4).fit(p, y), DirichletScaling(lam=100.0).fit(p, y)
    off = 1 - np.eye(k)
    assert np.abs(strong.W_ * off).sum() < 0.5 * np.abs(weak.W_ * off).sum()


def test_cv_lambda_is_on_the_grid_and_rows_sum_to_one():
    p, y, _ = _draw(900, 5, 5)
    d = DirichletScaling().fit(p, y)
    assert d.lam_ in DirichletScaling.LAMBDAS
    q = d.transform(p)
    assert np.allclose(q.sum(axis=1), 1.0) and (q > 0).all()


def test_dispatch_and_errors():
    p, y, _ = _draw(600, 3, 6)
    assert isinstance(fit_multiclass_map(p, y, "dirichlet"), DirichletScaling)
    with pytest.raises(ValueError):
        DirichletScaling(lam=-1)
    with pytest.raises(RuntimeError):
        DirichletScaling().transform(p)
