import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.special import softmax

from model_eval_calibration.multiclass import (
    MulticlassTemperature, VectorScaling, classwise_ece, multiclass_nll, top_label_ece,
)


def _overconfident(n=6000, k=3, t_true=2.5, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.5, size=(n, k))
    true_p = softmax(logits, axis=1)
    y = np.array([rng.choice(k, p=r) for r in true_p])
    return y, softmax(logits * t_true, axis=1)  # reported probs are too sharp by T = t_true


def test_temperature_recovers_the_sharpening_factor_and_keeps_argmax():
    y, p = _overconfident()
    m = MulticlassTemperature().fit(p[:3000], y[:3000])
    assert m.T_ == pytest.approx(2.5, rel=0.1)
    q = m.transform(p[3000:])
    np.testing.assert_array_equal(q.argmax(axis=1), p[3000:].argmax(axis=1))
    assert top_label_ece(y[3000:], q) < 0.5 * top_label_ece(y[3000:], p[3000:])
    assert multiclass_nll(y[3000:], q) < multiclass_nll(y[3000:], p[3000:])


def test_vector_scaling_fixes_a_class_bias_temperature_cannot():
    rng = np.random.default_rng(3)
    n, k = 8000, 3
    logits = rng.normal(0, 1.0, size=(n, k))
    y = np.array([rng.choice(k, p=r) for r in softmax(logits, axis=1)])
    biased = softmax(logits + np.array([1.0, 0.0, -1.0]), axis=1)  # class 0 over-, class 2 under-predicted
    tr, te = slice(0, 4000), slice(4000, None)
    t = MulticlassTemperature().fit(biased[tr], y[tr]).transform(biased[te])
    v = VectorScaling().fit(biased[tr], y[tr]).transform(biased[te])
    assert classwise_ece(y[te], v) < 0.5 * classwise_ece(y[te], t)


def test_vector_scaling_gradient_matches_finite_differences():
    y, p = _overconfident(n=300, seed=4)
    vs = VectorScaling(reg=0.01)
    z = np.log(p)
    onehot = np.eye(3)[y]

    def loss(theta):
        from scipy.special import log_softmax
        ls = log_softmax(z * theta[:3] + theta[3:], axis=1)
        return -np.sum(ls * onehot) / len(y) + 0.01 * (np.sum((theta[:3] - 1) ** 2) + np.sum(theta[3:] ** 2))

    vs.fit(p, y)
    theta = np.concatenate([vs.w_, vs.b_])
    assert np.abs(approx_fprime(theta, loss, 1e-6)).max() < 1e-3  # at the optimum


def test_identity_when_already_calibrated():
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(4), size=8000)
    y = np.array([rng.choice(4, p=r) for r in p])
    assert MulticlassTemperature().fit(p, y).T_ == pytest.approx(1.0, abs=0.08)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        MulticlassTemperature().transform(np.array([[0.5, 0.5]]))
    with pytest.raises(RuntimeError):
        VectorScaling().transform(np.array([[0.5, 0.5]]))
