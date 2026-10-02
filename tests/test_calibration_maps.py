"""Tests for post-hoc calibration maps (temperature / Platt / beta / isotonic)."""

import numpy as np
import pytest

from model_eval_calibration.calibration_maps import (
    BetaCalibration,
    IdentityMap,
    IsotonicMap,
    PlattScaling,
    TemperatureScaling,
    make_calibration_maps,
)
from model_eval_calibration.metrics import brier_score, expected_calibration_error


def _overconfident(n=6000, true_t=2.5, seed=0):
    """True probs q; model reports sigmoid(logit(q) * true_t) (overconfident)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.2, size=n)
    q = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n) < q).astype(float)
    p = 1.0 / (1.0 + np.exp(-z * true_t))
    return p, y


def test_temperature_recovers_known_overconfidence():
    p, y = _overconfident(true_t=2.5, seed=1)
    ts = TemperatureScaling().fit(p, y)
    assert ts.temperature_ == pytest.approx(2.5, rel=0.12)


def test_temperature_preserves_ranking_and_decision_boundary():
    p, y = _overconfident(seed=2)
    out = TemperatureScaling().fit_transform(p, y)
    assert np.array_equal(np.argsort(p, kind="stable"), np.argsort(out, kind="stable"))
    assert np.array_equal(p >= 0.5, out >= 0.5)


def test_beta_is_near_identity_on_calibrated_probs():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.98, size=8000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    bc = BetaCalibration().fit(p, y)
    a, b, c = bc.coef_
    assert a == pytest.approx(1.0, abs=0.15)
    assert b == pytest.approx(1.0, abs=0.15)
    assert c == pytest.approx(0.0, abs=0.15)


def test_beta_respects_sign_constraints_and_monotone():
    rng = np.random.default_rng(4)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < 0.3).astype(float)  # labels independent of p
    bc = BetaCalibration().fit(p, y)
    assert bc.coef_[0] >= 0 and bc.coef_[1] >= 0
    grid = np.linspace(0.01, 0.99, 50)
    assert np.all(np.diff(bc.transform(grid)) >= -1e-12)


def test_isotonic_is_monotone_and_bounded():
    p, y = _overconfident(n=1000, seed=5)
    iso = IsotonicMap().fit(p, y)
    grid = np.linspace(0, 1, 101)
    out = iso.transform(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() > 0 and out.max() < 1


@pytest.mark.parametrize("name", ["temperature", "platt", "beta", "isotonic"])
def test_every_map_reduces_ece_on_overconfident_holdout(name):
    p_cal, y_cal = _overconfident(n=4000, seed=10)
    p_te, y_te = _overconfident(n=4000, seed=11)
    m = make_calibration_maps()[name].fit(p_cal, y_cal)
    out = m.transform(p_te)
    assert expected_calibration_error(y_te, out) < 0.5 * expected_calibration_error(y_te, p_te)
    assert brier_score(y_te, out) < brier_score(y_te, p_te)


def test_sample_weight_changes_fit_toward_weighted_region():
    # region A (p<0.5) is calibrated, region B (p>=0.5) is overconfident.
    rng = np.random.default_rng(6)
    p = rng.uniform(size=6000)
    true = np.where(p < 0.5, p, 0.5 + 0.4 * (p - 0.5))
    y = (rng.uniform(size=p.size) < true).astype(float)
    w_b = np.where(p >= 0.5, 10.0, 0.1)
    unweighted = PlattScaling().fit(p, y)
    weighted = PlattScaling().fit(p, y, sample_weight=w_b)
    hi = p >= 0.5
    err_u = abs(unweighted.transform(p[hi]).mean() - y[hi].mean())
    err_w = abs(weighted.transform(p[hi]).mean() - y[hi].mean())
    assert err_w < err_u


def test_uniform_weights_equal_unweighted():
    p, y = _overconfident(n=800, seed=7)
    a = BetaCalibration().fit(p, y).coef_
    b = BetaCalibration().fit(p, y, sample_weight=np.full(p.size, 3.0)).coef_
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_identity_and_validation():
    p = np.array([0.2, 0.7])
    y = np.array([0.0, 1.0])
    assert np.array_equal(IdentityMap().fit(p, y).transform(p), p)
    with pytest.raises(RuntimeError):
        TemperatureScaling().transform(p)
    with pytest.raises(ValueError):
        PlattScaling().fit(p, np.array([0.0, 2.0]))
    with pytest.raises(ValueError):
        PlattScaling().fit(p, y, sample_weight=np.array([-1.0, 1.0]))
