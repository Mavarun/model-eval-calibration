import numpy as np
import pytest

from model_eval_calibration.metrics import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)


def test_brier_perfect_predictions_zero():
    y = np.array([0, 1, 0, 1], dtype=float)
    p = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    assert brier_score(y, p) == pytest.approx(0.0)


def test_brier_constant_half():
    y = np.array([0, 1, 0, 1], dtype=float)
    p = np.full(4, 0.5)
    assert brier_score(y, p) == pytest.approx(0.25)


def test_ece_perfectly_calibrated_is_near_zero():
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0.0, 1.0, size=3000)
    y_true = (rng.uniform(0.0, 1.0, size=3000) < y_prob).astype(float)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.05


def test_ece_constant_wrong_confidence_is_high():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    y_prob = np.full(8, 0.9)
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece > 0.3


def test_ece_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0, 1]), np.array([0.1]))


def test_reliability_curve_counts_sum_to_n():
    rng = np.random.default_rng(1)
    y_prob = rng.uniform(0, 1, size=200)
    y_true = (y_prob > 0.5).astype(float)
    curve = reliability_curve(y_true, y_prob, n_bins=10)
    assert int(curve.bin_counts.sum()) == 200
    assert curve.n_bins == 10


def test_reliability_empty_bins_are_nan():
    y_true = np.array([0, 0, 1, 1], dtype=float)
    y_prob = np.array([0.1, 0.15, 0.85, 0.9], dtype=float)
    curve = reliability_curve(y_true, y_prob, n_bins=10)
    assert np.isnan(curve.bin_accuracy).any()
    assert (curve.bin_counts == 0).any()
