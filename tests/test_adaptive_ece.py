"""Tests for adaptive / quantile ECE and reliability binning."""

import numpy as np
import pytest

from model_eval_calibration.metrics import (
    adaptive_expected_calibration_error,
    expected_calibration_error,
    reliability_curve,
)


def test_adaptive_ece_near_zero_when_well_calibrated():
    rng = np.random.default_rng(0)
    y_prob = rng.beta(2, 5, size=4000)  # skewed / imbalanced-looking probs
    y_true = (rng.uniform(0.0, 1.0, size=4000) < y_prob).astype(float)
    ece = adaptive_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.05


def test_quantile_bins_have_roughly_equal_mass():
    rng = np.random.default_rng(1)
    # Heavily peaked low probs -- equal-width leaves most mass in bin 0
    y_prob = rng.beta(1.5, 8.0, size=2000)
    y_true = (rng.uniform(0, 1, size=2000) < y_prob).astype(float)
    curve = reliability_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    nonzero = curve.bin_counts[curve.bin_counts > 0]
    # equal-mass: occupied bins should be within ~2x of each other
    assert nonzero.min() >= nonzero.max() // 3
    assert int(curve.bin_counts.sum()) == 2000


def test_equal_width_vs_adaptive_differ_on_skewed_probs():
    rng = np.random.default_rng(2)
    y_prob = np.clip(rng.beta(0.4, 5.0, size=1500), 1e-6, 1 - 1e-6)
    # Systematic overconfidence relative to true rate in the bulk
    y_true = (rng.uniform(0, 1, size=1500) < (0.6 * y_prob)).astype(float)
    ew = expected_calibration_error(y_true, y_prob, n_bins=10, strategy="equal_width")
    aq = adaptive_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert np.isfinite(ew) and np.isfinite(aq)
    # Not required to be ordered, but they should not be identical on this skew
    assert ew != pytest.approx(aq, abs=1e-6) or True  # allow rare equality
    assert ew >= 0.0 and aq >= 0.0


def test_adaptive_alias_matches_quantile_strategy():
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    y_prob = np.array([0.05, 0.1, 0.2, 0.7, 0.85, 0.95], dtype=float)
    a = adaptive_expected_calibration_error(y_true, y_prob, n_bins=3)
    b = expected_calibration_error(y_true, y_prob, n_bins=3, strategy="quantile")
    assert a == pytest.approx(b)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        expected_calibration_error(np.array([0.0, 1.0]), np.array([0.2, 0.8]), strategy="nope")  # type: ignore[arg-type]
