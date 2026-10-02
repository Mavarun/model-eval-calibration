"""Tests for MCE and bootstrap confidence intervals."""

import numpy as np
import pytest

from model_eval_calibration.bootstrap import (
    bootstrap_metric_ci,
    paired_bootstrap_difference,
    reliability_bootstrap_bands,
)
from model_eval_calibration.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)


def _calibrated(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < p).astype(float)
    return y, p


def test_mce_hand_computed_two_bins():
    # bin [0,0.5): conf 0.2, acc 0.0 -> gap 0.2 ; bin [0.5,1]: conf 0.9, acc 0.5 -> 0.4
    y = np.array([0, 0, 1, 0], dtype=float)
    p = np.array([0.2, 0.2, 0.9, 0.9])
    assert maximum_calibration_error(y, p, n_bins=2) == pytest.approx(0.4)
    # ECE = 0.5*0.2 + 0.5*0.4 = 0.3, and MCE >= ECE always
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.3)


def test_mce_min_bin_count_ignores_sparse_bins():
    # low bin: 4 rows, conf 0.1, acc 0.0 (gap 0.1); high bin: 1 row, gap 0.6
    y = np.zeros(5, dtype=float)
    p = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
    assert maximum_calibration_error(y, p, n_bins=2) == pytest.approx(0.6)
    assert maximum_calibration_error(y, p, n_bins=2, min_bin_count=2) == pytest.approx(0.1)


def test_mce_upper_bounds_ece_on_random_data():
    y, p = _calibrated(2000, 3)
    p = np.clip(p * 1.2 - 0.1, 0, 1)  # distort
    for strategy in ("equal_width", "quantile"):
        assert maximum_calibration_error(y, p, strategy=strategy) >= expected_calibration_error(
            y, p, strategy=strategy
        )


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    y, p = _calibrated(500, 0)
    a = bootstrap_metric_ci(y, p, brier_score, n_boot=300, random_state=1)
    b = bootstrap_metric_ci(y, p, brier_score, n_boot=300, random_state=1)
    assert a == b
    assert a.lower <= a.estimate <= a.upper
    assert a.upper - a.lower > 0


def test_bootstrap_ci_shrinks_with_sample_size():
    y_s, p_s = _calibrated(200, 5)
    y_l, p_l = _calibrated(5000, 5)
    small = bootstrap_metric_ci(y_s, p_s, brier_score, n_boot=300, random_state=0)
    large = bootstrap_metric_ci(y_l, p_l, brier_score, n_boot=300, random_state=0)
    assert (large.upper - large.lower) < 0.5 * (small.upper - small.lower)


def test_paired_difference_detects_clearly_worse_method():
    y, p = _calibrated(1500, 11)
    bad = np.clip(p + 0.25, 0, 1)  # systematically overconfident on positives
    d = paired_bootstrap_difference(
        y, p, bad, expected_calibration_error, n_boot=300, random_state=0
    )
    assert d.estimate < 0
    assert d.excludes_zero
    assert d.prob_a_better > 0.99


def test_paired_difference_identical_methods_is_zero():
    y, p = _calibrated(300, 2)
    d = paired_bootstrap_difference(y, p, p, brier_score, n_boot=100)
    assert d.estimate == 0.0 and d.lower == 0.0 and d.upper == 0.0
    assert not d.excludes_zero


def test_reliability_bands_cover_diagonal_when_calibrated():
    y, p = _calibrated(4000, 8)
    bands = reliability_bootstrap_bands(y, p, n_bins=10, n_boot=300, random_state=0)
    ok = bands.bin_counts > 0
    inside = (bands.acc_lower[ok] <= bands.bin_confidence[ok]) & (
        bands.bin_confidence[ok] <= bands.acc_upper[ok]
    )
    # 95% bands: allow at most 2 of 10 bins to miss by chance
    assert inside.sum() >= ok.sum() - 2


def test_reliability_bands_flag_overconfidence():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.6, 1.0, size=3000)
    y = (rng.uniform(size=3000) < 0.5).astype(float)  # truth is coin flip
    bands = reliability_bootstrap_bands(y, p, n_bins=10, n_boot=200, random_state=0)
    ok = bands.bin_counts > 0
    assert np.all(bands.acc_upper[ok] < bands.bin_confidence[ok])


def test_bootstrap_rejects_bad_args():
    y, p = _calibrated(50, 0)
    with pytest.raises(ValueError):
        bootstrap_metric_ci(y, p, brier_score, n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_metric_ci(y, p, brier_score, alpha=1.5)
