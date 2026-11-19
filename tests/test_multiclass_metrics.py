import numpy as np
import pytest

from model_eval_calibration.metrics import adaptive_expected_calibration_error, expected_calibration_error
from model_eval_calibration.multiclass import classwise_ece, multiclass_brier, multiclass_nll, top_label_ece


def _calibrated(n=20000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k) * 0.7, size=n)
    y = np.array([rng.choice(k, p=r) for r in p])
    return y, p


def test_perfectly_calibrated_sample_has_small_eces():
    y, p = _calibrated()
    assert top_label_ece(y, p) < 0.02
    assert classwise_ece(y, p) < 0.01


def test_overconfident_probs_raise_both_eces():
    y, p = _calibrated(seed=1)
    sharp = p ** 3 / np.sum(p ** 3, axis=1, keepdims=True)
    assert top_label_ece(y, sharp) > top_label_ece(y, p) + 0.05
    assert classwise_ece(y, sharp) > classwise_ece(y, p) + 0.01


def test_binary_case_matches_binary_metrics():
    rng = np.random.default_rng(2)
    q = rng.uniform(size=3000)
    y = (rng.uniform(size=3000) < q ** 1.5).astype(int)
    p = np.column_stack([1 - q, q])
    # classwise ECE with K = 2 averages two mirror-image binary ECEs that are equal for equal-width bins.
    assert classwise_ece(y, p, n_bins=10, adaptive=False) == pytest.approx(
        expected_calibration_error(y, q, n_bins=10), abs=1e-9)
    assert classwise_ece(y, p, n_bins=10, adaptive=True) == pytest.approx(
        0.5 * (adaptive_expected_calibration_error(y, q, n_bins=10)
               + adaptive_expected_calibration_error(1 - y, 1 - q, n_bins=10)), abs=0.01)


def test_brier_and_nll_known_values():
    y = np.array([0, 1])
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert multiclass_brier(y, p) == pytest.approx((0 + 0.5) / 2)
    assert multiclass_nll(y, p) == pytest.approx(np.log(2) / 2)


@pytest.mark.parametrize("y,p", [
    (np.array([0]), np.array([[0.6, 0.6]])),   # row does not sum to 1
    (np.array([0]), np.array([[1.0]])),        # K < 2
    (np.array([2]), np.array([[0.5, 0.5]])),   # label out of range
])
def test_validation(y, p):
    with pytest.raises(ValueError):
        top_label_ece(y, p)
