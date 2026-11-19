"""The debiased classwise calibration error added for slice G2."""

import numpy as np
import pytest

from model_eval_calibration.multiclass import classwise_ce_debiased, classwise_ece


def _calibrated(n, k, seed):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k), size=n)
    y = np.array([rng.choice(k, p=row) for row in p])
    return y, p


def test_calibrated_model_scores_near_zero_while_binned_ece_has_a_floor():
    # Single draws are noisy (the square root amplifies small values), so average over seeds.
    draws = [_calibrated(600, 5, s) for s in range(10)]
    ece = np.mean([classwise_ece(y, p) for y, p in draws])
    deb = np.mean([classwise_ce_debiased(y, p) for y, p in draws])
    assert ece > 0.03 and deb < 0.5 * ece


def test_overconfidence_is_detected():
    y, p = _calibrated(3000, 5, 1)
    sharp = p ** 3
    sharp /= sharp.sum(axis=1, keepdims=True)
    assert classwise_ce_debiased(y, sharp) > 3 * classwise_ce_debiased(y, p)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        classwise_ce_debiased([0, 1], np.array([[0.5, 0.6], [0.5, 0.5]]))
