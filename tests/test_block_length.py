import numpy as np
import pytest

from model_eval_calibration.block_length import optimal_block_length, pooled_block_length


def _ar1(phi, n, rng):
    e, x = rng.standard_normal(n), np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_white_noise_gets_a_short_block():
    rng = np.random.default_rng(0)
    b = [optimal_block_length(rng.standard_normal(500)) for _ in range(30)]
    assert np.median(b) < 2.0 and min(b) >= 1.0


def test_block_grows_with_autocorrelation_and_tracks_ar1_theory():
    rng = np.random.default_rng(1)
    n = 2000
    med = {phi: np.median([optimal_block_length(_ar1(phi, n, rng)) for _ in range(20)]) for phi in (0.3, 0.6, 0.85)}
    assert med[0.3] < med[0.6] < med[0.85]
    for phi, b in med.items():  # CB asymptotics for AR(1): (6 phi^2 / (1 - phi^2)^2)^(1/3) n^(1/3)
        theory = (6 * phi**2 / (1 - phi**2) ** 2) ** (1 / 3) * n ** (1 / 3)
        assert 0.5 * theory < b < 1.3 * theory, (phi, b, theory)


def test_cap_constant_series_and_short_input():
    rng = np.random.default_rng(2)
    n = 100
    assert optimal_block_length(_ar1(0.99, n, rng)) <= np.ceil(min(3 * np.sqrt(n), n / 3))
    assert optimal_block_length(np.ones(50)) == 1.0
    with pytest.raises(ValueError):
        optimal_block_length(np.arange(5.0))


def test_pooled_rule_is_an_integer_median():
    rng = np.random.default_rng(3)
    s = [_ar1(0.7, 400, rng) for _ in range(5)]
    b = pooled_block_length(s)
    assert isinstance(b, int) and b == int(np.ceil(np.median([optimal_block_length(x) for x in s])))
    with pytest.raises(ValueError):
        pooled_block_length([])
