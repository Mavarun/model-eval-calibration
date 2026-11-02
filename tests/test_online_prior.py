import numpy as np
import pytest

from model_eval_calibration.label_shift import true_posterior
from model_eval_calibration.drift_eval import sample_row_stream
from model_eval_calibration.online_prior import mixture_loglik, online_em_path, window_em_path
from model_eval_calibration.prior_estimation import em_prior

PI_S = 0.3


def _stream(priors, seed):
    X, y = sample_row_stream(np.asarray(priors, dtype=float), np.random.default_rng(seed))
    return true_posterior(X, PI_S), y  # calibrated source posteriors of target rows


def test_mixture_loglik_is_zero_at_source_prior_and_maximised_near_the_truth():
    p, _ = _stream(np.full(6000, 0.1), 0)
    assert np.allclose(mixture_loglik(p, PI_S, PI_S), 0)
    grid = np.linspace(0.02, 0.6, 59)
    ll = [mixture_loglik(p, PI_S, g).sum() for g in grid]
    best = grid[int(np.argmax(ll))]
    assert abs(best - 0.1) < 0.02 and abs(best - em_prior(p, PI_S).prior) < 0.011  # EM is the argmax


@pytest.mark.parametrize("tracker", [lambda p: online_em_path(p, PI_S, 0.01), lambda p: window_em_path(p, PI_S, 300, 5)])
def test_trackers_never_read_the_current_or_future_rows(tracker):
    p, _ = _stream(np.full(800, 0.2), 1)
    base = tracker(p)
    k = 400
    q = p.copy()
    q[k:] = 1.0 - q[k:]  # rewrite row k and everything after it
    alt = tracker(q)
    assert np.array_equal(base[: k + 1], alt[: k + 1])  # prior used at row k depends on rows < k only
    assert not np.array_equal(base[k + 1 :], alt[k + 1 :])


def test_online_em_tracks_a_constant_prior_and_reacts_to_a_step_faster_with_a_bigger_step():
    p, _ = _stream(np.r_[np.full(3000, 0.3), np.full(3000, 0.05)], 2)
    slow, fast = online_em_path(p, PI_S, 0.002), online_em_path(p, PI_S, 0.02)
    assert abs(slow[2000:3000].mean() - 0.3) < 0.03
    assert abs(fast[5000:].mean() - 0.05) < 0.02
    # EM steps shrink when the classes overlap (missing information), so a 0.002 step needs far
    # more than its nominal 500-row memory to reach a low prior: still 0.06-0.12 after 2500 rows
    assert fast[5500:].mean() < slow[5500:].mean() < 0.12

    def rows_to_half(path):
        return int(np.argmax(path[3000:] < 0.175))

    assert rows_to_half(fast) < 150 and rows_to_half(slow) > 400
    assert fast[2000:3000].std() > slow[2000:3000].std()  # the price of a short memory is noise


def test_window_em_matches_batch_em_on_its_window():
    p, _ = _stream(np.full(1200, 0.15), 3)
    w = window_em_path(p, PI_S, 500)
    assert abs(w[1000] - em_prior(p[500:1000], PI_S).prior) < 1e-4
    assert w[0] == PI_S
    with pytest.raises(ValueError):
        online_em_path(p, PI_S, 0.0)
    with pytest.raises(ValueError):
        window_em_path(p, PI_S, 0)


def test_cusum_detects_jumps_quickly_and_restarts_em_on_the_new_segment():
    from model_eval_calibration.online_prior import cusum_em_path

    p, _ = _stream(np.r_[np.full(3000, 0.3), np.full(3000, 0.05), np.full(3000, 0.5)], 2)
    used, alarms = cusum_em_path(p, PI_S, threshold=6.0)
    for cp in (3000, 6000):
        hit = alarms[(alarms >= cp) & (alarms < cp + 300)]
        assert hit.size >= 1, cp  # each jump is flagged within 300 rows
    assert abs(used[3400:6000].mean() - 0.05) < 0.02 and abs(used[6400:].mean() - 0.5) < 0.04
    online = online_em_path(p, PI_S, 0.005)
    assert np.abs(used[3000:3600] - 0.05).mean() < np.abs(online[3000:3600] - 0.05).mean()


def test_cusum_false_alarms_fall_with_the_threshold_and_it_never_looks_ahead():
    from model_eval_calibration.online_prior import cusum_em_path

    counts = {h: [] for h in (4.0, 10.0)}
    for s in range(3):
        p, _ = _stream(np.full(6000, 0.3), 10 + s)
        for h in counts:
            counts[h].append(len(cusum_em_path(p, PI_S, threshold=h)[1]))
    assert sum(counts[10.0]) <= 1 and sum(counts[4.0]) > sum(counts[10.0])
    p, _ = _stream(np.r_[np.full(500, 0.3), np.full(500, 0.05)], 4)
    base, _ = cusum_em_path(p, PI_S)
    q = p.copy()
    q[700:] = 0.99
    alt, _ = cusum_em_path(q, PI_S)
    assert np.array_equal(base[:701], alt[:701]) and not np.array_equal(base[701:], alt[701:])
    with pytest.raises(ValueError):
        cusum_em_path(p, PI_S, threshold=0)
