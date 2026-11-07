import numpy as np
import pytest

from model_eval_calibration.stream_eval import (
    SMALL_REGIME_LEVELS,
    detection_stats,
    make_target_stream,
    run_stream,
    select_cusum,
    stream_decisions,
    fit_source,
)


def test_small_regime_path_only_takes_small_jumps():
    priors, cps, X, y = make_target_stream(3, "small_regime", 9600)
    assert set(np.round(np.unique(priors), 2)) <= set(SMALL_REGIME_LEVELS)
    assert cps.size >= 2
    jumps = np.abs(priors[cps] - priors[cps - 1])
    assert (jumps > 0.04).all() and (jumps <= 0.2 + 1e-9).all()
    assert np.array_equal(np.flatnonzero(np.diff(priors)) + 1, cps)


def test_cusum_delta_reaches_the_tracker():
    model, cmap, p_cal, y_cal, pi_s = fit_source(0, "logistic")
    priors, _, X, _ = make_target_stream(0, "small_regime", 3000)
    p = cmap.transform(model.predict_proba(X)[:, 1])
    lab = np.zeros(p.size, dtype=int)
    _, a = stream_decisions(p, lab, pi_s, priors, p_cal, y_cal, trackers={"cusum_delta": 0.8})
    _, b = stream_decisions(p, lab, pi_s, priors, p_cal, y_cal, trackers={"cusum_delta": 0.3})
    assert not np.array_equal(a["cusum_em"], b["cusum_em"])
    assert np.array_equal(a["online_em"], b["online_em"])


def test_detection_stats_counts_true_and_false_alarms():
    d = detection_stats(np.array([50, 1010, 1300, 5000]), np.array([1000, 3000]), 10_000, horizon=400)
    assert d["detected"] == 0.5 and d["median_delay"] == 10
    assert d["false_alarms_per_1000"] == pytest.approx(0.2)  # 50 and 5000 are false; 1300 is late but within 400
    assert np.isnan(detection_stats(np.array([5]), np.array([], dtype=int), 100)["detected"])


def test_select_cusum_refuses_overlap_and_returns_the_dev_argmin():
    with pytest.raises(ValueError):
        select_cusum("logistic", dev_seeds=(1, 2), test_seeds=(2,))
    grid = {"cusum_threshold": (4.0,), "cusum_delta": (0.3, 0.8)}
    chosen, dev = select_cusum("logistic", dev_seeds=(1000,), paths=("small_regime",), grid=grid, n_rows=2000)
    score = dev.groupby("cusum_delta")["cost"].mean()
    assert chosen["cusum_delta"] == float(score.idxmin()) and chosen["cusum_threshold"] == 4.0
    assert {"detected", "median_delay", "false_alarms_per_1000"} <= set(dev.columns)


def test_run_stream_accepts_the_small_regime_path():
    out = run_stream(seed=0, path_kind="small_regime", n_rows=2000)
    assert out["post_change_share"].gt(0).any() and out.policy.nunique() == 10
