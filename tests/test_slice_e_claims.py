"""Directional slice E claims quoted in the README, re-checked at reduced scale (3 seeds, logistic)."""

import pandas as pd
import pytest

from model_eval_calibration.stream_eval import run_stream, summarise_stream

CHOSEN = {"online_step": 0.02, "window": 200, "cusum_threshold": 4.0}  # dev-seed selection in the full run


@pytest.fixture(scope="module")
def summ():
    long = pd.concat([run_stream(seed=s, path_kind=k, trackers=CHOSEN) for k in ("smooth", "regime") for s in range(3)],
                     ignore_index=True)
    return summarise_stream(long).set_index(["path", "policy"])


def test_batch_em_degrades_as_batches_shrink(summ):
    r = summ["regret_mean"]
    for k in ("smooth", "regime"):
        assert r[(k, "batch25_em_current")] > r[(k, "batch400_em_current")] + 0.02


def test_per_row_online_em_is_close_to_waiting_for_a_400_row_batch(summ):
    r = summ["regret_mean"]
    for k in ("smooth", "regime"):
        assert abs(r[(k, "online_em")] - r[(k, "batch400_em_current")]) < 0.01
        assert r[(k, "online_em")] < r[(k, "batch25_em_current")]


def test_cusum_helps_after_jumps_but_costs_on_smooth_drift(summ):
    assert summ.loc[("regime", "cusum_em"), "post_change_regret_mean"] < summ.loc[("regime", "window_em"), "post_change_regret_mean"]
    assert summ.loc[("smooth", "cusum_em"), "regret_mean"] > summ.loc[("smooth", "online_em"), "regret_mean"]


def test_label_free_trackers_beat_late_labels_after_jumps(summ):
    late = summ.loc[("regime", "delayed_label_threshold"), "regret_mean"]
    for pol in ("online_em", "window_em", "cusum_em"):
        assert summ.loc[("regime", pol), "regret_mean"] < late - 0.03
