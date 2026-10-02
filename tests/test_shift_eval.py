"""Tests for the covariate-shift calibration / conformal harness."""

import numpy as np
import pandas as pd
import pytest

from model_eval_calibration.shift_eval import (
    run_shift_experiment,
    run_shift_grid,
    summarise_grid,
    win_rate_vs,
)

SMALL = dict(n_train=600, n_cal=400, n_source_test=400, n_target_pool=400,
             n_target_test=600, n_boot=30)


def test_single_run_structure_and_barriers():
    r = run_shift_experiment(shift=1.0, seed=0, **SMALL)
    r.splits.assert_disjoint()
    assert len(r.splits.cal) == 400 and len(r.splits.target_test) == 600
    methods = set(r.calibration["method"])
    assert {"raw", "temperature", "platt", "beta", "isotonic"} <= methods
    assert {"iw_temperature", "iw_platt", "iw_beta", "iw_isotonic"} <= methods
    assert "iw_raw" not in methods
    assert set(r.calibration["domain"]) == {"source", "target"}
    assert set(r.conformal["method"]) == {"split", "weighted_estimated", "weighted_oracle"}
    c = r.calibration
    assert (c["ece_adaptive_lo"] <= c["ece_adaptive_hi"]).all()
    # Brier is a smooth mean, so its percentile CI brackets the estimate;
    # binned ECE is biased upward under resampling, so its percentile CI
    # need not (documented in shift_eval and README).
    assert (c["brier_lo"] <= c["brier"]).all() and (c["brier"] <= c["brier_hi"]).all()
    assert (c["mce"] >= 0).all() and (c["mce"] <= 1).all()


def test_reproducible_for_fixed_seed():
    a = run_shift_experiment(shift=1.0, seed=3, **SMALL)
    b = run_shift_experiment(shift=1.0, seed=3, **SMALL)
    pd.testing.assert_frame_equal(a.calibration, b.calibration)
    pd.testing.assert_frame_equal(a.conformal, b.conformal)


def test_overlap_split_does_not_raise_and_overlap_is_detected():
    r = run_shift_experiment(shift=0.5, seed=1, **SMALL)
    r.splits.cal = np.r_[r.splits.cal, r.splits.train[:1]]
    with pytest.raises(AssertionError):
        r.splits.assert_disjoint()


def test_source_conformal_coverage_near_nominal_across_seeds():
    covs = []
    for seed in range(8):
        r = run_shift_experiment(shift=1.0, seed=seed, alpha=0.1, **SMALL)
        row = r.conformal.query("domain == 'source' and method == 'split'").iloc[0]
        covs.append(row["coverage"])
    assert np.mean(covs) == pytest.approx(0.90, abs=0.02)


def test_zero_shift_importance_weights_near_uniform():
    r = run_shift_experiment(shift=0.0, seed=2, **SMALL)
    # with no shift the estimated weights should keep most of the sample
    assert r.diagnostics["ess_cal_estimated"] > 0.9 * r.diagnostics["n_cal"]


def test_grid_summary_and_win_rate():
    cal, conf = run_shift_grid(shifts=(0.0, 1.5), seeds=range(2),
                               base_models=("logistic",), **SMALL)
    summ = summarise_grid(cal, ["ece_adaptive", "brier"])
    assert {"ece_adaptive_mean", "ece_adaptive_std", "brier_mean"} <= set(summ.columns)
    assert len(summ) == 2 * 2 * 9  # shifts x domains x methods
    wr = win_rate_vs(cal, "ece_adaptive", "iw_platt", "platt")
    assert wr.iloc[:, -1].between(0, 1).all()
    assert set(conf["method"]) == {"split", "weighted_estimated", "weighted_oracle"}
