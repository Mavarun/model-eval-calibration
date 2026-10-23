"""Directional slice D claims quoted in the README, re-checked at reduced scale (3 seeds)."""

import numpy as np
import pytest

from model_eval_calibration.drift_eval import paired_policy_difference, run_walk_forward
from model_eval_calibration.label_shift_eval import run_label_shift_grid


@pytest.fixture(scope="module")
def grid():
    return run_label_shift_grid(target_priors=(0.05, 0.5), seeds=range(3), base_models=("logistic",))


def test_prior_correction_recovers_most_of_the_cost_gap(grid):
    dec, _ = grid
    b = dec[(dec.policy == "bayes") & (dec.calibration == "raw") & (dec.target_prior == 0.05)]
    none = b[b.estimator == "none"]["regret"].mean()
    for est in ("em", "bbse"):
        assert b[b.estimator == est]["regret"].mean() < 0.25 * none, est
    tuned = dec[(dec.policy == "tuned_source") & (dec.target_prior == 0.05)]["regret"].mean()
    assert tuned > 0.5 * none  # a source-tuned threshold does not fix label shift


def test_tuned_threshold_cost_is_map_invariant(grid):
    dec, _ = grid
    t = dec[dec.policy == "tuned_source"].groupby(["target_prior", "seed"])["cost"].nunique()
    assert (t <= 2).all()  # ties at isotonic plateaus may differ by one cut at most


def test_mondrian_keeps_per_class_coverage_where_split_does_not(grid):
    _, conf = grid
    m = conf.groupby(["target_prior", "method"])[["coverage", "coverage_pos", "coverage_neg"]].mean()
    for p in (0.05, 0.5):
        assert m.loc[(p, "mondrian"), "coverage_pos"] > 0.85 and m.loc[(p, "mondrian"), "coverage_neg"] > 0.87
        assert m.loc[(p, "split"), "coverage_pos"] < 0.8
    assert m.loc[(0.5, "split"), "coverage"] < 0.88


def test_label_free_current_batch_em_beats_delayed_label_threshold():
    import pandas as pd

    wf = pd.concat([run_walk_forward(seed=s) for s in range(3)], ignore_index=True)
    r = paired_policy_difference(wf, "em_current", "delayed_label_threshold", n_boot=500)
    assert r["estimate"] < 0 and r["prob_a_better"] > 0.9  # README: CI below 0 in all 6 cells at 10 seeds
    assert np.isfinite(r["lower"])
