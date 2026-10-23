import numpy as np
import pytest

from model_eval_calibration.label_shift_eval import run_label_shift_experiment


@pytest.fixture(scope="module")
def cell():
    return run_label_shift_experiment(target_prior=0.1, seed=0, base_model="logistic")


def test_splits_are_disjoint_and_target_labels_only_score(cell):
    cell["split"].assert_disjoint()
    s = cell["split"]
    assert len(s.train) + len(s.cal) == 4500 and len(s.pool) + len(s.test) == 5000
    assert abs(cell["diagnostics"]["pi_t_test"] - 0.1) < 0.02


def test_rows_and_oracle_floor(cell):
    d = cell["decision"]
    assert set(d["policy"]) == {"bayes", "tuned_source", "oracle_posterior"}
    assert set(d.loc[d.policy == "bayes", "estimator"]) == {"none", "em", "bbse", "oracle"}
    assert d.loc[d.policy == "oracle_posterior", "regret"].item() == 0.0


def test_prior_correction_cuts_cost_versus_ignoring_the_shift(cell):
    d = cell["decision"].set_index(["calibration", "estimator", "policy"])
    for m in ("raw", "platt", "isotonic"):
        none = d.loc[(m, "none", "bayes"), "cost"]
        for est in ("em", "bbse", "oracle"):
            assert d.loc[(m, est, "bayes"), "cost"] < none - 0.05, (m, est)
        assert d.loc[(m, "none", "tuned_source"), "cost"] > d.loc[(m, "bbse", "bayes"), "cost"]
    assert d.loc[("raw", "em", "bayes"), "prior_abs_err"] < 0.03
    assert d.loc[("raw", "bbse", "bayes"), "prior_abs_err"] < 0.03


def test_same_seed_reproduces(cell):
    again = run_label_shift_experiment(target_prior=0.1, seed=0, base_model="logistic")
    assert np.allclose(again["decision"]["cost"], cell["decision"]["cost"])


def test_no_shift_cell_keeps_prior_estimates_near_source():
    r = run_label_shift_experiment(target_prior=0.3, seed=1, base_model="logistic")
    d = r["decision"]
    for est in ("em", "bbse"):
        row = d[(d.policy == "bayes") & (d.estimator == est) & (d.calibration == "raw")]
        assert row["prior_abs_err"].item() < 0.03, est
    c = r["conformal"].set_index("method")
    assert abs(c.loc["split", "coverage"] - 0.9) < 0.03
