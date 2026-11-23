"""Slice H post-hoc (decides nothing): CV lambda at the grid edge, and a wider-lambda probe on dev seeds only."""

import json
from pathlib import Path

import pandas as pd

from model_eval_calibration import multiclass_h as H

RES = Path(__file__).resolve().parents[1] / "results"


def test_edge_counts_recompute():
    e = H.posthoc_lambda_edge(pd.read_csv(RES / "slice_h_grid.csv"))
    assert e == json.loads((RES / "slice_h_posthoc_edge.json").read_text())
    assert e["n_at_top"] == 38 and e["n_fits"] == 60


def test_probe_uses_dev_seeds_only_and_larger_lambda_helps_digits():
    t = pd.read_csv(RES / "slice_h_posthoc_lambda.csv").set_index(["dataset", "model", "map"])
    assert not set(H.DEV_SEEDS) & set(H.EVAL_SEEDS)
    for m in ("logistic", "random_forest", "naive_bayes"):
        assert t.loc[("digits", m, "dirichlet lam=100"), "nll"] < t.loc[("digits", m, "dirichlet lam=1"), "nll"]
    # still worse than temperature on digits logistic even at lambda 100
    assert t.loc[("digits", "logistic", "dirichlet lam=100"), "nll"] > t.loc[("digits", "logistic", "temperature"), "nll"]


def test_probe_recomputes_one_cell():
    # digits logistic, not digits naive Bayes: GaussianNB's digits probabilities saturate (log p sits on the
    # 1e-12 floor for most classes), and that cell's mean NLL came out 0.548 instead of 0.521 on the CI's
    # Python 3.11 runner while reproducing exactly on a local 3.11 install with the same numpy / scipy /
    # scikit-learn versions; the logistic cell is well conditioned.
    w = H.posthoc_wider_lambda(seeds=H.DEV_SEEDS, lams=(100.0,))
    got = w[(w.dataset == "digits") & (w.model == "logistic") & (w["map"] == "dirichlet lam=100")].nll.mean()
    t = pd.read_csv(RES / "slice_h_posthoc_lambda.csv").set_index(["dataset", "model", "map"])
    assert abs(got - t.loc[("digits", "logistic", "dirichlet lam=100"), "nll"]) < 1e-3
