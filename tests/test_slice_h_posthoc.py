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
    # The ODIR fit stops at L-BFGS-B's default tolerance, before full convergence, and where it stops
    # depends on the platform: the CI's Python 3.11 runner lands nearer the converged optimum
    # (digits logistic 0.1444 vs the recorded 0.1423; a fit run to ftol 1e-15 gives 0.1447), digits naive
    # Bayes by +0.028. So the recompute uses the well-conditioned logistic cell with a tolerance that
    # covers the measured stopping gap (tests/test_dirichlet_convergence.py), not 1e-3.
    w = H.posthoc_wider_lambda(seeds=H.DEV_SEEDS, lams=(100.0,))
    got = w[(w.dataset == "digits") & (w.model == "logistic") & (w["map"] == "dirichlet lam=100")].nll.mean()
    t = pd.read_csv(RES / "slice_h_posthoc_lambda.csv").set_index(["dataset", "model", "map"])
    assert abs(got - t.loc[("digits", "logistic", "dirichlet lam=100"), "nll"]) < 5e-3
