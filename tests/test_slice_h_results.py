"""Slice H committed results: the decisions recompute from the grids, and a one-seed recompute matches."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_eval_calibration import multiclass_h as H

RES = Path(__file__).resolve().parents[1] / "results"
GRID = pd.read_csv(RES / "slice_h_grid.csv")
GDEB = pd.read_csv(RES / "slice_g_debiased.csv")
DEC = json.loads((RES / "slice_h_decisions.json").read_text())


def test_grids_cover_the_registered_seeds():
    assert sorted(GRID.seed.unique()) == list(H.EVAL_SEEDS) and len(GRID) == 6 * 10 * 3
    assert sorted(GDEB.seed.unique()) == list(range(10)) and set(GDEB["map"]) == {"none", "temperature"}


def test_decisions_recompute():
    h3, h4 = H.decide_h3(GRID), H.decide_h4(GDEB)
    assert h3["H3"] is False and h3["recommended"] == "temperature" and DEC["H3"]["H3"] is False
    assert h4["H1_debiased"] is True and DEC["H4"]["H1_debiased"] is True
    wins = {(r["dataset"], r["model"]): r["ece_wins"] for r in h3["dirichlet_vs_temperature"]}
    assert sum(w >= 7 for w in wins.values()) == 3
    assert wins[("digits", "naive_bayes")] == 10


def test_dirichlet_overfits_digits_logistic():
    c = {(r["dataset"], r["model"]): r for r in DEC["H3"]["dirichlet_vs_temperature"]}
    assert c[("digits", "logistic")]["nll_diff"] > 0.3 and c[("digits", "logistic")]["acc_diff"] < -0.01


def test_one_seed_recompute():
    rows = pd.DataFrame(H.run_cell("synth5", "random_forest", 20))
    ref = GRID[(GRID.dataset == "synth5") & (GRID.model == "random_forest") & (GRID.seed == 20)]
    np.testing.assert_allclose(rows.sort_values("map").nll.to_numpy(), ref.sort_values("map").nll.to_numpy(), atol=1e-3)
