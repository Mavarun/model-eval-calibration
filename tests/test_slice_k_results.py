"""Slice K's committed results: the reading recomputes from the grid, and the NB holdout swing is real."""

import json
from pathlib import Path

import pandas as pd

from model_eval_calibration import multiclass_j as J
from model_eval_calibration import multiclass_k as K

ROOT = Path(__file__).resolve().parents[1]
GK = pd.read_csv(ROOT / "results" / "slice_k_grid.csv")
GJ = pd.read_csv(ROOT / "results" / "slice_j_grid.csv")
DK = json.loads((ROOT / "results" / "slice_k_decisions.json").read_text())
DJ = json.loads((ROOT / "results" / "slice_j_decisions.json").read_text())


def test_reading_recomputes():
    conv = J.decide(GK)
    assert K.read(conv, DJ) == DK["read"]
    assert DK["read"] == {"K1": True, "H6_pass": False, "H7_pass": False, "H6_cells": 1, "H7_cells": 3,
                          "recommended": "temperature"}
    assert DK["preregistered"] == K.PREREGISTERED


def test_temperature_rows_identical_to_slice_j():
    tk = GK[GK["map"] == "temperature"].reset_index(drop=True)
    tj = GJ[GJ["map"] == "temperature"].reset_index(drop=True)
    pd.testing.assert_frame_equal(tk.drop(columns="lam"), tj.drop(columns="lam"), check_exact=False, atol=1e-6)


def test_naive_bayes_holdout_flips_sign_on_nll():
    c = pd.DataFrame(DK["vs_slice_j"]).set_index(["hyp", "dataset", "model"])
    r = c.loc[("H7", "digits", "naive_bayes")]
    assert r.nll_diff_j < -0.04 and r.nll_diff_k > 0.01
    sel = lambda g: g[(g.dataset == "digits") & (g.model == "naive_bayes") & (g["map"] == "dirichlet_holdout")].set_index("seed")
    j, k = sel(GJ), sel(GK)
    smaller = [s for s in j.index if k.loc[s, "lam"] < j.loc[s, "lam"]]
    assert smaller == [43, 45]
    assert all(k.loc[s, "nll"] - j.loc[s, "nll"] > 0.2 for s in smaller)
