"""Slice J: committed grid reproduces in one cell and the decisions recompute."""

import json
from pathlib import Path

import pandas as pd

from model_eval_calibration import multiclass_j as J

ROOT = Path(__file__).resolve().parents[1]
GRID = pd.read_csv(ROOT / "results" / "slice_j_grid.csv")
DEC = json.loads((ROOT / "results" / "slice_j_decisions.json").read_text())


def test_shape():
    assert len(GRID) == 2 * 3 * 10 * 3 and sorted(GRID.seed.unique()) == list(J.EVAL_SEEDS)


def test_decisions_recompute():
    d = J.decide(GRID)
    assert d["H6"]["pass"] is DEC["H6"]["pass"] is False
    assert d["H7"]["pass"] is DEC["H7"]["pass"] is False
    assert d["H6"]["cells_ece_wins_ge7"] == 1 and d["H7"]["cells_ece_wins_ge7"] == 3
    assert d["recommended"] == DEC["recommended"] == "temperature"


def test_both_fail_on_digits_nll_even_the_fixed_map():
    for h in ("H6", "H7"):
        cells = {(c["dataset"], c["model"]): c for c in DEC[h]["cells"]}
        for m in ("logistic", "random_forest"):
            assert 0.025 < cells[("digits", m)]["nll_diff"] < 0.035
        assert all(c["acc_diff"] >= -0.005 for c in cells.values())


def test_one_cell_reproduces():
    fresh = pd.DataFrame([r for s in J.EVAL_SEEDS[:2] for r in J.run_cell("synth5", "naive_bayes", s)])
    old = GRID[(GRID.dataset == "synth5") & (GRID.model == "naive_bayes") & GRID.seed.isin(J.EVAL_SEEDS[:2])]
    cols = ["classwise_ece", "nll", "accuracy"]
    pd.testing.assert_frame_equal(fresh[cols].reset_index(drop=True), old[cols].reset_index(drop=True), atol=1e-5)
