"""Slice I results: one cell reproduces, and the decision recomputes from the committed grid."""

import json
from pathlib import Path

import pandas as pd
import pytest

from model_eval_calibration import multiclass_i as I

ROOT = Path(__file__).resolve().parents[1]
GRID = pd.read_csv(ROOT / "results" / "slice_i_grid.csv")
DEC = json.loads((ROOT / "results" / "slice_i_decisions.json").read_text())


def test_grid_shape():
    assert len(GRID) == 6 * 10 * 2 and sorted(GRID.seed.unique()) == list(I.EVAL_SEEDS)


def test_decision_recomputes():
    d = I.decide_h5(GRID)
    assert d["H5"] is DEC["H5"]["H5"] is False and d["cells_ece_wins_ge7"] == 3
    assert d["recommended"] == "temperature"


def test_one_cell_reproduces():
    fresh = pd.DataFrame(I.run_cell("synth5", "naive_bayes", 33))
    old = GRID[(GRID.dataset == "synth5") & (GRID.model == "naive_bayes") & (GRID.seed == 33)].reset_index(drop=True)
    for col in ("accuracy", "classwise_ece", "nll", "lam"):
        assert fresh[col].to_numpy() == pytest.approx(old[col].to_numpy(), abs=1e-5, nan_ok=True)
