"""Slice I post-hoc edge reading recomputes from the grid."""

import json
from pathlib import Path

import pandas as pd

from model_eval_calibration.multiclass_i import posthoc_edge

ROOT = Path(__file__).resolve().parents[1] / "results"


def test_posthoc_matches_and_reads_as_stated():
    p = posthoc_edge(pd.read_csv(ROOT / "slice_i_grid.csv"))
    assert p == json.loads((ROOT / "slice_i_posthoc_edge.json").read_text())
    assert p["n_at_top"] == 24 and p["digits_n_at_top"] == 16
    # on digits the NLL loss is concentrated in fits where CV chose a smaller (more flexible) lambda
    assert p["digits_nll_gap_below_top"] > 5 * p["digits_nll_gap_at_top"] > 0
