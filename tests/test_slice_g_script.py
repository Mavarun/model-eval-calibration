"""The slice-G script runs on dev seeds into a temp dir; the committed artifacts are consistent."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

from model_eval_calibration.multiclass_eval import decide, run_cell

ROOT = Path(__file__).resolve().parents[1]


def test_script_runs_on_one_dev_seed(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("run_mc", ROOT / "scripts" / "run_multiclass_slice.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = mod.main(seeds=(100,), out=tmp_path)
    out = capsys.readouterr().out
    assert "Pre-registered decisions" in out and (tmp_path / "slice_g_grid.csv").exists()
    assert d["H1"] is False  # 7 of 10 seeds cannot hold on one seed


def test_committed_grid_shape_and_one_cell_recomputes():
    g = pd.read_csv(ROOT / "results" / "slice_g_grid.csv")
    assert sorted(g.seed.unique()) == list(range(10)) and len(g) == 10 * 2 * 3 * 3
    ref = g[(g.seed == 7) & (g.dataset == "digits") & (g.model == "naive_bayes")].set_index("map")
    for r in run_cell("digits", "naive_bayes", 7):
        assert abs(ref.loc[r["map"], "classwise_ece"] - r["classwise_ece"]) < 1e-6
        assert abs(ref.loc[r["map"], "nll"] - r["nll"]) < 1e-5


def test_committed_decisions_recompute():
    g = pd.read_csv(ROOT / "results" / "slice_g_grid.csv")
    d = decide(g)
    rec = json.loads((ROOT / "results" / "slice_g_decisions.json").read_text())
    assert (d["H1"], d["H2"], d["recommended"]) == (rec["H1"], rec["H2"], rec["recommended"]) == (True, False, "temperature")
