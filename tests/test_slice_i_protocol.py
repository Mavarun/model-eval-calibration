"""Slice I protocol: fresh seeds, unchanged bar, the rule on a toy grid, a dev-seed smoke run."""

import pandas as pd

from model_eval_calibration import multiclass_eval as G, multiclass_h as H, multiclass_i as I
from model_eval_calibration.multiclass import DirichletScaling


def test_seeds_are_fresh_and_grid_is_the_wide_one():
    assert I.EVAL_SEEDS == tuple(range(30, 40))
    assert not set(I.EVAL_SEEDS) & (set(G.EVAL_SEEDS) | set(H.EVAL_SEEDS) | set(H.DEV_SEEDS))
    assert DirichletScaling.WIDE_LAMBDAS[-1] == 1000.0 and set(DirichletScaling.LAMBDAS) < set(DirichletScaling.WIDE_LAMBDAS)


def _toy(wins_cells, acc_drop=0.0):
    rows = []
    for c, (d, m) in enumerate([(d, m) for d in G.DATASETS for m in G.MODELS]):
        for s in range(10):
            better = c < wins_cells
            rows.append({"seed": s, "dataset": d, "model": m, "map": "temperature", "classwise_ece": 0.05,
                         "nll": 0.3, "accuracy": 0.9})
            rows.append({"seed": s, "dataset": d, "model": m, "map": "dirichlet_wide",
                         "classwise_ece": 0.04 if better else 0.06, "nll": 0.3, "accuracy": 0.9 - acc_drop})
    return pd.DataFrame(rows)


def test_rule():
    assert I.decide_h5(_toy(4))["H5"]
    assert not I.decide_h5(_toy(3))["H5"]
    assert not I.decide_h5(_toy(6, acc_drop=0.01))["H5"]


def test_dev_smoke():
    rows = I.run_cell("synth5", "naive_bayes", H.DEV_SEEDS[0])
    assert [r["map"] for r in rows] == list(I.MAPS)
    assert rows[1]["lam"] in DirichletScaling.WIDE_LAMBDAS
