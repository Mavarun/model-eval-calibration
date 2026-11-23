import pandas as pd

from model_eval_calibration import multiclass_j as J


def test_frozen_design():
    assert J.EVAL_SEEDS == tuple(range(40, 50)) and J.FIXED_LAMBDA == 100.0
    assert not set(J.EVAL_SEEDS) & set(J.DEV_SEEDS) and not set(J.EVAL_SEEDS) & set(range(0, 40))
    assert J.MAPS == ("temperature", "dirichlet_fixed", "dirichlet_holdout")


def _long(fixed_ece, hold_ece, fixed_nll=0.0):
    rows = []
    for d in ("a", "b"):
        for m in ("x", "y", "z"):
            for s in range(10):
                for mp, e, n in (("temperature", 0.02, 0.5), ("dirichlet_fixed", fixed_ece, 0.5 + fixed_nll),
                                 ("dirichlet_holdout", hold_ece, 0.5)):
                    rows.append({"dataset": d, "model": m, "seed": s, "map": mp, "classwise_ece": e, "nll": n,
                                 "accuracy": 0.9})
    return pd.DataFrame(rows)


def test_rules_on_hand_made_rows():
    d = J.decide(_long(0.01, 0.01))
    assert d["H6"]["pass"] and d["H7"]["pass"] and d["recommended"] == "dirichlet_fixed"
    d = J.decide(_long(0.01, 0.03, fixed_nll=0.02))
    assert not d["H6"]["pass"] and not d["H7"]["pass"] and d["recommended"] == "temperature"
    d = J.decide(_long(0.03, 0.01))
    assert d["recommended"] == "dirichlet_holdout"


def test_one_cell_runs():
    rows = J.run_cell("synth5", "naive_bayes", 100)
    assert [r["map"] for r in rows] == list(J.MAPS) and rows[1]["lam"] == 100.0
