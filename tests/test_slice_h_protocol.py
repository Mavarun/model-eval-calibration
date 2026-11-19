"""Slice H protocol: fresh seeds, the H3 / H4 rules on toy tables, and a dev-seed smoke run."""

import pandas as pd

from model_eval_calibration import multiclass_eval as G
from model_eval_calibration import multiclass_h as H


def _long(rows):
    return pd.DataFrame(rows)


def _rows(cells, a, b, a_ece, b_ece, n=10, acc=(0.9, 0.9), nll=(0.5, 0.5), key="classwise_ece"):
    out = []
    for d, m in cells:
        for s in range(n):
            out.append({"dataset": d, "model": m, "seed": s, "map": a, key: a_ece, "accuracy": acc[0], "nll": nll[0]})
            out.append({"dataset": d, "model": m, "seed": s, "map": b, key: b_ece, "accuracy": acc[1], "nll": nll[1]})
    return out


CELLS = [(d, m) for d in G.DATASETS for m in G.MODELS]


def test_seeds_are_fresh():
    assert not set(H.EVAL_SEEDS) & set(G.EVAL_SEEDS) and H.DEV_SEEDS == G.DEV_SEEDS


def test_h3_rule():
    assert H.decide_h3(_long(_rows(CELLS, "temperature", "dirichlet", 0.03, 0.02)))["H3"]
    assert not H.decide_h3(_long(_rows(CELLS[:3], "temperature", "dirichlet", 0.03, 0.02)
                                 + _rows(CELLS[3:], "temperature", "dirichlet", 0.02, 0.03)))["H3"]
    # wins everywhere but costs accuracy in one cell
    rows = _rows(CELLS[:5], "temperature", "dirichlet", 0.03, 0.02) + _rows(
        CELLS[5:], "temperature", "dirichlet", 0.03, 0.02, acc=(0.9, 0.89))
    assert not H.decide_h3(_long(rows))["H3"]


def test_h4_rule_reads_the_debiased_column():
    rows = []
    for r in _rows(CELLS, "none", "temperature", 0.05, 0.01, key="classwise_ce_debiased"):
        rows.append({**r, "classwise_ece": 0.0})
    assert H.decide_h4(_long(rows))["H1_debiased"]


def test_dev_seed_smoke():
    rows = H.run_cell("digits", "logistic", H.DEV_SEEDS[0])
    assert [r["map"] for r in rows] == list(H.MAPS)
    assert rows[2]["lam"] in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
