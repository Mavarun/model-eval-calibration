"""Slice I protocol: Dirichlet scaling with a wider CV lambda grid (frozen 2026-11-20).

Slice H's Dirichlet arm failed H3 (it beat temperature on classwise ECE in 3 of 6 cells; digits
logistic lost 1.5 points of accuracy) and its CV chose the top of the grid (lambda 1) in 38 of 60
fits, so the grid may have stopped it short. Slice H's post-hoc probe on dev seeds 100-102 found
lambda 100 helped digits but still trailed temperature on logistic. This slice re-asks H3 with
``DirichletScaling.WIDE_LAMBDAS`` (1e-4 .. 1e3), nothing else changed. Committed before
evaluation seeds 30-39 were run; same datasets, models and splits as slices G and H.

Expectation written before the run (from the dev probe): H5 fails again, on the logistic cells.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from model_eval_calibration.multiclass import (
    DirichletScaling, MulticlassTemperature, classwise_ce_debiased, classwise_ece, multiclass_nll,
)
from model_eval_calibration.multiclass_eval import DATASETS, MODELS, _cell_stats, _renorm, load, make_model, split

MAPS = ("temperature", "dirichlet_wide")
EVAL_SEEDS = tuple(range(30, 40))

PREREGISTERED = {
    "H5_wide_dirichlet_over_temperature": (
        "On fresh seeds 30-39, Dirichlet scaling with the wide grid replaces temperature as the recommended "
        "default iff, in >= 4 of the 6 (dataset, model) cells, it lowers test classwise ECE vs temperature on "
        ">= 7 of 10 seeds, AND in every cell its mean test accuracy is no more than 0.005 below temperature's and "
        "its mean NLL no more than 0.01 above (slice H's H3 bar, unchanged). Otherwise it stays OFF."
    ),
    "reported_not_decided": "Debiased classwise CE, the chosen lambdas and how often the new top (1e3) is chosen.",
}


def run_cell(dataset: str, model: str, seed: int) -> list[dict]:
    x, y = load(dataset, seed)
    x_tr, y_tr, x_cal, y_cal, x_te, y_te = split(x, y, seed)
    m = make_model(model, seed).fit(x_tr, y_tr)
    p_cal, p_te = _renorm(m.predict_proba(x_cal)), _renorm(m.predict_proba(x_te))
    out = []
    for name in MAPS:
        if name == "temperature":
            t = MulticlassTemperature().fit(p_cal, y_cal)
            q, extra = t.transform(p_te), {"T": t.T_}
        else:
            d = DirichletScaling(seed=seed, lambdas=DirichletScaling.WIDE_LAMBDAS).fit(p_cal, y_cal)
            q, extra = d.transform(p_te), {"lam": d.lam_}
        out.append({"seed": seed, "dataset": dataset, "model": model, "map": name, "n_test": len(y_te),
                    "accuracy": float((q.argmax(axis=1) == y_te).mean()), "classwise_ece": classwise_ece(y_te, q),
                    "classwise_ce_debiased": classwise_ce_debiased(y_te, q), "nll": multiclass_nll(y_te, q), **extra})
    return out


def run_grid(seeds: Iterable[int] = EVAL_SEEDS, datasets=DATASETS, models=MODELS) -> pd.DataFrame:
    return pd.DataFrame([r for d in datasets for m in models for s in seeds for r in run_cell(d, m, s)])


def decide_h5(long: pd.DataFrame) -> dict:
    v = _cell_stats(long, "temperature", "dirichlet_wide")
    h5 = bool((v.ece_wins >= 7).sum() >= 4 and (v.acc_diff >= -0.005).all() and (v.nll_diff <= 0.01).all())
    return {"H5": h5, "recommended": "dirichlet_wide" if h5 else "temperature",
            "cells_ece_wins_ge7": int((v.ece_wins >= 7).sum()), "dirichlet_wide_vs_temperature": v.to_dict("records")}


def posthoc_edge(grid: pd.DataFrame) -> dict:
    """Post-hoc, decides nothing: how often CV chose the new top lambda (1e3), and the Dirichlet - temperature
    NLL gap on digits split by whether the top was chosen (is the loss an edge effect or not?)."""
    top = max(DirichletScaling.WIDE_LAMBDAS)
    d = grid[grid["map"] == "dirichlet_wide"].set_index(["dataset", "model", "seed"])
    t = grid[grid["map"] == "temperature"].set_index(["dataset", "model", "seed"])
    gap = (d.nll - t.nll).rename("nll_gap").to_frame().join(d.lam)
    dig = gap.loc["digits"]
    return {"n_fits": int(len(d)), "n_at_top": int((d.lam == top).sum()),
            "per_cell_at_top": {f"{k[0]}/{k[1]}": int(v) for k, v in
                                d.groupby(["dataset", "model"]).lam.apply(lambda s: (s == top).sum()).items()},
            "digits_nll_gap_at_top": float(dig[dig.lam == top].nll_gap.mean()),
            "digits_nll_gap_below_top": float(dig[dig.lam < top].nll_gap.mean()),
            "digits_n_at_top": int((dig.lam == top).sum())}
