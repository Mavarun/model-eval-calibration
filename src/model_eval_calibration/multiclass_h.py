"""Slice H protocol: Dirichlet (matrix) scaling, and slice G's H1 under a second estimator.

Committed before evaluation seeds 20-29 were run. Development seeds 100-102 only check that the
harness runs. Same datasets, models and splits as slice G (:mod:`multiclass_eval`).

Maps: ``none``, ``temperature`` (the shipped default), ``dirichlet``
(:class:`~model_eval_calibration.multiclass.DirichletScaling`, CV-chosen ODIR lambda).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from model_eval_calibration.multiclass import (
    DirichletScaling, MulticlassTemperature, classwise_ce_debiased, classwise_ece, multiclass_brier, multiclass_nll,
    top_label_ece,
)
from model_eval_calibration.multiclass_eval import DATASETS, MODELS, _cell_stats, _renorm, load, make_model, split

MAPS = ("none", "temperature", "dirichlet")
EVAL_SEEDS = tuple(range(20, 30))
DEV_SEEDS = (100, 101, 102)

PREREGISTERED = {
    "H3_dirichlet_over_temperature": (
        "On fresh seeds 20-29, Dirichlet scaling replaces temperature as the recommended default iff, in >= 4 "
        "of the 6 (dataset, model) cells, it lowers test classwise ECE vs temperature on >= 7 of 10 seeds, AND "
        "in every cell its mean test accuracy is no more than 0.005 below temperature's and its mean NLL no "
        "more than 0.01 above. Otherwise OFF (same bar as slice G's H2)."
    ),
    "H4_h1_under_debiased_estimator": (
        "Slice G's H1 (temperature vs none) is re-scored on the same seeds 0-9 with classwise_ce_debiased in "
        "place of binned classwise ECE. These seeds are already seen, so this cannot change the default: if H1 "
        "fails under the debiased estimator, the README states that the slice G call depends on the estimator."
    ),
    "reported_not_decided": "Debiased CE for H3, top-label ECE, Brier and the chosen ODIR lambdas are context only.",
}


def run_cell(dataset: str, model: str, seed: int, maps: Iterable[str] = MAPS) -> list[dict]:
    x, y = load(dataset, seed)
    x_tr, y_tr, x_cal, y_cal, x_te, y_te = split(x, y, seed)
    m = make_model(model, seed).fit(x_tr, y_tr)
    p_cal, p_te = _renorm(m.predict_proba(x_cal)), _renorm(m.predict_proba(x_te))
    out = []
    for name in maps:
        extra = {}
        if name == "none":
            q = p_te
        elif name == "temperature":
            t = MulticlassTemperature().fit(p_cal, y_cal)
            q, extra = t.transform(p_te), {"T": t.T_}
        elif name == "dirichlet":
            d = DirichletScaling(seed=seed).fit(p_cal, y_cal)
            q, extra = d.transform(p_te), {"lam": d.lam_}
        else:
            raise ValueError(name)
        out.append({"seed": seed, "dataset": dataset, "model": model, "map": name, "n_test": len(y_te),
                    "accuracy": float((q.argmax(axis=1) == y_te).mean()), "classwise_ece": classwise_ece(y_te, q),
                    "classwise_ce_debiased": classwise_ce_debiased(y_te, q), "top_label_ece": top_label_ece(y_te, q),
                    "brier": multiclass_brier(y_te, q), "nll": multiclass_nll(y_te, q), **extra})
    return out


def run_grid(seeds: Iterable[int] = EVAL_SEEDS, maps: Iterable[str] = MAPS, datasets=DATASETS,
             models=MODELS) -> pd.DataFrame:
    return pd.DataFrame([r for d in datasets for m in models for s in seeds for r in run_cell(d, m, s, maps)])


def decide_h3(long: pd.DataFrame) -> dict:
    v = _cell_stats(long[long["map"].isin(["temperature", "dirichlet"])], "temperature", "dirichlet")
    h3 = bool((v.ece_wins >= 7).sum() >= 4 and (v.acc_diff >= -0.005).all() and (v.nll_diff <= 0.01).all())
    return {"H3": h3, "recommended": "dirichlet" if h3 else "temperature", "dirichlet_vs_temperature": v.to_dict("records")}


def decide_h4(long_g: pd.DataFrame) -> dict:
    """H1 with the debiased estimator: ``long_g`` holds none / temperature rows of seeds 0-9."""
    sub = long_g[long_g["map"].isin(["none", "temperature"])].copy()
    sub["classwise_ece"] = sub["classwise_ce_debiased"]
    t = _cell_stats(sub, "none", "temperature")
    h1 = bool((t.ece_wins >= 7).sum() >= 4 and (t.nll_diff <= 0.01).all())
    return {"H1_debiased": h1, "temperature_vs_none_debiased": t.to_dict("records")}


def posthoc_lambda_edge(grid: pd.DataFrame) -> dict:
    """Post-hoc, decides nothing: how often CV picked the largest lambda on the grid (an edge solution)."""
    d = grid[grid["map"] == "dirichlet"]
    top = max(DirichletScaling.LAMBDAS)
    per = d.groupby(["dataset", "model"]).lam.apply(lambda s: int((s == top).sum())).to_dict()
    return {"n_fits": int(len(d)), "n_at_top": int((d.lam == top).sum()),
            "per_cell_at_top": {f"{k[0]}/{k[1]}": v for k, v in per.items()}}


def posthoc_wider_lambda(seeds: Iterable[int] = DEV_SEEDS, lams=(1.0, 10.0, 100.0)) -> pd.DataFrame:
    """Post-hoc probe on the *dev* seeds only (never seeds 20-29): fixed larger lambdas vs temperature.

    Decides nothing; it says whether a future pre-registration should widen the grid.
    """
    rows = []
    for d_ in DATASETS:
        for m_ in MODELS:
            for s in seeds:
                x, y = load(d_, s)
                x_tr, y_tr, x_cal, y_cal, x_te, y_te = split(x, y, s)
                mod = make_model(m_, s).fit(x_tr, y_tr)
                p_cal, p_te = _renorm(mod.predict_proba(x_cal)), _renorm(mod.predict_proba(x_te))
                t = MulticlassTemperature().fit(p_cal, y_cal).transform(p_te)
                base = {"dataset": d_, "model": m_, "seed": s, "k": p_te.shape[1], "n_cal": len(y_cal)}
                rows.append({**base, "map": "temperature", "nll": multiclass_nll(y_te, t),
                             "classwise_ece": classwise_ece(y_te, t)})
                for lam in lams:
                    q = DirichletScaling(lam=lam).fit(p_cal, y_cal).transform(p_te)
                    rows.append({**base, "map": f"dirichlet lam={lam:g}", "nll": multiclass_nll(y_te, q),
                                 "classwise_ece": classwise_ece(y_te, q)})
    return pd.DataFrame(rows)
