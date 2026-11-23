"""Slice J protocol: Dirichlet with a fixed large lambda, or lambda chosen on a holdout (frozen 2026-11-24).

Slice I's post-hoc reading put the digits NLL loss on the 3-fold CV choice picking too flexible a
map, and named "a fixed large lambda, or selection on a held-out split" as the next thing to
pre-register. This slice tests both against temperature, on fresh seeds 40-49, with slice H's H3
bar unchanged. Same datasets, models and splits as slices G-I.

Dev probe (seeds 100-102, never used for a decision): a fixed lambda of 100 beat 1000 on mean NLL
in all six cells, so 100 is the fixed arm; both arms still trailed temperature on digits logistic
NLL by 0.03-0.04.

Expectation written before the run: H6 and H7 both fail, on the digits logistic NLL.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from model_eval_calibration.multiclass import (
    DirichletScaling, MulticlassTemperature, classwise_ce_debiased, classwise_ece, multiclass_nll,
)
from model_eval_calibration.multiclass_eval import DATASETS, MODELS, _cell_stats, _renorm, load, make_model, split

FIXED_LAMBDA = 100.0
MAPS = ("temperature", "dirichlet_fixed", "dirichlet_holdout")
EVAL_SEEDS = tuple(range(40, 50))
DEV_SEEDS = (100, 101, 102)

PREREGISTERED = {
    "H6_fixed_lambda_over_temperature": (
        "On fresh seeds 40-49, Dirichlet scaling with lambda fixed at 100 replaces temperature iff, in >= 4 of the "
        "6 (dataset, model) cells, it lowers test classwise ECE vs temperature on >= 7 of 10 seeds, AND in every "
        "cell its mean test accuracy is no more than 0.005 below temperature's and its mean NLL no more than 0.01 "
        "above."
    ),
    "H7_holdout_lambda_over_temperature": (
        "The same bar for Dirichlet with the wide grid and lambda chosen on one stratified 30% holdout of the "
        "calibration rows. If both pass, the one with more ECE-winning cells is recommended (ties: fixed, simpler)."
    ),
    "reported_not_decided": "Debiased classwise CE, the holdout-chosen lambdas.",
}


def _maps(seed: int):
    return {
        "temperature": MulticlassTemperature(),
        "dirichlet_fixed": DirichletScaling(lam=FIXED_LAMBDA),
        "dirichlet_holdout": DirichletScaling(seed=seed, lambdas=DirichletScaling.WIDE_LAMBDAS, select="holdout"),
    }


def run_cell(dataset: str, model: str, seed: int) -> list[dict]:
    x, y = load(dataset, seed)
    x_tr, y_tr, x_cal, y_cal, x_te, y_te = split(x, y, seed)
    m = make_model(model, seed).fit(x_tr, y_tr)
    p_cal, p_te = _renorm(m.predict_proba(x_cal)), _renorm(m.predict_proba(x_te))
    out = []
    for name, mp in _maps(seed).items():
        mp.fit(p_cal, y_cal)
        q = mp.transform(p_te)
        extra = {"T": mp.T_} if name == "temperature" else {"lam": mp.lam_}
        out.append({"seed": seed, "dataset": dataset, "model": model, "map": name, "n_test": len(y_te),
                    "accuracy": float((q.argmax(axis=1) == y_te).mean()), "classwise_ece": classwise_ece(y_te, q),
                    "classwise_ce_debiased": classwise_ce_debiased(y_te, q), "nll": multiclass_nll(y_te, q), **extra})
    return out


def run_grid(seeds: Iterable[int] = EVAL_SEEDS, datasets=DATASETS, models=MODELS) -> pd.DataFrame:
    return pd.DataFrame([r for d in datasets for m in models for s in seeds for r in run_cell(d, m, s)])


def _bar(v: pd.DataFrame) -> bool:
    return bool((v.ece_wins >= 7).sum() >= 4 and (v.acc_diff >= -0.005).all() and (v.nll_diff <= 0.01).all())


def decide(long: pd.DataFrame) -> dict:
    out = {}
    for h, arm in (("H6", "dirichlet_fixed"), ("H7", "dirichlet_holdout")):
        v = _cell_stats(long[long["map"].isin(["temperature", arm])], "temperature", arm)
        out[h] = {"pass": _bar(v), "cells_ece_wins_ge7": int((v.ece_wins >= 7).sum()), "cells": v.to_dict("records")}
    passing = [arm for h, arm in (("H6", "dirichlet_fixed"), ("H7", "dirichlet_holdout")) if out[h]["pass"]]
    if len(passing) == 2:
        rec = "dirichlet_fixed" if out["H6"]["cells_ece_wins_ge7"] >= out["H7"]["cells_ece_wins_ge7"] else "dirichlet_holdout"
    else:
        rec = passing[0] if passing else "temperature"
    out["recommended"] = rec
    return out
