"""Slice G harness: multiclass calibration on two datasets x three base models.

* Datasets: ``digits`` (scikit-learn, 1797 rows, 10 classes) and ``synth5``
  (``make_classification``, 6000 rows, 5 classes, 10% label noise ``flip_y``).
* Base models: multinomial logistic regression (scaled), random forest (200 trees) and
  Gaussian naive Bayes (known to be over-confident).
* Per seed: stratified train 50% / calibration 20% / test 30%. The base model sees train only,
  the maps see calibration only, every number is on test.
* Maps: ``none``, ``temperature`` and ``vector`` (:mod:`model_eval_calibration.multiclass`).

:data:`PREREGISTERED` was committed before evaluation seeds 0-9 were run. Development seeds
100-102 only checked that the harness runs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_eval_calibration.multiclass import (
    MulticlassTemperature, VectorScaling, classwise_ece, multiclass_brier, multiclass_nll, top_label_ece,
)

DATASETS = ("digits", "synth5")
MODELS = ("logistic", "random_forest", "naive_bayes")
MAPS = ("none", "temperature", "vector")
EVAL_SEEDS = tuple(range(10))
DEV_SEEDS = (100, 101, 102)

PREREGISTERED = {
    "H1_temperature": (
        "Temperature scaling becomes the recommended multiclass default iff, in >= 4 of the 6 (dataset, model) "
        "cells, it lowers test classwise ECE vs 'none' on >= 7 of 10 seeds, AND in no cell does it raise mean test "
        "NLL by more than 0.01. Otherwise the default stays 'none'."
    ),
    "H2_vector_over_temperature": (
        "Vector scaling replaces temperature as the recommended default iff, in >= 4 of 6 cells, it lowers test "
        "classwise ECE vs temperature on >= 7 of 10 seeds, AND in every cell its mean test accuracy is no more than "
        "0.005 below temperature's and its mean NLL no more than 0.01 above. Otherwise OFF."
    ),
    "reported_not_decided": "Top-label ECE, multiclass Brier and accuracy are reported for context only.",
}


def load(name: str, seed: int):
    if name == "digits":
        d = load_digits()
        return d.data.astype(float), d.target.astype(int)
    if name == "synth5":
        return make_classification(n_samples=6000, n_features=20, n_informative=8, n_redundant=4, n_classes=5,
                                   n_clusters_per_class=2, flip_y=0.10, class_sep=1.0, random_state=10_000 + seed)
    raise ValueError(name)


def make_model(name: str, seed: int):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=1, random_state=seed, n_jobs=1)
    if name == "naive_bayes":
        return GaussianNB()
    raise ValueError(name)


def split(x, y, seed):
    x_tr, x_rest, y_tr, y_rest = train_test_split(x, y, train_size=0.5, stratify=y, random_state=seed)
    x_cal, x_te, y_cal, y_te = train_test_split(x_rest, y_rest, train_size=0.4, stratify=y_rest, random_state=seed)
    return x_tr, y_tr, x_cal, y_cal, x_te, y_te


def _renorm(p):
    p = np.clip(p, 0.0, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def run_cell(dataset: str, model: str, seed: int) -> list[dict]:
    x, y = load(dataset, seed)
    x_tr, y_tr, x_cal, y_cal, x_te, y_te = split(x, y, seed)
    m = make_model(model, seed).fit(x_tr, y_tr)
    p_cal, p_te = _renorm(m.predict_proba(x_cal)), _renorm(m.predict_proba(x_te))
    out = []
    for name in MAPS:
        if name == "none":
            q, extra = p_te, {}
        elif name == "temperature":
            t = MulticlassTemperature().fit(p_cal, y_cal)
            q, extra = t.transform(p_te), {"T": t.T_}
        else:
            q = VectorScaling().fit(p_cal, y_cal).transform(p_te)
            extra = {}
        out.append({"seed": seed, "dataset": dataset, "model": model, "map": name, "n_test": len(y_te),
                    "accuracy": float((q.argmax(axis=1) == y_te).mean()), "classwise_ece": classwise_ece(y_te, q),
                    "top_label_ece": top_label_ece(y_te, q), "brier": multiclass_brier(y_te, q),
                    "nll": multiclass_nll(y_te, q), **extra})
    return out


def run_grid(seeds: Iterable[int] = EVAL_SEEDS, datasets=DATASETS, models=MODELS) -> pd.DataFrame:
    return pd.DataFrame([r for d in datasets for m in models for s in seeds for r in run_cell(d, m, s)])


def _cell_stats(long, a, b):
    rows = []
    for (d, m), sub in long.groupby(["dataset", "model"], sort=False):
        w = sub.pivot(index="seed", columns="map")
        rows.append({"dataset": d, "model": m,
                     "ece_wins": int((w["classwise_ece"][b] < w["classwise_ece"][a]).sum()),
                     "ece_diff": float((w["classwise_ece"][b] - w["classwise_ece"][a]).mean()),
                     "nll_diff": float((w["nll"][b] - w["nll"][a]).mean()),
                     "acc_diff": float((w["accuracy"][b] - w["accuracy"][a]).mean()),
                     "n_seeds": int(len(w))})
    return pd.DataFrame(rows)


def decide(long: pd.DataFrame) -> dict:
    """Apply :data:`PREREGISTERED`."""
    t = _cell_stats(long, "none", "temperature")
    h1 = bool((t.ece_wins >= 7).sum() >= 4 and (t.nll_diff <= 0.01).all())
    v = _cell_stats(long, "temperature", "vector")
    h2 = bool((v.ece_wins >= 7).sum() >= 4 and (v.acc_diff >= -0.005).all() and (v.nll_diff <= 0.01).all())
    default = ("vector" if h2 else "temperature") if h1 else "none"
    return {"H1": h1, "H2": h2, "recommended": default,
            "temperature_vs_none": t.to_dict("records"), "vector_vs_temperature": v.to_dict("records")}


def posthoc_saturation(seeds: Iterable[int] = EVAL_SEEDS, datasets=DATASETS, models=MODELS,
                       cut: float = 0.999) -> pd.DataFrame:
    """Post-hoc reading that decides nothing: share of test rows whose base max probability exceeds ``cut``.

    A base model that outputs near-0/1 vectors leaves a single temperature little to work with:
    softening the top class also spreads mass evenly over the K - 1 others.
    """
    rows = []
    for d in datasets:
        for m in models:
            for s in seeds:
                x, y = load(d, s)
                x_tr, y_tr, _, _, x_te, _ = split(x, y, s)
                p = _renorm(make_model(m, s).fit(x_tr, y_tr).predict_proba(x_te))
                rows.append({"seed": s, "dataset": d, "model": m, "share_saturated": float((p.max(axis=1) > cut).mean())})
    return pd.DataFrame(rows)
