"""Slice K protocol: re-score slice J with converged Dirichlet fits (frozen 2026-11-24, before the re-run).

Slice J's weaknesses: the ODIR fit's default L-BFGS-B stop lands at a platform-dependent point short
of the optimum (digits logistic seed 100: test NLL 0.149 locally, 0.159 on the CI 3.11 runner, 0.157
converged on both), so whether any of slice J's OFF decisions would flip under converged fits was
unknown. This slice re-runs slice J's grid (seeds 40-49, same datasets, models, splits, lambdas,
holdout) with ``converge=True`` for both Dirichlet arms, and applies slice J's H6 / H7 bar unchanged.
Temperature is a 1-parameter bounded fit and is not re-run differently.

It re-reads a decision already made on these seeds, so it is a robustness check, not a new test: a
flip would reopen slice J, not ship anything.

Pre-stated reading:

- **K1:** H6 and H7 keep slice J's verdicts (both fail) under converged fits.
- **K2 (reported):** per cell, the change in mean test NLL and in ECE-winning seeds vs slice J.

Expectation written before the run: K1 holds; converged fits lose a little more NLL on digits
(the dev probe's seed-100 digits logistic NLL rose 0.149 -> 0.157 locally).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from model_eval_calibration import multiclass_j as J
from model_eval_calibration.multiclass import DirichletScaling, MulticlassTemperature

EVAL_SEEDS = J.EVAL_SEEDS

PREREGISTERED = {
    "K1_verdicts_hold": "Slice J's H6 and H7 both still fail when both Dirichlet arms are fitted with converge=True.",
    "K2_reported": "Per (dataset, model) cell: mean test NLL and ECE-winning seeds, converged minus slice J.",
    "expectation": "K1 holds; converged fits lose a little more NLL on digits.",
}


def _maps(seed: int):
    return {
        "temperature": MulticlassTemperature(),
        "dirichlet_fixed": DirichletScaling(lam=J.FIXED_LAMBDA, converge=True),
        "dirichlet_holdout": DirichletScaling(seed=seed, lambdas=DirichletScaling.WIDE_LAMBDAS, select="holdout",
                                              converge=True),
    }


def run_grid(seeds: Iterable[int] = EVAL_SEEDS) -> pd.DataFrame:
    orig = J._maps
    J._maps = _maps
    try:
        return J.run_grid(seeds)
    finally:
        J._maps = orig


def compare(conv: dict, slice_j: dict) -> pd.DataFrame:
    rows = []
    for h in ("H6", "H7"):
        for a, b in zip(slice_j[h]["cells"], conv[h]["cells"]):
            assert (a["dataset"], a["model"]) == (b["dataset"], b["model"])
            rows.append({"hyp": h, "dataset": a["dataset"], "model": a["model"],
                         "nll_diff_j": a["nll_diff"], "nll_diff_k": b["nll_diff"],
                         "ece_wins_j": a["ece_wins"], "ece_wins_k": b["ece_wins"]})
    return pd.DataFrame(rows)


def read(conv: dict, slice_j: dict) -> dict:
    k1 = conv["H6"]["pass"] == slice_j["H6"]["pass"] and conv["H7"]["pass"] == slice_j["H7"]["pass"]
    return {"K1": bool(k1), "H6_pass": conv["H6"]["pass"], "H7_pass": conv["H7"]["pass"],
            "H6_cells": conv["H6"]["cells_ece_wins_ge7"], "H7_cells": conv["H7"]["cells_ece_wins_ge7"],
            "recommended": conv["recommended"]}
