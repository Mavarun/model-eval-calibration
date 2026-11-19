#!/usr/bin/env python3
"""Slice H: Dirichlet scaling vs temperature (fresh seeds 20-29) and H1 under the debiased estimator.

Writes results/slice_h_grid.csv, results/slice_g_debiased.csv and results/slice_h_decisions.json.
Rules: model_eval_calibration.multiclass_h.PREREGISTERED (committed before this run).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_eval_calibration import multiclass_eval as G  # noqa: E402
from model_eval_calibration import multiclass_h as H  # noqa: E402


def main() -> int:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    grid = H.run_grid()
    grid.to_csv(out / "slice_h_grid.csv", index=False, float_format="%.6f")
    g = H.run_grid(seeds=G.EVAL_SEEDS, maps=("none", "temperature"))
    g.to_csv(out / "slice_g_debiased.csv", index=False, float_format="%.6f")
    dec = {"H3": H.decide_h3(grid), "H4": H.decide_h4(g), "preregistered": H.PREREGISTERED}
    (out / "slice_h_decisions.json").write_text(json.dumps(dec, indent=2) + "\n")
    means = grid.groupby(["dataset", "model", "map"], sort=False)[["classwise_ece", "classwise_ce_debiased", "nll", "accuracy"]].mean()
    print(means.round(4).to_string())
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in dec.items() if k != "preregistered"}))
    for k in ("dirichlet_vs_temperature",):
        for r in dec["H3"][k]:
            print(r)
    for r in dec["H4"]["temperature_vs_none_debiased"]:
        print(r)
    print(grid[grid["map"] == "dirichlet"].groupby(["dataset", "model"]).lam.agg(lambda s: s.value_counts().to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
