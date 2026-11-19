#!/usr/bin/env python3
"""Slice I: Dirichlet scaling with the wide lambda grid vs temperature (fresh seeds 30-39).

Writes results/slice_i_grid.csv and results/slice_i_decisions.json.
Rules: model_eval_calibration.multiclass_i.PREREGISTERED (committed before this run).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_eval_calibration import multiclass_i as I  # noqa: E402


def main() -> int:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    grid = I.run_grid()
    grid.to_csv(out / "slice_i_grid.csv", index=False, float_format="%.6f")
    dec = {"H5": I.decide_h5(grid), "preregistered": I.PREREGISTERED}
    (out / "slice_i_decisions.json").write_text(json.dumps(dec, indent=2) + "\n")
    print(grid.groupby(["dataset", "model", "map"], sort=False)[["classwise_ece", "nll", "accuracy"]].mean().round(4).to_string())
    for r in dec["H5"]["dirichlet_wide_vs_temperature"]:
        print(r)
    print("H5", dec["H5"]["H5"])
    print(grid[grid["map"] == "dirichlet_wide"].groupby(["dataset", "model"]).lam.agg(lambda s: s.value_counts().to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
