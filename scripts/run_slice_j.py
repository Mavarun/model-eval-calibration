#!/usr/bin/env python3
"""Slice J: Dirichlet with lambda fixed at 100, or chosen on a holdout, vs temperature (fresh seeds 40-49).

Writes results/slice_j_grid.csv and results/slice_j_decisions.json.
Rules: model_eval_calibration.multiclass_j.PREREGISTERED (committed before this run).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_eval_calibration import multiclass_j as J  # noqa: E402


def main() -> int:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    grid = J.run_grid()
    grid.to_csv(out / "slice_j_grid.csv", index=False, float_format="%.6f")
    dec = {**J.decide(grid), "preregistered": J.PREREGISTERED}
    (out / "slice_j_decisions.json").write_text(json.dumps(dec, indent=2) + "\n")
    print(grid.groupby(["dataset", "model", "map"], sort=False)[["classwise_ece", "nll", "accuracy"]].mean().round(4).to_string())
    for h in ("H6", "H7"):
        print(h, dec[h]["pass"], dec[h]["cells_ece_wins_ge7"])
        for r in dec[h]["cells"]:
            print("  ", r)
    print("recommended", dec["recommended"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
