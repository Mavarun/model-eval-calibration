#!/usr/bin/env python3
"""Slice K: slice J re-scored with converged Dirichlet fits (seeds 40-49).

Writes results/slice_k_grid.csv and results/slice_k_decisions.json.
Rules: model_eval_calibration.multiclass_k.PREREGISTERED (committed before this run).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_eval_calibration import multiclass_j as J  # noqa: E402
from model_eval_calibration import multiclass_k as K  # noqa: E402


def main() -> int:
    out = ROOT / "results"
    grid = K.run_grid()
    grid.to_csv(out / "slice_k_grid.csv", index=False, float_format="%.6f")
    conv = J.decide(grid)
    sj = json.loads((out / "slice_j_decisions.json").read_text())
    cmp_ = K.compare(conv, sj)
    dec = {**conv, "read": K.read(conv, sj), "vs_slice_j": cmp_.to_dict("records"), "preregistered": K.PREREGISTERED}
    (out / "slice_k_decisions.json").write_text(json.dumps(dec, indent=2) + "\n")
    print(cmp_.round(4).to_string(index=False))
    print(dec["read"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
