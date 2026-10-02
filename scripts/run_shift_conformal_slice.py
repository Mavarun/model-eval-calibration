#!/usr/bin/env python3
"""Slice C: calibration maps and conformal coverage under covariate shift.

Hypotheses (research demo; synthetic data with known ground truth):
1. A calibrator fitted on source rows does not stay calibrated when P(x)
   shifts, even though P(y|x) is unchanged, if the base model is misspecified.
2. Importance-weighting the calibration rows by an estimated density ratio
   (domain classifier on unlabelled target features) reduces target ECE.
3. Split conformal loses its nominal coverage on the target; weighted
   conformal (Tibshirani et al., 2019) moves it back toward 1 - alpha.

Design: shifts {0, 1, 2} x 10 seeds x {logistic, random_forest}; per cell
2000 train / 1000 cal / 1000 source-test source rows, 1000 unlabelled
target-pool rows, 2000 labelled target-test rows; alpha = 0.1.
Outputs markdown tables to stdout and long CSVs under artifacts/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from model_eval_calibration.bootstrap import reliability_bootstrap_bands  # noqa: E402
from model_eval_calibration.shift_eval import (  # noqa: E402
    run_shift_experiment,
    run_shift_grid,
    summarise_grid,
    win_rate_vs,
)

SHIFTS = (0.0, 1.0, 2.0)
SEEDS = range(10)
METHODS = ["raw", "temperature", "platt", "beta", "isotonic",
           "iw_temperature", "iw_platt", "iw_beta", "iw_isotonic"]


def _pm(mean, std, nd=4):
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def main() -> None:
    cal, conf = run_shift_grid(shifts=SHIFTS, seeds=SEEDS, n_boot=100)
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    cal.to_csv(out_dir / "shift_calibration_long.csv", index=False)
    conf.to_csv(out_dir / "shift_conformal_long.csv", index=False)

    summ = summarise_grid(cal, ["ece_adaptive", "brier", "mce"])
    tgt = summ[summ["domain"] == "target"]
    for base in ("logistic", "random_forest"):
        print(f"\n### Target-domain calibration: {base} (mean ± std over {len(SEEDS)} seeds)\n")
        print("| method | " + " | ".join(f"ECE-adapt s={s:g}" for s in SHIFTS)
              + " | " + " | ".join(f"Brier s={s:g}" for s in SHIFTS) + " |")
        print("| --- |" + " ---: |" * (2 * len(SHIFTS)))
        for m in METHODS:
            cells_e, cells_b = [], []
            for s in SHIFTS:
                r = tgt[(tgt.base_model == base) & (tgt["shift"] == s) & (tgt.method == m)].iloc[0]
                cells_e.append(_pm(r.ece_adaptive_mean, r.ece_adaptive_std))
                cells_b.append(_pm(r.brier_mean, r.brier_std))
            print(f"| {m} | " + " | ".join(cells_e + cells_b) + " |")

    print("\n### Seed win-rates on target adaptive ECE (share of seeds where A < B)\n")
    pairs = [("iw_platt", "platt"), ("iw_beta", "beta"), ("iw_isotonic", "isotonic"),
             ("iw_temperature", "temperature"), ("beta", "platt"), ("temperature", "raw"),
             ("isotonic", "raw")]
    print("| A vs B | " + " | ".join(f"{b} s={s:g}" for b in ("logistic", "random_forest")
                                       for s in SHIFTS) + " |")
    print("| --- |" + " ---: |" * (2 * len(SHIFTS)))
    for a, b in pairs:
        wr = win_rate_vs(cal, "ece_adaptive", a, b).set_index(["base_model", "shift"]).iloc[:, 0]
        cells = [f"{wr.loc[(base, s)]:.1f}" for base in ("logistic", "random_forest") for s in SHIFTS]
        print(f"| {a} vs {b} | " + " | ".join(cells) + " |")

    csum = summarise_grid(conf, ["coverage", "mean_set_size"])
    print("\n### Conformal (alpha = 0.10, nominal coverage 0.90)\n")
    print("| base | shift | domain | method | coverage | mean set size |")
    print("| --- | ---: | --- | --- | ---: | ---: |")
    for _, r in csum.iterrows():
        print(f"| {r.base_model} | {r['shift']:g} | {r.domain} | {r.method} | "
              f"{_pm(r.coverage_mean, r.coverage_std, 3)} | "
              f"{_pm(r.mean_set_size_mean, r.mean_set_size_std, 3)} |")

    ess = conf.groupby("shift")["ess_cal_estimated"].mean()
    print("\nMean Kish ESS of the 1000 importance-weighted cal rows: "
          + ", ".join(f"shift {s:g}: {v:.0f}" for s, v in ess.items()))

    # Reliability bands for one illustrative cell (logistic, shift 2, seed 0)
    r = run_shift_experiment(shift=2.0, seed=0, base_model="logistic", n_boot=50)
    from model_eval_calibration.shift import make_covariate_shift_data
    from model_eval_calibration.shift_eval import build_base_model

    d = make_covariate_shift_data(n_source=4000, n_target=3000, shift=2.0, random_state=0)
    Xs, ys = d.X_source, d.y_source
    model = build_base_model("logistic").fit(Xs[r.splits.train], ys[r.splits.train])
    p_t = model.predict_proba(d.X_target[r.splits.target_test])[:, 1]
    y_t = d.y_target[r.splits.target_test]
    bands = reliability_bootstrap_bands(y_t, p_t, n_bins=10, strategy="quantile",
                                        n_boot=500, random_state=0)
    ok = bands.bin_counts > 0
    miss = ~((bands.acc_lower <= bands.bin_confidence) & (bands.bin_confidence <= bands.acc_upper))
    print(f"\nReliability bands (logistic raw, target, shift 2, seed 0): "
          f"{int(np.sum(miss & ok))}/{int(ok.sum())} quantile bins have a 95% "
          f"bootstrap band that excludes their mean confidence.")


if __name__ == "__main__":
    main()
