#!/usr/bin/env python3
"""Slice D: label shift -- prior estimation, cost-aware decisions, Mondrian conformal, walk-forward drift.

Hypotheses (research demo; synthetic data with known ground truth):
1. Under label shift a source-calibrated model is miscalibrated on target and
   its cost-optimal threshold moves; correcting the prior (EM or BBSE on
   unlabelled target rows) recovers most of the cost gap to the oracle.
2. EM needs calibrated posteriors; BBSE does not. A source calibration map
   should therefore help EM -- measured, not assumed.
3. Pooled split conformal loses per-class coverage under label shift;
   Mondrian (class-conditional) conformal keeps it.
4. Walk-forward under a drifting prior: label-free prior tracking beats a
   static model and competes with re-tuning on labels that arrive one batch late.

Grid: pi_s = 0.3; pi_t in {0.05, 0.1, 0.3, 0.5} x 10 seeds x {logistic,
random_forest}. Walk-forward: 24 batches x 400 rows, 10 seeds x 2 models x
{raw, platt, isotonic}. Costs c_fp = 1, c_fn = 5. Long CSVs under artifacts/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_eval_calibration.drift_eval import (  # noqa: E402
    POLICIES,
    paired_policy_difference,
    run_walk_forward,
    summarise_walk_forward,
)
from model_eval_calibration.label_shift_eval import run_label_shift_grid  # noqa: E402

import pandas as pd  # noqa: E402

PRIORS = (0.05, 0.1, 0.3, 0.5)
SEEDS = range(10)
MODELS = ("logistic", "random_forest")
MAPS = ("raw", "platt", "isotonic")


def _pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main() -> None:
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    dec, conf = run_label_shift_grid(target_priors=PRIORS, seeds=SEEDS, base_models=MODELS)
    dec.to_csv(out / "label_shift_decisions_long.csv", index=False)
    conf.to_csv(out / "label_shift_conformal_long.csv", index=False)

    bayes = dec[dec.policy == "bayes"]
    for base in MODELS:
        print(f"\n### Prior estimation |pi_hat - pi_t| and target cost regret: {base} (mean ± std, {len(SEEDS)} seeds)\n")
        print("| map | estimator | " + " | ".join(f"err pi={p:g}" for p in PRIORS) + " | "
              + " | ".join(f"regret pi={p:g}" for p in PRIORS) + " |")
        print("| --- | --- |" + " ---: |" * (2 * len(PRIORS)))
        b = bayes[bayes.base_model == base]
        for m in MAPS:
            for est in ("none", "em", "bbse", "oracle"):
                g = b[(b.calibration == m) & (b.estimator == est)].groupby("target_prior")
                e, r = g["prior_abs_err"].agg(["mean", "std"]), g["regret"].agg(["mean", "std"])
                print(f"| {m} | {est} | " + " | ".join(_pm(*e.loc[p]) for p in PRIORS) + " | "
                      + " | ".join(_pm(*r.loc[p]) for p in PRIORS) + " |")
            t = dec[(dec.base_model == base) & (dec.calibration == m) & (dec.policy == "tuned_source")]
            r = t.groupby("target_prior")["regret"].agg(["mean", "std"])
            print(f"| {m} | tuned on source | " + " | ".join("-" for _ in PRIORS) + " | "
                  + " | ".join(_pm(*r.loc[p]) for p in PRIORS) + " |")
        o = dec[(dec.base_model == base) & (dec.policy == "oracle_posterior")].groupby("target_prior")["cost"].mean()
        print("\nOracle-posterior cost per row (the floor): " + ", ".join(f"pi={p:g}: {o.loc[p]:.3f}" for p in PRIORS))

    print("\n### Conformal on target (alpha = 0.10), mean over seeds\n")
    print("| base | pi_t | method | coverage | cov. negatives | cov. positives | set size |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    cm = conf.groupby(["base_model", "target_prior", "method"], sort=False)[
        ["coverage", "coverage_neg", "coverage_pos", "mean_set_size"]].mean()
    for (b, p, m), r in cm.iterrows():
        print(f"| {b} | {p:g} | {m} | {r.coverage:.3f} | {r.coverage_neg:.3f} | {r.coverage_pos:.3f} | {r.mean_set_size:.3f} |")

    wf = pd.concat([run_walk_forward(seed=s, base_model=b, calibration=m)
                    for b in MODELS for m in MAPS for s in SEEDS], ignore_index=True)
    wf.drop(columns="decisions_hash").to_csv(out / "walk_forward_long.csv", index=False)
    summ = summarise_walk_forward(wf)
    print("\n### Walk-forward under prior drift (24 batches x 400 rows; mean per-row cost and regret over seeds)\n")
    print("| base | map | " + " | ".join(POLICIES) + " |")
    print("| --- | --- |" + " ---: |" * len(POLICIES))
    for b in MODELS:
        for m in MAPS:
            s = summ[(summ.base_model == b) & (summ.calibration == m)].set_index("policy")
            print(f"| {b} | {m} | " + " | ".join(_pm(s.loc[k, 'regret_mean'], s.loc[k, 'regret_std']) for k in POLICIES) + " |")
    # paired win-rate: label-free em_window vs delayed-label threshold
    per = wf.groupby(["base_model", "calibration", "seed", "policy"])["cost"].mean().unstack("policy")
    wins = (per["em_window"] < per["delayed_label_threshold"]).groupby(level=["base_model", "calibration"]).mean()
    wins2 = (per["em_current"] < per["delayed_label_threshold"]).groupby(level=["base_model", "calibration"]).mean()
    print("\nSeed win-rate (lower mean cost), em_window vs delayed_label_threshold: "
          + ", ".join(f"{b}/{m} {v:.1f}" for (b, m), v in wins.items()))
    print("Seed win-rate, em_current vs delayed_label_threshold: "
          + ", ".join(f"{b}/{m} {v:.1f}" for (b, m), v in wins2.items()))
    print("\n### Paired cost differences (moving-block bootstrap over batches, block 4, seeds resampled; 95% CI)\n")
    print("| base | map | a - b | estimate | CI | P(a cheaper) |")
    print("| --- | --- | --- | ---: | --- | ---: |")
    for b in MODELS:
        for m in MAPS:
            sub = wf[(wf.base_model == b) & (wf.calibration == m)]
            for pa, pb in (("em_current", "delayed_label_threshold"), ("em_window", "delayed_label_threshold"),
                           ("em_prev", "em_window"), ("em_current", "static")):
                r = paired_policy_difference(sub, pa, pb)
                print(f"| {b} | {m} | {pa} - {pb} | {r['estimate']:+.4f} | [{r['lower']:+.4f}, {r['upper']:+.4f}] | "
                      f"{r['prob_a_better']:.2f} |")


if __name__ == "__main__":
    main()
