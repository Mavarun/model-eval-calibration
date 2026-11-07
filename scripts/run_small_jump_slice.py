#!/usr/bin/env python3
"""Slice F: small prior jumps and the CUSUM drift size.

Hypotheses (written before any test stream was scored; synthetic label shift,
c_fp = 1, c_fn = 5, source prior 0.3):
1. On small jumps (levels 0.2-0.4, every switch 0.05-0.2) slice E's CUSUM
   (h = 4, delta = 0.8 logit) detects few changes and is not cheaper than
   online EM.
2. A CUSUM whose (threshold, delta) is chosen on development seeds pooled over
   smooth, large-jump and small-jump streams is cheaper than slice E's setting
   on small jumps, without a cost on smooth drift (paired CI covering or below 0).
3. Every tracker's gain over the static prior shrinks on small jumps, because
   the static prior is never far from the truth.

Protocol: (threshold, delta) on a 3 x 4 grid chosen per base model on dev seeds
1000-1005 (all three shapes pooled); online step and window keep slice E's dev
choice (0.02, 200). Then 10 disjoint test seeds x 3 paths x 2 models, 9600 rows.
Paired differences: moving-block bootstrap over 100-row chunks, Politis-White
block. Detection stats count an alarm as true within 400 rows of a change.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model_eval_calibration.drift_eval import paired_policy_difference  # noqa: E402
from model_eval_calibration.online_prior import cusum_em_path  # noqa: E402
from model_eval_calibration.stream_eval import (  # noqa: E402
    PATH_KINDS,
    detection_stats,
    fit_source,
    make_target_stream,
    run_stream,
    select_cusum,
    summarise_stream,
)

SEEDS = range(10)
MODELS = ("logistic", "random_forest")
SLICE_E = {"online_step": 0.02, "window": 200, "cusum_threshold": 4.0, "cusum_delta": 0.8}
ORDER = ("static", "online_em", "window_em", "cusum_em_slice_e", "cusum_em_tuned", "oracle_prior")


def _pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def _detect(seed: int, b: str, kind: str, tr: dict) -> dict:
    model, cmap, _, _, pi_s = fit_source(seed, b)
    _, cps, X, _ = make_target_stream(seed, kind)
    p = cmap.transform(model.predict_proba(X)[:, 1])
    _, alarms = cusum_em_path(p, pi_s, threshold=tr["cusum_threshold"], delta_logit=tr["cusum_delta"])
    return detection_stats(alarms, cps, p.size)


def main() -> None:
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    chosen, dev = {}, []
    for b in MODELS:
        c, d = select_cusum(b, test_seeds=SEEDS)
        chosen[b] = {**SLICE_E, **c}
        dev.append(d.assign(base_model=b))
    dev = pd.concat(dev)
    dev.to_csv(out / "small_jump_dev_selection.csv", index=False)
    print("### CUSUM (threshold, delta) chosen on dev seeds 1000-1005 (mean per-row cost, 3 shapes pooled)\n")
    grid = dev.groupby(["base_model", "cusum_threshold", "cusum_delta"])["cost"].mean().unstack("cusum_delta")
    print("| base | threshold | " + " | ".join(f"delta {d:g}" for d in grid.columns) + " |")
    print("| --- | ---: |" + " ---: |" * len(grid.columns))
    for (b, h), r in grid.iterrows():
        print(f"| {b} | {h:g} | " + " | ".join(f"{v:.4f}" for v in r) + " |")
    for b in MODELS:
        print(f"- {b}: threshold {chosen[b]['cusum_threshold']}, delta {chosen[b]['cusum_delta']}")

    parts = []
    for b in MODELS:
        for k in PATH_KINDS:
            for s in SEEDS:
                e = run_stream(seed=s, base_model=b, path_kind=k, trackers=SLICE_E)
                t = run_stream(seed=s, base_model=b, path_kind=k, trackers=chosen[b])
                e.loc[e.policy == "cusum_em", "policy"] = "cusum_em_slice_e"
                t = t[t.policy == "cusum_em"].assign(policy="cusum_em_tuned")
                parts += [e, t]
    long = pd.concat(parts, ignore_index=True)
    long = long[long.policy.isin(ORDER)]
    long.drop(columns="decisions_hash").to_csv(out / "small_jump_long.csv", index=False)
    summ = summarise_stream(long)
    print("\n### Regret per row on test streams (mean ± std over 10 seeds)\n")
    cols = [(k, b) for k in PATH_KINDS for b in MODELS]
    print("| policy | " + " | ".join(f"{k} {b}" for k, b in cols) + " |")
    print("| --- |" + " ---: |" * len(cols))
    s = summ.set_index(["path", "base_model", "policy"])
    for pol in ORDER:
        print(f"| {pol} | " + " | ".join(_pm(s.loc[(k, b, pol), "regret_mean"], s.loc[(k, b, pol), "regret_std"])
                                        for k, b in cols) + " |")
    print("\n### Post-change regret per row (400 rows after each jump)\n")
    pcols = [(k, b) for k in ("regime", "small_regime") for b in MODELS]
    print("| policy | " + " | ".join(f"{k} {b}" for k, b in pcols) + " |")
    print("| --- |" + " ---: |" * len(pcols))
    for pol in ORDER:
        print(f"| {pol} | " + " | ".join(_pm(s.loc[(k, b, pol), "post_change_regret_mean"],
                                             s.loc[(k, b, pol), "post_change_regret_std"]) for k, b in pcols) + " |")

    print("\n### CUSUM alarms on test streams (mean over 10 seeds; true = within 400 rows of a change)\n")
    print("| path | base | setting | detected | median delay | false alarms / 1000 rows |")
    print("| --- | --- | --- | ---: | ---: | ---: |")
    for k in PATH_KINDS:
        for b in MODELS:
            for name, tr in (("slice E (4, 0.8)", SLICE_E), (f"tuned ({chosen[b]['cusum_threshold']:g}, "
                                                            f"{chosen[b]['cusum_delta']:g})", chosen[b])):
                d = pd.DataFrame([_detect(sd, b, k, tr) for sd in SEEDS])
                det = "-" if k == "smooth" else f"{d['detected'].mean():.2f}"
                dl = "-" if k == "smooth" else f"{np.nanmedian(d['median_delay']):.0f}"
                print(f"| {k} | {b} | {name} | {det} | {dl} | {d['false_alarms_per_1000'].mean():.2f} |")

    print("\n### Paired cost differences a - b per row (block bootstrap, Politis-White block; 95% CI)\n")
    print("| path | base | a - b | estimate | CI | block |")
    print("| --- | --- | --- | ---: | --- | ---: |")
    pairs = (("cusum_em_tuned", "cusum_em_slice_e"), ("cusum_em_tuned", "online_em"),
             ("cusum_em_slice_e", "online_em"), ("online_em", "static"))
    for k in PATH_KINDS:
        for b in MODELS:
            sub = long[(long.path == k) & (long.base_model == b)]
            for pa, pb in pairs:
                r = paired_policy_difference(sub, pa, pb, block="auto")
                print(f"| {k} | {b} | {pa} - {pb} | {r['estimate']:+.4f} | [{r['lower']:+.4f}, {r['upper']:+.4f}] | "
                      f"{r['block']} |")


if __name__ == "__main__":
    main()
