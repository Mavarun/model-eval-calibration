#!/usr/bin/env python3
"""Slice E: per-row decisions under smooth and abrupt prior drift.

Hypotheses (synthetic label shift with known ground truth, c_fp = 1, c_fn = 5):
1. Slice D's best policy (EM on the batch being decided) depends on batch
   size: its regret grows as batches shrink, so it does not carry over to
   per-row decisions.
2. Per-row label-free trackers (online EM, sliding-window EM) using only
   past rows get close to the 400-row batch policy without making rows wait.
3. Under abrupt jumps a change-point-restarted EM (CUSUM) beats fixed-memory
   trackers, mostly in the rows right after a change; on smooth drift it
   should cost little or nothing.
4. All label-free trackers beat a threshold re-tuned on 400-row-late labels.

Protocol: tracker settings chosen on 6 development seeds (both drift shapes
pooled) per base model; then 10 disjoint test seeds x {smooth, regime} x
{logistic, random_forest}, 9600 rows each. Paired differences use the
moving-block bootstrap over 100-row chunks with a Politis-White block.
Also re-checks slice D's fixed block of 4 batches with the same rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from model_eval_calibration.drift_eval import paired_policy_difference, run_walk_forward  # noqa: E402
from model_eval_calibration.stream_eval import run_stream, select_trackers, summarise_stream  # noqa: E402

SEEDS = range(10)
MODELS = ("logistic", "random_forest")
PATHS = ("smooth", "regime")
ORDER = ("static", "batch400_em_current", "batch100_em_current", "batch25_em_current", "batch400_em_prev",
         "online_em", "window_em", "cusum_em", "delayed_label_threshold", "oracle_prior")
PAIRS = (("cusum_em", "batch400_em_current"), ("online_em", "batch400_em_current"), ("cusum_em", "online_em"),
         ("cusum_em", "window_em"), ("cusum_em", "delayed_label_threshold"), ("online_em", "batch25_em_current"))


def _pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main() -> None:
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    chosen, dev = {}, []
    for b in MODELS:
        chosen[b], score = select_trackers(b, test_seeds=SEEDS)
        dev.append(score.assign(base_model=b))
    pd.concat(dev).to_csv(out / "stream_dev_selection.csv", index=False)
    print("### Tracker settings chosen on development seeds 1000-1005 (mean per-row cost, both paths pooled)\n")
    for b in MODELS:
        print(f"- {b}: {chosen[b]}")

    long = pd.concat([run_stream(seed=s, base_model=b, path_kind=k, trackers=chosen[b])
                      for b in MODELS for k in PATHS for s in SEEDS], ignore_index=True)
    long.drop(columns="decisions_hash").to_csv(out / "stream_long.csv", index=False)
    summ = summarise_stream(long)
    for k in PATHS:
        print(f"\n### {k} path: regret per row (mean ± std over {len(SEEDS)} test seeds)\n")
        print("| policy | " + " | ".join(MODELS) + (" | " + " | ".join(f"post-change {b}" for b in MODELS) if k == "regime" else "") + " |")
        print("| --- |" + " ---: |" * (len(MODELS) * (2 if k == "regime" else 1)))
        s = summ[summ.path == k].set_index(["base_model", "policy"])
        for pol in ORDER:
            cells = [_pm(s.loc[(b, pol), "regret_mean"], s.loc[(b, pol), "regret_std"]) for b in MODELS]
            if k == "regime":
                cells += [_pm(s.loc[(b, pol), "post_change_regret_mean"], s.loc[(b, pol), "post_change_regret_std"]) for b in MODELS]
            print(f"| {pol} | " + " | ".join(cells) + " |")
        fl = long[(long.path == k) & (long.policy == "oracle_prior")]
        print(f"\nFloor (exact posterior) cost per row: {(fl['cost'] - fl['regret']).mean():.3f}")

    print("\n### Paired cost differences a - b per row (block bootstrap over 100-row chunks, Politis-White block; 95% CI)\n")
    print("| path | base | a - b | estimate | CI | block | P(a cheaper) |")
    print("| --- | --- | --- | ---: | --- | ---: | ---: |")
    for k in PATHS:
        for b in MODELS:
            sub = long[(long.path == k) & (long.base_model == b)]
            for pa, pb in PAIRS:
                r = paired_policy_difference(sub, pa, pb, block="auto")
                print(f"| {k} | {b} | {pa} - {pb} | {r['estimate']:+.4f} | [{r['lower']:+.4f}, {r['upper']:+.4f}] | "
                      f"{r['block']} | {r['prob_a_better']:.2f} |")

    print("\n### Slice D re-check: fixed block 4 vs Politis-White block (24 batches x 400 rows, raw maps)\n")
    print("| base | a - b | block 4 CI | auto block | auto CI |")
    print("| --- | --- | --- | ---: | --- |")
    for b in MODELS:
        wf = pd.concat([run_walk_forward(seed=s, base_model=b, calibration="raw") for s in SEEDS], ignore_index=True)
        for pa, pb in (("em_current", "delayed_label_threshold"), ("em_window", "delayed_label_threshold"),
                       ("em_prev", "em_window"), ("em_current", "static")):
            f = paired_policy_difference(wf, pa, pb, block=4)
            a = paired_policy_difference(wf, pa, pb, block="auto")
            print(f"| {b} | {pa} - {pb} | [{f['lower']:+.4f}, {f['upper']:+.4f}] | {a['block']} | "
                  f"[{a['lower']:+.4f}, {a['upper']:+.4f}] |")


if __name__ == "__main__":
    main()
