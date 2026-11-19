"""Slice G: multiclass calibration (temperature vs vector scaling), pre-registered decisions.

    python scripts/run_multiclass_slice.py      # 10 evaluation seeds, ~1 min on CPU

Writes results/slice_g_grid.csv and results/slice_g_decisions.json and prints the README tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_eval_calibration.multiclass_eval import DATASETS, EVAL_SEEDS, MAPS, MODELS, decide, run_grid  # noqa: E402


def main(seeds=EVAL_SEEDS, out: Path = ROOT / "results"):
    long = run_grid(seeds=seeds)
    out.mkdir(exist_ok=True)
    long.to_csv(out / "slice_g_grid.csv", index=False, float_format="%.6f")
    n = long.seed.nunique()
    agg = long.groupby(["dataset", "model", "map"]).mean(numeric_only=True)
    print(f"### Test metrics (mean over {n} seeds)\n")
    print("| dataset | model | map | accuracy | classwise ECE | top-label ECE | Brier | NLL |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for d in DATASETS:
        for m in MODELS:
            for mp in MAPS:
                r = agg.loc[(d, m, mp)]
                print(f"| {d} | {m} | {mp} | {r.accuracy:.3f} | {r.classwise_ece:.4f} | {r.top_label_ece:.4f} | "
                      f"{r.brier:.4f} | {r.nll:.3f} |")
    dec = decide(long)
    for key, title in (("temperature_vs_none", "temperature - none"), ("vector_vs_temperature", "vector - temperature")):
        print(f"\n### Paired per-cell effects: {title} ({n} seeds)\n")
        print("| dataset | model | seeds with lower classwise ECE | classwise ECE diff | NLL diff | accuracy diff |")
        print("| --- | --- | ---: | ---: | ---: | ---: |")
        for r in dec[key]:
            print(f"| {r['dataset']} | {r['model']} | {r['ece_wins']}/{r['n_seeds']} | {r['ece_diff']:+.4f} | "
                  f"{r['nll_diff']:+.3f} | {r['acc_diff']:+.4f} |")
    print("\n### Pre-registered decisions\n")
    print(json.dumps({k: dec[k] for k in ("H1", "H2", "recommended")}))
    (out / "slice_g_decisions.json").write_text(json.dumps(dec, indent=2) + "\n")
    return dec


if __name__ == "__main__":
    main()
