"""Every slice H number in the README is recomputed from results/."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SEC = README[README.index("## Slice H hypothesis"): README.index("## Slice G hypothesis")]
RES = ROOT / "results"


def test_table_rows():
    g = pd.read_csv(RES / "slice_h_grid.csv")
    dec = json.loads((RES / "slice_h_decisions.json").read_text())
    m = g.groupby(["dataset", "model", "map"])[["classwise_ece", "nll"]].mean()
    for r in dec["H3"]["dirichlet_vs_temperature"]:
        d, mo = r["dataset"], r["model"]
        cells = " | ".join(f"{m.loc[(d, mo, x), 'classwise_ece']:.4f} / {m.loc[(d, mo, x), 'nll']:.3f}"
                           for x in ("none", "temperature", "dirichlet"))
        assert f"| {d} | {mo} | {cells} | {r['ece_wins']}/10 | {r['acc_diff']:+.3f} |" in SEC


def test_calls_and_posthoc():
    dec = json.loads((RES / "slice_h_decisions.json").read_text())
    assert not dec["H3"]["H3"] and "**H3 fails, so Dirichlet stays OFF.**" in SEC
    assert dec["H4"]["H1_debiased"] and "**H4 passes.**" in SEC
    per = " | ".join(f"{x['ece_wins']}/10" for x in dec["H4"]["temperature_vs_none_debiased"])
    assert f"(per cell: {per})" in SEC
    e = json.loads((RES / "slice_h_posthoc_edge.json").read_text())
    assert f"largest grid lambda in {e['n_at_top']} of {e['n_fits']} fits" in SEC.replace("\n", " ")
    t = pd.read_csv(RES / "slice_h_posthoc_lambda.csv").set_index(["dataset", "model", "map"])
    lg = t.loc[("digits", "logistic")]
    assert f"logistic\n0.{round(lg.loc['dirichlet lam=1', 'nll'] * 1000):03d} → 0.{round(lg.loc['dirichlet lam=100', 'nll'] * 1000):03d}" in SEC
    assert f"trails temperature (0.{round(lg.loc['temperature', 'nll'] * 1000):03d})" in SEC
