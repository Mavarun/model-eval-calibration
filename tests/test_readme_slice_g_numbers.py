"""Every slice-G number in the README is recomputed from the committed results."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
G = pd.read_csv(ROOT / "results" / "slice_g_grid.csv")
A = G.groupby(["dataset", "model", "map"]).mean(numeric_only=True)
D = json.loads((ROOT / "results" / "slice_g_decisions.json").read_text())
CELLS = [(d, m) for d in ("digits", "synth5") for m in ("logistic", "random_forest", "naive_bayes")]


def test_ece_nll_table():
    for d, m in CELLS:
        cells = [f"{A.loc[(d, m, mp)].classwise_ece:.4f} / {A.loc[(d, m, mp)].nll:.3f}" for mp in ("none", "temperature", "vector")]
        assert f"| {d} | {m} | " + " | ".join(cells) + " |" in README


def test_win_table_and_decisions():
    tv = {(r["dataset"], r["model"]): r["ece_wins"] for r in D["temperature_vs_none"]}
    vt = {(r["dataset"], r["model"]): r["ece_wins"] for r in D["vector_vs_temperature"]}
    for c in CELLS:
        assert f"| {c[0]} | {c[1]} | {tv[c]}/10 | {vt[c]}/10 |" in README
    assert D["H1"] and not D["H2"]
    assert "**H1 passes: temperature ships ON**" in README and "**H2 fails: vector scaling stays OFF.**" in README


def test_posthoc_numbers():
    t = G[G["map"] == "temperature"].groupby(["dataset", "model"])["T"].mean()
    assert f"mean T {t[('digits', 'random_forest')]:.2f} on digits, {t[('synth5', 'random_forest')]:.2f} on synth5" in README
    assert f"(T {t[('digits', 'naive_bayes')]:.1f} on digits)" in README
    nb = lambda mp: A.loc[("digits", "naive_bayes", mp)].nll
    assert f"from {nb('none'):.2f} to {nb('temperature'):.2f}" in README
    sat = pd.read_csv(ROOT / "results" / "slice_g_saturation.csv").groupby(["dataset", "model"]).share_saturated.mean()
    assert f"{100 * sat[('digits', 'naive_bayes')]:.0f}% of its test rows" in README
    assert G[(G.dataset == "digits")].n_test.iloc[0] == 540 and "540 test rows" in README
