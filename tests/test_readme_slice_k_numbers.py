"""Every slice-K number in the README comes from the committed results."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
SEC = README[README.index("## Slice K (2026-11-24)"): README.index("## Slice J (2026-11-24)")]
D = json.loads((ROOT / "results" / "slice_k_decisions.json").read_text())


def test_overview_and_order():
    assert "11. **Slice K (this PR)**" in README and "10. **Slice J**:" in README
    assert README.index("## Slice K (2026-11-24)") < README.index("## Slice J (2026-11-24)")
    assert "Slice K re-scores slice J converged: both verdicts hold." in README


def test_table_and_prose():
    for r in pd.DataFrame(D["vs_slice_j"]).itertuples():
        row = (f"| {r.hyp} | {r.dataset} | {r.model} | {r.nll_diff_j:+.3f} / {r.nll_diff_k:+.3f} | "
               f"{r.ece_wins_j}/10 / {r.ece_wins_k}/10 |")
        assert row in SEC, row
    assert f"H6 still wins ECE in {D['read']['H6_cells']} of 6 cells and H7 in {D['read']['H7_cells']}" in SEC
    gj, gk = (pd.read_csv(ROOT / "results" / f"slice_{s}_grid.csv") for s in ("j", "k"))
    sel = lambda g: g[(g.dataset == "digits") & (g.model == "naive_bayes") & (g["map"] == "dirichlet_holdout")].set_index("seed")
    j, k = sel(gj), sel(gk)
    assert f"rises by {k.loc[43, 'nll'] - j.loc[43, 'nll']:.2f} and {k.loc[45, 'nll'] - j.loc[45, 'nll']:.2f}" in SEC
    assert (j.loc[43, "lam"], k.loc[43, "lam"], j.loc[45, "lam"], k.loc[45, "lam"]) == (1.0, 0.1, 0.1, 0.001)
