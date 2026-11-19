"""Every slice-I number in the README comes from the committed results."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
README = (ROOT / "README.md").read_text()
SEC = README[README.index("## Slice I (2026-11-20)"): README.index("## Slice H hypothesis")]


def test_table_and_decision():
    g = pd.read_csv(RES / "slice_i_grid.csv")
    dec = json.loads((RES / "slice_i_decisions.json").read_text())["H5"]
    p = json.loads((RES / "slice_i_posthoc_edge.json").read_text())
    m = g.groupby(["dataset", "model", "map"]).mean(numeric_only=True)
    for r in dec["dirichlet_wide_vs_temperature"]:
        d, mo = r["dataset"], r["model"]
        t, w = m.loc[(d, mo, "temperature")], m.loc[(d, mo, "dirichlet_wide")]
        row = (f"| {d} | {mo} | {t.classwise_ece:.4f} / {w.classwise_ece:.4f} | {r['ece_wins']}/10 | "
               f"{r['nll_diff']:+.3f} | {r['acc_diff']:+.4f} | {p['per_cell_at_top'][d + '/' + mo]}/10 |")
        assert row in SEC, row
    assert not dec["H5"] and f"It wins {dec['cells_ece_wins_ge7']} cells" in SEC
    digits = [r["nll_diff"] for r in dec["dirichlet_wide_vs_temperature"] if r["dataset"] == "digits"]
    assert f"(+{min(digits):.2f} to +{max(digits):.2f})" in SEC


def test_posthoc_prose():
    p = json.loads((RES / "slice_i_posthoc_edge.json").read_text())
    assert f"chose 1e3 in {p['n_at_top']} of {p['n_fits']} fits" in SEC
    assert f"{p['digits_nll_gap_at_top']:+.3f} in the {p['digits_n_at_top']} fits" in SEC
    assert f"{p['digits_nll_gap_below_top']:+.3f} in the {30 - p['digits_n_at_top']} fits" in SEC


def test_overview_lists_slice_i():
    assert "9. **Slice I (this PR)**" in README and "8. **Slice H**:" in README
