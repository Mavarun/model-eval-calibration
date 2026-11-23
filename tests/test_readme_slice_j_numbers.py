"""Every slice-J number in the README comes from the committed results."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
README = (ROOT / "README.md").read_text()
SEC = README[README.index("## Slice J (2026-11-24)"): README.index("## Slice I (2026-11-20)")]
G = pd.read_csv(RES / "slice_j_grid.csv")
D = json.loads((RES / "slice_j_decisions.json").read_text())


def test_table():
    m = G.groupby(["dataset", "model", "map"]).mean(numeric_only=True)
    f = {(c["dataset"], c["model"]): c for c in D["H6"]["cells"]}
    h = {(c["dataset"], c["model"]): c for c in D["H7"]["cells"]}
    for (d, mo), a in f.items():
        b = h[(d, mo)]
        t, x, y = (m.loc[(d, mo, k)] for k in ("temperature", "dirichlet_fixed", "dirichlet_holdout"))
        row = (f"| {d} | {mo} | {t.classwise_ece:.4f} / {x.classwise_ece:.4f} / {y.classwise_ece:.4f} | "
               f"{a['ece_wins']}/10 / {b['ece_wins']}/10 | {a['nll_diff']:+.3f} / {b['nll_diff']:+.3f} | "
               f"{a['acc_diff']:+.4f} / {b['acc_diff']:+.4f} |")
        assert row in SEC, row


def test_prose():
    assert not D["H6"]["pass"] and not D["H7"]["pass"] and D["recommended"] == "temperature"
    assert "wins on ECE in 1\ncell" in SEC and D["H6"]["cells_ece_wins_ge7"] == 1
    assert f"holdout choice in {D['H7']['cells_ece_wins_ge7']}, not 4" in SEC
    f = {(c["dataset"], c["model"]): c for c in D["H6"]["cells"]}
    assert "loses +0.030 NLL on digits logistic" in SEC and round(f[("digits", "logistic")]["nll_diff"], 3) == 0.030
    hd = G[(G["map"] == "dirichlet_holdout") & (G.dataset == "digits")]
    assert f"chose lambda >= 100 in {int((hd.lam >= 100).sum())} of its 30 digits fits" in SEC
    nb = [c for c in D["H6"]["cells"] + D["H7"]["cells"] if c["model"] == "naive_bayes" and c["dataset"] == "digits"]
    assert all(c["ece_wins"] == 10 and 0.035 < c["acc_diff"] < 0.045 for c in nb)


def test_overview():
    assert "10. **Slice J**:" in README
