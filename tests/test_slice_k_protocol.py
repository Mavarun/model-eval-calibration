"""Slice K's frozen protocol: same seeds as slice J, both Dirichlet arms converged, J's bar unchanged."""

from model_eval_calibration import multiclass_j as J
from model_eval_calibration import multiclass_k as K


def test_frozen():
    assert K.EVAL_SEEDS == J.EVAL_SEEDS == tuple(range(40, 50))
    assert set(K.PREREGISTERED) == {"K1_verdicts_hold", "K2_reported", "expectation"}
    m = K._maps(40)
    assert m["dirichlet_fixed"].converge and m["dirichlet_holdout"].converge
    assert m["dirichlet_fixed"].lam == J.FIXED_LAMBDA and m["dirichlet_holdout"].select == "holdout"


def test_read():
    j = {"H6": {"pass": False, "cells_ece_wins_ge7": 1}, "H7": {"pass": False, "cells_ece_wins_ge7": 3}}
    assert K.read({**j, "recommended": "temperature"}, j)["K1"]
    flip = {"H6": {"pass": True, "cells_ece_wins_ge7": 4}, "H7": j["H7"], "recommended": "dirichlet_fixed"}
    assert not K.read(flip, j)["K1"]


def test_run_grid_uses_converged_maps_and_restores(monkeypatch):
    seen = {}

    def fake(seeds):
        seen["maps"] = J._maps(40)
        return list(seeds)

    before = J._maps
    monkeypatch.setattr(J, "run_grid", fake)
    assert K.run_grid(seeds=(40,)) == [40]
    assert seen["maps"]["dirichlet_fixed"].converge and J._maps is before
