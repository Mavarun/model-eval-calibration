import numpy as np

from model_eval_calibration.multiclass_eval import (
    DEV_SEEDS, EVAL_SEEDS, MAPS, PREREGISTERED, decide, load, run_cell, run_grid, split,
)


def test_preregistration_fixed_and_seeds_disjoint():
    assert set(PREREGISTERED) == {"H1_temperature", "H2_vector_over_temperature", "reported_not_decided"}
    assert ">= 4 of the 6" in PREREGISTERED["H1_temperature"] and "0.005" in PREREGISTERED["H2_vector_over_temperature"]
    assert not set(EVAL_SEEDS) & set(DEV_SEEDS)


def test_split_is_disjoint_stratified_and_sized():
    x, y = load("digits", 0)
    x_tr, y_tr, x_cal, y_cal, x_te, y_te = split(x, y, 0)
    assert len(y_tr) + len(y_cal) + len(y_te) == len(y)
    assert abs(len(y_cal) / len(y) - 0.2) < 0.01 and abs(len(y_te) / len(y) - 0.3) < 0.01
    assert set(np.unique(y_cal)) == set(range(10))


def test_temperature_keeps_accuracy_exactly():
    rows = {r["map"]: r for r in run_cell("synth5", "naive_bayes", DEV_SEEDS[0])}
    assert set(rows) == set(MAPS)
    assert rows["temperature"]["accuracy"] == rows["none"]["accuracy"]
    assert rows["temperature"]["T"] > 0


def test_decide_shape_on_dev_seeds():
    g = run_grid(seeds=DEV_SEEDS[:2], datasets=("digits",), models=("naive_bayes", "logistic"))
    assert len(g) == 2 * 2 * 3
    d = decide(g)
    assert d["H1"] is False and d["recommended"] == "none"  # 4 of 6 cells cannot hold with 2 cells
