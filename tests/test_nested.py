"""Tests for nested CV calibration and the outer-test barrier."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from model_eval_calibration.data import load_hard_binary_classification
from model_eval_calibration.nested import (
    assert_nested_barrier,
    run_naive_same_fold_calibration,
    run_naive_vs_nested_comparison,
    run_nested_calibration,
)


def test_assert_nested_barrier_passes_on_clean_split():
    outer_train = np.arange(0, 80)
    outer_test = np.arange(80, 100)
    cal_fit = np.arange(40, 80)  # subset of train
    assert_nested_barrier(outer_train, outer_test, cal_fit)


def test_assert_nested_barrier_detects_leakage():
    outer_train = np.arange(0, 80)
    outer_test = np.arange(80, 100)
    cal_fit = np.array([10, 20, 85])  # 85 is outer-test
    with pytest.raises(AssertionError, match="barrier violated|must be subset"):
        assert_nested_barrier(outer_train, outer_test, cal_fit)


def test_nested_calibrator_fit_never_uses_outer_test_indices():
    """Explicit barrier: simulate nested index sets and assert no leakage."""
    X, y = make_classification(
        n_samples=200, n_features=8, n_informative=5, weights=[0.7, 0.3], random_state=0
    )
    from sklearn.model_selection import StratifiedKFold

    outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    for train_idx, test_idx in outer.split(X, y):
        # mimic nested: calibrator fit indices drawn only from train
        rng = np.random.default_rng(0)
        cal_fit = rng.choice(train_idx, size=max(10, len(train_idx) // 4), replace=False)
        assert_nested_barrier(train_idx, test_idx, cal_fit)
        # and a deliberate leak must fail
        leaked = np.concatenate([cal_fit[:5], test_idx[:2]])
        with pytest.raises(AssertionError):
            assert_nested_barrier(train_idx, test_idx, leaked)


def test_nested_vs_naive_runs_and_returns_both_paths():
    X, y, meta = load_hard_binary_classification(random_state=0, n_samples=400)
    assert meta["name"] == "make_classification_hard_binary"
    est = RandomForestClassifier(n_estimators=30, random_state=0)
    result = run_naive_vs_nested_comparison(
        X, y, estimator=est, n_outer=3, n_inner=2, random_state=0, dataset_name=meta["name"]
    )
    paths = {m.path for m in result.methods}
    assert paths == {"naive_same_fold", "nested"}
    frame = result.summary_frame()
    assert {"ece_equal_width", "ece_adaptive", "brier"}.issubset(frame.columns)
    assert len(frame) == 6  # 3 methods x 2 paths


def test_naive_and_nested_produce_finite_metrics():
    X, y, _ = load_hard_binary_classification(random_state=1, n_samples=350)
    est = RandomForestClassifier(n_estimators=25, random_state=1)
    naive = run_naive_same_fold_calibration(X, y, est, n_splits=3, random_state=1)
    nested = run_nested_calibration(X, y, est, n_outer=3, n_inner=2, random_state=1)
    for m in naive + nested:
        assert np.isfinite(m.ece_equal_width)
        assert np.isfinite(m.ece_adaptive)
        assert m.ece_equal_width >= 0.0
        assert m.ece_adaptive >= 0.0
