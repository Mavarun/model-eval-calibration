import numpy as np
from sklearn.datasets import make_classification

from model_eval_calibration.evaluate import (
    build_logistic_pipeline,
    build_random_forest,
    reliability_for_method,
    run_calibration_comparison,
)


def test_run_calibration_comparison_returns_three_methods():
    X, y = make_classification(
        n_samples=150, n_features=8, n_informative=5, random_state=0
    )
    result = run_calibration_comparison(
        X, y, estimator=build_random_forest(0), n_splits=3, random_state=0
    )
    names = [m.name for m in result.methods]
    assert names == ["raw", "platt_sigmoid", "isotonic"]
    frame = result.summary_frame()
    assert set(frame.columns) >= {"accuracy", "brier", "ece", "roc_auc"}
    assert len(frame) == 3


def test_logistic_pipeline_comparison_runs():
    X, y = make_classification(
        n_samples=120, n_features=6, n_informative=4, random_state=3
    )
    result = run_calibration_comparison(
        X,
        y,
        estimator=build_logistic_pipeline(0),
        n_splits=3,
        random_state=0,
        dataset_name="synthetic",
    )
    assert result.dataset == "synthetic"
    for m in result.methods:
        assert 0.0 <= m.accuracy <= 1.0
        assert 0.0 <= m.brier <= 1.0
        assert m.ece >= 0.0


def test_reliability_for_method():
    X, y = make_classification(
        n_samples=100, n_features=6, n_informative=4, random_state=4
    )
    result = run_calibration_comparison(
        X, y, estimator=build_random_forest(0), n_splits=3, random_state=0
    )
    curve = reliability_for_method(result.methods[0], n_bins=5)
    assert curve.n_bins == 5
    assert int(curve.bin_counts.sum()) == len(y)
