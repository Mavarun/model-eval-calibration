import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from model_eval_calibration.calibration import ProbabilityCalibrator, fit_calibrator
from model_eval_calibration.metrics import brier_score, expected_calibration_error


def test_platt_fit_predict_proba_shape():
    X, y = make_classification(
        n_samples=120, n_features=8, n_informative=5, random_state=0
    )
    cal = fit_calibrator(
        LogisticRegression(max_iter=1000), X, y, method="sigmoid", n_splits=3
    )
    proba = cal.predict_proba(X)
    assert proba.shape == (120, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_isotonic_fit_predict():
    X, y = make_classification(
        n_samples=120, n_features=8, n_informative=5, random_state=1
    )
    cal = ProbabilityCalibrator(
        estimator=RandomForestClassifier(n_estimators=20, random_state=0),
        method="isotonic",
        n_splits=3,
        random_state=0,
    )
    cal.fit(X, y)
    pred = cal.predict(X[:10])
    assert pred.shape == (10,)


def test_calibrator_requires_estimator():
    with pytest.raises(ValueError, match="estimator"):
        ProbabilityCalibrator().fit(np.zeros((10, 2)), np.zeros(10, dtype=int))


def test_calibration_can_change_ece_on_overconfident_forest():
    """Sanity: wrapping RF in calibration yields finite ECE/Brier on holdout."""
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=2,
    )
    split = 280
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    raw = RandomForestClassifier(n_estimators=50, random_state=0)
    raw.fit(X_tr, y_tr)
    raw_p = raw.predict_proba(X_te)[:, 1]

    cal = fit_calibrator(
        RandomForestClassifier(n_estimators=50, random_state=0),
        X_tr,
        y_tr,
        method="sigmoid",
        n_splits=3,
    )
    cal_p = cal.predict_proba(X_te)[:, 1]

    raw_ece = expected_calibration_error(y_te, raw_p)
    cal_ece = expected_calibration_error(y_te, cal_p)
    raw_brier = brier_score(y_te, raw_p)
    cal_brier = brier_score(y_te, cal_p)

    assert np.isfinite(raw_ece) and np.isfinite(cal_ece)
    assert np.isfinite(raw_brier) and np.isfinite(cal_brier)
    # Do not assert cal always wins — measure-only contract.
