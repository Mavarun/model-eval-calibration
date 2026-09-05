"""Probability calibration wrappers (Platt / isotonic) for research demos.

Fit calibrators on held-out folds only. Never claim that calibration always
improves ECE — measure it on a proper held-out split.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


CalibrationMethod = Literal["sigmoid", "isotonic"]


class ProbabilityCalibrator(ClassifierMixin, BaseEstimator):
    """Wrap an estimator with sklearn CalibratedClassifierCV.

    ``method='sigmoid'`` is Platt scaling; ``method='isotonic'`` is
    isotonic regression. Inner CV uses StratifiedKFold on the *fit*
    sample only — callers must not pass outer test rows into ``fit``.
    """

    def __init__(
        self,
        estimator: Any = None,
        method: CalibrationMethod = "sigmoid",
        n_splits: int = 3,
        random_state: int = 0,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        if self.estimator is None:
            raise ValueError("estimator is required")
        y = np.asarray(y).ravel()
        n_splits = min(self.n_splits, int(np.min(np.bincount(y.astype(int)))))
        if n_splits < 2:
            raise ValueError("need >=2 samples per class to calibrate with CV")
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        self.calibrated_ = CalibratedClassifierCV(
            estimator=clone(self.estimator),
            method=self.method,
            cv=cv,
        )
        self.calibrated_.fit(X, y)
        self.classes_ = self.calibrated_.classes_
        return self

    def predict_proba(self, X):
        return self.calibrated_.predict_proba(X)

    def predict(self, X):
        return self.calibrated_.predict(X)


def fit_calibrator(
    estimator: Any,
    X_train,
    y_train,
    method: CalibrationMethod = "sigmoid",
    n_splits: int = 3,
    random_state: int = 0,
) -> ProbabilityCalibrator:
    """Fit a ProbabilityCalibrator and return it."""
    cal = ProbabilityCalibrator(
        estimator=estimator,
        method=method,
        n_splits=n_splits,
        random_state=random_state,
    )
    cal.fit(X_train, y_train)
    return cal
