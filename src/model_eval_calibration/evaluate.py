"""Eval harness: raw vs Platt vs isotonic on held-out folds.

Compares accuracy, Brier, and ECE so miscalibration is visible even when
accuracy looks fine. Research demo only — not a production accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_eval_calibration.calibration import ProbabilityCalibrator
from model_eval_calibration.metrics import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)


@dataclass
class MethodResult:
    name: str
    accuracy: float
    roc_auc: float
    brier: float
    ece: float
    y_true: np.ndarray = field(repr=False)
    y_prob: np.ndarray = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "brier": self.brier,
            "ece": self.ece,
            "n_test": int(self.y_true.size),
        }


@dataclass
class SliceResult:
    dataset: str
    base_estimator: str
    n_splits: int
    methods: list[MethodResult]

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame([m.as_dict() for m in self.methods])

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "base_estimator": self.base_estimator,
            "n_splits": self.n_splits,
            "methods": [m.as_dict() for m in self.methods],
        }


def build_logistic_pipeline(random_state: int = 0) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, random_state=random_state),
            ),
        ]
    )


def build_random_forest(random_state: int = 0) -> RandomForestClassifier:
    """Uncalibrated trees often produce overconfident leaf probabilities."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        random_state=random_state,
    )


def _score_probs(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> MethodResult:
    pred = (y_prob >= 0.5).astype(int)
    return MethodResult(
        name=name,
        accuracy=float(accuracy_score(y_true, pred)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        brier=float(brier_score(y_true, y_prob)),
        ece=float(expected_calibration_error(y_true, y_prob)),
        y_true=y_true,
        y_prob=y_prob,
    )


def run_calibration_comparison(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Optional[Any] = None,
    n_splits: int = 5,
    random_state: int = 0,
    dataset_name: str = "breast_cancer",
    ece_bins: int = 10,
) -> SliceResult:
    """Out-of-fold compare raw / Platt / isotonic probabilities.

    For each outer StratifiedKFold split:
      - fit base estimator on train
      - fit Platt (sigmoid) and isotonic calibrators on train only
      - score all three on the held-out fold

    OOF predictions are concatenated so metrics reflect true held-out
    performance. Calibration is never fit on test rows.
    """
    del ece_bins  # reserved; metrics default to 10 bins
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if estimator is None:
        estimator = build_random_forest(random_state=random_state)
        base_name = "RandomForestClassifier"
    else:
        base_name = type(estimator).__name__
        if hasattr(estimator, "steps"):
            base_name = "Pipeline(StandardScaler+LogisticRegression)"

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "raw": [],
        "platt": [],
        "isotonic": [],
    }

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        raw = clone(estimator)
        raw.fit(X_tr, y_tr)
        raw_prob = raw.predict_proba(X_te)[:, 1]
        buckets["raw"].append((y_te, raw_prob))

        platt = ProbabilityCalibrator(
            estimator=clone(estimator),
            method="sigmoid",
            n_splits=3,
            random_state=random_state,
        )
        platt.fit(X_tr, y_tr)
        buckets["platt"].append((y_te, platt.predict_proba(X_te)[:, 1]))

        iso = ProbabilityCalibrator(
            estimator=clone(estimator),
            method="isotonic",
            n_splits=3,
            random_state=random_state,
        )
        iso.fit(X_tr, y_tr)
        buckets["isotonic"].append((y_te, iso.predict_proba(X_te)[:, 1]))

    methods: list[MethodResult] = []
    labels = {
        "raw": "raw",
        "platt": "platt_sigmoid",
        "isotonic": "isotonic",
    }
    for key, label in labels.items():
        y_all = np.concatenate([p[0] for p in buckets[key]])
        p_all = np.concatenate([p[1] for p in buckets[key]])
        methods.append(_score_probs(y_all, p_all, label))

    return SliceResult(
        dataset=dataset_name,
        base_estimator=base_name,
        n_splits=n_splits,
        methods=methods,
    )


def reliability_for_method(result: MethodResult, n_bins: int = 10):
    """Convenience: reliability curve for one MethodResult."""
    return reliability_curve(result.y_true, result.y_prob, n_bins=n_bins)
