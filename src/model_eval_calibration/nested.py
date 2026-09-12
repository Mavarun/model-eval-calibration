"""Nested CV calibration evaluator vs a deliberately optimistic naive path.

Hypothesis: fitting Platt/isotonic on the same fold used for ECE metrics
produces optimistic (too-low) ECE. Nested CV -- inner fold for calibrator
fit, outer fold for scoring -- reduces that optimism.

Barrier rule: outer-test rows must never enter calibrator ``fit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from model_eval_calibration.metrics import (
    adaptive_expected_calibration_error,
    brier_score,
    expected_calibration_error,
)


CalibrationMethod = Literal["sigmoid", "isotonic"]
PathName = Literal["naive_same_fold", "nested"]


@dataclass
class NestedMethodResult:
    path: PathName
    method: str  # raw | platt_sigmoid | isotonic
    accuracy: float
    roc_auc: float
    brier: float
    ece_equal_width: float
    ece_adaptive: float
    y_true: np.ndarray = field(repr=False)
    y_prob: np.ndarray = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "method": self.method,
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "brier": self.brier,
            "ece_equal_width": self.ece_equal_width,
            "ece_adaptive": self.ece_adaptive,
            "n_test": int(self.y_true.size),
        }


@dataclass
class NestedSliceResult:
    dataset: str
    base_estimator: str
    n_outer: int
    n_inner: int
    methods: list[NestedMethodResult]

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame([m.as_dict() for m in self.methods])

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "base_estimator": self.base_estimator,
            "n_outer": self.n_outer,
            "n_inner": self.n_inner,
            "methods": [m.as_dict() for m in self.methods],
        }


def _score(
    path: PathName,
    method: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> NestedMethodResult:
    pred = (y_prob >= 0.5).astype(int)
    return NestedMethodResult(
        path=path,
        method=method,
        accuracy=float(accuracy_score(y_true, pred)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        brier=float(brier_score(y_true, y_prob)),
        ece_equal_width=float(
            expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy="equal_width")
        ),
        ece_adaptive=float(
            adaptive_expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        ),
        y_true=y_true,
        y_prob=y_prob,
    )


def _fit_calibrator_prefit(
    estimator: Any,
    X_fit,
    y_fit,
    X_cal,
    y_cal,
    method: CalibrationMethod,
) -> CalibratedClassifierCV:
    """Fit base on ``fit`` split, then calibrate on ``cal`` split only.

    sklearn>=1.6 removed ``cv='prefit'``; wrap the fitted base in
    ``FrozenEstimator`` so CalibratedClassifierCV only learns the map
    on the calibration split.
    """
    base = clone(estimator)
    base.fit(X_fit, y_fit)
    cal = CalibratedClassifierCV(estimator=FrozenEstimator(base), method=method)
    cal.fit(X_cal, y_cal)
    return cal


def run_naive_same_fold_calibration(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    n_splits: int = 5,
    random_state: int = 0,
    dataset_name: str = "hard_binary",
    n_bins: int = 10,
) -> list[NestedMethodResult]:
    """Optimistic path: calibrate on the same outer-test fold used for metrics.

    For each outer fold:
      1. Fit base estimator on train.
      2. Fit Platt/isotonic on **test** rows (FrozenEstimator) -- leakage.
      3. Score ECE on that same test fold.

    This intentionally violates the hold-out barrier to quantify optimism.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "raw": [],
        "platt_sigmoid": [],
        "isotonic": [],
    }

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        raw = clone(estimator)
        raw.fit(X_tr, y_tr)
        raw_p = raw.predict_proba(X_te)[:, 1]
        buckets["raw"].append((y_te, raw_p))

        # LEAKAGE: calibrator sees outer-test labels/features
        for method, key in (("sigmoid", "platt_sigmoid"), ("isotonic", "isotonic")):
            base = clone(estimator)
            base.fit(X_tr, y_tr)
            cal = CalibratedClassifierCV(
                estimator=FrozenEstimator(base), method=method
            )
            cal.fit(X_te, y_te)  # barrier violation (deliberate)
            buckets[key].append((y_te, cal.predict_proba(X_te)[:, 1]))

    out: list[NestedMethodResult] = []
    for key in ("raw", "platt_sigmoid", "isotonic"):
        y_all = np.concatenate([p[0] for p in buckets[key]])
        p_all = np.concatenate([p[1] for p in buckets[key]])
        out.append(_score("naive_same_fold", key, y_all, p_all, n_bins=n_bins))
    return out


def run_nested_calibration(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    n_outer: int = 5,
    n_inner: int = 3,
    random_state: int = 0,
    dataset_name: str = "hard_binary",
    n_bins: int = 10,
    inner_frac: float = 0.25,
) -> list[NestedMethodResult]:
    """Proper nested path: inner hold-out for calibrator, outer for score.

    For each outer fold:
      - Fit base on full outer-train; score raw on outer-test.
      - Split outer-train into fit/cal (stratified); fit base on fit,
        calibrate on cal only; score on outer-test.
      - Outer-test never enters calibrator ``fit`` (barrier).

    ``n_inner`` is retained for API symmetry / documentation; the default
    implementation uses a single stratified cal split of size
    ``inner_frac`` of the outer-train (faster, clearer barrier). When
    ``n_inner >= 2``, we instead average calibrators from an inner
    StratifiedKFold (still never touching outer-test).
    """
    del dataset_name  # carried by NestedSliceResult wrapper
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "raw": [],
        "platt_sigmoid": [],
        "isotonic": [],
    }

    for fold_i, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Barrier assertions for callers / tests
        assert set(train_idx).isdisjoint(set(test_idx))

        raw = clone(estimator)
        raw.fit(X_tr, y_tr)
        buckets["raw"].append((y_te, raw.predict_proba(X_te)[:, 1]))

        for method, key in (("sigmoid", "platt_sigmoid"), ("isotonic", "isotonic")):
            if n_inner >= 2:
                # Average calibrated probs from inner CV folds on outer-train
                inner = StratifiedKFold(
                    n_splits=min(
                        n_inner,
                        int(np.min(np.bincount(y_tr.astype(int)))),
                    ),
                    shuffle=True,
                    random_state=random_state + fold_i,
                )
                # Collect per-outer-test predictions averaged across inner models
                pred_sum = np.zeros(len(test_idx), dtype=float)
                n_models = 0
                for fit_idx, cal_idx in inner.split(X_tr, y_tr):
                    # cal_idx is a subset of outer-train only -- never outer-test
                    cal = _fit_calibrator_prefit(
                        estimator,
                        X_tr[fit_idx],
                        y_tr[fit_idx],
                        X_tr[cal_idx],
                        y_tr[cal_idx],
                        method=method,
                    )
                    pred_sum += cal.predict_proba(X_te)[:, 1]
                    n_models += 1
                buckets[key].append((y_te, pred_sum / max(n_models, 1)))
            else:
                X_fit, X_cal, y_fit, y_cal = train_test_split(
                    X_tr,
                    y_tr,
                    test_size=inner_frac,
                    stratify=y_tr,
                    random_state=random_state + fold_i,
                )
                cal = _fit_calibrator_prefit(
                    estimator, X_fit, y_fit, X_cal, y_cal, method=method
                )
                buckets[key].append((y_te, cal.predict_proba(X_te)[:, 1]))

    out: list[NestedMethodResult] = []
    for key in ("raw", "platt_sigmoid", "isotonic"):
        y_all = np.concatenate([p[0] for p in buckets[key]])
        p_all = np.concatenate([p[1] for p in buckets[key]])
        out.append(_score("nested", key, y_all, p_all, n_bins=n_bins))
    return out


def run_naive_vs_nested_comparison(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Optional[Any] = None,
    n_outer: int = 5,
    n_inner: int = 3,
    random_state: int = 0,
    dataset_name: str = "hard_binary",
    n_bins: int = 10,
) -> NestedSliceResult:
    """Compare naive same-fold calibration vs nested CV on one dataset."""
    from model_eval_calibration.evaluate import build_random_forest

    if estimator is None:
        estimator = build_random_forest(random_state=random_state)
        base_name = "RandomForestClassifier"
    else:
        base_name = type(estimator).__name__
        if hasattr(estimator, "steps"):
            base_name = "Pipeline(StandardScaler+LogisticRegression)"

    naive = run_naive_same_fold_calibration(
        X,
        y,
        estimator=estimator,
        n_splits=n_outer,
        random_state=random_state,
        dataset_name=dataset_name,
        n_bins=n_bins,
    )
    nested = run_nested_calibration(
        X,
        y,
        estimator=estimator,
        n_outer=n_outer,
        n_inner=n_inner,
        random_state=random_state,
        dataset_name=dataset_name,
        n_bins=n_bins,
    )
    return NestedSliceResult(
        dataset=dataset_name,
        base_estimator=base_name,
        n_outer=n_outer,
        n_inner=n_inner,
        methods=naive + nested,
    )


def assert_nested_barrier(
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    calibrator_fit_idx: np.ndarray,
) -> None:
    """Raise if any outer-test index appears in the calibrator fit set."""
    outer_train = set(int(i) for i in np.asarray(outer_train_idx).ravel())
    outer_test = set(int(i) for i in np.asarray(outer_test_idx).ravel())
    cal_fit = set(int(i) for i in np.asarray(calibrator_fit_idx).ravel())
    if not cal_fit.issubset(outer_train):
        raise AssertionError("calibrator fit indices must be subset of outer-train")
    leaked = cal_fit & outer_test
    if leaked:
        raise AssertionError(
            f"nested barrier violated: {len(leaked)} outer-test indices in calibrator fit"
        )
