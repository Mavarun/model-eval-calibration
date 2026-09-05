"""Public toy datasets for calibration research demos."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer


def load_binary_classification(
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Load sklearn breast_cancer as a binary classification matrix.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,) with labels in {0, 1}
    feature_names : list of feature name strings

    This is a public research toy set, not clinical data and not a
    production accuracy claim.
    """
    bundle = load_breast_cancer()
    X = np.asarray(bundle.data, dtype=float)
    y = np.asarray(bundle.target, dtype=int).ravel()
    # sklearn breast_cancer: 0=malignant, 1=benign — keep as-is
    rng = np.random.default_rng(random_state)
    order = rng.permutation(X.shape[0])
    X = X[order]
    y = y[order]
    names = list(bundle.feature_names)
    return X, y, names
