"""Public toy / synthetic datasets for calibration research demos."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification


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
    # sklearn breast_cancer: 0=malignant, 1=benign -- keep as-is
    rng = np.random.default_rng(random_state)
    order = rng.permutation(X.shape[0])
    X = X[order]
    y = y[order]
    names = list(bundle.feature_names)
    return X, y, names


def load_hard_binary_classification(
    random_state: int = 0,
    n_samples: int = 1200,
    n_features: int = 24,
    n_informative: int = 6,
    n_redundant: int = 4,
    n_repeated: int = 0,
    weights: tuple[float, float] = (0.72, 0.28),
    flip_y: float = 0.08,
    class_sep: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Harder binary task than breast_cancer for calibration stress tests.

    Uses ``sklearn.datasets.make_classification`` with:
      - class imbalance (default ~72/28),
      - label noise (``flip_y``),
      - modest ``class_sep``,
      - many non-informative / redundant features.

    Documented as a **synthetic** stand-in for noisy tabular credit/adult-style
    problems (no network fetch; reproducible offline). Not a production
    accuracy claim and not real financial/demographic data.

    Returns
    -------
    X, y, meta
        ``meta`` records generation hyperparameters for the README/script.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=2,
        n_clusters_per_class=2,
        weights=list(weights),
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=random_state,
    )
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).ravel()
    rng = np.random.default_rng(random_state)
    order = rng.permutation(X.shape[0])
    X = X[order]
    y = y[order]
    meta = {
        "name": "make_classification_hard_binary",
        "description": (
            "Synthetic imbalanced binary task with label noise; harder "
            "probability quality than sklearn breast_cancer. Offline stand-in "
            "for noisy credit/adult-style tabular problems."
        ),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_informative": int(n_informative),
        "n_redundant": int(n_redundant),
        "weights": list(weights),
        "flip_y": float(flip_y),
        "class_sep": float(class_sep),
        "positive_rate": float(y.mean()),
        "random_state": int(random_state),
    }
    return X, y, meta
