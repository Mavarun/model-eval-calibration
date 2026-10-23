import numpy as np
import pytest

from model_eval_calibration.conformal import (
    class_conditional_coverage,
    mondrian_prediction_sets,
    mondrian_thresholds,
    nonconformity_scores,
    prediction_sets,
    split_conformal_threshold,
)
from model_eval_calibration.label_shift import sample_label_shift, true_posterior


def test_mondrian_thresholds_are_per_class_split_quantiles():
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.05, 0.6, 0.7, 0.8])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    q0, q1 = mondrian_thresholds(s, y, alpha=0.2)
    assert q0 == split_conformal_threshold(s[:4], 0.2) and q1 == split_conformal_threshold(s[4:], 0.2)
    sets = mondrian_prediction_sets(np.array([0.5]), (0.4, 0.6))
    assert sets.tolist() == [[False, True]]


def test_class_conditional_coverage_survives_label_shift_where_split_conformal_does_not():
    """Averaged over 150 calibration draws; the score model is fixed and source-calibrated."""
    alpha, n_cal = 0.1, 400
    rng = np.random.default_rng(7)
    Xt, yt = sample_label_shift(20000, 0.1, rng)
    pt = true_posterior(Xt, 0.3)  # source posterior applied to target rows
    cov_split, cov_mond = [], []
    for _ in range(150):
        Xc, yc = sample_label_shift(n_cal, 0.3, rng)
        sc = nonconformity_scores(true_posterior(Xc, 0.3), yc)
        cov_split.append(class_conditional_coverage(prediction_sets(pt, split_conformal_threshold(sc, alpha)), yt))
        cov_mond.append(class_conditional_coverage(mondrian_prediction_sets(pt, mondrian_thresholds(sc, yc, alpha)), yt))
    for c in (0, 1):
        assert np.mean([d[c] for d in cov_mond]) >= 1 - alpha - 0.01
    # plain split conformal under-covers the minority class (pooled quantile is set by the majority)
    assert np.mean([d[1] for d in cov_split]) < 1 - alpha - 0.03


def test_mondrian_input_checks():
    with pytest.raises(ValueError):
        mondrian_thresholds(np.array([0.1, 0.2]), np.array([0, 0]), 0.1)
    with pytest.raises(ValueError):
        mondrian_thresholds(np.array([0.1]), np.array([0, 1]), 0.1)
