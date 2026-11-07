"""Reduced-scale re-check of slice F's directional claims (3 seeds, logistic)."""

import numpy as np
import pytest

from model_eval_calibration.costs import bayes_threshold, decisions
from model_eval_calibration.drift_eval import C_FN, C_FP
from model_eval_calibration.online_prior import cusum_em_path, online_em_path
from model_eval_calibration.prior_estimation import adjust_posterior
from model_eval_calibration.stream_eval import detection_stats, fit_source, make_target_stream

SEEDS = (0, 1, 2)


def _cost(y, d):
    return (C_FP * ((d == 1) & (y == 0)) + C_FN * ((d == 0) & (y == 1))).mean()


@pytest.fixture(scope="module")
def runs():
    t = bayes_threshold(C_FP, C_FN)
    out = {}
    for s in SEEDS:
        model, cmap, _, _, pi_s = fit_source(s, "logistic")
        for kind in ("regime", "small_regime"):
            priors, cps, X, y = make_target_stream(s, kind)
            p = cmap.transform(model.predict_proba(X)[:, 1])
            path, alarms = cusum_em_path(p, pi_s, threshold=4.0, delta_logit=0.8)
            out[(s, kind)] = {
                "detected": detection_stats(alarms, cps, p.size)["detected"],
                "static": _cost(y, decisions(p, t)),
                "online": _cost(y, decisions(adjust_posterior(p, pi_s, online_em_path(p, pi_s, 0.02)), t)),
                "oracle": _cost(y, decisions(adjust_posterior(p, pi_s, priors), t)),
            }
    return out


def test_slice_e_cusum_detects_fewer_small_jumps(runs):
    big = np.mean([runs[(s, "regime")]["detected"] for s in SEEDS])
    small = np.mean([runs[(s, "small_regime")]["detected"] for s in SEEDS])
    assert small < big


def test_tracking_gain_over_static_shrinks_on_small_jumps(runs):
    gain = {k: np.mean([runs[(s, k)]["static"] - runs[(s, k)]["online"] for s in SEEDS])
            for k in ("regime", "small_regime")}
    assert gain["regime"] > 0 and gain["small_regime"] < gain["regime"] / 2


def test_static_is_close_to_the_oracle_only_on_small_jumps(runs):
    gap = {k: np.mean([runs[(s, k)]["static"] - runs[(s, k)]["oracle"] for s in SEEDS])
           for k in ("regime", "small_regime")}
    assert gap["small_regime"] < gap["regime"] / 2
