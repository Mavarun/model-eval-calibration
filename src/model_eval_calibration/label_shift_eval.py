"""Slice D harness: calibration, prior correction, costs and conformal coverage under label shift.

One cell = (target prior, seed, base model). Protocol:

* source rows (pi_s = 0.3) -> 2500 train / 2000 cal; target rows (pi_t) ->
  2000 unlabelled pool / 3000 labelled test. Target labels enter **no** fit;
  they are used only to score. Split disjointness is asserted.
* base model fit on train; every map of ``make_calibration_maps`` fit on the
  source cal rows (frozen base probabilities).
* prior estimators on the unlabelled pool: ``none`` (keep pi_s), ``em``
  (Saerens EM on that map's pool probabilities), ``bbse`` (hard base
  predictions + source cal confusion matrix; map-independent), ``oracle``
  (true pi_t, a ceiling, not a method).
* decisions at costs c_fp = 1, c_fn = 5: ``bayes`` = Bayes threshold on the
  prior-adjusted probabilities; also ``tuned_source`` = threshold tuned on
  the source cal rows for that map, applied to unadjusted target scores, and
  ``oracle_posterior`` = Bayes decision on the exact target posterior (the
  floor no method can beat in expectation).
* metrics on target test: |pi_hat - pi_t|, Brier, adaptive ECE, expected
  cost per row, and regret = cost - oracle_posterior cost.
* conformal on raw base scores (alpha = 0.1): pooled split vs Mondrian,
  per-class coverage and mean set size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from model_eval_calibration.calibration_maps import make_calibration_maps
from model_eval_calibration.conformal import (
    class_conditional_coverage,
    mondrian_prediction_sets,
    mondrian_thresholds,
    nonconformity_scores,
    prediction_sets,
    set_diagnostics,
    split_conformal_threshold,
)
from model_eval_calibration.costs import bayes_threshold, decisions, expected_cost, tune_threshold
from model_eval_calibration.label_shift import make_label_shift_data
from model_eval_calibration.metrics import adaptive_expected_calibration_error, brier_score
from model_eval_calibration.prior_estimation import adjust_posterior, bbse_prior, em_prior
from model_eval_calibration.shift_eval import build_base_model

C_FP, C_FN = 1.0, 5.0
SOURCE_PRIOR = 0.3
ESTIMATORS = ("none", "em", "bbse", "oracle")


@dataclass
class LabelShiftSplit:
    train: np.ndarray
    cal: np.ndarray
    pool: np.ndarray
    test: np.ndarray

    def assert_disjoint(self) -> None:
        if set(self.train.tolist()) & set(self.cal.tolist()):
            raise AssertionError("source train and cal overlap")
        if set(self.pool.tolist()) & set(self.test.tolist()):
            raise AssertionError("target pool and test overlap")


def run_label_shift_experiment(
    target_prior: float = 0.1,
    seed: int = 0,
    base_model: str = "logistic",
    n_train: int = 2500,
    n_cal: int = 2000,
    n_pool: int = 2000,
    n_test: int = 3000,
    alpha: float = 0.1,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
) -> dict:
    data = make_label_shift_data(n_source=n_train + n_cal, n_target=n_pool + n_test,
                                 source_prior=SOURCE_PRIOR, target_prior=target_prior, random_state=seed)
    rng = np.random.default_rng(seed + 20_000)
    sp, tp = rng.permutation(n_train + n_cal), rng.permutation(n_pool + n_test)
    split = LabelShiftSplit(sp[:n_train], sp[n_train:], tp[:n_pool], tp[n_pool:])
    split.assert_disjoint()
    Xs, ys, Xt, yt = data.X_source, data.y_source, data.X_target, data.y_target
    X_tr, y_tr, X_cal, y_cal = Xs[split.train], ys[split.train], Xs[split.cal], ys[split.cal]
    X_pool, X_te, y_te = Xt[split.pool], Xt[split.test], yt[split.test]
    pi_s = float(y_tr.mean())  # the prior the model was trained under (estimated from labels we have)

    model = clone(build_base_model(base_model, random_state=seed)).fit(X_tr, y_tr)
    p_cal, p_pool, p_te = (model.predict_proba(X)[:, 1] for X in (X_cal, X_pool, X_te))
    pi_bbse = bbse_prior((p_cal >= 0.5).astype(int), y_cal, (p_pool >= 0.5).astype(int))

    t_bayes = bayes_threshold(c_fp, c_fn)
    oracle_p = data.oracle_target_posterior(X_te)
    oracle_cost = expected_cost(y_te, decisions(oracle_p, t_bayes), c_fp, c_fn)
    rows = []
    for name, cmap in make_calibration_maps().items():
        cmap.fit(p_cal, y_cal)
        q_cal, q_pool, q_te = cmap.transform(p_cal), cmap.transform(p_pool), cmap.transform(p_te)
        priors = {"none": pi_s, "em": em_prior(q_pool, pi_s).prior, "bbse": pi_bbse, "oracle": float(target_prior)}
        for est, pi_hat in priors.items():
            q = adjust_posterior(q_te, pi_s, pi_hat)
            rows.append({
                "calibration": name, "estimator": est, "policy": "bayes",
                "prior_hat": pi_hat, "prior_abs_err": abs(pi_hat - target_prior),
                "brier": brier_score(y_te, q), "ece_adaptive": adaptive_expected_calibration_error(y_te, q),
                "cost": expected_cost(y_te, decisions(q, t_bayes), c_fp, c_fn),
            })
        t_tuned = tune_threshold(q_cal, y_cal, c_fp, c_fn)
        rows.append({
            "calibration": name, "estimator": "none", "policy": "tuned_source",
            "prior_hat": pi_s, "prior_abs_err": abs(pi_s - target_prior),
            "brier": np.nan, "ece_adaptive": np.nan,
            "cost": expected_cost(y_te, decisions(q_te, t_tuned), c_fp, c_fn),
        })
    rows.append({"calibration": "oracle", "estimator": "oracle", "policy": "oracle_posterior",
                 "prior_hat": float(target_prior), "prior_abs_err": 0.0,
                 "brier": brier_score(y_te, oracle_p), "ece_adaptive": adaptive_expected_calibration_error(y_te, oracle_p),
                 "cost": oracle_cost})
    decision = pd.DataFrame(rows)
    decision["regret"] = decision["cost"] - oracle_cost

    s_cal = nonconformity_scores(p_cal, y_cal)
    conf = []
    for method, sets in (("split", prediction_sets(p_te, split_conformal_threshold(s_cal, alpha))),
                         ("mondrian", mondrian_prediction_sets(p_te, mondrian_thresholds(s_cal, y_cal, alpha)))):
        d = set_diagnostics(sets, y_te).as_dict()
        cc = class_conditional_coverage(sets, y_te)
        conf.append({"method": method, "coverage": d["coverage"], "coverage_neg": cc[0], "coverage_pos": cc[1],
                     "mean_set_size": d["mean_set_size"]})
    conformal = pd.DataFrame(conf)
    for f in (decision, conformal):
        f["target_prior"], f["seed"], f["base_model"] = float(target_prior), seed, base_model
    return {"decision": decision, "conformal": conformal, "split": split,
            "diagnostics": {"pi_s_train": pi_s, "pi_t_test": float(y_te.mean()), "oracle_cost": oracle_cost}}


def run_label_shift_grid(target_priors=(0.05, 0.1, 0.3, 0.5), seeds=range(10),
                         base_models=("logistic", "random_forest"), **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    dec, conf = [], []
    for b in base_models:
        for pt in target_priors:
            for s in seeds:
                r = run_label_shift_experiment(target_prior=pt, seed=s, base_model=b, **kw)
                dec.append(r["decision"])
                conf.append(r["conformal"])
    return pd.concat(dec, ignore_index=True), pd.concat(conf, ignore_index=True)
