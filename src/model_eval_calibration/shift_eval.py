"""Calibration and conformal coverage under covariate shift: experiment harness.

Protocol (one seed, one shift level)
------------------------------------
1. Draw labelled source rows and a separate target sample from
   :func:`make_covariate_shift_data` (same P(y|x), shifted P(x)).
2. Split source into disjoint ``train`` / ``cal`` / ``source_test`` rows.
   Split target into an *unlabelled* ``target_pool`` (features only, used
   to fit the domain classifier) and a labelled ``target_test`` used only
   for scoring.
3. Fit the base model on ``train``. Fit every calibration map on ``cal``
   twice: unweighted, and importance-weighted with estimated w(x) from the
   domain classifier (``iw_`` prefix).
4. Score every map on ``source_test`` and ``target_test``: Brier,
   equal-width ECE, adaptive ECE, MCE (bins with >= 10 rows), with
   percentile bootstrap CIs on adaptive ECE and Brier.
5. Conformal (LAC, alpha): unweighted split conformal on the raw base
   probabilities; on the target also weighted conformal with the estimated
   ratio and with the oracle analytic ratio.

Caveat on the ECE intervals: binned ECE is a biased (upward) estimator
and that bias grows when rows are resampled with replacement, so the
percentile bootstrap interval for ECE can sit entirely above the point
estimate. Read ``ece_adaptive_lo/hi`` as resampling spread, not as a
coverage-calibrated interval for the true calibration error.

Barriers: target labels never enter any fit; ``cal`` rows never enter the
base model; ``source_test`` / ``target_test`` never enter any fit. Target
*features* are used for the ratio (that is the point of unsupervised
shift adaptation) and for per-point weighted conformal thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from model_eval_calibration.bootstrap import bootstrap_metric_ci
from model_eval_calibration.calibration_maps import make_calibration_maps
from model_eval_calibration.conformal import (
    nonconformity_scores,
    prediction_sets,
    set_diagnostics,
    split_conformal_threshold,
    weighted_conformal_thresholds,
)
from model_eval_calibration.metrics import (
    adaptive_expected_calibration_error,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from model_eval_calibration.shift import (
    DomainClassifierRatio,
    effective_sample_size,
    make_covariate_shift_data,
)


def build_base_model(name: str, random_state: int = 0):
    if name == "logistic":
        return LogisticRegression(max_iter=2000)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=1, random_state=random_state, n_jobs=1
        )
    raise ValueError(f"unknown base model {name!r}")


@dataclass
class ShiftSplitIndices:
    train: np.ndarray
    cal: np.ndarray
    source_test: np.ndarray
    target_pool: np.ndarray
    target_test: np.ndarray

    def assert_disjoint(self) -> None:
        src = [set(self.train.tolist()), set(self.cal.tolist()), set(self.source_test.tolist())]
        for i in range(3):
            for j in range(i + 1, 3):
                if src[i] & src[j]:
                    raise AssertionError("source splits overlap")
        if set(self.target_pool.tolist()) & set(self.target_test.tolist()):
            raise AssertionError("target pool and target test overlap")


@dataclass
class ShiftRunResult:
    shift: float
    seed: int
    base_model: str
    calibration: pd.DataFrame
    conformal: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)
    splits: Optional[ShiftSplitIndices] = None


def _score_block(domain, method, y, p, n_boot, seed):
    ece_ci = bootstrap_metric_ci(
        y, p, adaptive_expected_calibration_error, n_boot=n_boot, random_state=seed
    )
    brier_ci = bootstrap_metric_ci(y, p, brier_score, n_boot=n_boot, random_state=seed)
    return {
        "domain": domain,
        "method": method,
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": brier_ci.estimate,
        "brier_lo": brier_ci.lower,
        "brier_hi": brier_ci.upper,
        "ece_equal_width": float(expected_calibration_error(y, p)),
        "ece_adaptive": ece_ci.estimate,
        "ece_adaptive_lo": ece_ci.lower,
        "ece_adaptive_hi": ece_ci.upper,
        "mce": float(maximum_calibration_error(y, p, strategy="quantile", min_bin_count=10)),
        "n": int(y.size),
    }


def run_shift_experiment(
    shift: float = 1.5,
    seed: int = 0,
    base_model: str = "logistic",
    n_train: int = 2000,
    n_cal: int = 1000,
    n_source_test: int = 1000,
    n_target_pool: int = 1000,
    n_target_test: int = 2000,
    alpha: float = 0.1,
    n_boot: int = 200,
    ratio_degree: int = 1,
    ratio_clip_quantile: Optional[float] = 0.99,
) -> ShiftRunResult:
    """Run one (shift, seed, base_model) cell of the experiment."""
    n_src = n_train + n_cal + n_source_test
    n_tgt = n_target_pool + n_target_test
    data = make_covariate_shift_data(
        n_source=n_src, n_target=n_tgt, shift=shift, random_state=seed
    )
    rng = np.random.default_rng(seed + 10_000)
    src_perm, tgt_perm = rng.permutation(n_src), rng.permutation(n_tgt)
    idx = ShiftSplitIndices(
        train=src_perm[:n_train],
        cal=src_perm[n_train : n_train + n_cal],
        source_test=src_perm[n_train + n_cal :],
        target_pool=tgt_perm[:n_target_pool],
        target_test=tgt_perm[n_target_pool:],
    )
    idx.assert_disjoint()

    Xs, ys, Xt, yt = data.X_source, data.y_source, data.X_target, data.y_target
    X_tr, y_tr = Xs[idx.train], ys[idx.train]
    X_cal, y_cal = Xs[idx.cal], ys[idx.cal]
    X_st, y_st = Xs[idx.source_test], ys[idx.source_test]
    X_pool = Xt[idx.target_pool]  # features only
    X_tt, y_tt = Xt[idx.target_test], yt[idx.target_test]

    model = clone(build_base_model(base_model, random_state=seed)).fit(X_tr, y_tr)
    p_cal = model.predict_proba(X_cal)[:, 1]
    p_st = model.predict_proba(X_st)[:, 1]
    p_tt = model.predict_proba(X_tt)[:, 1]

    ratio = DomainClassifierRatio(
        degree=ratio_degree, clip_quantile=ratio_clip_quantile, random_state=seed
    ).fit(X_tr, X_pool)
    w_cal = ratio.weights(X_cal)
    w_tt = ratio.weights(X_tt)

    rows = []
    for weighted in (False, True):
        for name, cal_map in make_calibration_maps().items():
            if weighted and name == "raw":
                continue
            cal_map.fit(p_cal, y_cal, sample_weight=w_cal if weighted else None)
            label = f"iw_{name}" if weighted else name
            rows.append(_score_block("source", label, y_st, cal_map.transform(p_st), n_boot, seed))
            rows.append(_score_block("target", label, y_tt, cal_map.transform(p_tt), n_boot, seed))
    calib = pd.DataFrame(rows)

    s_cal = nonconformity_scores(p_cal, y_cal)
    q = split_conformal_threshold(s_cal, alpha)
    conf_rows = []
    for domain, p, y in (("source", p_st, y_st), ("target", p_tt, y_tt)):
        d = set_diagnostics(prediction_sets(p, q), y).as_dict()
        conf_rows.append({"domain": domain, "method": "split", **d})
    q_est = weighted_conformal_thresholds(s_cal, w_cal, w_tt, alpha)
    conf_rows.append(
        {"domain": "target", "method": "weighted_estimated",
         **set_diagnostics(prediction_sets(p_tt, q_est), y_tt).as_dict()}
    )
    q_orc = weighted_conformal_thresholds(
        s_cal, data.true_density_ratio(X_cal), data.true_density_ratio(X_tt), alpha
    )
    conf_rows.append(
        {"domain": "target", "method": "weighted_oracle",
         **set_diagnostics(prediction_sets(p_tt, q_orc), y_tt).as_dict()}
    )
    conformal = pd.DataFrame(conf_rows)
    conformal["alpha"] = alpha

    diagnostics = {
        "ess_cal_estimated": effective_sample_size(w_cal),
        "ess_cal_oracle": effective_sample_size(data.true_density_ratio(X_cal)),
        "n_cal": int(n_cal),
        "positive_rate_source_test": float(y_st.mean()),
        "positive_rate_target_test": float(y_tt.mean()),
        "ratio_clip_value": ratio.clip_value_,
    }
    return ShiftRunResult(shift, seed, base_model, calib, conformal, diagnostics, idx)


def run_shift_grid(
    shifts: Iterable[float] = (0.0, 1.0, 2.0),
    seeds: Iterable[int] = range(5),
    base_models: Iterable[str] = ("logistic", "random_forest"),
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat :func:`run_shift_experiment` and return long-format frames.

    Returns (calibration_long, conformal_long) with ``shift``, ``seed`` and
    ``base_model`` columns; aggregate with :func:`summarise_grid`.
    """
    cal_frames, conf_frames = [], []
    for base in base_models:
        for shift in shifts:
            for seed in seeds:
                r = run_shift_experiment(shift=shift, seed=seed, base_model=base, **kwargs)
                for frame, bucket in ((r.calibration, cal_frames), (r.conformal, conf_frames)):
                    f = frame.copy()
                    f["shift"], f["seed"], f["base_model"] = shift, seed, base
                    f["ess_cal_estimated"] = r.diagnostics["ess_cal_estimated"]
                    bucket.append(f)
    return pd.concat(cal_frames, ignore_index=True), pd.concat(conf_frames, ignore_index=True)


def summarise_grid(long: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean and std across seeds per (base_model, shift, domain, method)."""
    keys = ["base_model", "shift", "domain", "method"]
    agg = long.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def win_rate_vs(
    long: pd.DataFrame, metric: str, method: str, baseline: str, domain: str = "target"
) -> pd.DataFrame:
    """Share of seeds where ``method`` has a lower ``metric`` than ``baseline``."""
    sub = long[long["domain"] == domain]
    piv = sub.pivot_table(
        index=["base_model", "shift", "seed"], columns="method", values=metric
    )
    wins = (piv[method] < piv[baseline]).groupby(level=["base_model", "shift"]).mean()
    return wins.rename(f"win_rate_{method}_vs_{baseline}").reset_index()
