"""Walk-forward decisions under a drifting class prior (slice D, part 2).

The deployed model is trained and calibrated once on source rows
(pi_s = 0.3). Target rows then arrive in batches whose prior drifts
(``prior_path``). At batch t each policy picks a prior (or a threshold)
using only what would be available *before* labels of batch t exist, then
decides every row of batch t at costs c_fp = 1, c_fn = 5 and is scored on
batch t's labels:

* ``static`` -- keep pi_s (ignore drift);
* ``em_prev`` -- Saerens EM on the unlabelled features of batch t-1;
* ``em_window`` -- EM on batches t-W..t-1 pooled (W = ``window``);
* ``em_current`` -- EM on batch t's own *unlabelled* features (label-free,
  so usable when decisions are made per batch rather than per row);
* ``bbse_window`` -- BBSE on batches t-W..t-1;
* ``delayed_label_threshold`` -- labels of batch t-1 arrive before batch t;
  re-tune the cost threshold on them (uses labels, one batch late);
* ``oracle_prior`` -- the true pi_t (a ceiling for prior tracking, not a
  method).

The leakage barrier is structural: policies receive batch t's labels only
through ``score`` and the delayed-label policy reads labels < t only; a
test corrupts the labels of batch t and checks that no decision at t moves.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import clone

from model_eval_calibration.calibration_maps import make_calibration_maps
from model_eval_calibration.costs import bayes_threshold, decisions, expected_cost, tune_threshold
from model_eval_calibration.label_shift import sample_label_shift, true_posterior
from model_eval_calibration.prior_estimation import adjust_posterior, bbse_prior, em_prior
from model_eval_calibration.shift_eval import build_base_model

C_FP, C_FN = 1.0, 5.0
SOURCE_PRIOR = 0.3
POLICIES = ("static", "em_prev", "em_window", "em_current", "bbse_window", "delayed_label_threshold", "oracle_prior")


def prior_path(n_batches: int = 24, low: float = 0.06, high: float = 0.5, start: float = SOURCE_PRIOR) -> np.ndarray:
    """Smooth drift: from ``start`` down to ``low``, up to ``high``, back toward ``start`` (one cosine cycle)."""
    t = np.arange(n_batches) / max(n_batches - 1, 1)
    mid, amp = (low + high) / 2, (high - low) / 2
    path = mid - amp * np.sin(2 * np.pi * t + np.arcsin((mid - start) / amp))
    return np.clip(path, 1e-3, 1 - 1e-3)


def make_stream(priors: np.ndarray, batch_size: int, rng: np.random.Generator):
    return [sample_label_shift(batch_size, float(p), rng) for p in priors]


def choose_priors(p_batches: list[np.ndarray], yhat_cal: np.ndarray, y_cal: np.ndarray,
                  yhat_batches: list[np.ndarray], pi_s: float, priors_true: np.ndarray,
                  window: int = 3) -> dict[str, np.ndarray]:
    """Per-batch prior used by each prior-tracking policy (labels of batches are never read)."""
    T = len(p_batches)
    out = {k: np.full(T, pi_s) for k in ("static", "em_prev", "em_window", "em_current", "bbse_window")}
    out["oracle_prior"] = np.asarray(priors_true, dtype=float).copy()
    for t in range(T):
        out["em_current"][t] = em_prior(p_batches[t], pi_s).prior
        if t >= 1:
            out["em_prev"][t] = em_prior(p_batches[t - 1], pi_s).prior
            lo = max(0, t - window)
            out["em_window"][t] = em_prior(np.concatenate(p_batches[lo:t]), pi_s).prior
            try:
                out["bbse_window"][t] = bbse_prior(yhat_cal, y_cal, np.concatenate(yhat_batches[lo:t]))
            except ValueError:
                out["bbse_window"][t] = out["bbse_window"][t - 1]
    return out


def run_walk_forward(seed: int = 0, base_model: str = "logistic", calibration: str = "raw",
                     n_batches: int = 24, batch_size: int = 400, window: int = 3,
                     n_train: int = 2500, n_cal: int = 2000, c_fp: float = C_FP, c_fn: float = C_FN,
                     corrupt_labels_of: int | None = None) -> pd.DataFrame:
    """One seed: per-batch cost and regret of every policy (long format)."""
    rng = np.random.default_rng(seed + 30_000)
    Xtr, ytr = sample_label_shift(n_train, SOURCE_PRIOR, rng)
    Xcal, ycal = sample_label_shift(n_cal, SOURCE_PRIOR, rng)
    priors = prior_path(n_batches)
    stream = make_stream(priors, batch_size, rng)
    labels = [y.copy() for _, y in stream]
    if corrupt_labels_of is not None:  # leakage probe: flip batch t's labels in the policy-visible copy
        labels[corrupt_labels_of] = 1 - labels[corrupt_labels_of]
    pi_s = float(ytr.mean())
    model = clone(build_base_model(base_model, random_state=seed)).fit(Xtr, ytr)
    cmap = make_calibration_maps()[calibration]
    p_cal_raw = model.predict_proba(Xcal)[:, 1]
    cmap.fit(p_cal_raw, ycal)
    p_cal = cmap.transform(p_cal_raw)
    p_b = [cmap.transform(model.predict_proba(X)[:, 1]) for X, _ in stream]
    yhat_b = [(model.predict_proba(X)[:, 1] >= 0.5).astype(int) for X, _ in stream]
    chosen = choose_priors(p_b, (p_cal_raw >= 0.5).astype(int), ycal, yhat_b, pi_s, priors, window)
    t_bayes = bayes_threshold(c_fp, c_fn)
    rows = []
    for t, (X, y_true) in enumerate(stream):
        oracle_cost = expected_cost(y_true, decisions(true_posterior(X, float(priors[t])), t_bayes), c_fp, c_fn)
        dec = {k: decisions(adjust_posterior(p_b[t], pi_s, v[t]), t_bayes) for k, v in chosen.items()}
        thr = tune_threshold(p_cal, ycal, c_fp, c_fn) if t == 0 else tune_threshold(p_b[t - 1], labels[t - 1], c_fp, c_fn)
        dec["delayed_label_threshold"] = decisions(p_b[t], thr)
        for k in POLICIES:
            c = expected_cost(y_true, dec[k], c_fp, c_fn)
            rows.append({"seed": seed, "base_model": base_model, "calibration": calibration, "batch": t,
                         "true_prior": float(priors[t]), "policy": k,
                         "prior_used": float(chosen[k][t]) if k in chosen else np.nan,
                         "cost": c, "regret": c - oracle_cost, "decisions_hash": hashlib.sha1(dec[k].tobytes()).hexdigest()[:16]})
    return pd.DataFrame(rows)


def summarise_walk_forward(long: pd.DataFrame) -> pd.DataFrame:
    """Mean per-row cost / regret over batches, then mean and std over seeds."""
    per_seed = long.groupby(["base_model", "calibration", "policy", "seed"], sort=False)[["cost", "regret"]].mean()
    agg = per_seed.groupby(level=["base_model", "calibration", "policy"], sort=False).agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def _block_resample_mean(d: np.ndarray, block: int, rng: np.random.Generator) -> float:
    """Mean of one moving-block bootstrap resample of a series (circular blocks)."""
    n = d.size
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
    return float(d[idx].mean())


def paired_policy_difference(long: pd.DataFrame, policy_a: str, policy_b: str, block: int = 4,
                             n_boot: int = 2000, alpha: float = 0.05, random_state: int = 0) -> dict:
    """Mean per-row cost difference a - b over batches and seeds, with a moving-block bootstrap CI.

    Batches of one seed share a drift path, so per-batch differences are
    autocorrelated; resampling whole blocks of ``block`` consecutive batches
    within each seed (seeds resampled too) keeps that dependence. ``block=1``
    is the i.i.d. bootstrap. ``prob_a_better`` is the bootstrap share with
    a cheaper than b (descriptive, not a p-value).
    """
    if block < 1 or n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("block >= 1, n_boot >= 1 and 0 < alpha < 1 required")
    piv = long.pivot_table(index=["seed", "batch"], columns="policy", values="cost")
    d = (piv[policy_a] - piv[policy_b]).rename("d").reset_index()
    series = [g.sort_values("batch")["d"].to_numpy() for _, g in d.groupby("seed")]
    rng = np.random.default_rng(random_state)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(series), size=len(series))
        boots[i] = np.mean([_block_resample_mean(series[j], block, rng) for j in pick])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"policy_a": policy_a, "policy_b": policy_b, "estimate": float(np.mean([s.mean() for s in series])),
            "lower": float(lo), "upper": float(hi), "excludes_zero": bool(lo > 0 or hi < 0),
            "prob_a_better": float(np.mean(boots < 0)), "block": block, "n_seeds": len(series)}
