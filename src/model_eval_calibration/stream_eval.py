"""Slice E: per-row decisions under smooth and abrupt prior drift.

Same world as slice D (label shift, c_fp = 1, c_fn = 5, model trained and
calibrated once on source rows at pi_s = 0.3), but rows are decided one at a
time and the prior can jump. One stream = 9600 target rows (the volume of
slice D's 24 x 400 batches) on either the smooth cosine path or an abrupt
regime-switching path (``drift_eval.regime_row_path``).

Policies (``stream_decisions``):

* ``static`` -- keep pi_s;
* ``batch{B}_em_current`` for B in ``batch_sizes`` -- slice D's best policy:
  EM on the whole B-row batch's unlabelled features, so every row of a batch
  waits until the batch is complete (not available for per-row decisions;
  kept as the batch-latency reference);
* ``batch400_em_prev`` -- EM on the previous 400-row batch (per-row usable);
* ``online_em``, ``window_em``, ``cusum_em`` -- ``online_prior`` trackers;
  row i uses unlabelled rows < i only;
* ``delayed_label_threshold`` -- labels arrive ``label_delay`` rows late; the
  cost threshold is re-tuned every 25 rows on the latest ``label_window``
  labelled rows (source calibration rows before enough labels exist);
* ``oracle_prior`` -- the true per-row prior (a ceiling, not a method).

Regret is cost minus the exact-posterior cost on the same rows. Results are
aggregated into 100-row chunks (column ``batch``) so slice D's paired
block bootstrap can be reused unchanged.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import clone

from model_eval_calibration.calibration_maps import make_calibration_maps
from model_eval_calibration.costs import bayes_threshold, decisions, tune_threshold
from model_eval_calibration.drift_eval import (
    C_FN,
    C_FP,
    SOURCE_PRIOR,
    regime_row_path,
    sample_row_stream,
    smooth_row_path,
)
from model_eval_calibration.label_shift import log_likelihood_ratio, sample_label_shift
from model_eval_calibration.online_prior import cusum_em_path, em_on, online_em_path, window_em_path
from model_eval_calibration.prior_estimation import adjust_posterior
from model_eval_calibration.shift_eval import build_base_model

N_ROWS = 9600
CHUNK = 100
POST_CHANGE_ROWS = 400
DEFAULT_TRACKERS = {"online_step": 0.01, "window": 400, "cusum_threshold": 6.0}


def make_target_stream(seed: int, path_kind: str, n_rows: int = N_ROWS):
    """Per-row priors, change points (empty for smooth), features and labels for one target stream."""
    rng = np.random.default_rng(seed + 50_000)
    if path_kind == "smooth":
        priors, cps = smooth_row_path(n_rows), np.array([], dtype=int)
    elif path_kind == "regime":
        priors, cps = regime_row_path(n_rows, rng)
    else:
        raise ValueError(f"unknown path kind {path_kind!r}")
    X, y = sample_row_stream(priors, rng)
    return priors, cps, X, y


def fit_source(seed: int, base_model: str, calibration: str = "raw", n_train: int = 2500, n_cal: int = 2000):
    rng = np.random.default_rng(seed + 40_000)
    Xtr, ytr = sample_label_shift(n_train, SOURCE_PRIOR, rng)
    Xcal, ycal = sample_label_shift(n_cal, SOURCE_PRIOR, rng)
    model = clone(build_base_model(base_model, random_state=seed)).fit(Xtr, ytr)
    cmap = make_calibration_maps()[calibration]
    cmap.fit(model.predict_proba(Xcal)[:, 1], ycal)
    p_cal = cmap.transform(model.predict_proba(Xcal)[:, 1])
    return model, cmap, p_cal, ycal, float(ytr.mean())


def _batch_em(p: np.ndarray, pi_s: float, size: int, shift: int) -> np.ndarray:
    """Per-row prior from EM on batch k (shift=0, waits for the batch) or batch k-1 (shift=1)."""
    out = np.full(p.size, pi_s)
    starts = np.arange(0, p.size, size)
    est = [em_on(p[s : s + size], pi_s, pi_s) for s in starts]
    for k, s in enumerate(starts):
        if k - shift >= 0:
            out[s : s + size] = est[k - shift]
    return out


def _delayed_threshold(p: np.ndarray, labels: np.ndarray, p_cal: np.ndarray, y_cal: np.ndarray,
                       delay: int, window: int, every: int = 25) -> np.ndarray:
    thr = np.empty(p.size)
    current = tune_threshold(p_cal, y_cal, C_FP, C_FN)
    for i in range(p.size):
        if i % every == 0:
            hi = i - delay  # labels of rows < hi have arrived before row i
            if hi >= window:
                current = tune_threshold(p[hi - window : hi], labels[hi - window : hi], C_FP, C_FN)
        thr[i] = current
    return thr


def stream_decisions(p: np.ndarray, labels_visible: np.ndarray, pi_s: float, priors_true: np.ndarray,
                     p_cal: np.ndarray, y_cal: np.ndarray, trackers: dict | None = None,
                     batch_sizes: tuple[int, ...] = (400, 100, 25), label_delay: int = 400,
                     label_window: int = 400) -> tuple[dict, dict]:
    """Decisions of every policy for every row; also the prior each prior-tracking policy used."""
    tr = {**DEFAULT_TRACKERS, **(trackers or {})}
    priors = {"static": np.full(p.size, pi_s)}
    for b in batch_sizes:
        priors[f"batch{b}_em_current"] = _batch_em(p, pi_s, b, 0)
    priors["batch400_em_prev"] = _batch_em(p, pi_s, 400, 1)
    priors["online_em"] = online_em_path(p, pi_s, tr["online_step"])
    priors["window_em"] = window_em_path(p, pi_s, int(tr["window"]), refit_every=10)
    priors["cusum_em"] = cusum_em_path(p, pi_s, threshold=tr["cusum_threshold"])[0]
    priors["oracle_prior"] = np.asarray(priors_true, dtype=float)
    t_bayes = bayes_threshold(C_FP, C_FN)
    dec = {k: decisions(adjust_posterior(p, pi_s, v), t_bayes) for k, v in priors.items()}
    thr = _delayed_threshold(p, labels_visible, p_cal, y_cal, label_delay, label_window)
    dec["delayed_label_threshold"] = (p >= thr).astype(int)
    return dec, priors


def _row_cost(y: np.ndarray, d: np.ndarray) -> np.ndarray:
    return C_FP * ((d == 1) & (y == 0)) + C_FN * ((d == 0) & (y == 1))


def run_stream(seed: int = 0, base_model: str = "logistic", path_kind: str = "regime", calibration: str = "raw",
               trackers: dict | None = None, n_rows: int = N_ROWS, corrupt_labels_from: int | None = None,
               **kw) -> pd.DataFrame:
    """One seed and path: per-chunk mean cost / regret of each policy (long format)."""
    model, cmap, p_cal, y_cal, pi_s = fit_source(seed, base_model, calibration)
    priors_true, cps, X, y = make_target_stream(seed, path_kind, n_rows)
    p = cmap.transform(model.predict_proba(X)[:, 1])
    vis = y.copy()
    if corrupt_labels_from is not None:  # leakage probe on the policy-visible label copy
        vis[corrupt_labels_from:] = 1 - vis[corrupt_labels_from:]
    dec, used = stream_decisions(p, vis, pi_s, priors_true, p_cal, y_cal, trackers, **kw)
    floor = _row_cost(y, decisions(_oracle_post(X, priors_true), bayes_threshold(C_FP, C_FN)))
    post = np.zeros(n_rows, dtype=bool)
    for c in cps:
        post[c : c + POST_CHANGE_ROWS] = True
    chunk = np.arange(n_rows) // CHUNK
    rows = []
    for k, d in dec.items():
        c = _row_cost(y, d)
        df = pd.DataFrame({"batch": chunk, "cost": c, "regret": c - floor, "post": post,
                           "prior_used": used.get(k, np.full(n_rows, np.nan)), "true_prior": priors_true})
        g = df.groupby("batch")
        agg = g[["cost", "regret", "prior_used", "true_prior"]].mean()
        agg["post_change_share"] = g["post"].mean()
        agg["post_change_regret_sum"] = df[df.post].groupby("batch")["regret"].sum().reindex(agg.index, fill_value=0.0)
        agg = agg.reset_index().assign(seed=seed, base_model=base_model, path=path_kind, calibration=calibration,
                                       policy=k, decisions_hash=hashlib.sha1(d.tobytes()).hexdigest()[:16])
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def _oracle_post(X: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Exact per-row posterior under each row's own prior."""
    lg = np.log(priors / (1.0 - priors))
    return 1.0 / (1.0 + np.exp(-(log_likelihood_ratio(X) + lg)))


def summarise_stream(long: pd.DataFrame) -> pd.DataFrame:
    """Per-row mean cost / regret over the stream, then mean and std over seeds; plus post-change regret per row."""
    per_seed = long.groupby(["base_model", "path", "policy", "seed"], sort=False).apply(
        lambda g: pd.Series({"cost": g["cost"].mean(), "regret": g["regret"].mean(),
                             "post_change_regret": (g["post_change_regret_sum"].sum()
                                                    / max((g["post_change_share"] * CHUNK).sum(), 1.0))}),
        include_groups=False)
    agg = per_seed.groupby(level=["base_model", "path", "policy"], sort=False).agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()
