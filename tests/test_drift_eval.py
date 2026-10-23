import numpy as np
import pytest

from model_eval_calibration.drift_eval import POLICIES, prior_path, run_walk_forward, summarise_walk_forward


def test_prior_path_starts_at_source_and_spans_the_band():
    p = prior_path(24)
    assert abs(p[0] - 0.3) < 1e-9 and p.min() == pytest.approx(0.06, abs=0.01) and p.max() == pytest.approx(0.5, abs=0.01)


@pytest.fixture(scope="module")
def run():
    return run_walk_forward(seed=0, n_batches=12, batch_size=300)


def test_every_policy_scored_on_every_batch(run):
    assert set(run["policy"]) == set(POLICIES)
    assert run.groupby("policy")["batch"].nunique().eq(12).all()


def test_no_decision_at_t_reads_batch_t_labels(run):
    t = 5
    probe = run_walk_forward(seed=0, n_batches=12, batch_size=300, corrupt_labels_of=t)
    a = run[run.batch == t].set_index("policy")["decisions_hash"]
    b = probe[probe.batch == t].set_index("policy")["decisions_hash"]
    assert (a == b).all()  # labels of batch t never reach a decision at t
    # ...while the delayed-label policy at t+1 does read them (one batch late)
    a1 = run[run.batch == t + 1].set_index("policy")["decisions_hash"]
    b1 = probe[probe.batch == t + 1].set_index("policy")["decisions_hash"]
    assert a1["delayed_label_threshold"] != b1["delayed_label_threshold"]
    assert (a1.drop("delayed_label_threshold") == b1.drop("delayed_label_threshold")).all()


def test_tracking_the_prior_beats_ignoring_drift():
    long = run_walk_forward(seed=1, n_batches=24, batch_size=400)
    s = summarise_walk_forward(long).set_index("policy")["cost_mean"]
    assert s["oracle_prior"] < s["static"]
    assert s["em_window"] < s["static"] and s["em_current"] < s["static"]


def test_prior_used_follows_the_drift_and_the_window_lags_it():
    long = run_walk_forward(seed=2, n_batches=24, batch_size=400)

    def corr(policy, lag):
        w = long[long.policy == policy]
        return np.corrcoef(w["prior_used"].iloc[3:], w["true_prior"].shift(lag).iloc[3:])[0, 1]

    assert corr("em_current", 0) > 0.95
    # a W=3 window of past batches is centred two batches back: it tracks the lagged prior best
    assert corr("em_window", 2) > 0.98 and corr("em_window", 2) > corr("em_window", 0)


def _synthetic_long(diffs_by_seed):
    rows = []
    for s, d in enumerate(diffs_by_seed):
        for t, v in enumerate(d):
            rows += [{"seed": s, "batch": t, "policy": "a", "cost": 1.0 + v},
                     {"seed": s, "batch": t, "policy": "b", "cost": 1.0}]
    import pandas as pd
    return pd.DataFrame(rows)


def test_paired_difference_zero_when_policies_agree_and_detects_a_clear_gap():
    from model_eval_calibration.drift_eval import paired_policy_difference

    same = paired_policy_difference(_synthetic_long([np.zeros(24)] * 5), "a", "b", n_boot=200)
    assert same["estimate"] == 0 and same["lower"] == same["upper"] == 0 and not same["excludes_zero"]
    long = run_walk_forward(seed=3, n_batches=24, batch_size=400)
    r = paired_policy_difference(long, "oracle_prior", "static", n_boot=300)
    assert r["estimate"] < 0 and r["excludes_zero"] and r["prob_a_better"] > 0.95


def test_block_bootstrap_is_wider_than_iid_on_autocorrelated_differences():
    from model_eval_calibration.drift_eval import paired_policy_difference

    rng = np.random.default_rng(0)
    series = []
    for _ in range(6):
        e, x = rng.standard_normal(48), np.zeros(48)
        for t in range(1, 48):
            x[t] = 0.9 * x[t - 1] + e[t]  # AR(1), strongly autocorrelated
        series.append(0.1 * x)
    long = _synthetic_long(series)
    iid = paired_policy_difference(long, "a", "b", block=1, n_boot=1000)
    blk = paired_policy_difference(long, "a", "b", block=8, n_boot=1000)
    assert (blk["upper"] - blk["lower"]) > 1.3 * (iid["upper"] - iid["lower"])
    with pytest.raises(ValueError):
        paired_policy_difference(long, "a", "b", block=0)
