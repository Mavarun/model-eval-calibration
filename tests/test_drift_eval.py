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
