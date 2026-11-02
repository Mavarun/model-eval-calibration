import numpy as np
import pytest

from model_eval_calibration.stream_eval import (
    CHUNK,
    fit_source,
    make_target_stream,
    run_stream,
    stream_decisions,
    summarise_stream,
)

N = 3000


@pytest.fixture(scope="module")
def regime_run():
    return run_stream(seed=0, path_kind="regime", n_rows=N)


def test_every_policy_scored_on_every_chunk(regime_run):
    pol = regime_run.groupby("policy")["batch"].nunique()
    assert len(pol) == 10 and (pol == N // CHUNK).all()
    s = summarise_stream(regime_run).set_index("policy")
    assert s.loc["oracle_prior", "regret_mean"] < s.loc["static", "regret_mean"]


def test_labels_reach_only_the_delayed_label_policy_and_only_after_the_delay():
    model, cmap, p_cal, y_cal, pi_s = fit_source(1, "logistic")
    priors, _, X, y = make_target_stream(1, "regime", N)
    p = cmap.transform(model.predict_proba(X)[:, 1])
    r, delay = 1500, 400
    vis = y.copy()
    vis[r:] = 1 - vis[r:]
    a, _ = stream_decisions(p, y, pi_s, priors, p_cal, y_cal, label_delay=delay)
    b, _ = stream_decisions(p, vis, pi_s, priors, p_cal, y_cal, label_delay=delay)
    for k in a:
        assert np.array_equal(a[k][: r + delay], b[k][: r + delay]), k  # nothing reads labels earlier
        if k != "delayed_label_threshold":
            assert np.array_equal(a[k], b[k]), k  # label-free policies never read labels at all
    assert not np.array_equal(a["delayed_label_threshold"], b["delayed_label_threshold"])


def test_run_stream_leakage_probe_moves_only_the_label_policy(regime_run):
    probe = run_stream(seed=0, path_kind="regime", n_rows=N, corrupt_labels_from=2000)
    h0 = regime_run.groupby("policy")["decisions_hash"].first()
    h1 = probe.groupby("policy")["decisions_hash"].first()
    moved = set(h0.index[h0 != h1])
    assert moved == {"delayed_label_threshold"}


def test_post_change_rows_are_flagged_only_on_the_regime_path(regime_run):
    smooth = run_stream(seed=0, path_kind="smooth", n_rows=N)
    assert smooth["post_change_share"].eq(0).all()
    _, cps, _, _ = make_target_stream(0, "regime", N)
    expect = np.zeros(N, dtype=bool)
    for c in cps:
        expect[c : c + 400] = True
    got = regime_run[regime_run.policy == "static"]["post_change_share"].to_numpy()
    assert np.allclose(got, expect.reshape(-1, CHUNK).mean(1))
    with pytest.raises(ValueError):
        make_target_stream(0, "sawtooth")


def test_streams_are_reproducible_per_seed():
    a = make_target_stream(3, "regime", 1000)
    b = make_target_stream(3, "regime", 1000)
    c = make_target_stream(4, "regime", 1000)
    assert all(np.array_equal(u, v) for u, v in zip(a, b)) and not np.array_equal(a[3], c[3])
