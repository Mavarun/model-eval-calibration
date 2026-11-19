"""Post-hoc slice-G reading (decides nothing): fitted temperatures and base-model saturation."""

from pathlib import Path

import pandas as pd

from model_eval_calibration.multiclass_eval import posthoc_saturation

ROOT = Path(__file__).resolve().parents[1]


def _t():
    g = pd.read_csv(ROOT / "results" / "slice_g_grid.csv")
    return g[g["map"] == "temperature"].groupby(["dataset", "model"])["T"].mean()


def test_forests_are_under_confident_and_naive_bayes_over_confident():
    t = _t()
    assert t[("digits", "random_forest")] < 0.3 and t[("synth5", "random_forest")] < 0.7   # T < 1 sharpens
    assert t[("digits", "naive_bayes")] > 5 and t[("synth5", "naive_bayes")] > 1.4          # T > 1 softens


def test_digits_naive_bayes_is_saturated_which_is_where_temperature_fails_classwise():
    s = pd.read_csv(ROOT / "results" / "slice_g_saturation.csv").groupby(["dataset", "model"]).share_saturated.mean()
    assert s[("digits", "naive_bayes")] > 0.85
    assert s.drop(("digits", "naive_bayes")).max() < 0.25


def test_saturation_recomputes_for_one_seed():
    got = posthoc_saturation(seeds=(2,), datasets=("digits",), models=("naive_bayes",)).share_saturated.iloc[0]
    ref = pd.read_csv(ROOT / "results" / "slice_g_saturation.csv")
    ref = ref[(ref.seed == 2) & (ref.dataset == "digits") & (ref.model == "naive_bayes")].share_saturated.iloc[0]
    assert abs(got - ref) < 1e-6
