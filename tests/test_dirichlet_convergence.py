"""The ODIR fit's default stop is short of the optimum on digits, and where it stops depends on the platform.

Found when slice J's CI run on Python 3.11 recomputed slice H's dev probe 0.002-0.028 NLL away from
the recorded values. Every Dirichlet number in slices H-J comes from the default stop.
"""

import numpy as np
from scipy.optimize import minimize

import model_eval_calibration.multiclass as M
from model_eval_calibration.multiclass import DirichletScaling, multiclass_nll
from model_eval_calibration.multiclass_eval import _renorm, load, make_model, split


def _tight(*a, **k):
    k = dict(k)
    k["options"] = {"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10}
    return minimize(*a, **k)


def test_converged_digits_fit_is_reproducible_and_the_default_stop_is_not(monkeypatch):
    x, y = load("digits", 100)
    x_tr, y_tr, x_cal, y_cal, x_te, y_te = split(x, y, 100)
    m = make_model("logistic", 100).fit(x_tr, y_tr)
    pc, pt = _renorm(m.predict_proba(x_cal)), _renorm(m.predict_proba(x_te))
    default = DirichletScaling(lam=100.0).fit(pc, y_cal)
    monkeypatch.setattr(M, "minimize", _tight)
    tight = DirichletScaling(lam=100.0).fit(pc, y_cal)
    cal_default, cal_tight = multiclass_nll(y_cal, default.transform(pc)), multiclass_nll(y_cal, tight.transform(pc))
    te_default, te_tight = multiclass_nll(y_te, default.transform(pt)), multiclass_nll(y_te, tight.transform(pt))
    assert cal_tight <= cal_default + 1e-9  # closer to the optimum on the rows it is fitted to
    # The converged fit gives 0.15659 locally and 0.15659 on the CI 3.11 runner; the default stop gave 0.1492
    # locally and 0.1594 on that runner, so the sign of (converged - default) on test NLL is not stable.
    assert abs(te_tight - 0.15659) < 2e-4
    assert abs(te_tight - te_default) < 0.012
    assert not np.allclose(default.W_, tight.W_, atol=1e-3)


def test_converge_option_matches_the_patched_tight_fit(monkeypatch):
    x, y = load("digits", 100)
    x_tr, y_tr, x_cal, y_cal, x_te, y_te = split(x, y, 100)
    m = make_model("logistic", 100).fit(x_tr, y_tr)
    pc, pt = _renorm(m.predict_proba(x_cal)), _renorm(m.predict_proba(x_te))
    opt = DirichletScaling(lam=100.0, converge=True).fit(pc, y_cal)
    assert abs(multiclass_nll(y_te, opt.transform(pt)) - 0.15659) < 2e-4
    monkeypatch.setattr(M, "minimize", _tight)
    patched = DirichletScaling(lam=100.0).fit(pc, y_cal)
    assert np.allclose(opt.W_, patched.W_, atol=1e-8)


def test_converge_defaults_off():
    assert DirichletScaling().converge is False
