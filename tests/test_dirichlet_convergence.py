"""The ODIR fit's default stop is short of the optimum on digits; converging only raises test NLL.

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


def test_converged_digits_fit_has_lower_train_loss_and_higher_test_nll(monkeypatch):
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
    assert te_tight > te_default  # and worse out of sample: the default stop acts as early stopping
    assert te_tight - te_default < 0.01  # 0.149 -> 0.157 on this seed; +0.0024 on the mean over seeds 100-102
    assert not np.allclose(default.W_, tight.W_, atol=1e-3)
