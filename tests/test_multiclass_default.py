import json
from pathlib import Path

import numpy as np
import pytest

from model_eval_calibration.multiclass import (
    RECOMMENDED_MULTICLASS_MAP, MulticlassTemperature, VectorScaling, fit_multiclass_map,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_follows_the_recorded_decision():
    rec = json.loads((ROOT / "results" / "slice_g_decisions.json").read_text())
    assert RECOMMENDED_MULTICLASS_MAP == rec["recommended"] == "temperature"


def test_fit_multiclass_map_dispatch():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), size=500)
    y = rng.integers(0, 3, size=500)
    assert isinstance(fit_multiclass_map(p, y), MulticlassTemperature)
    assert isinstance(fit_multiclass_map(p, y, "vector"), VectorScaling)
    np.testing.assert_array_equal(fit_multiclass_map(p, y, "none").transform(p), p)
    with pytest.raises(ValueError):
        fit_multiclass_map(p, y, "isotonic")
