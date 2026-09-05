"""Model evaluation beyond accuracy: Brier, ECE, reliability, calibration."""

from model_eval_calibration.metrics import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
]

__version__ = "0.1.0"
