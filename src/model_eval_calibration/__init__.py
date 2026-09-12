"""Model evaluation beyond accuracy: Brier, ECE, reliability, calibration."""

from model_eval_calibration.metrics import (
    adaptive_expected_calibration_error,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "adaptive_expected_calibration_error",
    "reliability_curve",
]

__version__ = "0.2.0"
