#!/usr/bin/env python3
"""Run the Brier / ECE / reliability calibration slice.

Hypothesis (research demo, not production claims):
1. Accuracy alone hides miscalibration.
2. Brier + ECE + reliability diagrams diagnose it.
3. Platt / isotonic on held-out folds can improve ECE when base probs
   are bad — measure, don't assume.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_eval_calibration.data import load_binary_classification
from model_eval_calibration.evaluate import (
    build_logistic_pipeline,
    build_random_forest,
    reliability_for_method,
    run_calibration_comparison,
)
from model_eval_calibration.metrics import plot_reliability_diagram


def _print_table(result) -> None:
    frame = result.summary_frame()
    print(f"\nBase estimator: {result.base_estimator}")
    print(f"Dataset: {result.dataset} | outer StratifiedKFold={result.n_splits}")
    print(
        frame.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
            columns=["name", "accuracy", "roc_auc", "brier", "ece"],
        )
    )


def main() -> None:
    X, y, _ = load_binary_classification(random_state=0)
    n_splits = 5

    rf_result = run_calibration_comparison(
        X,
        y,
        estimator=build_random_forest(random_state=0),
        n_splits=n_splits,
        random_state=0,
        dataset_name="sklearn.datasets.load_breast_cancer",
    )
    log_result = run_calibration_comparison(
        X,
        y,
        estimator=build_logistic_pipeline(random_state=0),
        n_splits=n_splits,
        random_state=0,
        dataset_name="sklearn.datasets.load_breast_cancer",
    )

    payload = {
        "hypothesis": [
            "Accuracy alone hides miscalibration.",
            "Brier + ECE + reliability diagnose it.",
            "Platt/isotonic on held-out folds can improve ECE when base probs are bad — measure, don't assume.",
        ],
        "random_forest": rf_result.as_dict(),
        "logistic": log_result.as_dict(),
    }
    print(json.dumps(payload, indent=2))
    _print_table(rf_result)
    _print_table(log_result)

    # Optional reliability figure for the typically-miscalibrated RF raw probs
    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, method in zip(axes, rf_result.methods):
            curve = reliability_for_method(method, n_bins=10)
            plot_reliability_diagram(
                curve, ax=ax, title=f"RF {method.name}"
            )
        fig.tight_layout()
        out = art / "reliability_rf_breast_cancer.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\nWrote reliability diagram: {out}")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"\nSkipping reliability plot ({exc})")

    print(
        "\nNote: metrics are OOF under StratifiedKFold on a public toy set. "
        "Not clinical performance, not a production accuracy claim."
    )


if __name__ == "__main__":
    main()
