#!/usr/bin/env python3
"""Run adaptive-ECE + nested-CV calibration research slice.

Hypothesis (research demo, not production claims):
1. Equal-width ECE is unstable on imbalanced probs; adaptive/quantile
   binning gives a more stable calibration read.
2. Nested CV (inner fold for Platt/isotonic, outer for score) reduces
   optimistic ECE estimates vs fitting calibration on the same fold
   used for metrics.
3. On a harder binary task than breast_cancer, nested calibrated ECE
   will differ from the naive path -- report honestly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_eval_calibration.data import load_hard_binary_classification
from model_eval_calibration.evaluate import build_logistic_pipeline, build_random_forest
from model_eval_calibration.nested import run_naive_vs_nested_comparison


def _print_table(result) -> None:
    frame = result.summary_frame()
    print(f"\nBase estimator: {result.base_estimator}")
    print(
        f"Dataset: {result.dataset} | outer={result.n_outer} inner={result.n_inner}"
    )
    cols = [
        "path",
        "method",
        "accuracy",
        "roc_auc",
        "brier",
        "ece_equal_width",
        "ece_adaptive",
    ]
    print(
        frame.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
            columns=cols,
        )
    )


def main() -> None:
    X, y, meta = load_hard_binary_classification(random_state=0)
    n_outer, n_inner = 5, 3

    rf_result = run_naive_vs_nested_comparison(
        X,
        y,
        estimator=build_random_forest(random_state=0),
        n_outer=n_outer,
        n_inner=n_inner,
        random_state=0,
        dataset_name=meta["name"],
        n_bins=10,
    )
    log_result = run_naive_vs_nested_comparison(
        X,
        y,
        estimator=build_logistic_pipeline(random_state=0),
        n_outer=n_outer,
        n_inner=n_inner,
        random_state=0,
        dataset_name=meta["name"],
        n_bins=10,
    )

    payload = {
        "hypothesis": [
            "Equal-width ECE is unstable on imbalanced probs; adaptive/quantile binning gives a more stable calibration read.",
            "Nested CV (inner fold for Platt/isotonic, outer for score) reduces optimistic ECE estimates vs fitting calibration on the same fold used for metrics.",
            "On a harder binary task than breast_cancer, nested calibrated ECE will differ from the naive path -- report honestly.",
        ],
        "dataset": meta,
        "random_forest": rf_result.as_dict(),
        "logistic": log_result.as_dict(),
    }
    print(json.dumps(payload, indent=2))
    _print_table(rf_result)
    _print_table(log_result)

    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    out_json = art / "nested_calibration_slice_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote metrics JSON: {out_json}")
    print(
        "\nNote: synthetic hard binary via make_classification. "
        "Not a production accuracy claim."
    )


if __name__ == "__main__":
    main()
