# model-eval-calibration

Research slice: **model evaluation beyond accuracy** — **Brier score**, **Expected Calibration Error (ECE)**, and **reliability diagrams**, plus a fair comparison of **Platt (sigmoid)** vs **isotonic** calibration on held-out folds.

Public toy data only (`sklearn.datasets.load_breast_cancer`). Methodology demo, **not** a clinical or production accuracy claim.

## Hypothesis

1. **Accuracy alone hides miscalibration.** A model can be mostly correct while still being over-/under-confident.
2. **Brier + ECE + reliability diagrams diagnose it.** Brier is a proper scoring rule; ECE and reliability curves show *where* confidence diverges from frequency.
3. **Platt / isotonic on held-out folds can improve ECE when base probabilities are bad — measure, don't assume.** Calibration is fit only inside train folds; we report whether it helps on this dataset.

## Method

- **Data**: shuffled `load_breast_cancer` binary labels (public research set).
- **Base models**:
  - `RandomForestClassifier` — leaf probabilities are often overconfident (good stress test for calibration).
  - `Pipeline(StandardScaler → LogisticRegression)` — usually better calibrated already.
- **Protocol**: outer `StratifiedKFold` (k=5). For each fold, fit raw / Platt / isotonic on **train only**, score on the held-out fold, concatenate OOF predictions.
- **Metrics**: accuracy, ROC-AUC, Brier, equal-width ECE (10 bins), reliability diagram coordinates (optional PNG under `artifacts/`).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/run_calibration_slice.py
```

## Results (this slice, StratifiedKFold k=5, n=569)

### RandomForestClassifier

| Method | accuracy | ROC-AUC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| raw | 0.956 | 0.990 | 0.0335 | 0.0284 |
| platt_sigmoid | 0.949 | 0.989 | **0.0319** | 0.0310 |
| isotonic | 0.953 | 0.989 | 0.0327 | **0.0098** |

Accuracy stays ~0.95 across methods while ECE moves a lot — accuracy alone would have hidden that. Isotonic cut ECE sharply on RF; Platt slightly helped Brier but not ECE.

### LogisticRegression pipeline

| Method | accuracy | ROC-AUC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| raw | 0.975 | 0.996 | **0.0195** | **0.0146** |
| platt_sigmoid | 0.972 | 0.996 | 0.0232 | 0.0448 |
| isotonic | 0.977 | 0.991 | 0.0201 | 0.0192 |

Raw logistic was already well calibrated. Extra calibration **hurt** ECE/Brier here — a real negative result for hypothesis (3) on this base model, which is exactly why we measure.

## Assumptions and limits

- Equal-width ECE is binning- and sample-size-sensitive; treat it as a diagnostic beside Brier.
- Isotonic calibration can overfit small calibration folds; Platt is parametric and stabler on tiny data.
- Breast cancer is an easy public set — do not read metrics as domain performance.
- No nested hyperparameter search, no production monitoring, no accuracy claim.

## Layout

```
src/model_eval_calibration/
  metrics.py        # Brier, ECE, reliability curve + optional plot
  calibration.py    # Platt / isotonic wrappers (CalibratedClassifierCV)
  evaluate.py       # OOF raw vs calibrated comparison harness
  data.py           # breast_cancer loader
scripts/run_calibration_slice.py
tests/
```

## Next slices (not done)

- Adaptive / quantile binning for ECE
- Multiclass calibration and classwise reliability
- Temperature scaling for neural nets
- OpenML datasets with harder probability quality
