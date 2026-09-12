# model-eval-calibration

Research slices on **model evaluation beyond accuracy**:

1. **Slice A (main)**: Brier, equal-width ECE, reliability diagrams; Platt vs isotonic on `breast_cancer`.
2. **Slice B (this PR)**: **adaptive / quantile ECE** + **nested CV calibration** on a harder synthetic binary task.

Public / synthetic data only. Methodology demos -- **not** production accuracy claims.

---

## Slice B hypothesis

1. Equal-width ECE is unstable on imbalanced probs; adaptive/quantile binning gives a more stable calibration read.
2. Nested CV (inner fold for Platt/isotonic, outer for score) reduces optimistic ECE estimates vs fitting calibration on the same fold used for metrics.
3. On a harder binary task than breast_cancer (e.g. sklearn credit/adult subsample or make_classification with noise), nested calibrated ECE will differ from the naive path -- report honestly.

## Slice B method

- **Data**: `load_hard_binary_classification` -- `sklearn.datasets.make_classification` with class imbalance (~72/28), `flip_y=0.08`, modest `class_sep`, redundant features. Offline synthetic stand-in for noisy credit/adult-style tabular problems (no network fetch).
- **Base models**: `RandomForestClassifier` and `Pipeline(StandardScaler -> LogisticRegression)`.
- **Paths**:
  - **naive_same_fold**: fit base on outer-train, fit calibrator on **outer-test** (`FrozenEstimator` + `CalibratedClassifierCV`), score on that same outer-test (deliberate leakage / optimism).
  - **nested**: outer `StratifiedKFold` for scoring; within outer-train, inner CV fit/cal splits for Platt/isotonic; **outer-test never enters calibrator fit** (barrier).
- **Metrics**: accuracy, ROC-AUC, Brier, equal-width ECE (10 bins), adaptive/quantile ECE (10 bins).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/run_calibration_slice.py          # slice A (breast_cancer)
python scripts/run_nested_calibration_slice.py   # slice B (hard binary + nested)
```

## Slice B results (make_classification_hard_binary, n=1200, outer=5, inner=3)

### RandomForestClassifier -- naive vs nested

| path | method | accuracy | ROC-AUC | Brier | ECE equal-width | ECE adaptive |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| naive_same_fold | raw | 0.8608 | 0.8982 | 0.1141 | 0.0762 | 0.0730 |
| naive_same_fold | platt_sigmoid | 0.8542 | 0.8990 | 0.1054 | 0.0185 | 0.0188 |
| naive_same_fold | isotonic | 0.8683 | 0.9166 | 0.0962 | **0.0000** | **0.0000** |
| nested | raw | 0.8608 | 0.8982 | 0.1141 | 0.0762 | 0.0730 |
| nested | platt_sigmoid | 0.8542 | 0.8947 | 0.1091 | 0.0288 | 0.0270 |
| nested | isotonic | 0.8492 | 0.8910 | 0.1086 | **0.0225** | **0.0201** |

Naive isotonic ECE collapses to ~0 (fit+score on the same fold). Nested isotonic ECE stays ~0.02 -- the optimistic gap the hypothesis predicted.

### LogisticRegression pipeline -- naive vs nested

| path | method | accuracy | ROC-AUC | Brier | ECE equal-width | ECE adaptive |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| naive_same_fold | raw | 0.7600 | 0.8157 | 0.1568 | 0.0385 | 0.0465 |
| naive_same_fold | platt_sigmoid | 0.7700 | 0.8210 | 0.1546 | 0.0330 | 0.0404 |
| naive_same_fold | isotonic | 0.7867 | 0.8472 | 0.1411 | **0.0000** | **0.0000** |
| nested | raw | 0.7600 | 0.8157 | 0.1568 | 0.0385 | 0.0465 |
| nested | platt_sigmoid | 0.7625 | 0.8157 | 0.1569 | 0.0543 | 0.0517 |
| nested | isotonic | 0.7633 | 0.8171 | 0.1541 | **0.0345** | **0.0312** |

Same pattern: naive isotonic looks perfectly calibrated; nested reads a non-zero residual ECE. Equal-width vs adaptive disagree slightly on raw logistic (0.0385 vs 0.0465) under imbalanced probs -- adaptive is the stabler diagnostic here.

---

## Slice A hypothesis (prior)

1. **Accuracy alone hides miscalibration.** A model can be mostly correct while still being over-/under-confident.
2. **Brier + ECE + reliability diagrams diagnose it.** Brier is a proper scoring rule; ECE and reliability curves show *where* confidence diverges from frequency.
3. **Platt / isotonic on held-out folds can improve ECE when base probabilities are bad -- measure, don't assume.** Calibration is fit only inside train folds; we report whether it helps on this dataset.

## Slice A method

- **Data**: shuffled `load_breast_cancer` binary labels (public research set).
- **Base models**:
  - `RandomForestClassifier` -- leaf probabilities are often overconfident (good stress test for calibration).
  - `Pipeline(StandardScaler -> LogisticRegression)` -- usually better calibrated already.
- **Protocol**: outer `StratifiedKFold` (k=5). For each fold, fit raw / Platt / isotonic on **train only**, score on the held-out fold, concatenate OOF predictions.
- **Metrics**: accuracy, ROC-AUC, Brier, equal-width ECE (10 bins), reliability diagram coordinates (optional PNG under `artifacts/`).

## Slice A results (StratifiedKFold k=5, n=569)

### RandomForestClassifier

| Method | accuracy | ROC-AUC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| raw | 0.956 | 0.990 | 0.0335 | 0.0284 |
| platt_sigmoid | 0.949 | 0.989 | **0.0319** | 0.0310 |
| isotonic | 0.953 | 0.989 | 0.0327 | **0.0098** |

Accuracy stays ~0.95 across methods while ECE moves a lot -- accuracy alone would have hidden that. Isotonic cut ECE sharply on RF; Platt slightly helped Brier but not ECE.

### LogisticRegression pipeline

| Method | accuracy | ROC-AUC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| raw | 0.975 | 0.996 | **0.0195** | **0.0146** |
| platt_sigmoid | 0.972 | 0.996 | 0.0232 | 0.0448 |
| isotonic | 0.977 | 0.991 | 0.0201 | 0.0192 |

Raw logistic was already well calibrated. Extra calibration **hurt** ECE/Brier here -- a real negative result for hypothesis (3) on this base model, which is exactly why we measure.

## Assumptions and limits / weaknesses

- Equal-width ECE is binning- and sample-size-sensitive; adaptive/quantile ECE helps under skewed probs but still depends on `n_bins` and collapses duplicate quantile edges.
- Naive same-fold calibration is an intentionally broken baseline; real pipelines should never fit calibrators on the evaluation fold.
- Nested CV is honest but higher variance / cost; small inner cal folds can make isotonic unstable.
- Hard binary is **synthetic** (`make_classification`), not real credit/adult rows -- treat metrics as a stress test, not domain performance.
- Isotonic can overfit small calibration folds; Platt is parametric and stabler on tiny data.
- Breast cancer is an easy public set -- do not read slice A metrics as clinical performance.
- No nested hyperparameter search, no production monitoring, no accuracy claim.

## Layout

```
src/model_eval_calibration/
  metrics.py        # Brier, equal-width + adaptive ECE, reliability
  calibration.py    # Platt / isotonic wrappers (CalibratedClassifierCV)
  nested.py         # naive same-fold vs nested CV evaluator + barrier helper
  evaluate.py       # OOF raw vs calibrated comparison harness (slice A)
  data.py           # breast_cancer + hard make_classification loaders
scripts/run_calibration_slice.py
scripts/run_nested_calibration_slice.py
tests/
```

## Next slices (not done)

- Multiclass calibration and classwise reliability
- Temperature scaling for neural nets
- Real OpenML adult/credit datasets (with license notes)
