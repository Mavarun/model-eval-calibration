# model-eval-calibration

Research slices on **model evaluation beyond accuracy**:

1. **Slice A**: Brier, equal-width ECE, reliability diagrams; Platt vs isotonic on `breast_cancer`.
2. **Slice B**: **adaptive / quantile ECE** + **nested CV calibration** on a harder synthetic binary task.
3. **Slice C (this PR)**: **calibration and conformal coverage under covariate shift** -- temperature / Platt / beta / isotonic maps, importance-weighted calibration, split vs weighted conformal sets, MCE and bootstrap CIs.

Public / synthetic data only. Methodology demos -- **not** production accuracy claims.


## Slice C hypothesis

1. A calibrator fitted on source rows does not stay calibrated when P(x) shifts (P(y|x) fixed) if the base model is misspecified.
2. Importance-weighting the calibration rows with a density ratio estimated from **unlabelled** target features reduces target ECE.
3. Split conformal loses nominal coverage on the target; weighted conformal (Tibshirani, Barber, Candes & Ramdas, 2019) moves it back toward 1 - alpha.

## Slice C method

- **Data** (`shift.py`): source x ~ N(0, I_6), target x ~ N(mu, I_6) with |mu| = shift along (1, 0.5, 0, ...). Shared labelling P(y=1|x) = sigmoid(1.2 x0 - x1 + 0.7 (x0^2 - 1) - 0.5). The quadratic term makes a linear-logit model misspecified. The true density ratio is analytic (`exp(mu.x - |mu|^2/2)`), so an **oracle** weighted conformal baseline is available.
- **Splits per cell**: 2000 train / 1000 cal / 1000 source-test (source), 1000 unlabelled target-pool + 2000 labelled target-test. Target labels never enter any fit; split disjointness is asserted.
- **Base models**: `LogisticRegression` (misspecified) and `RandomForestClassifier(200 trees)`.
- **Calibration maps** (`calibration_maps.py`, fitted on frozen base probabilities): temperature (1 param), Platt on logit, beta calibration with a, b >= 0 (Kull et al., 2017), isotonic. Each map is fitted unweighted and importance-weighted (`iw_*`) with weights from a logistic domain classifier (source-train vs target-pool), clipped at the 99th percentile.
- **Metrics**: Brier, equal-width ECE, adaptive ECE, MCE (quantile bins with >= 10 rows), percentile bootstrap CIs (`bootstrap.py`), per-bin reliability bands, seed win-rates.
- **Conformal** (`conformal.py`): LAC score 1 - p(y|x), alpha = 0.10; split conformal vs weighted conformal with estimated and oracle ratios.
- **Grid**: shift in {0, 1, 2} x 10 seeds x 2 base models. `python scripts/run_shift_conformal_slice.py` (~80 s on CPU, writes long CSVs under `artifacts/`).

## Slice C results

### Target-domain calibration: logistic (mean ± std over 10 seeds)

| method | ECE-adapt s=0 | ECE-adapt s=1 | ECE-adapt s=2 | Brier s=0 | Brier s=1 | Brier s=2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.0481 ± 0.0082 | 0.0765 ± 0.0130 | 0.1729 ± 0.0228 | 0.1741 ± 0.0037 | 0.1589 ± 0.0064 | 0.1516 ± 0.0139 |
| temperature | 0.0490 ± 0.0078 | 0.0771 ± 0.0128 | 0.1730 ± 0.0231 | 0.1741 ± 0.0038 | 0.1591 ± 0.0065 | 0.1517 ± 0.0140 |
| platt | 0.0473 ± 0.0094 | 0.0823 ± 0.0146 | 0.1800 ± 0.0285 | 0.1740 ± 0.0038 | 0.1605 ± 0.0064 | 0.1549 ± 0.0167 |
| beta | 0.0291 ± 0.0071 | 0.0816 ± 0.0139 | 0.1710 ± 0.0300 | 0.1720 ± 0.0039 | 0.1601 ± 0.0072 | 0.1565 ± 0.0186 |
| isotonic | 0.0328 ± 0.0104 | 0.0731 ± 0.0179 | 0.1643 ± 0.0311 | 0.1735 ± 0.0041 | 0.1603 ± 0.0078 | 0.1557 ± 0.0193 |
| iw_temperature | 0.0494 ± 0.0083 | 0.0598 ± 0.0161 | 0.1561 ± 0.0314 | 0.1741 ± 0.0039 | 0.1554 ± 0.0073 | 0.1473 ± 0.0165 |
| iw_platt | 0.0478 ± 0.0098 | 0.0326 ± 0.0119 | 0.0540 ± 0.0170 | 0.1741 ± 0.0038 | 0.1522 ± 0.0046 | 0.1197 ± 0.0050 |
| iw_beta | 0.0292 ± 0.0072 | 0.0304 ± 0.0141 | 0.0544 ± 0.0163 | 0.1720 ± 0.0040 | 0.1521 ± 0.0045 | 0.1200 ± 0.0053 |
| iw_isotonic | 0.0330 ± 0.0109 | 0.0397 ± 0.0140 | 0.0660 ± 0.0126 | 0.1735 ± 0.0041 | 0.1549 ± 0.0059 | 0.1251 ± 0.0076 |

### Target-domain calibration: random_forest (mean ± std over 10 seeds)

| method | ECE-adapt s=0 | ECE-adapt s=1 | ECE-adapt s=2 | Brier s=0 | Brier s=1 | Brier s=2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.0293 ± 0.0062 | 0.0296 ± 0.0075 | 0.0595 ± 0.0140 | 0.1691 ± 0.0046 | 0.1476 ± 0.0059 | 0.1142 ± 0.0051 |
| temperature | 0.0299 ± 0.0073 | 0.0366 ± 0.0076 | 0.0718 ± 0.0158 | 0.1692 ± 0.0045 | 0.1482 ± 0.0060 | 0.1163 ± 0.0062 |
| platt | 0.0303 ± 0.0071 | 0.0433 ± 0.0116 | 0.0846 ± 0.0192 | 0.1690 ± 0.0046 | 0.1491 ± 0.0057 | 0.1192 ± 0.0063 |
| beta | 0.0271 ± 0.0074 | 0.0386 ± 0.0116 | 0.0722 ± 0.0212 | 0.1688 ± 0.0049 | 0.1486 ± 0.0057 | 0.1170 ± 0.0065 |
| isotonic | 0.0277 ± 0.0060 | 0.0418 ± 0.0153 | 0.0718 ± 0.0248 | 0.1703 ± 0.0055 | 0.1499 ± 0.0057 | 0.1178 ± 0.0067 |
| iw_temperature | 0.0298 ± 0.0073 | 0.0326 ± 0.0073 | 0.0577 ± 0.0180 | 0.1692 ± 0.0045 | 0.1478 ± 0.0059 | 0.1136 ± 0.0068 |
| iw_platt | 0.0302 ± 0.0072 | 0.0374 ± 0.0106 | 0.0587 ± 0.0277 | 0.1690 ± 0.0046 | 0.1485 ± 0.0057 | 0.1148 ± 0.0085 |
| iw_beta | 0.0271 ± 0.0075 | 0.0382 ± 0.0120 | 0.0611 ± 0.0296 | 0.1688 ± 0.0049 | 0.1484 ± 0.0058 | 0.1154 ± 0.0095 |
| iw_isotonic | 0.0285 ± 0.0059 | 0.0390 ± 0.0096 | 0.0617 ± 0.0210 | 0.1703 ± 0.0055 | 0.1500 ± 0.0056 | 0.1176 ± 0.0094 |

### Seed win-rates on target adaptive ECE (share of seeds where A < B)

| A vs B | logistic s=0 | logistic s=1 | logistic s=2 | random_forest s=0 | random_forest s=1 | random_forest s=2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iw_platt vs platt | 0.5 | 1.0 | 1.0 | 0.6 | 0.7 | 0.7 |
| iw_beta vs beta | 0.4 | 1.0 | 1.0 | 0.5 | 0.4 | 0.7 |
| iw_isotonic vs isotonic | 0.7 | 1.0 | 1.0 | 0.4 | 0.6 | 0.5 |
| iw_temperature vs temperature | 0.6 | 1.0 | 1.0 | 0.6 | 0.7 | 0.7 |
| beta vs platt | 0.9 | 0.4 | 1.0 | 0.7 | 0.8 | 0.9 |
| temperature vs raw | 0.2 | 0.3 | 0.3 | 0.5 | 0.0 | 0.0 |
| isotonic vs raw | 0.9 | 0.6 | 0.6 | 0.8 | 0.1 | 0.3 |

### Conformal (alpha = 0.10, nominal coverage 0.90)

| base | shift | domain | method | coverage | mean set size |
| --- | ---: | --- | --- | ---: | ---: |
| logistic | 0 | source | split | 0.906 ± 0.011 | 1.401 ± 0.031 |
| logistic | 0 | target | split | 0.903 ± 0.012 | 1.400 ± 0.032 |
| logistic | 0 | target | weighted_estimated | 0.904 ± 0.012 | 1.403 ± 0.033 |
| logistic | 0 | target | weighted_oracle | 0.903 ± 0.012 | 1.400 ± 0.032 |
| logistic | 1 | source | split | 0.906 ± 0.011 | 1.401 ± 0.031 |
| logistic | 1 | target | split | 0.933 ± 0.008 | 1.437 ± 0.037 |
| logistic | 1 | target | weighted_estimated | 0.908 ± 0.014 | 1.340 ± 0.036 |
| logistic | 1 | target | weighted_oracle | 0.908 ± 0.014 | 1.340 ± 0.033 |
| logistic | 2 | source | split | 0.906 ± 0.011 | 1.401 ± 0.031 |
| logistic | 2 | target | split | 0.940 ± 0.015 | 1.423 ± 0.045 |
| logistic | 2 | target | weighted_estimated | 0.911 ± 0.029 | 1.319 ± 0.078 |
| logistic | 2 | target | weighted_oracle | 0.915 ± 0.033 | 1.394 ± 0.083 |
| random_forest | 0 | source | split | 0.902 ± 0.009 | 1.383 ± 0.039 |
| random_forest | 0 | target | split | 0.905 ± 0.013 | 1.387 ± 0.037 |
| random_forest | 0 | target | weighted_estimated | 0.905 ± 0.012 | 1.389 ± 0.036 |
| random_forest | 0 | target | weighted_oracle | 0.905 ± 0.013 | 1.387 ± 0.037 |
| random_forest | 1 | source | split | 0.902 ± 0.009 | 1.383 ± 0.039 |
| random_forest | 1 | target | split | 0.912 ± 0.009 | 1.319 ± 0.038 |
| random_forest | 1 | target | weighted_estimated | 0.909 ± 0.012 | 1.310 ± 0.051 |
| random_forest | 1 | target | weighted_oracle | 0.907 ± 0.013 | 1.304 ± 0.058 |
| random_forest | 2 | source | split | 0.902 ± 0.009 | 1.383 ± 0.039 |
| random_forest | 2 | target | split | 0.939 ± 0.006 | 1.261 ± 0.049 |
| random_forest | 2 | target | weighted_estimated | 0.925 ± 0.027 | 1.221 ± 0.106 |
| random_forest | 2 | target | weighted_oracle | 0.908 ± 0.040 | 1.275 ± 0.147 |

Mean Kish ESS of the 1000 importance-weighted cal rows: shift 0: 991, shift 1: 460, shift 2: 138

Reliability bands (logistic raw, target, shift 2, seed 0): 9/10 quantile bins have a 95% bootstrap band that excludes their mean confidence.

### Reading the results

- **H1 supported for the misspecified logistic model**: target adaptive ECE rises from 0.048 (shift 0) to 0.173 (shift 2) for raw probabilities, and source-fitted temperature/Platt/beta/isotonic barely move it (0.164-0.180 at shift 2). 9 of 10 quantile bins have bootstrap bands that exclude their mean confidence in the illustrated cell.
- **H2 supported for logistic, not for random forest**: importance-weighted Platt/beta cut target ECE to ~0.03 (shift 1) and ~0.054 (shift 2) and beat their unweighted versions in 10/10 seeds at shifts 1 and 2. Temperature scaling, with no bias term, recovers much less (0.156 at shift 2). For the RF, raw probabilities were within seed noise of the best map at every shift; at shifts 1 and 2 every unweighted source-fitted map made target ECE *worse* than raw (temperature beat raw in 0/10 seeds), and weighting only partly undid that (iw-vs-unweighted win-rates 0.4-0.7, i.e. noise-level).
- **H3 partly supported, with a twist**: in this design unweighted split conformal **over**-covers on the target (0.93-0.94 vs nominal 0.90) rather than under-covering, because the shifted mass lands where the base model is more confident. Weighted conformal pulls coverage back to 0.908-0.915 for logistic with smaller sets (1.32-1.34 vs 1.42-1.44). For RF at shift 2 the estimated-ratio version only reaches 0.925 and the oracle 0.908, with seed std ~0.03-0.04 -- the variance cost of weights with ESS ~138 of 1000.
- **Controls**: at shift 0, weighted and unweighted results coincide (ESS 991/1000) and source-test split conformal covers 0.902-0.906, consistent with the finite-sample band [0.90, 0.901 + MC noise].

### Test design (slice C)

- Calibration maps: recover a known temperature (2.5 within 12%), beta ~ identity on calibrated probabilities, sign constraints and monotonicity, every map halves ECE on an overconfident hold-out, weighted fits move toward the up-weighted region, uniform weights == unweighted.
- Shift: marginal moves by mu while P(y|x) is equal in a shared x-region; domain-classifier ratio has Spearman > 0.95 with the analytic ratio and its importance-weighted mean recovers the target mean; ESS falls as shift grows.
- Conformal: hand-computed quantile rank; inf threshold for tiny n; **marginal coverage averaged over 300 independent calibration draws lies in [1 - alpha, 1 - alpha + 1/(n+1)] +/- 0.006** for alpha in {0.1, 0.2}; weighted == split with uniform weights; with the oracle ratio, weighted coverage on a shifted target is within 0.015 of nominal (20 seeds) and closer than unweighted.
- Harness: split disjointness, reproducibility for a fixed seed, near-nominal source coverage over 8 seeds, near-uniform weights at shift 0.

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
python scripts/run_shift_conformal_slice.py     # slice C (covariate shift + conformal)
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

- **Slice C is fully synthetic** (Gaussian marginals, one hand-written logit). Density-ratio estimation is easy here: the true log-ratio is linear, which is exactly what the logistic domain classifier models. Real shifts (non-Gaussian, high-dimensional, partial support overlap) will give worse ratios and larger weight variance.
- Importance weighting assumes pure covariate shift and support overlap; it cannot fix label shift or concept drift, and the 99th-percentile weight clip trades bias for variance without a principled choice.
- ESS of the weighted cal set falls to ~14% at shift 2; weighted results there carry seed std 2-4x larger than unweighted.
- ECE bootstrap intervals are percentile intervals of a biased binned estimator -- they describe resampling spread and can sit above the point estimate; they are not coverage-calibrated CIs. Bootstrap also ignores refit variance of the model and calibrator.
- Conformal guarantees are marginal only; class-conditional coverage is not controlled, and with binary labels set sizes are coarse (1 or 2).
- 10 seeds per cell; win-rates of 0.4-0.7 should be read as "no detectable difference".

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
  metrics.py        # Brier, equal-width + adaptive ECE, MCE, reliability
  bootstrap.py      # percentile + paired bootstrap CIs, reliability bands
  calibration.py    # Platt / isotonic wrappers (CalibratedClassifierCV)
  calibration_maps.py # temperature / Platt / beta / isotonic maps (+ sample_weight)
  shift.py          # covariate-shift generator, domain-classifier density ratio, ESS
  conformal.py      # split + weighted conformal sets and diagnostics
  shift_eval.py     # slice C harness: shift grid, win-rates, conformal coverage
  nested.py         # naive same-fold vs nested CV evaluator + barrier helper
  evaluate.py       # OOF raw vs calibrated comparison harness (slice A)
  data.py           # breast_cancer + hard make_classification loaders
scripts/run_calibration_slice.py
scripts/run_nested_calibration_slice.py
scripts/run_shift_conformal_slice.py
.github/workflows/tests.yml   # pytest on push / PR (py3.11, py3.12)
tests/
```

## Next slices (not done)

- Multiclass calibration and classwise reliability
- Temperature scaling for neural nets (binary temperature is done in slice C; multiclass logits are not)
- Label shift (EM prior correction) and real-data shift benchmarks (e.g. time-split public tabular sets)
- Class-conditional (Mondrian) conformal and conditional-coverage diagnostics
- Real OpenML adult/credit datasets (with license notes)
