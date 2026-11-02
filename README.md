# model-eval-calibration

Research slices on **model evaluation beyond accuracy**:

1. **Slice A**: Brier, equal-width ECE, reliability diagrams; Platt vs isotonic on `breast_cancer`.
2. **Slice B**: **adaptive / quantile ECE** + **nested CV calibration** on a harder synthetic binary task.
3. **Slice C**: **calibration and conformal coverage under covariate shift** -- temperature / Platt / beta / isotonic maps, importance-weighted calibration, split vs weighted conformal sets, MCE and bootstrap CIs.
4. **Slice D**: **label shift and cost-aware decisions** -- EM and BBSE target-prior estimation, Bayes vs source-tuned thresholds at c_fn = 5 c_fp, pooled vs Mondrian conformal, and a **walk-forward** run under a drifting prior with a label-leakage barrier and block-bootstrap CIs.
5. **Slice E (this PR)**: **per-row decisions under smooth and abrupt prior drift** -- online EM, sliding-window EM and CUSUM-restarted EM trackers, batch-latency and delayed-label baselines, dev-seed tuning, and Politis-White bootstrap block lengths.

Public / synthetic data only. Methodology demos -- **not** production accuracy claims.


## Slice E hypothesis

Slice D's best walk-forward policy ran EM on the 400-row batch being decided, so every row waited for its batch, and the prior only drifted smoothly. Slice E asks what happens when rows must be decided one at a time and the prior can jump.

1. Batch EM degrades as batches shrink, so slice D's winner does not carry over to small batches or per-row decisions.
2. Per-row label-free trackers that use only past rows (online EM, sliding-window EM) get close to the 400-row batch policy without making rows wait.
3. Under abrupt jumps, EM restarted at CUSUM-detected change points beats fixed-memory trackers, mostly in the rows right after a change, and costs little on smooth drift.
4. Label-free tracking beats re-tuning a threshold on labels that arrive 400 rows late, and the gap is largest after jumps.

## Slice E method

- **Streams** (`drift_eval.py`, `stream_eval.py`): the slice D world (pi_s = 0.3, c_fp = 1, c_fn = 5, model trained on 2500 and calibrated on 2000 source rows). 9600 target rows per stream (slice D's volume) on either the **smooth** cosine path evaluated per row or an **abrupt regime** path: piecewise-constant prior in {0.05, 0.1, 0.3, 0.5}, segment length 200 + geometric (mean 1600 rows), every switch a real jump, change points known.
- **Trackers** (`online_prior.py`): every tracker's prior for row i is a function of rows < i (unlabelled features only; tested by rewriting the future). `online_em_path`: constant-step stochastic-approximation EM (Cappe & Moulines, 2009). `window_em_path`: Saerens EM on the last W rows (refit every 10). `cusum_em_path`: EM on rows since the last detected change plus a two-sided Page CUSUM on the label-shift mixture log-likelihood (prior moved by +/- 0.8 logit vs the current estimate); on an alarm EM restarts where the statistic last left zero.
- **Policies**: static; `batch{400,100,25}_em_current` (slice D's policy, which waits for the batch); `batch400_em_prev` (EM on the previous batch, per-row usable); the three trackers; `delayed_label_threshold` (labels arrive 400 rows late, threshold re-tuned every 25 rows on the latest 400 labelled rows); oracle prior (ceiling). **Regret** = cost per row minus the exact-posterior cost on the same rows.
- **Tuning out of sample**: each tracker's setting (online step in {0.002 .. 0.05}, window in {100 .. 1600}, CUSUM threshold in {3 .. 16}) is chosen per base model by mean cost on **6 development seeds (1000-1005)**, with both drift shapes pooled. Test seeds 0-9 are disjoint (checked). The choice was step 0.02, window 200 and threshold 4 for both models.
- **Uncertainty**: paired per-row cost differences aggregated into 100-row chunks, moving-block bootstrap with seeds resampled. The block length is now picked by the **Politis-White rule** (`block_length.py`, with the Patton-Politis-White 2009 circular-block correction) instead of being fixed.
- **Grid**: 10 test seeds x {smooth, regime} x {logistic, random forest}. `python scripts/run_drift_stream_slice.py` (~3 min CPU, including dev selection and a slice D re-check).

## Slice E results

### Regret per row on test streams (mean ± std over 10 seeds; c_fp = 1, c_fn = 5)

| policy | smooth logistic | smooth RF | regime logistic | regime RF | regime post-change logistic | regime post-change RF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static | 0.098 ± 0.008 | 0.100 ± 0.011 | 0.113 ± 0.028 | 0.112 ± 0.022 | 0.141 ± 0.018 | 0.135 ± 0.022 |
| batch400 EM current (waits) | **0.032 ± 0.005** | **0.035 ± 0.010** | 0.045 ± 0.016 | 0.042 ± 0.011 | 0.063 ± 0.022 | 0.061 ± 0.020 |
| batch100 EM current (waits) | 0.046 ± 0.005 | 0.046 ± 0.007 | 0.045 ± 0.013 | 0.041 ± 0.007 | **0.052 ± 0.021** | **0.045 ± 0.016** |
| batch25 EM current (waits) | 0.079 ± 0.007 | 0.079 ± 0.010 | 0.077 ± 0.013 | 0.071 ± 0.009 | 0.079 ± 0.022 | 0.073 ± 0.021 |
| batch400 EM previous | 0.038 ± 0.009 | 0.039 ± 0.006 | 0.087 ± 0.040 | 0.079 ± 0.031 | 0.234 ± 0.099 | 0.214 ± 0.076 |
| online EM (step 0.02) | 0.034 ± 0.006 | 0.036 ± 0.008 | 0.048 ± 0.020 | 0.042 ± 0.011 | 0.089 ± 0.035 | 0.082 ± 0.027 |
| window EM (200 rows) | 0.038 ± 0.007 | 0.038 ± 0.008 | 0.050 ± 0.018 | 0.045 ± 0.010 | 0.086 ± 0.029 | 0.076 ± 0.024 |
| CUSUM EM (h = 4) | 0.041 ± 0.005 | 0.040 ± 0.007 | **0.045 ± 0.020** | **0.041 ± 0.012** | 0.076 ± 0.032 | 0.071 ± 0.023 |
| delayed-label threshold (400 late) | 0.048 ± 0.004 | 0.048 ± 0.011 | 0.115 ± 0.044 | 0.115 ± 0.042 | 0.293 ± 0.084 | 0.287 ± 0.083 |
| oracle prior | 0.028 ± 0.006 | 0.028 ± 0.008 | 0.029 ± 0.013 | 0.025 ± 0.008 | 0.037 ± 0.018 | 0.027 ± 0.013 |

Floor (exact-posterior) cost per row: 0.350 smooth, 0.343 regime. "Post-change" = regret per row over the 400 rows after each jump. Bold marks the best non-oracle policy in each column; batch policies make rows wait for their batch.

### Paired cost differences a - b (block bootstrap over 100-row chunks, Politis-White block 2-6; 95% CI)

| path | a - b | logistic | random forest |
| --- | --- | --- | --- |
| smooth | online EM - batch400 EM current | +0.002 [-0.002, +0.006] | +0.001 [-0.003, +0.006] |
| smooth | CUSUM EM - online EM | **+0.006 [+0.003, +0.010]** | **+0.0045 [+0.001, +0.009]** |
| smooth | CUSUM EM - delayed-label threshold | -0.008 [-0.014, -0.002] | -0.008 [-0.015, -0.002] |
| smooth | online EM - batch25 EM current | -0.044 [-0.052, -0.038] | -0.043 [-0.050, -0.037] |
| regime | CUSUM EM - batch400 EM current | +0.000 [-0.006, +0.007] | -0.001 [-0.007, +0.006] |
| regime | CUSUM EM - online EM | -0.003 [-0.007, +0.002] | -0.001 [-0.005, +0.003] |
| regime | CUSUM EM - window EM | -0.005 [-0.009, -0.001] | -0.004 [-0.008, -0.000] |
| regime | CUSUM EM - delayed-label threshold | -0.070 [-0.099, -0.045] | -0.074 [-0.104, -0.049] |
| regime | online EM - batch25 EM current | -0.029 [-0.038, -0.020] | -0.029 [-0.037, -0.021] |

Slice D re-check: on slice D's 24 x 400 walk-forward the Politis-White rule picks blocks of 2-3 batches, and every slice D CI moves by at most 0.001 against the fixed block of 4. The slice D conclusions stand.

### Reading the results

- **H1 supported.** EM on the batch being decided goes from 0.032 regret (400 rows) to 0.046 (100) and 0.079 (25) on smooth drift. With 25 rows, prior-estimation noise costs most of what tracking saves (static 0.098). After jumps the ranking flips between 400 and 100 rows (0.063 vs 0.052 post-change), because a 400-row batch that straddles a jump averages two regimes.
- **H2 supported.** Per-row online EM (step 0.02, only past rows) is within +0.000 to +0.003 of the 400-row batch policy on both paths, and every CI covers 0. It removes the wait without a measurable cost. It beats the 25-row batch by 0.029-0.044 per row (CIs exclude 0). EM on the *previous* 400-row batch is fine on smooth drift (0.038) but costly after jumps (0.21-0.23 post-change), because it reacts one full batch late.
- **H3 mostly not supported.** CUSUM-restarted EM does cut post-change regret (0.076 / 0.071 vs online 0.089 / 0.082 and window 0.086 / 0.076). Overall on the regime path it beats the 200-row window (CIs just exclude 0) but **not** online EM: -0.003 / -0.001, CIs cover 0. On smooth drift it is a **regression**: +0.006 / +0.0045 (logistic / RF) vs online EM, CIs exclude 0, because false alarms restart EM on short segments when nothing jumped. On this evidence online EM is the safer default, and change detection only helps when jumps are known to dominate.
- **H4 supported.** Every label-free tracker beats the 400-row-late label threshold: by 0.008-0.014 per row on smooth drift (point estimates; the CUSUM CI excludes 0) and by 0.07 per row overall on the regime path (post-change 0.29 vs 0.07-0.09), where late labels describe the previous regime.
- **Tuning.** On the dev seeds online EM preferred a short memory (0.02, nominally 50 rows; cost 0.3678 vs 0.3973 at 0.002), and the CUSUM threshold had a flat optimum (3-6 within 0.002). The test ranking matches the dev ranking, and no setting was re-picked after looking at test seeds.

### Test design (slice E)

- Block length: white noise gets a median block < 2. On AR(1) the block grows with phi and lands within [0.5, 1.3] of the circular-block asymptotic (6 phi^2 / (1 - phi^2)^2)^(1/3) n^(1/3). Cap, constant series and short input are covered; `block="auto"` is long on AR(1) differences and short on i.i.d. ones.
- Paths: the regime path is piecewise constant, its change points are exactly where the level moves, segments are >= 200 rows with a mean of about 1600, and the per-row sampler keeps P(x|y) fixed.
- Trackers: the mixture log-likelihood is 0 at pi_s and maximised at the EM estimate; no tracker reads row i or later (future rewrite probe); a bigger online step reacts faster and is noisier; window EM equals batch EM on its window; CUSUM flags both jumps of a 0.3 -> 0.05 -> 0.5 stream within 300 rows and tracks the new level better than online EM right after the jump, and its false alarms on a constant stream fall with the threshold.
- Harness: labels reach only the delayed-label policy and only after the delay (row-level check plus a whole-stream hash probe); post-change flags match the true change points; streams are reproducible per seed; tuning refuses overlapping dev/test seeds and returns the dev argmin.
- Claims: `tests/test_slice_e_claims.py` re-checks each directional claim above on 3 seeds.

---

## Slice D hypothesis

1. Under label shift (P(y) moves, P(x|y) fixed) a source-calibrated model is miscalibrated on the target by a known logit offset and its cost-optimal threshold moves; estimating the target prior from **unlabelled** target rows (EM, BBSE) and adjusting the posterior recovers most of the cost gap to the oracle.
2. EM (Saerens et al., 2002) needs calibrated posteriors and BBSE (Lipton et al., 2018) does not, so a source calibration map should help EM.
3. Pooled split conformal loses per-class coverage under label shift; Mondrian (class-conditional) conformal keeps it.
4. Walk-forward with a drifting prior: label-free prior tracking beats a static model and is competitive with re-tuning the threshold on labels that arrive one batch late.

## Slice D method

- **Data** (`label_shift.py`): x|y=0 ~ N(0, I_6), x|y=1 ~ N(m, diag(s^2)) with unequal variances, so the log-likelihood ratio is quadratic and logistic regression is misspecified. `true_posterior(X, pi)` is the exact Bayes posterior for any prior (oracle).
- **Cell** (`label_shift_eval.py`): source pi_s = 0.3: 2500 train / 2000 cal. Target pi_t in {0.05, 0.1, 0.3, 0.5}: 2000 **unlabelled** pool / 3000 labelled test; target labels only score. Base models: logistic, random forest (200 trees). Maps: raw / temperature / Platt / beta / isotonic fit on source cal.
- **Priors** (`prior_estimation.py`): none (keep pi_s), EM on the pool, BBSE (hard base predictions + source confusion matrix), oracle pi_t. `adjust_posterior` moves the logit by the prior log-odds.
- **Costs** (`costs.py`): c_fp = 1, c_fn = 5. Policies: Bayes threshold 1/6 on adjusted probabilities; a threshold tuned on source cal (no adjustment); the exact target posterior as the floor. **Regret** = cost per row - floor.
- **Conformal**: LAC scores on raw base probabilities, alpha = 0.1; pooled split vs Mondrian (`mondrian_thresholds`).
- **Walk-forward** (`drift_eval.py`): train and calibrate once on source; 24 target batches x 400 rows whose prior follows one cosine cycle 0.3 -> 0.06 -> 0.5 -> 0.3. At batch t policies use only pre-label information: static; EM on batch t-1; EM on a 3-batch past window; EM on batch t's own unlabelled features (label-free); BBSE on the window; a threshold re-tuned on batch t-1's labels (delayed labels); the true prior (ceiling). A test flips batch t's labels and checks no decision at t moves. Paired differences use a moving-block bootstrap (block 4 batches, seeds resampled).
- **Grid**: 10 seeds per cell. `python scripts/run_label_shift_slice.py` (~80 s CPU; long CSVs under `artifacts/`).

## Slice D results

### Prior error and cost regret on target (mean ± std over 10 seeds; regret per row at c_fp = 1, c_fn = 5)

Logistic (misspecified):

| map | estimator | abs err pi_t=0.05 | abs err 0.1 | abs err 0.5 | regret 0.05 | regret 0.1 | regret 0.3 (no shift) | regret 0.5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | none | 0.252 | 0.202 | 0.198 | 0.248 ± 0.022 | 0.143 ± 0.019 | 0.031 ± 0.012 | 0.098 ± 0.031 |
| raw | em | **0.016** | **0.014** | **0.017** | 0.016 ± 0.014 | 0.009 ± 0.009 | 0.037 ± 0.009 | 0.049 ± 0.013 |
| raw | bbse | 0.014 | 0.019 | 0.027 | 0.015 ± 0.027 | 0.009 ± 0.014 | 0.036 ± 0.010 | 0.048 ± 0.012 |
| raw | oracle prior | 0 | 0 | 0 | 0.006 ± 0.005 | 0.006 ± 0.007 | 0.031 ± 0.011 | 0.049 ± 0.014 |
| raw | tuned on source | - | - | - | 0.209 ± 0.051 | 0.115 ± 0.039 | 0.043 ± 0.013 | 0.149 ± 0.067 |
| platt | em | 0.019 | 0.026 | 0.030 | 0.016 ± 0.013 | 0.011 ± 0.011 | 0.035 ± 0.016 | 0.051 ± 0.016 |
| isotonic | em | 0.016 | 0.024 | 0.029 | 0.014 ± 0.014 | 0.007 ± 0.009 | 0.042 ± 0.017 | 0.054 ± 0.011 |

Random forest: raw-EM regret 0.012 / 0.024 / 0.036 / 0.031 at pi_t = 0.05 / 0.1 / 0.3 / 0.5 vs 0.268 / 0.157 / 0.034 / 0.095 ignoring the shift; BBSE within 0.007 of EM; source-tuned threshold 0.266 / 0.157 / 0.038 / 0.102. Floor cost per row: 0.159 / 0.274 / 0.417 / 0.382.

### Conformal on target (alpha = 0.10; logistic, mean over 10 seeds; RF within 0.015)

| pi_t | method | coverage | negatives | positives | set size |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.05 | split | 0.959 | 0.970 | **0.738** | 1.157 |
| 0.05 | mondrian | 0.900 | 0.901 | **0.885** | 1.338 |
| 0.3 | split | 0.902 | 0.970 | 0.739 | 1.192 |
| 0.3 | mondrian | 0.902 | 0.903 | 0.900 | 1.321 |
| 0.5 | split | **0.855** | 0.971 | 0.736 | 1.219 |
| 0.5 | mondrian | 0.904 | 0.907 | 0.901 | 1.308 |

### Walk-forward under prior drift (regret per row, mean ± std over 10 seeds)

| base | map | static | EM t-1 | EM window 3 | EM current (label-free) | BBSE window | delayed-label threshold | oracle prior |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | raw | 0.093 ± 0.012 | 0.038 ± 0.008 | 0.049 ± 0.010 | **0.034 ± 0.005** | 0.049 ± 0.010 | 0.044 ± 0.007 | 0.028 ± 0.006 |
| logistic | isotonic | 0.096 ± 0.010 | 0.041 ± 0.007 | 0.051 ± 0.011 | **0.035 ± 0.005** | 0.050 ± 0.013 | 0.042 ± 0.008 | 0.030 ± 0.005 |
| random_forest | raw | 0.096 ± 0.011 | 0.037 ± 0.009 | 0.047 ± 0.012 | **0.033 ± 0.005** | 0.049 ± 0.012 | 0.047 ± 0.012 | 0.029 ± 0.006 |
| random_forest | isotonic | 0.100 ± 0.012 | 0.044 ± 0.010 | 0.057 ± 0.015 | **0.039 ± 0.007** | 0.054 ± 0.016 | 0.051 ± 0.010 | 0.033 ± 0.006 |

Paired cost differences (block bootstrap, 95% CI): EM-current minus delayed-label threshold is -0.007 to -0.014 per row with every CI below 0 (6/6 model x map cells); EM-window minus delayed-label is +0.000 to +0.009 (CI excludes 0 only for logistic/isotonic, against EM-window); EM t-1 beats the 3-batch window in all 6 cells (-0.010 to -0.013); EM-current vs static -0.059 to -0.064.

### Reading the results

- **H1 supported.** Ignoring a 0.3 -> 0.05 shift costs 0.25-0.27 per row above the floor (logistic, RF); EM or BBSE on 2000 unlabelled rows cuts that to 0.01-0.02. A threshold tuned on source labels does not help (0.21 / 0.27): it is tied to the source prior, exactly like the unadjusted Bayes cut. The source-tuned threshold is identical across maps because a monotone map does not change which rows a tuned cut separates.
- **H2 not supported here (negative result).** Source calibration maps did not help EM: raw-logistic EM has the lowest prior error (0.014-0.017) and Platt / isotonic EM are 0.016-0.030, never lower than raw. Logistic regression is already calibrated in the large on source, and a map fitted on 2000 rows adds estimation noise that EM's fixed point amplifies. The unit test shows the mechanism still holds when miscalibration is real: an underconfident (T = 0.4) score drives EM from 0.10 to > 0.5.
- **Correcting when nothing shifted costs a little.** At pi_t = 0.3 EM/BBSE regret is 0.036-0.037 vs 0.031 for doing nothing (estimation noise of about 0.015 in the prior).
- **H3 supported.** Pooled split conformal covers positives only 0.72-0.74 at every prior (its single quantile is set by the majority class), so marginal coverage falls to 0.855 when positives become half the target. Mondrian keeps both classes at 0.885-0.907 for the price of larger sets (1.31-1.34 vs 1.16-1.24).
- **H4 partly supported.** Label-free EM on the current batch beats re-tuning on one-batch-late labels in every cell. Tracking from a past window lags the drift (it follows the prior about two batches late) and is no better than the delayed-label threshold; the one-batch-back EM beats the 3-batch window everywhere, so in this drift lag costs more than variance.

### Test design (slice D)

- Generator: priors move while class-conditional means stay put; the logit moves by exactly the prior log-odds; the oracle posterior is calibrated; the LLR is non-affine.
- Estimators: EM recovers pi_t within 0.015 from calibrated posteriors for pi_t in {0.05, 0.15, 0.5, 0.8}; same-ranking miscalibration biases EM (underconfident > 0.5, +0.8 logit offset > 0.3 for a true 0.1); BBSE recovers 0.1 within 0.02 from a deliberately miscalibrated hard rule; singular confusion matrices are refused.
- Costs: hand-computed cost; Bayes threshold is optimal on the oracle posterior; the tuned threshold is the empirical minimiser; the shifted threshold on source scores equals the Bayes threshold on adjusted scores (0.167 -> 0.435 for 0.3 -> 0.1).
- Conformal: Mondrian per-class coverage 0.904 / 0.906 over 150 calibration draws under 0.3 -> 0.1, pooled split 0.726 on positives.
- Walk-forward: label-flip leakage probe; window lag (correlation > 0.98 with the prior two batches back); block bootstrap wider than i.i.d. on AR(1) differences.

---

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
python scripts/run_label_shift_slice.py         # slice D (label shift, costs, walk-forward)
python scripts/run_drift_stream_slice.py        # slice E (per-row decisions, abrupt drift, online priors)
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

- **Slice E is fully synthetic** and keeps exact label shift: P(x|y) never moves, so every tracker is estimating the right quantity. Real streams mix in covariate and concept drift; the CUSUM statistic then alarms on drift that a prior correction cannot fix.
- The regime path draws its jumps from four fixed levels with segments of about 1600 rows. Shorter regimes, small jumps (for example 0.3 -> 0.25) or gradual-then-abrupt mixtures are not tested, and the CUSUM drift size (0.8 logit) is fixed rather than tuned.
- Tracker settings are tuned on only 6 development seeds, and one setting per model is shared across both drift shapes. A per-shape choice would need a drift-type detector, which is not built.
- The delayed-label baseline uses one delay (400 rows) and one window (400). Shorter label delays would narrow its gap.
- The block bootstrap runs on 100-row chunk means. The Politis-White rule picked blocks of 2-6 chunks; within-chunk dependence is averaged out, not modelled.

- **Slice D is fully synthetic** with exact label shift (P(x|y) identical by construction) and one cost ratio (c_fn = 5 c_fp). Real data mixes label shift with covariate and concept drift; EM and BBSE are then biased and nothing here measures by how much.
- The slice D prior path is smooth and its batches are large (400 rows); "EM on the current batch" assumes decisions can wait for a whole batch. Slice E measures abrupt drift, smaller batches and per-row decisions.
- Binary only; BBSE is the 2x2 case and EM's known failure under miscalibration is shown on synthetic score distortions, not measured on a real miscalibrated model.
- The slice D block length (4 batches) was a fixed choice; slice E re-checked it with the Politis-White rule (2-3 batches, CIs move <= 0.001). Seeds are only 10.

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
  label_shift.py    # slice D: prior-shift generator with exact Bayes posterior
  prior_estimation.py # EM (Saerens 2002), BBSE (Lipton 2018), posterior adjustment
  costs.py          # expected cost, Bayes / tuned / shifted thresholds
  label_shift_eval.py # slice D harness: maps x estimators x cost policies, Mondrian
  drift_eval.py     # walk-forward drift policies, per-row / regime prior paths, leakage probe, block bootstrap
  block_length.py   # Politis-White (PPW 2009) circular-block bootstrap block length
  online_prior.py   # slice E: online EM, window EM, CUSUM-restarted EM (past rows only)
  stream_eval.py    # slice E harness: per-row policies, dev-seed tuning, post-change regret
  nested.py         # naive same-fold vs nested CV evaluator + barrier helper
  evaluate.py       # OOF raw vs calibrated comparison harness (slice A)
  data.py           # breast_cancer + hard make_classification loaders
scripts/run_calibration_slice.py
scripts/run_nested_calibration_slice.py
scripts/run_shift_conformal_slice.py
scripts/run_label_shift_slice.py
scripts/run_drift_stream_slice.py
.github/workflows/tests.yml   # pytest on push / PR (py3.11, py3.12)
tests/
```

## Next slices (not done)

- Multiclass calibration and classwise reliability
- Temperature scaling for neural nets (binary temperature is done in slice C; multiclass logits are not)
- Real-data shift benchmarks (e.g. time-split public tabular sets) for the slice C / D methods
- Mixed label + covariate shift; multiclass BBSE / EM with a regularised confusion matrix
- Conditional-coverage diagnostics beyond class (Mondrian is done in slice D)
- Drift streams that mix label and covariate shift, so that a CUSUM alarm can be told apart from a prior jump (slice E assumes pure label shift)
- Real OpenML adult/credit datasets (with license notes)
