### Test metrics (mean over 10 seeds)

| dataset | model | map | accuracy | classwise ECE | top-label ECE | Brier | NLL |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| digits | logistic | none | 0.964 | 0.0043 | 0.0279 | 0.0569 | 0.126 |
| digits | logistic | temperature | 0.964 | 0.0036 | 0.0182 | 0.0555 | 0.122 |
| digits | logistic | vector | 0.964 | 0.0043 | 0.0180 | 0.0569 | 0.130 |
| digits | random_forest | none | 0.971 | 0.0400 | 0.2388 | 0.1383 | 0.365 |
| digits | random_forest | temperature | 0.971 | 0.0032 | 0.0173 | 0.0428 | 0.092 |
| digits | random_forest | vector | 0.972 | 0.0046 | 0.0329 | 0.0444 | 0.099 |
| digits | naive_bayes | none | 0.844 | 0.0267 | 0.1470 | 0.2965 | 2.551 |
| digits | naive_bayes | temperature | 0.844 | 0.0282 | 0.0443 | 0.2442 | 0.584 |
| digits | naive_bayes | vector | 0.874 | 0.0183 | 0.0348 | 0.1987 | 0.562 |
| synth5 | logistic | none | 0.510 | 0.0321 | 0.0462 | 0.6362 | 1.249 |
| synth5 | logistic | temperature | 0.510 | 0.0329 | 0.0505 | 0.6368 | 1.250 |
| synth5 | logistic | vector | 0.512 | 0.0336 | 0.0521 | 0.6369 | 1.249 |
| synth5 | random_forest | none | 0.724 | 0.0792 | 0.1929 | 0.4597 | 0.936 |
| synth5 | random_forest | temperature | 0.724 | 0.0290 | 0.0518 | 0.4085 | 0.852 |
| synth5 | random_forest | vector | 0.724 | 0.0294 | 0.0543 | 0.4086 | 0.853 |
| synth5 | naive_bayes | none | 0.512 | 0.0586 | 0.1238 | 0.6537 | 1.361 |
| synth5 | naive_bayes | temperature | 0.512 | 0.0368 | 0.0544 | 0.6326 | 1.246 |
| synth5 | naive_bayes | vector | 0.518 | 0.0335 | 0.0469 | 0.6272 | 1.238 |

### Paired per-cell effects: temperature - none (10 seeds)

| dataset | model | seeds with lower classwise ECE | classwise ECE diff | NLL diff | accuracy diff |
| --- | --- | ---: | ---: | ---: | ---: |
| digits | logistic | 10/10 | -0.0006 | -0.004 | +0.0000 |
| digits | random_forest | 10/10 | -0.0369 | -0.273 | +0.0000 |
| digits | naive_bayes | 3/10 | +0.0015 | -1.967 | +0.0000 |
| synth5 | logistic | 4/10 | +0.0008 | +0.000 | +0.0000 |
| synth5 | random_forest | 10/10 | -0.0502 | -0.084 | +0.0000 |
| synth5 | naive_bayes | 10/10 | -0.0218 | -0.116 | +0.0000 |

### Paired per-cell effects: vector - temperature (10 seeds)

| dataset | model | seeds with lower classwise ECE | classwise ECE diff | NLL diff | accuracy diff |
| --- | --- | ---: | ---: | ---: | ---: |
| digits | logistic | 4/10 | +0.0007 | +0.008 | +0.0000 |
| digits | random_forest | 0/10 | +0.0015 | +0.007 | +0.0009 |
| digits | naive_bayes | 10/10 | -0.0099 | -0.022 | +0.0302 |
| synth5 | logistic | 4/10 | +0.0007 | -0.000 | +0.0011 |
| synth5 | random_forest | 2/10 | +0.0004 | +0.000 | +0.0001 |
| synth5 | naive_bayes | 8/10 | -0.0033 | -0.008 | +0.0055 |

### Pre-registered decisions

{"H1": true, "H2": false, "recommended": "temperature"}
