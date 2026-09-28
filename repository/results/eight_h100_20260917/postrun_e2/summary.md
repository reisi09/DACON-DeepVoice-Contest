# E2 Music/File post-run analysis

This is the preregistered CPU-only analysis over the verified attempt6 feature archive and frozen E1 heads. A positive response is the fake SONICS backing score minus the mean score of two real FMA backings for the same vocal and level.

## Matched response results

| Arm | Split | Music response | Music real-backing control | Truth-changing File response | File real-backing control | Already-fake File response |
|---|---|---:|---:|---:|---:|---:|
| parent | dev | +0.184111 (8 pairs) | +0.154973 | +0.029755 (4 pairs) | -0.183287 | +0.368319 |
| balanced | dev | +0.135048 (8 pairs) | +0.232443 | +0.026527 (4 pairs) | -0.191272 | +0.302147 |
| parent | test | +0.439034 (8 pairs) | -0.398171 | +0.380183 (4 pairs) | -0.807171 | -0.114292 |
| balanced | test | +0.363382 (8 pairs) | -0.383070 | +0.381545 (4 pairs) | -0.789576 | -0.236273 |

All test Music deltas and all test truth-changing File deltas were positive for both arms. Dev truth-changing File deltas were positive for only half of pairs. Each dev/test slice is one provenance-connected component, so every CI is null and every contrast is marked uncertain.

The real-backing controls are large and sometimes exceed the target response. This prevents a causal claim that the score change is specifically due to fake backing truth. The diagnostic also retains source confounding, separator effects, and PANNs presence gating.

## Threshold diagnostics

| Arm | Split | Music n (REAL/FAKE) | Music AUC / EER | Fixed-threshold FPR / recall | File n (REAL/FAKE) | File AUC / EER | Fixed-threshold FPR / recall |
|---|---|---:|---:|---:|---:|---:|---:|
| parent | dev | 24 (16/8) | 0.578125 / 0.500 | 0.250 / 0.500 | 12 (8/4) | 0.468750 / 0.500 | 0.500 / 0.000 |
| balanced | dev | 24 (16/8) | 0.523438 / 0.375 | 0.375 / 0.500 | 12 (8/4) | 0.468750 / 0.500 | 0.375 / 0.000 |
| parent | test | 24 (16/8) | 1.000000 / 0.000 | 0.000 / 0.750 | 12 (8/4) | 0.750000 / 0.500 | 0.250 / 0.500 |
| balanced | test | 24 (16/8) | 0.968750 / 0.125 | 0.000 / 0.625 | 12 (8/4) | 0.750000 / 0.500 | 0.000 / 0.500 |

Thresholds were fixed from calibration before held-out features were scored. These small, source-confounded slices are diagnostics and are not an official score.

## Integrity and scope

- Manifest SHA-256: `fc21a54a7d9aeec69575ad1b6a9c0d0c35910f13ba317fde434587686309b537`.
- Normalizers used 831 core-music train rows and 212 core-speech train rows.
- The lock verified 12 main E1 checkpoints and `test_used=false` before held-out scoring.
- Predictions: 224 rows (96 calibration, 64 dev, 64 test; two arms).
- Contrasts: 56 rows, including 16 dev and 16 test arm-specific contrasts.
- Forty noise rows were excluded from truth contrasts; no noise truth labels were fabricated.
- `fixed_experiment_population_denominators.json` corrects report-only E1/E3/E4/E6/E8 population metadata. It does not alter scores or fitted models.
