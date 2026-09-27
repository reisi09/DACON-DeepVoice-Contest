# R1 — length shift and window aggregation pilot (2026-09-27, CPU only)

All numbers come from files in this folder. Everything ran on CPU in fp32. Peak RSS was at most 1.8 GB and each step took at most 78 s.
Project files were only read. Nothing was written outside `research/top15/work/r1/` (no new `.pyc` in project `__pycache__`).

## Inputs and anchors
- Per-window DB-head logits for val (4,931) and the held-out test (6,450) are in `windows.npz` (`extract.py`). They come from
  `_workspace/cache_v4/{cache_tv,cache_test}` and `_keep_best_20260926/train_duration_balanced` (`full_bank_evaluate.load_selected_heads`).
- Reproduction check: the DB val/test EERs are reproduced exactly (val FILE .07003 / V .05096 / M .12015; test .10940 / .11071 / .20005).
  They match `file_ensemble.report.json`. Per-file scores differ by at most 0.0045 (bf16 GPU vs fp32 CPU). See `a.json.repro`.
- EER uses `data_v4/file_ensemble.eer`, the DACON formula. FILE is refit on val for every pooling with `file_ensemble.fit`, using equal weights unless the grid gains at least 0.002.
  Presence comes from the DB decision CSV columns.
- The held-out test already covers the official length range: median 9.6 s, p90 24.7 s, p99 55.9 s. p50/p90/p99 voice windows are 3/7/14.

## (a) Inference-only pooling on DB heads (`agg.py a` -> `a.json`)
| pooling (voice / music) | val FILE, V, M, ADS | test FILE, V, M, ADS |
|---|---|---|
| max / LME (current) | .07003 .05096 .12015 .91875 | .10940 .11071 .20005 .86314 |
| top3 / LME | .07714 .04679 — .91603 | .11391 .10393 — .86224 |
| top30% / LME | .07065 .05256 — .91812 | .10972 .11208 — .86271 |
| mean-logit / LME | .07977 .05555 — .91296 | .11655 .11482 — .85875 |
| LME / LME | .07265 .04918 — .91779 | .11049 .10974 — .86279 |
| max / max | .07853 — .13619 .90969 | .11514 — .19826 .86081 |
| max / mean | .06477 — .12590 .91965 | .11391 — .21655 .85594 |

- No inference-only pooling beats the current one on ADS, on either split.
- top3 improves only the VOICE route: test −.0068 (group bootstrap CI [−.0114, −.0022]) and val −.0042. Fed into FILE, it hurts FILE.

## (b) Synthetic long files (`agg.py b` -> `b.json`; retrained variants in `b_grid_*`, `b_music3_*`, `b_s3music_*`)
- Each held-out anchor is extended with k−1 files of the same source and label, drawn from other groups. The result is capped at 60 s (15 voice / 29 music windows).
- The threshold is the k=1 (true file) EER threshold. We use 3 draws.
- Partner sets are shared across poolings, so comparisons are paired. About 2% of anchors at k=8 had to reuse a group.
- Metrics:
  - FPR and FNR as k goes from 1 to 8.
  - Mixed-k EER, with k uniform on 1..8 for both classes.
  - RL/FS EER: reals at k=8, fakes at k=1.
  - RS/FL EER: reals at k=1, fakes at k=8.

| route / pooling (DB heads, inference-only) | FPR k1→k8 | FNR k1→k8 | mixed | RL/FS | RS/FL |
|---|---|---|---|---|---|
| voice max (current) | .111→.392 | .111→.020 | .0851 | .176 | .046 |
| voice LME | .110→.180 | .110→.037 | .0799 | .128 | .061 |
| voice top30% | .112→.200 | .112→.035 | .0811 | .133 | .067 |
| voice mean-logit | .115→.040 | .115→.073 | .0756 | .090 | .090 |
| music LME (current) | .200→.398 | .200→.020 | .1010 | .252 | .064 |
| music max | .198→.686 | .198→.006 | .1183 | .323 | .043 |
| FILE (voice max, music LME) | .110→.300 | .110→.006 | .0515 | .153 | .0245 |

- Max makes long reals look fake at a fixed threshold.
- LME is not length-normalised in practice: music FPR doubles. This contradicts lane3's statement that the music path is length-normalised.

## (c) Retrained heads (pooling in training = pooling in inference)
- `retrain.py` uses `full_bank_train.read_items`, `step` and `predict` unchanged, with the DB weights, duration balance, init, optimizer, selection and patience. Only `T.forward` pooling is patched.
- The max and LME controls reproduce the H100 DB runs:
  - Voice seeds 41 and 43 give identical best epoch and val EER.
  - Music seeds 42 and 43 are identical.
  - The 3-seed music LME ensemble gives exactly the DB val and test metrics (`music3.json`).
- Head retraining therefore needs no GPU: 20–40 s per seed.

3-seed grid, DB protocol (`eval_grid.py` -> `grid.json`). The control is voice max / music LME, CPU retrain.

| voice / music | val ADS | test ADS | test FILE | test V | test M |
|---|---|---|---|---|---|
| max / LME (control) | .91868 | .86348 | .10895 | .11014 | .20005 |
| max / mean | .92302 | .86743 | .10630 | .11014 | .19129 |
| LME / LME | .92007 | .86442 | .10785 | .10821 | .20005 |
| **LME / mean** | **.92404** | **.86837** | .10521 | .10821 | .19129 |
| mean / LME | .91811 | .86216 | .11049 | .11288 | .20005 |
| mean / mean | .92115 | .86479 | .11049 | .11288 | .19129 |

- The ranking of the six configurations is identical on val and test. Val alone therefore picks LME / mean, and the choice does not depend on the test.
- LME / mean vs control, FILE EER bootstrap: val CI [−.0109, −.0015] (P(Δ≤0) .994), test CI [−.0096, +.0017] (P .932).
- Music mean vs LME: test CI [−.0214, +.0035] (P .927), val P .922.
- Per-seed music test EER: LME .2057 / .2127 / .1962 vs mean .1843 / .2145 / .1854.
- Voice mean pooling is seed-unstable. Seeds 42 and 43 stop early at val EER about .055, against .051 for max. It is rejected.
- LME voice beats max on val in 3 of 3 seeds: .04918 / .04937 / .04937 vs .05077 / .05096 / .05077.
- Length dependence on true held-out files, within-source Spearman for reals (control → LME / mean):
  - FILE .218 → .108
  - voice .241 → .140
  - music .116 → .007
- FILE FPR on real files, by duration (control → LME / mean):
  - 15–30 s: .150 → .114
  - 30–60 s: .220 → .163
- FILE FNR on fakes, by duration (control → LME / mean):
  - 15–30 s: .142 → .161
  - 30–60 s: .094 → .106
- Synthetic FILE, control → LME / mean:
  - FPR at k8: .300 → .060
  - mixed: .0518 → .0427
  - RL/FS: .154 → .091
  - RS/FL: .0245 → .0438. This case gets worse: max profits when fakes are the long files.

S3 features (`s3music.py` -> `s3music.json`, `b_s3music_*`): MERT 6+12 music heads, backbone-cache rows (2,149 test), 3 seeds.
- The LME control reproduces `_workspace/audit_20260927/music_layer_seeds.json` [6,12] per seed.
- Mean pooling:
  - Val ensemble .11194 → .10579.
  - Test ensemble .18467 → .18329, CI [−.0123, +.0153], P .51. Every single seed is worse on test.
  - Synthetic FPR at k8 .420 → .047, mixed .1048 → .0849, RL/FS .251 → .123, RS/FL .059 → .118.
- The held-out EER gain does not transfer to S3 music. Only the length robustness does.
- The S3 AntiDeepfake voice heads (max, `full_bank_submission.py:135`) were not retrained. That result is UNVERIFIED.

## Decision
- Keep max for any inference-only use of the existing heads, including the packaged P1/S3. Every inference-only alternative loses ADS.
- Change aggregation only by retraining: voice LME with music mean, 3 seeds. This is a cheap, principled add-on to the next retrained candidate:
  - DB held-out +.0049 ADS and val +.0054.
  - It removes most of the synthetic length drift.
  - It is not a top-15 lever. At lane2's ratio of .3–.6, it is worth about +.0015–.003 LB ADS.
  - The upside exists only if official reals are systematically longer than fakes. That is UNVERIFIED, and the opposite case costs a little.
- Code touch points:
  - Runtime: `full_bank_submission.py:114` (music), `:120` (voice), `:135` (AD).
  - Training: `full_bank_train.forward` (`full_bank_train.py:78-96`), used by `data_v4/train_backbone_heads.py`.
  - Refit afterwards: the `component_fusion` CDF tables and `file_ensemble`.
  - Check parity on the 10 QA files before spending a slot.

## Caveats
- Synthetic constituents come from distinct groups, so their windows are more independent than those of one long recording. This overstates both max inflation and evidence accumulation.
- In the FILE synthetic, Pv and Pm are the max over constituents, which is an approximation.
- There is one split. The candidate poolings for (c) were chosen from test-based (a)/(b), but the val ranking picks the same configuration.
