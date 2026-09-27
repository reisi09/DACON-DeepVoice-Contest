# Review v2: statistics (2026-09-27, CPU only, read-only on project files)

All numbers are official held-out ADS on the backbone-cache rows: 3,680 val and 4,822 test rows, 2,215 test groups (group_id).
The pipeline is ads_compare: DB-protocol heads, voice CDF fusion 1:0.5, FILE ensemble refit on val.
`core.py` re-implements it; `core.eer` equals sklearn roc_curve including ties and weights (selfcheck).
`boot.py` asserts all 7 point ADS against ads_compare.json and R2 ads_report.json (|diff| < 6e-6) and cross-checks S3 with `graph_scoring.official_metrics`.
Each run stayed at or under 10 min and 2.2 GB peak RSS. Scripts and logs are in this folder.

## Q1-Q3: S3-style vs baseline, sampling uncertainty (`boot.py` -> boot.json)
| delta (point) | test-group boot 2000 (fits fixed): pct CI / basic CI / P>0 | +val refit 500: pct CI / P>0 | Dirichlet source mix 2000: pct CI / P>0 | leave-one-source-out min..max |
|---|---|---|---|---|
| S3 - base (.00829) | [.0025,.0159] / [.0007,.0141] / .9965 | [.0026,.0158] / .996 | [-.0004,.0160] / .967 | .0067...0107 |
| MERT 6+12 only (.0031) | [-.0018,.0107] / [-.0045,.0080] / .921 | [-.0020,.0108] / .899 | [-.0060,.0110] / .825 | .0008...0063 |
| AD voice fusion part (.0052) | [.0024,.0075] / [.0029,.0080] / 1.0 | [.0024,.0077] / 1.0 | [.0004,.0102] / .983 | .0022...0063 |
| w=.5 - w=1 (.0006) | [-.0018,.0020] / .529 | / .55 | / .55 | -.0009...0015 |
| R2 MLP both - Linear (.0138) | [.0096,.0174] / [.0101,.0179] / 1.0 | [.0094,.0177] / 1.0 | [.0030,.0245] / .994 | .0085...0170 |

Bootstrap means sit above the point for S3 (.0092 vs .0083). The "+0.0092" quoted by lane3 is that bootstrap mean.
In 97.9% of val resamples the base FILE grid weights change. The S3 delta barely depends on the FILE rule: .00807 with equal weights on both sides and .00808 with the base weights on both (seedcheck d).

## Q4: seed noise, S3 vs base (`seeds.py`, `seedcheck.py` -> seedcheck.json)
Seeds 44-46 were retrained with the same H.train calls. Seeds 41-43 reproduce ads_compare_scores (max |diff| 6.6e-5).
- Single seeds 41-46, S3 - base: .0161 .0097 .0082 .0111 .0184 .0094 (mean .0122, sd .0041, 6/6 > 0).
  - MERT 6+12 part: 6/6 > 0 (mean .0077).
  - AD part: .0044 +/- .0003.
- Disjoint triples:
  - A (41-43, the shipped screen): .0083. Base ADS .86438.
  - B (44-46): .0157. Base ADS .85986.
  - 6-seed ensemble: s3 .87469 - base .86161 = .0131 (r2seen Q8).
- 150 random 3-of-6 per-component triples: mean .0124, 2.5-97.5% [.0086, .0175]. Triple A is at or below the 2.5th percentile.
  The main noise source is the music_12 baseline seeds.

## Q5: R2 MLP voice heads, seed noise (`r2seeds.py` 44-46 -> seeds_mlp.json; `r2seen.py` -> r2seen.json)
- Paired single seeds 41-46, MLP(df+ad) - Linear: .0137 .0079 .0126 .0098 .0120 .0131 (mean .0115, sd .0022, min .0079).
  - DF-MLP only: mean .0078, sd .0017, 6/6 > 0.
- Triples:
  - A-A .0138; A-B .0109; B-A .0124; B-B .0095.
  - B-B bootstrap: pct CI [.0064, .0150] / basic CI [.0040, .0126], P>0 1.0.
- 6-seed ensembles: MLP .88689 vs Linear .87469, delta .0122.
- MLP seeds 44-46 val EER: df .0313/.0347/.0310, ad .0278/.0260/.0278. Same range as R2's 41-43.

## Q6: R1 retrained pooling (`r1check.py` -> r1check.json; R1 windows, R1 agg scoring, all 6,450 test rows)
- Ensemble ADS reproduces r1/grid.json exactly.
- LME/mean - control:
  - Single seeds: +.0115, -.0007, +.0045 (sd .0061 > mean .0051).
  - Bootstrap: pct CI [-.0005, .0109], P>0 .964.
- Music-mean part: music EER per seed -.0214, +.0018, -.0108. R1 itself found every seed worse on S3's MERT-6+12 features.
- Voice LME only (LME/LME - control): +.0013, +.0004, +.0008 (3/3 > 0). Ensemble +.0009, CI [-.0003, .0030], P>0 .938.
- The val/test rank agreement over the 2x3 grid is effectively two factor-level comparisons. It is weak evidence.

## Q7: seen vs unseen by AD/DF-Arena/SONICS (fusion_fit.member_seen; fits per subset as orchestrator/s3_unseen_check.py)
- Unseen (2,706 test, 1,054 groups):
  - Triple A .0078. Reproduces the orchestrator.
  - Triple B .0120. Single-seed mean .0104 (sd .0050, 6/6 > 0).
  - Bootstrap pct CI [.0017, .0166], P>0 .993.
- Seen (2,116 test, 1,290 groups):
  - Triple A .0154, triple B .0208. Single-seed mean .0133 (sd .0061).
  - Bootstrap pct CI [-.0036, .0298].
- Seen - unseen: .0076, CI [-.014, .023], P>0 .69. Seen > unseen in 4/6 paired seeds. The difference cannot be told apart from noise.

## Not covered
- Uncertainty from training-set sampling: no bagging over train groups.
- P1-protocol refits: no held-out estimate exists.
- R3 song-aware VOICE: not bootstrapped here.
- LB-side noise: estimated only by scaling (interpretation, not measured).
