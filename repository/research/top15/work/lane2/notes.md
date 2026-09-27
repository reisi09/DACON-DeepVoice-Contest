# Lane 2 notes: repository and experiment history (observed 2026-09-27, KST)

Read-only. Files written here: build_ledger.py, experiment-ledger.csv, cache_coverage.py, notes.md. Two CPU checks were run
(stdlib csv plus a directory listing, under 1 s and 110 MB RSS each). Their questions are stated below.

## 1. Official LB timeline (team killerpjw)
Source: results/balanced7144_regression_20260919/live_submissions.json (authenticated table, observed 2026-09-19T05:38Z);
results/precision_audit_20260918/live_submissions_20260918.txt; DACON_LB_IMPROVEMENT_AUDIT_2026-09-24.md:19;
_workspace/lb_cpu_20260924/docs/evidence/panns_leaderboard_20260926/result.json; E08_REGRESSION_2026-09-24.md; memory file.

| sub | time (KST) | model (link strength) | total | ADS | CPS |
|---|---|---|---|---|---|
| 88418 | 09-13 00:35 | G0 baseline (verified name) | 0.6909110688 | .6577539683 | .9893249735 |
| 88630 | 09-13 11:25 | AST epoch17 (name) | 0.6453405503 | .6212619048 | .8620483598 |
| 88673 | 09-13 12:48 | AST full1029 (name) | 0.6174819841 | .5881111111 | .8818198413 |
| 88809 | 09-13 17:25 | G0 resubmit | 0.6909110688 | | |
| 89372 | 09-14 11:17 | V5_B1 | error: libtorchaudio.so load | | |
| 89524 | 09-14 15:48 | V5_B1_fixed | 0.6776485873 | .6451031746 | .9705573016 |
| 89989 | 09-15 07:22 | V6 | error: numpy dtype ABI | | |
| 90004 | 09-15 08:42 | V6 fixed | 0.6727822646 | .6376031746 | .9893940741 |
| 90217 | 09-15 14:06 | C3 | 0.6843536931 | .6504603175 | .9893940741 |
| 90692 | 09-16 02:34 | C08 | 0.7314822646 | .7028253968 | .9893940741 |
| 90735 | 09-16 07:05 | M330_MLP_full1206 | 0.7396465503 | .7118968254 | .9893940741 |
| 91238 | 09-16 23:06 | S2 EAT (user-identified) | 0.6378679788 | .5988095238 | .9893940741 |
| 91610 | 09-17 12:20 | submit.zip; B_full1696 packaged 12:15 (inferred) | 0.6837822646 | .6498253968 | .9893940741 |
| 92294 | 09-18 10:30 | MERT12_full1322 (memo has ZIP sha) | 0.7479394074 | .7211111111 | .9893940741 |
| 92644 | 09-18 22:18 | submit.zip; Expanded_MIL_3659 packaged 20:44 (inferred, unverified) | 0.7579036931 | .7321825397 | .9893940741 |
| 93093 | 09-19 14:11 | Balanced7144 (conversation context) | 0.7559347884 | .7299920635 | .9894193122 |
| 102411 | 09-24 02:09 | DataV4_DurationBalanced (sha on README, screen shows submit.zip) | 0.7912534339 | .7683174603 | .9976771958 |
| ? | 09-24 | E08_Partial5952_Top30 (user-reported ADS only) | UNVERIFIED | .7064761905 | UNVERIFIED |
| 104224 | 09-26 21:13 | PANNs R3 fix1 (contextual) | 0.7901426243 | .7674603175 | .9942833862 |
| ? | 09-27 ~02:10 | DataV4_P1Refit | ~0.7928 (brief) | .7700396825 | (same presence heads) |
| ? | 09-27 | DataV4_S2a_AIHubPhone | 0.7917391481 | .7688571429 | |
| ? | pending | DataV4_S3_Backbones | UNVERIFIED | | |

Existing9504_MIL: submissions/Existing9504_MIL/README.md says "No DACON submission was made"; no LB record found (grep over
*.md/*.json/*.txt). Its official score is UNVERIFIED (probably never submitted). Its zip (sha b19b88de) is base_archive_sha256 of
every DataV4 package (submit.json choice.base_archive_sha256).

## 2. Validation / held-out / LB relation (computed from file_ensemble.report.json val_official_decision)
| model | val ADS (4,927 FILE rows) | held-out ADS (6,444) | LB ADS |
|---|---|---|---|
| Existing9504 | .9311 (own split, exposed) | .7989 legacy route | not submitted |
| Balanced7144 | .9250 (own split) | .8021 legacy route | .7300 |
| DataV4 DurBal | .91875 | .86314 (total .8767) | .76832 |
| P1 | .92410 | in-sample | .77004 |
| S2a | .92107 | in-sample | .76886 |
Paired deltas (same val panel): DurBal->P1 dval +.00535 / dLB +.00172 (ratio .32); P1->S2a -.00302 / -.00118 (.39).
Held-out->LB: Balanced7144->DurBal dheld +.0611 / dLB +.0383 (.63), one pair, mixed model+presence+FILE changes.
Interpretation only (n=2 and n=1). Top-15 cut 0.8737 with CPS .99768 needs LB ADS ~.860 = our current held-out ADS.

## 3. Protocol defects found in this pass
- P1 file_ensemble.report.json "test_one_shot" ensemble_eer .05151 and fusion_20260927/p1_* baseline_test ADS .94459/.95523 are
  in-sample (P1 trained on bank_test rows, run_identity extra_train 50e993a1 = bank_test_as_val). Voice test EER stays .100 even
  in-sample while music drops to .032 -> interpretation: voice route is representation-limited, not data-starved.
- Backbone selection saw the held-out test: music_layer_seeds.json val means [3,6,9] .1156 < [6,12] .1167 but test .1843 > .1810;
  ads_compare.json val ADS w=1 .92984 > w=.5 .92912 but test .87209 < .87267. S3 took [6,12] and w=.5 (defaults in
  data_v4/train_backbone_heads.py:115,117; no rationale recorded). File mtimes: screen 07:29, layer seeds 07:33, ads_compare 07:37,
  final heads 07:40-07:41. Magnitude small (<=.0007 ADS), but the .8727 figure is not a clean one-shot estimate.
- Zero-shot fusion unseen runs db_unseen_1b and db_unseen_all used --select valtest (fusion_report.json "select"); the +.0017 test
  ADS quoted for w=.25 was selected with test. All reports were written 04:21-04:27, after component_fusion.py mtime 04:20:14.
- Backbone screen: voice_ad val .0485 < DF .0509 but test route .1395 > .1203 (screen_report.json): val->test inversion.
- S3 val metrics (validation_metrics.json) are on a 3,676-row feature subset; not comparable with 4,927-row val numbers.
- Echoes generator parse still collapsed in the bank: only (echoes, Echoes/TTA) 300 and (echoes, Echoes/ATA) 294
  (data_v4/build_manifest.py:141 `gen = parts[parts.index(mode) - 1]`, mtime 09-23 18:54). All DataV4/P1/S2a/S3 loss weights use it.
- Constructed rows skewed to held-out test (check 2 below): constructed_song 400 test / 0 train / 0 val; constructed_enhanced
  276 test / 17 train / 7 val. Explains the held-out "enhanced real speech 25% FPR" concentration and part of P1's gain.
- E08 compared against a freshly retrained E00, not the submitted DataV4 (E08_REGRESSION_2026-09-24.md).
- Balanced7144 val allowed source-only music AUC 1.0 and length-only AUC .887 (ANALYSIS.md s2).

## 4. CPU checks (questions answered)
1. cache_coverage.py: "Is the 74% backbone-cache coverage balanced, or is the S3 / DB-protocol comparison on a biased subset?"
   Result: 31,270/42,039 cached; split .74-.75, route .73-.75, FILE label .73-.76; no val/test source with n>=50 outside .5-.95.
   Coverage looks uniform (likely hash-ordered partial extraction), so subset bias is not evident at this granularity.
2. inline script: "Are constructed rows skewed toward the held-out test?" Result above (yes, strongly for song/enhanced).

## 5. Unresolved opportunities (evidence never produced)
- S2-B music-only extra data (Jamendo, Music Arena, AI-Hub 71470): dev features 515/515, train 401/1,913; never trained.
- S2-A AI-Hub channel-transplanted fakes: train features 3,186/5,341, dev 6/1,200; never trained.
- Codex 1,477 admitted rows (LibriSpeech 449, MusicArena 882, MLAAD 146): 0 cached (LOCAL_EXPERIMENTS_2026-09-25.md).
- Stacking v2 FILE meta-learner: CPU/synthetic only (mirror plan.md; EXPERIMENT_INDEX.md).
- Echoes generator fix + re-weighting; split-internal constructions (audit 2026-09-24 P0-2): not done.
- Voice capacity: heads hit 50-epoch cap (results.json voice_41/43 epoch 50/50) and learning curve still improving; no
  non-linear voice head on DF-Arena/AD embeddings evaluated on full data (codex residual MLP only on 5,952 seen rows).
- raw+stem / time-aligned voice-music features for singing: diagnosed (voice-with-music EER 20.6% vs 4.4%) but untested on v4.
- V5 B3/B4 temporal/joint heads, redesign C/D mixed training: never completed.
- S3 official score and L4 runtime: pending/unverified.
