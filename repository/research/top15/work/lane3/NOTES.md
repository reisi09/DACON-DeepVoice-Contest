# Lane 3: data and evaluation validity (2026-09-27, KST)

Every result below comes from read-only CPU checks. Each check took 10 s or less and used under 0.5 GB of RAM. Scripts and JSON outputs are in this folder:
- `lane3_checks.py split|duration|subset|coverage` writes `split.json`, `duration.json`, `subset.json` and `coverage.json`.
- `dur_adjust.py` writes `dur_adjust.json`. It fits a source-adjusted logistic model and runs a 200-rep group-cluster bootstrap.
- `s3_bootstrap.py` writes `s3_bootstrap.json`. It reproduces `ads_compare.json` exactly and runs a 1,000-rep test-group bootstrap.

Inputs:
- The bank: `results/data_v4_20260923/.../data/bank/bank_{trainval,test}.csv`.
- Durations: `samples/16000` from `_workspace/cache_v4/{cache_tv,cache_test}/index.json`. All 42,039 rows have one.
- DB-head predictions: `_keep_best_20260926/train_duration_balanced/{val,test}_predictions_decision.csv`. These have 4,927 and 6,444 rows. The rows dropped (4 in val, 6 in test) are exactly the rows with an unknown FILE_FAKE label.

## 1. Split integrity (full population, `split.json`)
- Across train/val/test there is no overlap in ID, orig_id, audio_sha256, source_sha256, group_id or component_id. There are also no duplicate audio_sha256 values within a split.
- There are 2,321 constructed rows carrying 4,342 parent references. Every reference resolves to a bank orig_id. None of them points to another split or another group, and no parent is used in more than one split.
- group_id == component_id for every row. There are 18,852 unique groups. The brief's figure of "21,817 groups" does not match the bank. Where that number comes from is UNVERIFIED.
- NEW: SpeechFake MD Korean CosyVoice, MeloTTS and OpenVoice files reuse Zeroth utterance IDs (`speechfake:MD/OpenVoice/ko/123_003_1960.wav` next to `zeroth:123_003_1960`). `build_manifest.ing_speechfake` (lines 104-131) groups them only by `speechfake_text:{stem}`, and `ing_zeroth` groups by `speaker:zeroth:{id}`. Nothing links the two sources. Effects:
  - 49 utterance IDs, meaning the same speaker and text, appear in more than one split.
  - 64 of 114 Zeroth speakers span splits.
  - 367 of 427 held-out test rows with Zeroth-style IDs (306 of them OpenVoice fakes) have a speaker who also appears in train.
  - This is a speaker/text lineage gap. It is not a label leak: the linked rows carry opposite labels, so the bias could go either way.
- ASVspoof 2019 LA (train only, 2,400 rows) and 2021 LA (train/val) are not linked (split_rules.json `_why`). Speaker overlap between them is UNVERIFIED because the bank has no speaker column. It affects val, not test, since test holds only 5 LA rows.

## 2. Duration bias, controlled for source (`duration.json`, `dur_adjust.json`)
Held-out test, DB heads. The global file-EER is 0.1094 at threshold 0.3347.
- Pooled over REAL files, Spearman(FILE score, duration) is +0.195. Within sources it is larger: the n-weighted mean is +0.218, against a within-source permutation null of mean -0.002 and sd 0.017 (p < 1/300).
- Per source (reals):

  | Source | Spearman | n | Groups |
  |---|---|---|---|
  | fleurs_ko | +0.29 | 1,500 | 1,150 |
  | vocalset | +0.39 | | 10 |
  | haim B1 | +0.26 | | 34 |
  | constructed_enhanced | +0.21 | | |
  | constructed_song | +0.15 | | 10 |
  | musan_music | +0.12 | | 77 |
  | constructed_mix | +0.12 (p = 0.18) | | |
  | zeroth | +0.11 (n.s.) | | 4 |
  | openmic | +0.03 (n.s.) | | |

  openmic durations are capped at 10 s.
- The correlation persists inside source × codec cells. In fleurs_ko it ranges from +0.16 to +0.42 across all 11 codecs.
- Source-fixed-effect logistic model (group bootstrap):
  - False-positive odds on reals rise by a factor of 1.22 per doubling of duration (95% CI 1.08-1.38).
  - Miss odds on fakes fall by a factor of 0.74 (0.66-0.85).
  - So longer files score higher in both classes. This fits the voice path's window max (`full_bank_evaluate.score_rows` line 58, `.max()`). The music path uses a length-normalised log-mean-exp (line 56). The mechanism is a hypothesis.
- Pooled FPR by duration bin: 7.4% at 4-8 s, 10.7% at 8-15 s, 15.2% at 15-30 s, 22.0% at 30-60 s. Much of this slope comes from the source mix (musan_music is long and has a high FPR). Within fleurs the rise is small (1.8% to 3.1% to 2.7%). Within musan_music it is 17% to 19% to 21% to 22%.
- Val shows a weaker effect: within-source rho is +0.05 (p = 0.037). It is negative within ASVspoof, where 85% of rows are under 4 s.
- Verdict: the length bias is real within sources. It is moderate in size and concentrated in long reals. The official test runs 4-60 s. Train has 26% of rows under 4 s; the held-out test has 4%.

## 3. Backbone-cache subset (31,270 rows, 74.4%) (`subset.json`)
- Coverage is 0.743 in train, 0.746 in val and 0.748 in test. By label it ranges 0.735-0.757.
- By major source it ranges 0.71-0.79, and by first hex digit of the ID 0.717-0.768. No source has zero coverage.
- The total-variation distance between the subset's source mix and the full bank's is 0.006-0.014.
- Duration: train and val match (KS p = 0.22 and 1.0). In test the subset runs slightly longer (KS D = 0.041, p = 0.03; medians 9.7 vs 9.3 s).
- The same DB-head predictions give ADS 0.86317 on the subset, 0.86314 on the full test and 0.86212 on the remainder.
- Verdict: the subset is representative. Subset-based deltas can be read as full-test deltas, to within about 0.001 ADS.

## 4. Selection validity of S3 (music MERT [6,12], AD weight 0.5, FILE refit)
Timeline from file mtimes (KST) and git:
- `backbone_heads_screen/screen_report.json` (07:29) prints val and test EER for each of 8 layers.
- `music_layer_seeds.json` (07:33) prints per-seed val and test for [12], [6], [6,12] and [3,6,9].
- `ads_compare.json` (07:37:40) prints val and test ADS for 4 configurations.
- `data_v4/train_backbone_heads.py` (07:39:58) first adds `--final`, `--music-layers` default [6,12] and `--ad-weight` default 0.5 (git diff b6dfcb4..3753e10).
- The final heads were written at 07:40-07:41.
- The commit message for 3753e10 justifies S3 with the held-out 0.8644 -> 0.8727.

So the held-out test was visible before both choices were made.

Where val and test disagree:
- Music combination: val would pick [3,6,9] (val mean 0.1156 against 0.1167 for [6,12]). Test picks [6,12] (0.1810 against 0.1843). The pick matches test.
- AD weight: val prefers w = 1 (ADS 0.92984 against 0.92912 for 0.5). Test prefers 0.5 (0.87267 against 0.87209). The pick matches test.

Where val and test agree:
- L6 is the best single layer on both.
- Every S3 configuration beats the baseline on val as well (+0.0045 to +0.0121 val ADS).

FILE refit: the ensemble is fit on val only (`file_ensemble.fit`, grid adopted when it gains at least 0.002). For S3 the grid won (0.05852 -> 0.05438), giving weights {noisy_or 9, max_nopresence 1}. No test data is used here, but the val FILE metric is in-sample.

Exposure:
- Test-guided tie-breaks can account for at most about 0.001 ADS (music: 0.3 × 0.0033) plus 0.0006 ADS (weight). That is about 0.0016 ADS of the claimed +0.0083.
- Bootstrap over 2,215 test groups (1,000 reps), change in ADS against the baseline:

  | Change | Mean | 95% CI | P(≤ 0) |
  |---|---|---|---|
  | Full S3 | +0.0092 | 0.0028 to 0.0161 | 0.001 |
  | Music [6,12] only | +0.0044 | -0.0014 to +0.0108 | 0.065 |
  | w = 0.5 minus w = 1 | +0.0001 | -0.0019 to +0.0019 | |

- On the held-out test alone, the music gain is not significant.

Caveats:
- The final package trains on train plus the held-out test (P1). No untouched held-out estimate exists for the shipped S3; the leaderboard is the only external check.
- Across the session the held-out test has been consulted many times (screens, fusion variants, learning curve, error analysis). It now works as a second validation set.
- Held-out ADS is about 0.863; LB ADS is about 0.77. Earlier held-out-style gains of this size moved the LB by only ±0.002.

## 5. Coverage and label quality (`coverage.json`)
Nominal rows against independent groups (group_id is a union-find component, so it is a lower bound on independent units):

| Source | Rows | Groups |
|---|---|---|
| asvspoof2019_la | 2,400 | 4 |
| asvspoof2021_la | 3,672 | 67 |
| asvspoof2021_df | 2,875 | 81 |
| zeroth | 3,000 | 114 |
| vocalset | 800 | 17 |
| musan_speech | 426 | 2 |
| csd | 200 | 1 |
| constructed_song | 400 | 10 |
| existing:constructed_music | 128 | 1 |
| musan_music | 645 | 77 |
| mlaad | 8,775 | 8,169 |
| speechfake | 3,600 | 2,774 |
| openmic | 2,764 | 2,510 |

Independent test units behind each component metric:
- Music EER, reals: 1,825 rows but only 229 groups.
- Music EER, fakes: 617 groups, of which HAIM accounts for 596.
- Voice EER, reals: 1,200 groups, 1,150 of them FLEURS-ko.

Coverage against the official test description:
- Duration: 26.1% of train is under 4 s, against 0% in the official test. The train p90 is 15.5 s; the official maximum is 60 s. The held-out test has 3 files over 60 s, reaching 75.1 s.
- Codecs: wav, flac, mp3 64/128/192, aac, opus and 4 telephone chains, each about 20% per split. The share of reals is flat across codecs (about 30% in train), so there is no codec shortcut.
- Channels (mono/stereo): no bank column. UNVERIFIED. The runtime downmixes to mono.
- Language: train voice in languages other than ko/en is 100% fake (2,005 fake, 0 real). English is 81% fake. This is a possible language shortcut (hypothesis).

Label checks:
- constructed_enhanced: 300 rows, all REAL. They are real zeroth/fleurs speech plus MUSAN noise at 0-15 dB, passed through HTDemucs vocals (`build_manifest.enhance_one`), which matches the DACON rule. 276 of the 300 are in test, so in the DB protocol only 17 were trained on.
- HAIM B1 (SonicMaster AI-mastered human music): REAL. This fits the rule wording, but SonicMaster is generative, so it is an ambiguous case.
- HAIM B2 (fx_encoder): FAKE. The local HAIM README defines B2 as "human_mastered_ai_dsp", meaning AI tracks with DSP applied, so FAKE is correct.
- Song vocals: SONICS 743, echoes 345 and HAIM A2/B2 438 rows have vocals labelled VOICE_FAKE = 1 (generated with the music). The existing:SONICS and existing:HAIM rows have VOICE_FAKE masked (NA).
- Masks: 1,007 rows are missing VOICE_PRESENT, 8,492 VOICE_FAKE and 29 FILE_FAKE (HAIM concat/crossfade).
