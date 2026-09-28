# Review packet v1 — DACON 236749 model audit (2026-09-28 21:10 KST)

Same packet for the coordinator (Claude Opus 5.5) and the independent reviewer (GPT-6-Astra, reasoning effort ultra).
Everything here is SOURCE EVIDENCE. The coordinator's own recommendations are deliberately NOT included; form your
conclusions from these files first.

## 0. Identity
- Project repository: local git, HEAD f20e8b3 (2026-09-28). Uncommitted: research/top15/work5/newdata/bcast_rows.py
  (an unstarted experiment script; not part of any model). History: history/git_log_60.txt.
- Excluded on purpose: AI-Hub audio, features, row lists and file identifiers (Korean data that must not leave the
  local machine); credentials; the coordinator's plan documents and debate transcripts.

## 1. Competition (verified text: history/research/top15/work/lane1/official_text.txt)
- Task: per audio file predict FILE_FAKE_PROB, VOICE_FAKE_PROB, MUSIC_FAKE_PROB, VOICE_PRESENT_PROB, MUSIC_PRESENT_PROB.
- Score = 0.9 x ADS + 0.1 x CPS; ADS = 0.5 (1 - FILE EER) + 0.2 (1 - VOICE EER) + 0.3 (1 - MUSIC EER);
  CPS = 0.5 AUC(voice presence) + 0.5 AUC(music presence). EER per the official snippet (argmin |fpr - fnr|).
  VERIFIED (official_text.txt:141-142): FAKE is the positive class; Voice EER only on files with voice present, Music EER
  only on files with music present.
- Hidden test: 1,200 files, 4-60 s, 16 kHz, mono/stereo, MP3/WAV/FLAC, "일부 샘플에는 전화채널 오디오"; types: speech,
  music (no vocals), mixed (voice and music "동시에 또는 순차적으로"; songs are mixed). FAKE = any AI-generated component;
  enhancement / denoise / loudness only stays REAL.
- Runtime: L4 22.4 GiB, 6 vCPU, 28 GB RAM, inference <= 60 min for 1,200 files, install <= 10 min, zip <= 10 GB (32 GB
  unpacked), offline. Files must be predicted independently.
- Deadline: leaderboard submissions close 2026-09-29 10:00 KST; 3 submissions per day; the remaining 3 fall on
  2026-09-29 00:00-10:00. VERIFIED (official_text.txt:144-147): Public = 100 % of the test data; Private = the Public
  score at close. Whether the team's best or last submission counts is not stated there (REPORTED: best).
  Top 15 enter a second round (reports + all training data files, incl. generated audio); top 7 win.

## 2. Reported leaderboard history (REPORTED by the user; scores typed from the DACON page)
| submission | package (receipts/) | sha256 prefix | LB ADS | LB total |
|---|---|---|---|---|
| DurationBalanced (09-24) | DataV4_DurationBalanced | see receipt | 0.7683 | 0.7912534339 |
| P1 (09-27) | DataV4_P1Refit | 1b9ba441 | 0.7700396825 | 0.7928 |
| S2a (09-27) | DataV4_S2a_AIHubPhone (not in packet: AI-Hub-trained) | c7c8b1c8 | 0.7688571429 | 0.7917 |
| S3 (09-27) | DataV4_S3_Backbones | 062fc236 | 0.7853015873 | - |
| S6 (09-28 ~02:20) | DataV4_S6_S5LabelCompletion | 7429b183 | 0.8220079365 | - |
| S7 (09-28 ~04:20) | DataV4_S7_SSLAMWavLM | 5239c912 | **0.8373968254 (best)** | ~0.8535 |
| S8 (09-28 ~07:15) | DataV4_S8_MixSSLAM | b0e87442 | 0.8339682540 | - |
Earlier history (09-13 .. 09-23, other pipelines): history/research/top15/experiment-ledger.csv and
history/research/top15/work/lane2/experiment-ledger.csv, history/_workspace/.../panns_leaderboard_20260926/result.json.
S4 (sha beeda9a6) and S5 (sha cf17eec9) were built and verified but, per the notes, not submitted (UNKNOWN: confirm).
The 09-26 top-15 cut was total 0.8737 (~ADS +0.022 over S7).

## 3. Current plan (baseline definitions only; not a recommendation)
- Incumbent / LB-best: S7 = submissions/DataV4_S7_SSLAMWavLM (receipt sha 5239c912...; runtime in runtime_pkg/DataV4_S7_SSLAMWavLM).
- Planned slot 1: s7_new = S7 recipe + 40,214 new rows (receipts/DataV4_s7_new, runtime_pkg/DataV4_s7_new).
  Planned slot 2: W_newB (W reweighting + newval in early stopping). Slot 3: W_new (or an alternative).
- New rows: data/new_rows.csv (+ data/new_data_summary.csv); S7 bank: data/bank/*.csv (35,589 trainval + 6,450 test).

## 4. Local evaluation sets in use (and their exposure)
- "old val" (4,931 bank val rows): early stopping and CDF tables of every head (all configs).
- E1 = DB held-out 4,822 rows of bank test (used to compare many configs today and on earlier days).
- E2 = newtest 12,426 rows of new data (80.9 % AT-ADD); 75 rows lineage-linked (results/.../leak_check.txt).
- newval = 3,841 new rows, used for early stopping only by W_newB (NEW_VAL=1).
- E3 = 257 AI-Hub Korean real files, Mac only; only aggregate metrics here (results/.../new_eval_e3.json).
- Results: results/research/top15/work5/newdata/{new_eval_newtest.json, virtual_configs_eval.txt, select_newval.txt,
  leak_check.txt, e2_subsets.txt}; results/research/top15/work4/rt_s6enc_report.json.

## 5. Resources and limits for the future execution phase
- GPU budget ceiling USD 50 total, but the VESSL credit is $18.75 (2026-09-28 20:54 KST); H100 x1 $2.98/h.
  Reuse workspace wsp-9683432r72m7 (stopped); CPU workspace wsp-56syy2qpy2oe ($0.30/h, stopped).
- Time: about 12.5 h until the last upload window closes. Feature dump ~18 audio-s/s per H100.
- The package is 9.745 GB of the 10 GB limit.

## 6. Review questions (answer in Korean; label every claim VERIFIED / REPORTED / HYPOTHESIS / ESTIMATE / UNKNOWN /
NOT RUN; cite file:line; separate confirmed defects from risks; list what you did NOT review)
1. Data leakage and adaptive overfitting across the pipeline: repeated held-out selection (E1/E2/old val/newval),
   lineage (speakers, songs, source tracks, generators, remix donors, derivatives), calibration/CDF tables and
   early stopping fitted on data later used for selection.
2. Evaluation validity: which local set, if any, is a defensible proxy of the hidden evaluator; single-class slices;
   all-positive slices (e.g. realV+fakeM); comparisons across differently calibrated score scales (CDF-mapped members).
3. Implementation correctness and training/inference integration: trace data -> labels/masks -> features -> training
   -> checkpoint selection -> CDF/fusion -> packaged inference (runtime_pkg/*/model/runtime/full_bank_submission.py,
   decision_heads.py, component_fusion.py) -> output. Specifically: missing-member handling and zero-filled scores,
   silent fallbacks, whether the SSLAM and WavLM heads actually affect packaged predictions, fp16 rounding, resampling,
   channel handling, segmentation, differences between the S7 runtime and the s7_new runtime.
4. Improvement feasibility within the limits of section 5: which improvements do not need new data (sampling, loss
   weights, masks, fusion/calibration with disjoint fitting, member removal), and which need data; overfitting risk.
5. What evidence would be needed to call any local set "independent final evaluation"; does such data exist here?
