# Shared research brief — DACON 236749 top-15 (observed 2026-09-27 13:28 KST)

Project root: `/Users/choiwuseck/Documents/ChatGPT/데이콘` (local git HEAD f81e24c, 934 uncommitted/untracked
entries — the working tree, not git, is the source of truth). Published mirror clone (code + small results):
`/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/repo` at 59c2318
(`repository/` subdir). Python env: `.venv/bin/python` (torch 2.7.1, transformers 4.57.6, timm 1.0.15).
No active training or extraction anywhere at 13:28 KST: local MPS idle, VESSL workspace `wsp-9683432r72m7` stopped.

## Competition (official sources saved locally)
- DACON 236749 "딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회". Page dumps: `/tmp/dacon_description.json`,
  `/tmp/dacon_data.json`, `/tmp/dacon_evaluation.json`, `/tmp/dacon_rules.json`, `/tmp/dacon_talk_full.txt`
  (talk board incl. official replies 417702, 417844, 417857, 417885), `/tmp/dacon_codeshare.json`, `/tmp/dacon_talkboard.json`.
- Per audio file predict FILE_FAKE_PROB, VOICE_FAKE_PROB, MUSIC_FAKE_PROB, VOICE_PRESENT_PROB, MUSIC_PRESENT_PROB.
  Score = 0.9·ADS + 0.1·CPS; ADS = 0.5(1−FileEER)+0.2(1−VoiceEER)+0.3(1−MusicEER); CPS = mean presence ROC-AUC.
  EER: roc_curve(drop_intermediate=False), argmin|fpr−fnr|, (fpr+fnr)/2; FAKE = positive; Voice/Music EER only on files
  where that component is present. Public = 100% of test; Private = Public at competition end; top-15 by Private go to
  a 2nd-round document review (model + training-data reports, training code, training data files, source attribution).
- Test: 1,200 files, 4–60 s, 16 kHz, mono/stereo, mp3/wav/flac, "some" phone-channel audio; vocals count as voice;
  enhancement/denoise/volume-only processing of real audio = REAL. Code submission: ≤60 min inference for 1,200 files on
  L4 22.4 GiB, 6 vCPU, 28 GB RAM, offline, ≤10 GB zip (≤32 GB unzipped), install ≤10 min, 3 submissions/day,
  submission deadline 2026-09-29 09:59:59 KST. Runtime error counts against the daily limit.
- Rules: any public resource usable at least non-commercially (NC pretrained OK); no training/tuning/pseudo-labeling on
  private test; per-file independence (fixed reference tables from training data allowed, 417885); LB probing discouraged
  (417702); AI-Hub allowed (417844). Official preinstalled packages listed in `/tmp/dacon_evaluation.json`.

## Leaderboard history (team killerpjw; official ADS/total as reported by the user)
| submission | ADS | total | notes |
|---|---|---|---|
| Balanced7144 | — | 0.7559 | earlier Codex model |
| (prev best) | — | 0.7579 | |
| DataV4_DurationBalanced (09-24) | 0.7683 | 0.7912534339 | data v4 heads + presence + FILE ensemble |
| PANNs R3 (Codex branch) | — | 0.7901426243 | regression |
| **DataV4_P1Refit (09-27)** | **0.7700396825** | 0.7928 | best so far; held-out 6,450 rows added to training |
| DataV4_S2a_AIHubPhone (09-27) | 0.7688571429 | 0.7917391481 | P1 + 800 AI-Hub phone reals |
| **DataV4_S3_Backbones (09-27, ~14:00 KST)** | **0.7853015873** | ≈0.8065 (CPS unchanged) | **new best: +0.0153 ADS vs P1**; MERT 6+12 music heads + AntiDeepfake-embedding voice heads fused with DF-Arena heads |
CPS ≈ 0.99768 (saturated). ADS values are multiples of 1/126000 (P1 97025, S2a 96876, S3 98948), a check that a reported number is ADS not total.
S3 LB gain (+0.0153) EXCEEDED its held-out gain (+0.0083) — first time LB > held-out; earlier data-v4 transfer was ~0.6x. On rows unseen by AntiDeepfake/DF-Arena/SONICS training data the S3-style held-out gain is +0.0078 (val +0.012), on seen rows +0.015 (research/top15/work/orchestrator/s3_unseen_check.py). 09-26 night snapshot: rank 120/463, #1 0.898715, top-15 cut 0.8737 (total). Gap ≈ +0.08 total.

## Pipeline (runtime `full_bank_submission.py`, packaged by `package_datav4_submission.py`)
load → peak cap 0.6 → HTDemucs vocal/backing stems → VOICE: DF-Arena 1B (`_workspace/assets_e9504/baseline/model/df_arena_1b`)
on 4 s voice segments, fc5 input 1280-d → 3 Linear heads (seeds 41/42/43), window max, seed mean; MUSIC: MERT-v1-330M
layer 12 time-mean per 4.04 s window (24 kHz) of the backing stem → WindowMIL MLP (1024→128→1) log-mean-exp, seed mean;
PRESENCE: PANNs CNN14 tag stats + stem energy → logistic heads (`presence_head.json`); FILE: equal-weight mean of val-CDF of
max(Pv·Fv,Pm·Fm), noisy-or, max(Fv,Fm) (`decision_heads.py`, `file_ensemble.json`). S3 adds MERT layers 6+12 music heads,
AntiDeepfake-1B voice-stem window embeddings → 3 Linear heads, VOICE = val-CDF fusion (ours 1, ad 0.5) mapped back to our
scale (`component_fusion.py`), FILE refit; 48-min fallback skips AntiDeepfake.
Training: `full_bank_train.py` (weighted BCE; loss_weights = class×source×generator balance; duration_balance; AdamW
lr 1e-3 music / 1e-4 voice, wd .01, 256 files per update, clip 1, patience 5, ≤50 epochs; val EER then BCE selection).
New-backbone heads: `data_v4/train_backbone_heads.py` (screen + `--final`).

## Data (see `repository/data/data_v4_bank` in mirror; local bank `results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank`)
- data v4: 42,039 rows / 21,817 groups; train 30,658, val 4,931, held-out test 6,450 (source-held-out split,
  `data_v4/split_rules.json`). Audio lives only on VESSL (/root/data_v4/out); local has feature caches only:
  `_workspace/cache_v4/{cache_tv,cache_test}` (DF-Arena fc5 inputs, MERT-12 windows, presence feats; H100 fp32) and
  `_workspace/backbone_cache` (31,270 rows, 74%: AntiDeepfake-1B window embeddings fp16, MERT layers 3..24 fp16).
- AI-Hub (local only, never upload/export): `_workspace/aihub/*`, usage index `repository/data/aihub_usage`. S2 extra rows
  `_workspace/s2_extra` (A = AI-Hub speech/channel transplants, B = Jamendo, Music Arena, AI-Hub 71470), partially featurized.
- Member scores (zero-shot AntiDeepfake 1B/2B + SONICS) for val+test: `_workspace/members_20260927/members_all.csv`.

## Established findings this session (evidence in `_workspace/audit_20260927`, mirror `repository/results/*`)
- S2a refuted AI-Hub phone reals as LB lever (−0.0012 ADS). Head-only data additions moved LB ±0.002.
- Zero-shot AntiDeepfake fusion and SONICS rejected (after fixing a CDF-scale FILE bug in `data_v4/fusion_fit.py`).
- Learning curve (nested groups, 1 seed): music held-out flat beyond 50% of groups; voice still improving; voice heads hit
  the 50-epoch cap.
- New-backbone heads (DB protocol, matched rows, 3 seeds): held-out official ADS 0.8644 → 0.8727 (music MERT 6+12 +
  voice AD-embedding fusion); music gain consistent over seeds. `_workspace/audit_20260927/ads_compare.json`.
- Duration shift: train p90 17.5 s, 22% < 4 s, 0% > 60 s vs test 4–60 s; held-out Spearman(score, duration | real voice)
  = 0.36 (not controlled for source). Held-out error concentration: HAIM fakes, MUSAN/constructed real music, FLEURS-ko
  real speech, enhanced real speech (25% FPR), MLAAD fakes.
- Device parity MPS/CPU vs H100 caches verified (10 files); bf16 vs fp32 heads negligible; metric recomputation exact;
  primary split has zero ID/bytes/component/group/parent overlap.
- VESSL credit $9.85 left (H100 $2.98/h); AGENTS.md: reuse wsp-9683432r72m7 only, stop after downloads.
- USER DIRECTIVE 2026-09-27: GPU budget = wall-clock time only; money is not a constraint (credit top-up may be needed beyond ~3.3 h).

## Hard constraints for every agent
No GPU/VESSL start, no paid compute, no downloads beyond small official documentation pages, no dataset uploads, no
git commit/push, no competition submission, no edits to project source files, caches or submissions. Write only inside
`research/top15/work/<your-lane>/`. CPU checks allowed if ≤10 min and ≤4 GB RAM; state the question they answer.
