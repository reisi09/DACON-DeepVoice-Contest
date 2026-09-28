# Round 15 — design ONE integrated model for the last submissions (2026-09-28 16:50 KST)

Goal: a single, normally trained candidate that is as strong as possible on every factor we believe drives the DACON
grader. All selection is done on the validation rows only; the held-out sets (E1/E2/E3) are used once, for confirmation
with pre-registered gates. The leaderboard is used only to submit finished best-effort models (no probing, no
deliberately weakened variants). No GPU is running now; this round is design only. Answer in Korean.

## 1. Fixed facts (VERIFIED)
Metric: ADS = 0.5(1 - FILE EER) + 0.2(1 - VOICE EER) + 0.3(1 - MUSIC EER); total = 0.9 ADS + 0.1 CPS (CPS ~0.998, saturated).
LB = whole test (1,200 files); best-ever counts. Deadline 2026-09-29 10:00 KST; 3 submissions left, all on 09-29.
Runtime: offline L4, <= 60 min (S7 took ~30 min), zip <= 10 GB (S7 = 9.74 GB, so NO new encoder fits), GPU memory:
S7 peak ~20 GB of the 22.4 GiB budget.

Architecture (S7; code in code_r14/: full_bank_submission.py, train_backbone_heads.py, run.py, voice_data.py):
- Frozen encoders: HTDemucs split -> voice stem / backing stem.
  voice stem: DF-Arena-1B fc5 input (1280-d per segment), AntiDeepfake XLS-R-1B (1280-d per window), WavLM-L hidden 8
  (1024-d per 4.04 s window, fp16-rounded stem). backing stem: MERT-330M h6+h12 (2x1024) | EAT-large b16 (1024) |
  SSLAM-pt b5 (768) per 4.04 s window -> 3840-d.
- Heads (3 seeds 41/42/43, probabilities averaged):
  voice DF: Standardize -> Linear(1280,256) -> GELU -> Linear(256,2); file = max over segments of the fake odds.
  voice AD: ResMLP (Linear init from the AntiDeepfake classifier + zero-init residual MLP 256).
  voice WavLM: MLP 1024 -> 256 -> 2. Label-completion rule L for DF/AD only.
  music: WindowMIL = per-window MLP on the standardized 3840-d features, log-mean-exp pooling over windows,
  + 0.25 x BCE localization loss on windows with known truth.
- Training: weighted BCE (weights s7 or W, data_v4/weighting.py), AdamW, lr 1e-3 music / 1e-4 voice, wd 0.01, batch 256,
  clip 1; early stopping on val EER (ties by BCE); music 50 epochs patience 5; voice 150 / 10.
- Voice fusion: weighted mean of member val-CDFs (1 / 0.5 / 1) mapped back onto the DF head's score scale.
- Presence: logistic heads on PANNs tags + stem statistics (unchanged since data v4).
- FILE: CDF-weighted mean of max(Pv*Fv, Pm*Fm) / noisy_or / max(Fv, Fm); S7 weights 0/9/1, tables refit on val.

Features of every training / val / held-out row are cached (npz), so head, fusion and FILE experiments do not touch the
encoders: one head run is minutes on one H100.

## 2. LB history (what changed -> LB ADS)
P1 0.7700 -> S2a 0.7689 (+800 AI-Hub phone reals: no effect) -> S3 0.7853 (AntiDeepfake voice member + MERT 6+12 music)
-> S6 0.8220 (MLP/ResMLP heads + EAT-16 music + rule L) -> S7 0.8374 (SSLAM-5 music + WavLM-8 voice member, best)
-> S8 0.8340 (raw-MIX SSLAM appended to music features + FILE refit to equal: DB +0.0043 but LB -0.0034).
Offline DB (E1) for S7 ~0.909 vs LB 0.837: the 0.07 gap is unexplained.

## 3. Offline evidence of today (DB protocol, vs the S7 recipe s7_ref; details new_eval_newtest.json, leak_check.txt)
| config | E1 (4,822) | E2 unlinked (12,351) | E2 open data (2,361) | E2 AT-ADD unseen generators (9,990) |
|---|---|---|---|---|
| s7_new (S7 weights + 40,214 new rows) | +0.0064 (CI .0013..0111) | +0.0285 | +0.0058 | +0.0349 |
| W_new (W weights + same rows) | +0.0033 | +0.0442 | +0.0058 | +0.0562 |
| W_newB (W_new + newval rows in early stopping) | +0.0032 | +0.0467 | +0.0071 | +0.0583 |
| s7_new_ob (s7_new + 400 AI-OpenBMAT) | +0.0049 | +0.0363 | +0.0049 | +0.0469 |
Weak spots that remain: AI-OpenBMAT broadcast mixes FILE EER 0.34-0.39 (all configs); AT-ADD singing FILE EER 0.13-0.15;
W variants worsen E1 VOICE EER (0.061 -> 0.079). E3 (AI-Hub Korean reals, 257, Mac only): FPR <= 0.04 for new-data
configs. Korean FAKE data: none. Packages built and verified: s7_new, W_new, W_newB (s7_new_ob: CUDA check pending).

## 4. Factors we believe drive the grader (INFERRED from sections 2-3)
1. AI music / song detection (biggest LB jumps came from backing-stem music features; raw-mix music features hurt).
2. Generalization to generators unseen in training (the DB-LB gap hypothesis; AT-ADD unseen-generator results).
3. The FILE combination rule (weight 0.5 of ADS; S8 changed it together with features, confounded).
Others: real commercial/mastered music false positives, broadcast speech-over-music mixes, the voice component
(W hurts E1 voice), Korean fakes (no data), presence (saturated).

## 5. Budget and time
VESSL credit $19.00 (H100x1 $2.98/h -> at most ~6 GPU-hours; H100x4 $11.92/h). Elice credit KRW 10,000 (AI-Hub only,
Korea-only processing). Packaging + CPU verify + parity + CUDA check ~1 h per package. A new candidate is useful only if
verified by ~06:00 on 09-29 (it would be the 3rd submission).

## 6. Questions
(A) Design: component by component (music head, voice members + fusion, FILE rule, presence, training data mix and
    weights, seeds/ensembling), what to change and why, with concrete options and defaults. Keep each component simple and
    independently tunable. Runtime code changes are allowed only if small and parity-checkable (the package must keep the
    same encoders). Is ensembling heads across configs (e.g. s7_new + W_new music heads) worth a runtime change?
(B) Experiment plan: candidate grid per component, val-only selection rule, how to limit selection overfitting on a
    4,931-row val, the order of experiments, pre-registered gates on E1/E2/E3, attribution (which component gave what),
    GPU-hours and dollars, and a timeline that ends with a verified package by ~06:00.
(C) Risks and what NOT to do. Mark VERIFIED / INFERRED / UNKNOWN; decisions and numbers, no generic advice.
