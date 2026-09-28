# DACON 236749 — Research results across all branches (2026-09-27)

This is the full record of the 2026-09-27 investigation. The ranked plan built on it is in
[TOP15_RESEARCH_PLAN_2026-09-27.md](TOP15_RESEARCH_PLAN_2026-09-27.md). This file keeps the evidence: every branch
slice, verifier verdict, experiment status, defect, research source and open question. The plan keeps only the
decisions.

No training, GPU job, data download or DACON submission was run for this work.

**Contents:** [1 Scope](#1-scope-and-method) · [2 Consolidated results](#2-consolidated-results) ·
[3 Results by branch slice](#3-results-by-branch-slice) · [4 Critique and resolved questions](#4-completeness-critique-and-resolved-questions) ·
[5 External research](#5-external-research) · [6 Direct checks](#6-direct-checks-in-this-session) ·
[7 Limitations](#7-evidence-limitations)

## 1. Scope and method

| Ref | Head | Relation to main | Coverage |
|---|---|---|---|
| `main` (= `claude/amazing-archimedes-a92agr` at start) | b6dfcb4, 09-27 04:56 KST | — | read directly + slice 7 |
| `cpu/lb-improvement-20260924` | 1c08b1b, 09-26 21:48 KST | 59 commits ahead, 1 behind (base c157c2f), never merged | slices 1–6 |
| `data-v4` | 435f153 | ancestor of main, 0 commits of its own | covered by main history |
| `pull/1` | 13a957e | ancestor of main, 0 commits of its own | covered by main history |

- **Direct reading:** the main pipeline code (training, features, runtime, decision layer, fusion), receipts, fusion
  reports, learning curve and audit evidence. The results are in section 2 and section 6.
- **Workflow (20 agents, no errors):** 7 slice readers → 1 adversarial verifier per slice (told to refute, and to
  answer "unverifiable" unless the underlying JSON/code was opened) → 2 targeted research agents → a completeness critic
  → 3 resolvers for the critic's decision-critical questions. The raw outputs are rendered in sections 3–5.
- **Evidence labels in section 2:** **[V]** checked against repo code or JSON; **[R]** recorded in the repo but resting
  on a transcribed screenshot or markdown; **[I]** inferred.

## 2. Consolidated results

### 2.1 Official leaderboard ledger

| ID | Time (KST, displayed) | Package | Total | ADS | CPS | Runtime | Identity |
|---|---|---|---:|---:|---:|---|---|
| 88418 / 88809 | 09-13 | official baseline G0 | 0.6909110688 | 0.6577539683 | 0.9893249735 | 21m59s | [R] |
| 88630 | 09-13 | AST epoch17 | 0.6453405503 | 0.6212619048 | 0.8620483598 | 1m21s | [R] |
| 88673 | 09-13 | AST full1029 | 0.6174819841 | 0.5881111111 | 0.8818198413 | 1m22s | [R] |
| 89524 | 09-14 | V5_B1_fixed | 0.6776485873 | | | 2m55s | [R] |
| — | 09-15 | V6_final_fixed | 0.6727822646 | | | | [R] |
| 90217 | 09-15 | C3 full clean | 0.6843536931 | 0.6504603175 | 0.9893940741 | 21m24s | [R] |
| — | 09-16 | C08 (music Conv1D swap) | 0.7314822646 | | | | [R] |
| — | 09-16 | M330 MLP (music swap) | 0.7396465503 | | | | [R] |
| — | 09-16 | S2 EAT full fine-tune | 0.6378679788 | | | | [R] |
| 91610 | ~09-17 | B_full1696 (noisy-OR) | 0.6838 | | | | [I] mtime only |
| 92294 | 09-17…19 | MERT12 full1322 | 0.7479394074 | | | | [I] |
| 92644 | 09-17…19 | Expanded | 0.7579036931 | 0.7321825397 | | | [I] |
| 93093 | 09-17…19 | Balanced7144 | 0.7559347884 | 0.7299920635 | 0.9894193122 | | [I] |
| **102411** | **09-24 02:09:50** | **DataV4_DurationBalanced** | **0.7912534339** | **0.7683174603** | **0.9976771958** | 17m32s | [R] score, [I] package by timing (+4m31s after c157c2f) |
| (no ID) | 09-24 ~23:00 | E08 partial-data voice top-30% | — | 0.7064761905 | — | — | [R] user-reported |
| error | 09-26 20:37:55 | PANNs R3 (NumPy 1.26 crash) | — | | | | [R] slot deducted |
| 104224 | 09-26 21:13:14 | PANNs R3 fix1 | 0.7901426243 | 0.7674603175 | 0.9942833862 | 26m16s | [R]/[I] |

Errored uploads also include V5_B1 (install-stage, not deducted) and V6 89989 (NumPy ABI, deducted). Packaged but
never submitted: `DataV4_P1Refit`, `DataV4_S2a_AIHubPhone`, Existing9504, E08 variants. Scoring is deterministic
(88418 = 88809 to 10 dp), and ADS values fall on a 1/126000 grid [R].

### 2.2 Main findings

1. **Incumbent.** The best recorded score is 0.7912534339 (102411). main's docs still list 0.7396, which is stale. [R]
2. **No validated improvement exists after 102411.** E08 was confounded: partial data, 1 seed, no duration balance, a
   music head retrained on 373 rows, a legacy decision layer and top-k pooling, all at once. PANNs 104224 changed only
   the presence path and regressed. The zero-shot AntiDeepfake/SONICS fusion was rejected on held-out data. [V]/[R]
3. **Where the errors are** (DataV4 phase-C heads on the 6,444-row held-out proxy, never trained on) [V by verifier]:
   - Music-containing files hold 538 of 705 FILE errors (18.88% vs 4.65% per-row error rate).
   - Voice EER is 0.206 with music present (n=1,544) and 0.0443 voice-only (n=3,595).
   - Music EER is 0.20005 (n=2,849).
   - Oracle music-fake labels cut FILE EER 0.1094 → 0.0445; oracle voice-fake labels cut it to 0.0607.
   - Lyria-3 Pro: vocal FNR 0.631 (n=222); instrumental-segment Music FNR ≈0.753 (n=77).
4. **Presence is saturated.** CPS 0.9977 leaves at most +0.00023 total. Oracle presence moves FILE EER only −0.0037
   (CI includes 0). Every presence change so far lowered CPS on the LB. [R]
5. **Singing is limited by separability as well as labels.** Legacy fully-AI songs (SONICS 751, suno 18, udio 14, …)
   have `VOICE_FAKE=null`, and fake singing gets about 1.5% of the voice fake-class weight. But within-singing AUC
   (Lyria vs real singing) is 0.75–0.78 for every head, whatever the weighting. [R]
6. **Cross-condition score shift is a small lever.** An oracle constant offset on singing rows improves proxy Voice EER
   at best 0.1107 → 0.1059, and makes val worse (0.0510 → 0.0542). [R]
7. **More of the same data does not help music.** Held-out music EER stays flat from 50% to 100% of training groups
   (0.198 → 0.206, 1 seed). Voice improves, but that is confounded with the number of optimizer updates. [V]
8. **P1Refit is the incumbent runtime with new heads.** It is byte-identical to the 102411-lineage runtime (c157c2f +
   candidate string), with the same presence assets; only the heads and file_ensemble differ. Paired val gain is
   +0.00548 ADS (both sides selection-biased). [V]
9. **Do not merge the branch into main.** A merge gives 3 textual conflicts and semantic breaks (the branch rejects heads
   without `feature_contract`, and changed hashes invalidate existing caches). [V by verifier, `git merge-tree`]
10. **Round-2 rights risks exist for the incumbent too:**
    - MLAAD (NC, admitted on the project's own reading of the rules) and JamendoMaxCaps (400 rows) are in DataV4.
    - AI-Hub ("no export abroad") is in S2a.
    - 900 MUSAN noise files were regenerated and 8 HAIM files are missing, so the exact training bytes cannot be
      handed over. [R]
11. **Proxy → LB transfer is weak and rests on one pair.** Balanced7144 → DataV4: proxy ΔADS 0.0611 vs LB ΔADS 0.0383
    (ratio 0.63). DataV4 goes from val ADS 0.918 to held-out 0.863 to LB 0.768. [R]

### 2.3 Score sensitivity

Each 0.01 of EER is worth 0.0045 (FILE), 0.0027 (music) or 0.0018 (voice) of total score. Presence AUC headroom is
≤ 0.00023 total. Because Private = Public on the same 1,200 files, a leaderboard gain is the final-ranking gain.

## 3. Results by branch slice

Each slice was read by one agent and its decision-critical claims were then checked by a separate adversarial verifier that opened the underlying JSON and code. Verdicts: confirmed / partially / refuted / unverifiable. Where a verifier corrected a claim, the corrected statement is the one to use.

### 3.1 LB improvement audit, plan, findings ledger, baseline lock (branch cpu/lb-improvement-20260924)

This slice was a read-only audit on 2026-09-24. It recovered 87 findings: 21 had prior Claude reviews, 16 critical/high items got new artifact and primary-source reviews, and 50 remained not_independently_verified. No finding is plainly "confirmed": 36 are partially_confirmed and 1 is conditionally_confirmed (VALIDATION_GAP-2, the LB column probe). Several sub-claims were rejected: the 69% music-error share (recomputed as 76.31% of error count and 76.09% of EER contribution), "length explains 40% of the LB gap", fixed 1.69x or 63% proxy-to-LB multipliers, the music-to-VOICE max rule, and "0 fake voice-only clips of 30 s or more" (there are 4). The baseline lock (W00, the only passed gate) is the historical DataV4 proxy on 6,444 predictions out of 6,450 bank rows: File EER 0.10939954 (n=6444), Voice EER 0.11070987 (n=5139, 261 labels unknown), Music EER 0.20005498 (n=2849), W_known_subset 0.13685824. The lock itself says "not independent or official score". It holds no submit.zip hash, so the link between submission 102411 and DataV4 is inferred, not proven. Role split roles-v3 covers 42,947 rows and 19,131 components: train 29,643, selection 2,585, calibration 2,202, development 8,517. Zero components cross roles and no sealed set exists. roles-v4 later replaced it: 28,319 / 2,435 / 2,073 / 10,120. What was executed: W00 passed. The code fixes landed: Echoes parser, per-output masks, role-gated construction, and the config/pooling contract. W01, W02 and W03 are blocked, and W05 to W11 are deferred in workflow_state. Later, E08 alone ran on a partial cache of 5,952 files with several other changes bundled in; it regressed to ADS 0.7064761905 (user-reported). Not executed: E01 to E07, E09 to E14, the sealed set, the refit, and the LB probe.

#### Verified claims

| Verdict | Claim | Corrected statement / evidence |
|---|---|---|
| **confirmed** | Locked baseline = historical DataV4 proxy, self-declared not official: File 0.10939954188481675 (n=6444), Voice 0.11070986873541894 (n=5139, 261 unknown), Music 0.20005497526113247 (n=2849), W_known_subset 0.13685823726783192; the lock holds no submit.zip hash, so 102411 (0.7912534339) identity is inferred. | As stated. Addition: the lock ties to the 05ae7868 package through member hashes (run_identity, candidate_lock). Only the package-to-102411 link is unverified (timing inference). The prediction CSVs behind the EERs are not in the repo. |
| **partially** | Proxy-to-LB transfer rests on one anchor: proxy dADS 0.06107174 vs LB 0.03832540 (ratio 0.62755); fixed multipliers (1.69x, 63%) rejected; paired bootstrap SD of proxy W differences ~0.0046, so proxy gaps <=~0.01 cannot rank candidates. | On the 6,444-row v4 proxy there is one attribution-inferred anchor (B7144 93093 → DataV4 102411): LB dADS 0.0383253968 (exact) vs proxy 0.06107174 (from the ledger, not recomputable), ratio 0.62755. The ledger rejects fixed multipliers and uniform thresholds. The bootstrap SD ≈0.0046 is pair-specific and its scripts are not in the repo. Treating proxy gaps ≲0.01 as unrankable is a reasonable heuristic, not a validated rule. The post-audit LB results (E08, PANNs) are further examples of weak transfer, but not proxy anchors. |
| **partially** | LB mechanics: ADS exact on 1/126000 grid, deterministic scoring, Public=100% and best score kept so regressions cost only a slot; CPS headroom only 0.00023228042; single-column constant-0.5 probe gives E_c = 0.5 + (Ac-A0)/w; not executed. | The grid, best-kept rule, CPS-only headroom 0.00023228042 and probe algebra all check out. The determinism evidence is a markdown table, since the raw JSON is not in the repo. Presence is not worthless: it feeds FILE fusion, which carries 0.45 of the total. A constant-0.5 probe is unexecuted, and DACON ruling 417702 (recorded on main) discourages LB probing. |
| **partially** | No post-audit candidate passed the plan's gates: W00 passed; W01-W03 blocked; W05-W11 deferred; no sealed set. The only audit experiment to reach LB (E08: 1 seed, no duration balance, retrained music head, max fusion) regressed to ADS 0.7064761905 (user-reported) from 0.7683174603. | workflow_state.json is a stale 09-24 14:55 snapshot: W00 passed, W01-W03 blocked, W04 ready, W05-W11 deferred. Afterwards, E00/E08 were trained on partial data and E08 was submitted without passing the gates. No sealed set exists. E08 changed voice pooling (top-30%) plus a retrained voice head, a retrained music head (local loss 0.25), 1 seed, no duration balance, legacy PANNs presence and max fusion. It scored ADS 0.7064761905 (user-reported). PANNs 104224, from a separate plan, also regressed. No validated improvement over DataV4 exists in the repo. |

#### Experiments and their status

- **Audit findings ledger: verdict map for all 87 findings** — *completed, not evaluated*
  - Change: Recovered 87 findings, reviewed 37 of them, and assigned verdicts. No code or model change.
  - Data/split: DataV4 proxy test (6,444 prediction IDs), val, manifest, public web sources
  - Metrics: partially_confirmed via prior review with synthesis corrections (20): ERROR_ANALYSIS-1(C),-2(C),-3(H),-4(H),-5(H); DATA_DISTRIBUTION-1(C),-2(C),-3(H),-4(H),-5(H),-7(H); ARCHITECTURE-1(C),-2(H),-3(H),-4(H); VALIDATION_GAP-1(C); HISTORY-1(C),-2(C),-3(H),-4(H). conditionally_confirmed (1): VALIDATION_GAP-2(H), historically 'confirmed'. partially_confirmed via current artifact/primary review (16): DATA_DISTRIBUTION-6, VALIDATION_GAP-3, VALIDATION_GAP-4, HISTORY-5, RUNTIME_BUDGET-1, -2, -4, DATA_ACQUISITION-1, DATA_ACQUISITION-2 [availability_confirmed_recommendation_conditional], -3 [license_and_label_defects], -4 [access_and_quality_gates_missing], -7 [core_restrictions_confirmed_blanket_clearance_rejected], -10 [historical_timing_confirmed_prospective_budget_inconsistent], SOTA_RESEARCH-1 [failure_mode_supported_max_rule_recommendation_rejected], -2 [model_facts_confirmed_dominance_and_deployment_unverified], -3 [rationale_supported_equivalence_and_gain_overstated]. not_independently_verified candidates (50): ERROR_ANALYSIS-6..10, DATA_DISTRIBUTION-8..13, ARCHITECTURE-5..11, VALIDATION_GAP-5..10, HISTORY-6..12, RUNTIME_BUDGET-3,5..11, DATA_ACQUISITION-5,6,8,9, SOTA_RESEARCH-4..10. Original severities: 10 critical, 27 high, 39 medium, 11 low.
  - Supported conclusion: On the proxy, errors concentrate in files that contain music: 538 of 705 FILE errors, per-row error rate 18.88% vs 4.65%. Main error cells are real-music false positives (MUSAN, OpenMIC, constructed songs/mixes) and Lyria-3 Pro (instrumental passages have Music FNR about 0.753, n=77; voice FNR 0.631, n=222). No hidden-set composition can be inferred from this. Rejected claims: length explains 40% of the gap; fixed transfer multipliers; the max(Fv, 0.5*Pm*Fm) voice fusion; blanket license clearance for datasets.
  - Evidence: `docs/evidence/lb_cpu_20260924/audit_findings_ledger.json`
- **W00 baseline lock (recompute historical DataV4 proxy metrics)** — *evaluated, valid protocol*
  - Change: None. Recomputed metrics from the saved test_predictions_decision.csv with bank masks.
  - Data/split: bank_test 6,450 rows; 6,444 prediction rows (6 HAIM IDs missing because of the FILE_FAKE_mask filter)
  - Protocol: Recomputed metrics must match the stored report within 1e-12. The audit md reports a difference under 1.74e-14; verification.json reports 0.
  - Metrics: file_eer 0.10939954188481675 (n=6444); voice_eer 0.11070986873541894 (n=5139, 261 unknown, coverage 0.9517); music_eer 0.20005497526113247 (n=2849); voice_presence_auc 0.9979462237360157 (n=6268, 176 unknown); music_presence_auc 0.9986542871104935 (n=6444); W_known_subset 0.13685823726783192; max_metric_abs_difference 0.0; surrogate 0.9*(1-W)+0.1*CPS = 0.8766576120 (not an official score)
  - Supported conclusion: Proves only that the stored DataV4 proxy metrics are numerically reproducible. Says nothing about LB quality. The proxy-to-LB ADS gap (1-W 0.86314 vs LB 0.7683174603) is about 0.0948.
  - Confounders: The proxy is not independent. Neither the submission (102411) nor the ZIP is hash-bound in this lock.
  - Evidence: `docs/evidence/lb_cpu_20260924/baseline_lock.json`
- **roles-v3 role assignment** — *completed, not evaluated*
  - Change: Mapped historical splits to roles: train to train, test to development, val components to selection or calibration (sha256(seed 20260924|component) mod 2). Cross-split components and unresolved Echoes aliases go to development.
  - Data/split: 42,947 manifest rows, 19,131 components
  - Protocol: data/prepare_roles.py:19-95 (role_reason field, historical_exposure propagated to the whole component)
  - Metrics: train 29,643 / selection 2,585 / calibration 2,202 / development 8,517; cross_role_components 0; conflicting components quarantined 40; alias components quarantined 276; unresolved_echoes_rows 1,238; legacy catalog matched 8,075/8,075; sealed unavailable; 121 voice-interval rows with alignment pending; 12 Echoes generators restored (594 rows). Later roles-v4: 28,319 / 2,435 / 2,073 / 10,120.
  - Supported conclusion: The leakage-safe role contract exists: optimizer uses train only, selection picks epochs, calibration fits CDFs, development compares candidates. No sealed holdout exists, and the existing test is development only.
  - Evidence: `docs/evidence/lb_cpu_20260924/roles_report.json; docs/evidence/lb_cpu_20260924/data_integrity.json; repository/data_v4/prepare_roles.py`
- **CPU training/eval contract verification** — *evaluated, valid protocol*
  - Change: Added experiment-config seeds, voice pooling (max/lme/topk), voice local loss, per-output masks, cache-provenance checks and package identity checks.
  - Data/split: Local CPU unit and contract tests; workspace wsp-9683432r72m7 stopped; 0 GPU jobs
  - Metrics: Microbatch parity atol 1e-7 rtol 1e-6; pooling train/runtime parity and file-order invariance atol 1e-7 rtol 1e-6; historical five-metric reconstruction max diff 0; quality_claims false; code_commit ad4a6ad
  - Supported conclusion: Numeric and wiring contracts only. Not validated: new model EER or LB, full encoder inference, L4, alignment of the 121 codec intervals, a sealed set, cache promotion.
  - Evidence: `docs/evidence/lb_cpu_20260924/verification.json; repository/full_bank_contract.py`
- **LibriSpeech long-speech pilot** — *partial*
  - Change: Collected real-speech candidates
  - Data/split: 200 selected out of 714 considered (514 rejected), 33 speakers, 2,590.78 s, 46,435,061 bytes
  - Metrics: Full-audio decode verified; overlap with the 40 legacy LibriSpeech speakers is 0; training_admitted false; sealed false
  - Supported conclusion: Candidate data only. Still pending: cross-corpus identity review, listening checks, fake long-speech coverage, role assignment.
  - Evidence: `docs/evidence/lb_cpu_20260924/librispeech_pilot.json; docs/evidence/lb_cpu_20260924/pilot_overlap.json`
- **FMA real-music pilot** — *partial*
  - Change: Selected FMA tracks
  - Data/split: target 200
  - Metrics: selected 0, bytes 0; failures pending_access
  - Supported conclusion: No FMA data acquired in this slice.
  - Evidence: `docs/evidence/lb_cpu_20260924/fma_selection_receipt.json`
- **E00 baseline retrain (seed 41, roles contract)** — *evaluated, invalid protocol*
  - Change: None relative to the recipe; config {seeds:[41]}
  - Data/split: Later partial A100 cache of 5,952 files: train 5,196 / selection 421 / calibration 335. Not the planned full E00 set (train 27,788 / selection 2,386 / calibration 2,023 / development 9,842).
  - Metrics: Selection: voice EER 0.0627014702 (n=401), music EER 0.1715686275 (n=35), file EER 0.0726953842 (n=421)
  - Supported conclusion: Not a valid stand-in for the historical baseline. Beating E00 does not imply beating DataV4.
  - Confounders: Extraction-order subset; fresh heads rather than the historical DataV4 heads; F0 calibration not fitted
  - Evidence: `repository/configs/lb_improvement/E00.json; docs/A100_PARTIAL_TRAINING_2026-09-24.md`
- **E08 voice top-30% logit pooling** — *evaluated, invalid protocol*
  - Change: voice_pooling topk, fraction 0.3 (full_bank_contract.py:48-54)
  - Data/split: Same partial cache as E00; submitted to the LB
  - Metrics: Selection: voice 0.0501611762 (n=401), file 0.0627458694 (n=421), music 0.1715686275 (n=35). LB ADS 0.7064761905, user-reported, versus 0.7683174603. CPU calibration (n=335/308/43): DataV4 File/Voice/Music 0.055970/0.023566/0.093074 vs E08 0.089552/0.061008/0.186147.
  - Supported conclusion: The E08 package is worse than DataV4. The effect of pooling alone was not isolated. Swapping the DataV4 music head into E08 improves calibration File EER to 0.072761.
  - Confounders: Several changes bundled: 1 seed instead of 3, no duration balance, a new music WindowMIL trained on 373 rows, voice head restarted from DF-Arena init, PANNs max presence, FILE max fusion, partial data
  - Evidence: `repository/configs/lb_improvement/E08_voice_topk.json; docs/E08_REGRESSION_2026-09-24.md`
- **E02 voice local loss 0.25** — *implemented, not run*
  - Change: voice_local_loss_weight 0.25
  - Data/split: 121 fake_voice_segments rows (train 101, val 18, test 2), all with alignment pending
  - Supported conclusion: Config and gradient path verified on CPU only. No supervised intervals are usable until codec alignment is verified.
  - Evidence: `repository/configs/lb_improvement/E02_voice_local.json; docs/evidence/lb_cpu_20260924/roles_report.json`
- **E09 voice LME pooling** — *implemented, not run*
  - Change: voice_pooling lme: logsumexp(odds) - log(n)
  - Supported conclusion: Not run
  - Evidence: `repository/configs/lb_improvement/E09_voice_lme.json`
- **E04 approved extra data (up to 4,400 rows)** — *partial*
  - Change: Add admitted rows to train
  - Data/split: The A100 run planned E04 with +1,477 train rows. Feature extraction stopped at 5,952 files and was never trained.
  - Supported conclusion: No E04 result exists
  - Evidence: `docs/A100_RUN_2026-09-24.md`
- **E01 singing/source reweight, E03 joint-feature head, E05/E06 pair and consistency losses, E07 group-OOF hard weighting, E10 combinations, E11-E14 raw+stem / multi-layer MERT / second backbone / partial fine-tune** — *proposed*
  - Change: See plan section 5 (W05-W08)
  - Supported conclusion: Not implemented or run on this branch as of workflow_state (W05-W11 deferred)
  - Evidence: `repository/docs/superpowers/plans/2026-09-24-dacon-lb-improvement-workflow.md; docs/evidence/lb_cpu_20260924/workflow_state.json`
- **Historical: training-free voice fusion max(Fv, 0.5*Pm*Fm)** — *evaluated, invalid protocol*
  - Change: Feed music evidence into VOICE_FAKE_PROB, with w chosen on val
  - Data/split: Val (only 5 rows of real vocal + AI music) and proxy test
  - Metrics: Proxy voice EER 0.1107 to 0.0872; Lyria FNR 0.631 to 0.252; constructed_song real FPR 0.06 to 0.202; hybrid errors 10 to 46 of 84
  - Supported conclusion: Rejected. It breaks component semantics (real vocal + AI backing must be VOICE_FAKE=0).
  - Evidence: `docs/evidence/lb_cpu_20260924/audit_findings_ledger.json (ERROR_ANALYSIS-3, SOTA_RESEARCH-1)`
- **Historical: val-fitted voice stacker** — *evaluated, valid protocol*
  - Change: Logistic stack of fv, fm, fm*pv
  - Data/split: Fitted on val, evaluated on proxy test (development)
  - Metrics: Voice EER 0.1107 to 0.0922; group bootstrap delta 95% CI [-0.0295, -0.0111]; errors: constructed_mix VF1/MF0 29 to 50, constructed_song VF0/MF1 10 to 22
  - Supported conclusion: The pooled gain comes entirely from all-fake AI songs, and partial-fake hybrids get worse. Not adopted.
  - Evidence: `docs/evidence/lb_cpu_20260924/audit_findings_ledger.json (ARCHITECTURE-3)`
- **Historical: FILE fusion variants** — *evaluated, valid protocol*
  - Change: Replace the CDF-ensemble fusion rule
  - Data/split: Proxy test n=6444; val grid
  - Metrics: Submitted CDF ensemble 0.10940; max 0.11004; noisy_or 0.10753; max_nopresence 0.13035; voice_only 0.19056; music_only 0.42103; val grid gain 0.0000 (< 0.002); group-5fold LR stack 0.1075, HGB 0.0942 (leaky); routed per-type calibration 0.1094 to 0.1372 (worse, not independently verified)
  - Supported conclusion: Fusion is roughly saturated (within +/-0.004 on the proxy). Gains must come from the component heads.
  - Evidence: `docs/evidence/lb_cpu_20260924/audit_findings_ledger.json (HISTORY-3, ARCHITECTURE-10, ERROR_ANALYSIS-1, SOTA_RESEARCH-10)`
- **Historical: oracle decomposition** — *evaluated, valid protocol*
  - Change: Replace one component with ground truth inside the fusion
  - Data/split: Proxy test; decision layer reconstructed with maxdiff 2.4e-13
  - Metrics: FILE EER 0.1094: oracle music 0.0411, oracle voice 0.0607, oracle presence 0.1063; Lyria perfect gives voice 0.0763, FILE 0.0968
  - Supported conclusion: Hypothetical headroom on the proxy only. The audit forbids converting it into an expected LB gain.
  - Evidence: `docs/evidence/lb_cpu_20260924/audit_findings_ledger.json (HISTORY-1, DATA_DISTRIBUTION-1)`
- **Historical: duration reweighting scenarios** — *evaluated, valid protocol*
  - Change: Reweight proxy rows toward uniform 4-60 s
  - Metrics: Claimed uniform reweight W 0.1369 to 0.1744. Source-preserving within-source reweight: Voice 0.1107 to 0.1160, W 0.1369 to 0.1277. Oracle making both classes length-invariant: voice 0.1107 to 0.1098.
  - Supported conclusion: The aggregate length effect is mostly source confounding. The claim that length explains 40% of the LB gap is rejected.
  - Evidence: `docs/evidence/lb_cpu_20260924/audit_findings_ledger.json (HISTORY-2, ARCHITECTURE-1)`
- **Historical: old/new voice head blends and singing offset** — *evaluated, valid protocol*
  - Change: Rank-blend the Existing9504/B7144 voice head with DataV4; add a logit offset for singing rows
  - Metrics: Rank blend w_old 0 to 1: voice 0.1107 to 0.1601; mean 0.1201; geo 0.1134. Singing offset: test minimum 0.1059 while val worsens 0.0510 to 0.0542.
  - Supported conclusion: No blend gain. The old head's 'music context = fake' bias explains its Lyria recall.
  - Evidence: `docs/evidence/lb_cpu_20260924/audit_findings_ledger.json (VALIDATION_GAP-1, ERROR_ANALYSIS-3, DATA_DISTRIBUTION-1)`
- **LB single-column diagnostic probe** — *proposed*
  - Change: Set one component column to constant 0.5 with all other columns fixed. Then E_component = 0.5 + (Ac - A0)/w.
  - Supported conclusion: Not executed. Costs one slot per component and measures without improving the score.
  - Evidence: `docs/evidence/lb_cpu_20260924/audit_findings_ledger.json (VALIDATION_GAP-2); repository/DACON_LB_IMPROVEMENT_AUDIT_2026-09-24.md:164`
- **All-data head refit (train+val+test)** — *proposed*
  - Change: Refit the 6 heads on 42,039 cached rows with a frozen recipe
  - Metrics: Historical head stage took 455 s (6 heads = 2 routes x 3 seeds); voice seeds 41/43 hit epoch 50/50
  - Supported conclusion: Gain unmeasured, and it would lose the evaluation and calibration sets. Not executed.
  - Evidence: `docs/evidence/lb_cpu_20260924/audit_findings_ledger.json (RUNTIME_BUDGET-2, DATA_DISTRIBUTION-3)`
- **Runtime parity probes (MERT layers[:13], DF-Arena in-file batching, bf16)** — *partial*
  - Change: Inference speedups
  - Metrics: MERT truncated to layers[:13] is bitwise equal for hidden_states[12]; CPU time 1.01 s to 0.54 s. DF batching is bitwise equal; bf16 prob max error 2.69e-3. ZIP headroom 3,649,279,387 B.
  - Supported conclusion: CPU probe only (candidate, not independently verified). Not packaged, and no L4 measurement.
  - Evidence: `docs/evidence/lb_cpu_20260924/audit_findings_ledger.json (RUNTIME_BUDGET-5, -6, -9)`

#### Defects

| Severity | Status | Location | Description | Affected results |
|---|---|---|---|---|
| minor | fixed | `origin/main repository/data_v4/file_ensemble.py:71 (audit cites :72)` | The FILE_FAKE_mask filter dropped whole rows, so 6 HAIM test rows lost all five outputs, including known MUSIC labels. | DataV4 proxy denominators: 6,444 rows vs 6,450 |
| major | fixed | `origin/main repository/data_v4/build_manifest.py:140-141 ing_echoes` | The generator was parsed as the directory before TTA/ATA. All 594 Echoes rows became generator 'Echoes/TTA' (300) or 'Echoes/ATA' (294). The group_key is per-item, not per reference song (59 of 273 reference songs span splits). | Per-generator caps and weights of all DataV4 heads (the submitted model was trained with the bug). Echoes rows are not re-selected (about 3,670 possible). |
| major | open | `origin/main repository/data_v4/build_manifest.py plan_songs max_partners=40 and post-construction component closure` | Closure after construction pulled all 400 constructed_song rows, 276 of 300 constructed_enhanced rows, 385 vocalset rows and 386 openmic rows into test. Train kept only 33 real simultaneous speech+music mixes. | The train distribution of DataV4 and every head trained from it |
| minor | open | `origin/main repository/data_v4/export_bank.py (event_intervals exports music events only); full_bank_train.py voice path` | The 121 fake_voice_segments rows exported empty intervals, and the voice local loss was unused. | E02 cannot run |
| major | open | `docs/evidence/lb_cpu_20260924/baseline_lock.json` | The baseline lock contains no submit.zip hash, and the submission-to-package binding is absent. The plan's W00 required ZIP hashes and the attribution of the screenshot score. | Identity of the best LB package (102411, assumed DataV4_DurationBalanced) |
| major | open | `docs/E08_REGRESSION_2026-09-24.md` | E08 violated the plan's single-change rule. Its comparator E00 was a fresh partial-data retrain, not historical DataV4. The package also changed seeds (3 to 1), removed duration balance, retrained the music head on 373 rows, and replaced presence calibration and FILE fusion with max. | E08 LB slot; attribution of top-k pooling |
| major | open | `docs/evidence/lb_cpu_20260924/workflow_state.json` | No sealed holdout exists ('sealed': 'unavailable'), so no candidate can pass the plan's final gate. The existing test is development and has been reused heavily. | All candidate selection before the deadline |
| minor | fixed | `historical audit claims (ledger HISTORY-4, ERROR_ANALYSIS-5, HISTORY-2, ERROR_ANALYSIS-1, DATA_DISTRIBUTION-2, RUNTIME_BUDGET-1/-2/-4)` | Earlier synthesis errors: 69% music share (actually 76.31% / 76.09%); '0 fake voice-only >=30 s' (actually 4); 18 heads (actually 6); 4,020 AntiDeepfake windows (actually 2,564 plus 1,456 summary values); a fixed 1.69x or 63% transfer; length explaining 40% of the gap. | Any plan built on the pre-audit numbers |
| minor | unknown | `repository/data_v4/build_manifest.py:563-564 and :555-556 (per candidate DATA_DISTRIBUTION-13)` | Sequential mixes insert exactly 0.2 s of digital silence, and 91.6% (285/311) of train sequential mixes are fake. Backing tracks loop via np.resize without crossfade. | Possible construction shortcut |
| minor | open | `Label semantics: HAIM B2 fx_encoder_style_transfer (VOICE_FAKE=1) and HAIM B1 sonicmaster (REAL)` | Ambiguous component labels: fx_encoder vocals may not be generated, and sonicmaster applies generative mastering yet is labelled REAL. | Voice and Music EER on HAIM cells |

#### Lessons

- The proxy overstates LB gains. The only anchor pair (Balanced7144 to DataV4) had proxy dADS 0.06107174 vs LB 0.03832540 (ratio 0.62755). The DataV4 surrogate gap vs LB is 0.08540418, 99.927% of it from ADS. Fixed multipliers were rejected.
- LB ADS values sit on a 1/126000 grid (13 values checked), and scoring is deterministic (88418 = 88809 = 0.6909110688). With Public = 100% and the best score kept, a regressing submission costs only a slot.
- CPS is saturated: the most any presence work can add to the total is 0.00023228042. Replacing presence was explicitly deprioritized by the audit.
- FILE fusion variants all fall within about +/-0.004 on the proxy (noisy_or 0.10753 vs submitted 0.10940). Only component-head changes can move ADS materially.
- On the proxy, music-containing files carry 538 of 705 FILE errors. Music real FPs come mostly from inherited parent-accompaniment FPs (P(FP|parent FP) 0.607 vs 0.149), not from vocal residue alone.
- Copying music evidence into VOICE lowers pooled voice EER but breaks the hybrid real-vocal + AI-backing semantics. Do not use it.
- Changing several factors at once (E08) produced an LB regression of -0.0618412698 ADS that cannot be attributed. Single-change comparisons against the actual historical heads are needed.
- Voice-only proxy has no file of 30 s or longer (real max 26.58 s, fake max 24.803 s), so long-file behaviour of max vs top-k pooling is unmeasured.
- Historical LB evidence: frozen-backbone small heads and data diversity helped (C08 +0.0405711958 over G0; M330 0.7396). Full fine-tuning (EAT S2 0.6378679788) and presence swaps regressed, though not every replacement is proven bad.
- A 'sealed' holdout was never built, so every post-audit decision rests on development or partial data plus LB feedback.

#### Open opportunities (as proposed by the slice; ranked in the plan, not here)

- Keep DataV4_DurationBalanced (0.7912534339) as the anchor or fallback. E08 is inferior on both LB and CPU calibration.
- Voice top-k pooling applied to the full DataV4 heads (3 seeds, duration balance, CDF fusion, presence head) as a true single-change candidate. Untested; pooling at training and inference must match.
- Swap only the FILE fusion rule to noisy_or (proxy 0.10753 vs 0.10940). Expected change is about +/-0.001; low value.
- Run the LB column probe (constant 0.5 for one component) to measure hidden Music vs Voice EER before spending slots. Costs 1 slot per component.
- Refit the DataV4 heads on train+val(+test) with a frozen recipe (heads take about 455 s on H100). Gain unmeasured, and it consumes the evaluation data.
- Music-head lever: real-music FP (MUSAN/OpenMIC/constructed songs) and Lyria instrumental FN. Re-select full Echoes (about 3,670 rows with correct generators) for diversity.
- Runtime headroom exists (17m32s of 60 min; ZIP headroom 3,649,279,387 B). MERT layers[:13] truncation is bitwise-identical if new encoders are added.
- Voice local loss (E02) and LME pooling (E09) are implemented but not run. E02 is blocked until the 121 intervals are aligned.

<details><summary>Files this reader did not open (10)</summary>

- `.omo/evidence/dacon-lb-improvement-audit-20260924/audit_receipt.json (cited as evidence_locator for the 16 current reviews): not in repo on either branch`
- `.omo/evidence/dacon-lb-improvement-audit-20260924/findings_ledger.json and chat_handoff.md: not in repo`
- `results/data_v4_20260923/.../compare/compare_report.json: not in repo`
- `results/data_v4_20260923/.../train_duration_balanced/file_ensemble.report.json: not in repo`
- `live_submissions.json (source of historical LB rows quoted in the ledger): not in repo`
- `docs/evidence/lb_cpu_20260924/fma_selection_receipt.json (only the head was read; selected 0 / pending_access)`
- `docs/evidence/lb_cpu_20260924/librispeech-pilot-v1_transfer.json (summary only)`
- `docs/evidence/lb_cpu_20260924/weights_transfer.json (head only)`
- `docs/GPU_PREPARATION_2026-09-24.md, docs/MLAAD_DECISION_2026-09-24.md, docs/CPU_ADMISSION_2026-09-24.md, docs/ADDITIONAL_DATA_COLLECTION_2026-09-24.md (other slices)`
- `PANNs docs/evidence (panns_*_2026092x) (other slices)`

</details>

### 3.2 Additional data collection, rights review, CPU admission (cpu/lb-improvement-20260924, 2026-09-24)

Collected 2,800 new originals (344,462.04 s total). Jamendo 1,000 real music; Music Arena 1,000 AI music (13 generators, 559 prompt groups); LibriSpeech 400 long real speech (86 speakers, 15.0-35.155 s); MLAAD 400 long English TTS (15.25-58.5 s). Together with the 200-file LibriSpeech pilot that makes 3,000 candidates. After lineage review (roles-v4), 247 of the 400 MLAAD files were linked to historical data, 216 of them to eval-role groups. A metadata-only grouped CV reached voice AUC 1.0 in all 5 folds (n=1000, FLAC real vs WAV fake). A CPU AST screen with a 0.5 threshold and 0 human listening admitted 15-s mono 16 kHz PCM16 crops as train-only. roles-v5 admitted LibriSpeech 449 and MusicArena 882 (1,331). roles-v6 added MLAAD 146 (1,477 total, 44,424-row manifest). Jamendo (0 admitted) was excluded over its TDM prior-written-authorization terms. Rights: LibriSpeech and MusicArena were closed as CC BY 4.0. MLAAD was first excluded, then admitted 39 minutes later on the project's own reading of the rules (explicit_publisher_approval=false). Gaps actually filled: AI music (fake only) and more real/fake English speech. Singing, real music, vocal/accompaniment authenticity combos and the sealed 600-row eval set were not filled, and admission cropped the 'long speech' clips to 15 s. E04 (E00 + 1,477 rows) was implemented, but feature extraction stopped at 5,952/43,516 with 0 new rows cached, and compare_data_admission.py never produced output. No trained or submitted model (102411, E08, PANNs 104224) used any of this data. Main commit b6dfcb4 builds rows from the Jamendo payload this branch excluded, and adds AI-Hub rows labelled 'no export abroad'. Both are handover risks for round 2.

#### Verified claims

| Verdict | Claim | Corrected statement / evidence |
|---|---|---|
| **confirmed** | No new data from this slice was used in any trained or scored model. The E08 partial cache has added_rows_cached=0 (5,952 cached rows). PANNs admission excludes LibriSpeech/MLAAD/MusicArena as 'Dplus_not_D0' (panns_adaptation_data.py:153). 102411 was packaged at c157c2f (2026-09-24 02:05 KST), before any branch commit (first 14:53 KST). | Confirmed as of the repo. Caveats: the 102411 to c157c2f link is inferred from timing, not archive-hash-verified. Main b6dfcb4 has an in-progress S2-B pipeline (Jamendo and MusicArena from this slice's payload). It has no trained result yet, but it would break this claim if completed. |
| **partially** | E04 (E00 + 1,477 admitted train clips) has no result. Feature extraction stopped at 5,952/43,516, no complete DataV4 feature cache was located, and compare_data_admission.py produced no output. | E04 was implemented but never run: extraction stopped at 5,952/43,516 and no comparison output exists. The 'no complete DataV4 cache' finding covers only the branch's 2026-09-24 17:49 search. Main (b6dfcb4) references a local complete original-split DataV4 cache (_workspace/cache_v4, not in repo) and a CPU/MPS feature path. That makes CPU/MPS extraction of just the 1,477 new rows plausible, though it is still unmeasured. |
| **confirmed** | On the raw new speech candidates (n=1000: 600 LibriSpeech FLAC real vs 400 MLAAD WAV fake), a metadata-only grouped 5-fold probe gives voice AUC 1.0 in every fold. Every new source is single-class, and the probe was not re-run on the admitted derivatives. | Confirmed. The fold AUCs are rounded to 3 dp. MLAAD candidates are English only. Admitted MusicArena music is fake-only, with no matching new real music. |
| **partially** | Rights: LibriSpeech 449 and MusicArena 882 were admitted as CC BY 4.0. MLAAD 146 was admitted only on the project's own interpretation (explicit_publisher_approval=false, reversed from an exclusion 39 minutes earlier). Jamendo was excluded over its TDM terms, yet main's build_extra.py uses the same Jamendo payload and AI-Hub rows labelled 'no export abroad'. | Rights facts are confirmed, except that the MLAAD reversal came 19m45s after the committed exclusion record (6edb00e to c951608), not 39 min. Main's build_extra.py reuses the branch's Jamendo payload path and its prepared MusicArena rows, including AST-rejected ones, plus AI-Hub 'no export abroad' and MLAAD-ko rows. The current best 102411's DataV4 base already holds MLAAD (8,775 manifest rows) and JamendoMaxCaps (400 rows), so rights clearance is already an issue for 102411 and does not separate the new additions from it. |

#### Experiments and their status

- **Additional raw data collection (additional-v1)** — *completed, not evaluated*
  - Change: Downloaded 2,800 unmodified originals: Jamendo 1,000 (5,530,358,532 B, 30.16-479.64 s, mp3, 1,000 artist IDs); MusicArena 1,000 (mp3, 19.97-310 s, 13 generators, 559 prompt groups, Lyria excluded); LibriSpeech 400 (dev-clean 134 + 'long' 94 + test-other 172, FLAC, 86 speakers, 26 shared with the 200-file pilot); MLAAD 400 (WAV, English only, 15.25-58.5 s, 371 source-text groups). 229 files flagged (MusicArena 24 clipping; Jamendo 179 clipping / 24 duration mismatch / 3 artist-name match = 205 unique).
  - Data/split: All role=unassigned, training_admitted=false, sealed=false at collection
  - Metrics: 2,800 files; 7,986,807,113 B; 344,462.0419374986 s; flagged 229
  - Supported conclusion: Data exists on VESSL objvol-a5gxdg7er70i under lb-improvement-20260924/datasets/additional-v1/ (13 files, all readback SHA-256 matched). Byte and PCM SHA-256 are unique within the batch (2,800 each). This says nothing about model quality.
  - Confounders: Collection-time claim of MLAAD group independence (0 overlap) was later falsified by reference-speaker linkage (247/400 linked)
  - Evidence: `branch:docs/evidence/lb_collection_20260924/collection_summary.json`
- **CPU lineage review / roles-v4 (cpu-review-v1)** — *completed, not evaluated*
  - Change: Joined 3,000 candidates (2,800 + 200 pilot) with 42,947 historical rows through MLAAD reference-speaker, M-AILABS text, LibriSpeech speaker-alias and PCM keys. 37 conflicting historical groups (4,054 rows) moved to development; 1,603 rows changed role.
  - Data/split: Trainval loader 32,197 rows, development 9,842 rows, 0 group intersection
  - Protocol: repository/data_v4/prepare_candidate_review.py (union-find components; blockers per row)
  - Supported conclusion: roles-v4: train 28,319 / selection 2,435 / calibration 2,073 / development 10,120. Candidate independent components: LibriSpeech 93 (600 files), MusicArena 559 (1,000), MLAAD 167 (400), Jamendo 1,000 (1,000). MLAAD 247 linked to history (216 to eval-role groups, 31 train-only). Unknown labels: VOICE_PRESENT 2,000 and VOICE_FAKE 2,000 (all music), MUSIC_FAKE 1,000.
  - Confounders: English MLAAD metadata only; cross-system voice aliases unresolved; 5 unresolved references, 48 preview-generator files
  - Evidence: `branch:docs/evidence/lb_cpu_followup_20260924/review_summary.json`
- **Metadata shortcut probe on new speech candidates** — *evaluated, valid protocol*
  - Change: Grouped 5-fold logistic regression on non-content metadata only (duration and file format) to predict FILE_FAKE
  - Data/split: n=1000 speech candidates; GroupKFold(5) by component
  - Protocol: repository/data_v4/split.py:73-100 (features crop_len_s, orig_duration_s, channels, gain_db, codec one-hot). The driver that mapped candidate fields onto these features is not in the repo.
  - Metrics: voice auc_folds [1.0,1.0,1.0,1.0,1.0]
  - Supported conclusion: On the raw candidates, file format and duration alone perfectly separate real (LibriSpeech FLAC, 600) from fake (MLAAD WAV, 400). Voice AUC was 1.0 in all 5 folds, n=1000. Music and mixed were NOT_EVALUATED (n=0). This is a source-label confound, not detector quality. The probe was not re-run on the admitted 15-s PCM16 WAV derivatives.
  - Confounders: Each source is single-class (LibriSpeech all real, MLAAD all fake). Normalising the format does not remove source, language or content cues.
  - Evidence: `branch:docs/evidence/lb_cpu_followup_20260924/metadata_shortcut_probe.json`
- **CPU AST admission roles-v5 (admitted-v2)** — *completed, not evaluated*
  - Change: Deterministic 15-s crop, mono 16 kHz PCM16, peak cap 0.97. Offline AST-AudioSet on CPU, min Speech or Music score over two 10-s windows must be >= 0.5. Train-only admission. LibriSpeech 600 -> 452 prepared (148 <15 s) -> 449 admitted. MusicArena 1,000 -> 928 prepared (72 = 48 preview + 24 clipping) -> 882. Jamendo 0 (rights). MLAAD 0 (rights, at that time).
  - Data/split: New rows all train; 0 cross-role components; 0 candidate evaluation rows
  - Protocol: repository/data_v4/admit_candidates_cpu.py (prepare lines 55-143, finalize 146-254); repository/data_v4/review_audio_cpu.py
  - Metrics: LibriSpeech passed 449/452 (below 3); MusicArena passed 882/928 (below 46); max quantisation error 1.52587890625e-05 <= 1.53587890625e-05
  - Supported conclusion: 1,331 derived clips (638,938,564 B) joined 42,947 rows, giving 44,278 rows. Roles: train 29,650 / selection 2,435 / calibration 2,073 / development 10,120. Loader reads trainval 33,528 and development 9,842. Label supervision: LibriSpeech gives voice real only, MusicArena gives music fake only, the opposite component is masked. Human listening: 0 files. AST inference 635.67 s.
  - Confounders: AST is used only as a presence screen. Long speech is cut to 15 s. The 72 historical bank rows with changed group IDs need cache revalidation.
  - Evidence: `branch:docs/evidence/lb_admission_20260924/integration_verification.json`
- **MLAAD admission roles-v6 (mlaad-admitted-v1)** — *completed, not evaluated*
  - Change: Rights hold reversed. Of 400 MLAAD files, 216 eval-linked and 5 unresolved were excluded, 179 went to AST, and 146 passed the Speech >= 0.5 threshold (33 below).
  - Data/split: New rows all train; voice-fake supervision only; music masked
  - Metrics: admitted 146 / excluded 254 (exclusion reasons overlap and sum to 473)
  - Supported conclusion: 146 train-only fake-voice clips from 50 generators. Manifest 44,424 rows: train 29,796 / selection 2,435 / calibration 2,073 / development 10,120. Bank trainval 33,674, test 9,842. All 44,278 historical rows unchanged. AST wall time 85.62 s on 8 threads.
  - Confounders: The rights basis is the project's own interpretation (explicit_publisher_approval=false). English only.
  - Evidence: `branch:docs/evidence/lb_mlaad_admission_20260924/integration_verification.json`
- **E04 = E00 + 1,477 admitted train rows (data-admission comparison)** — *partial*
  - Change: Add LibriSpeech 449 + MusicArena 882 + MLAAD 146 to E00 (train 27,788 / selection 2,386 / calibration 2,023 / development 9,842). F0 fitted on E00 calibration only.
  - Data/split: Development 9,842 shared; F0 fitted on E00 calibration
  - Protocol: repository/data_v4/compare_data_admission.py:86-164 (asserts E04 adds only train rows; non-train panels identical)
  - Supported conclusion: No E04 result exists. A100 feature extraction over the 43,516-row union was stopped by the user at 5,952 rows. That partial cache contains 0 of the 1,477 new rows (added_rows_cached=0). compare_data_admission.py has no output anywhere in the repo.
  - Confounders: A full-bank feature cache is not available (cache_inventory_summary.json: data_v4_complete_feature_cache_located=false)
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **Legacy MERT feature backup audit** — *evaluated, valid protocol*
  - Change: CPU verification of a local features.zip (1,322 NPZ, 3,966 windows)
  - Protocol: repository/data_v4/audit_feature_backup.py
  - Metrics: records_matching_current_audio 0; archive 58,617,233 B
  - Supported conclusion: 0 records match the audio SHA-256 of the current 42,947 rows. The backup lacks voice_embeddings and presence. It was not promoted and is not usable as the current training cache.
  - Evidence: `branch:docs/evidence/lb_cpu_followup_20260924/feature_backup_audit.json`
- **FMA CC-BY real music pilot** — *partial*
  - Change: Plan up to 200 CC-BY tracks from new artists
  - Protocol: repository/data_v4/prepare_fma_pilot.py
  - Supported conclusion: Direct links returned HTTP 403, so 0 downloads (pending_access). The real-music gap was not filled by FMA.
  - Evidence: `branch:docs/CPU_PREPARATION_2026-09-24.md`
- **W02 pilot (singing real/fake, 4 vocal/accompaniment authenticity combos, 500-row pilot) and sealed 600-row eval** — *proposed*
  - Change: Planned 4,400 derived training rows plus 600 sealed eval rows
  - Supported conclusion: Not built. No singing data collected (CtrSVDD held over its BY-NC-ND license). No mixes constructed. sealed=false / 'sealed unavailable'.
  - Evidence: `branch:repository/docs/superpowers/plans/2026-09-24-dacon-lb-improvement-workflow.md`
- **PANNs D+ (D0 + 1,477 supplemental rows)** — *proposed*
  - Change: Add the new rows after structure selection
  - Supported conclusion: Never run. The PANNs data admission hard-excludes the LibriSpeech/MLAAD/MusicArena sources as 'Dplus_not_D0'.
  - Evidence: `branch:repository/panns_adaptation_data.py`

#### Defects

| Severity | Status | Location | Description | Affected results |
|---|---|---|---|---|
| major | open | `repository/data_v4/admit_candidates_cpu.py:23,106-108` | Admission turns every candidate into a single deterministic 15-s crop. The 'long speech' sources (LibriSpeech up to 35.155 s, MLAAD up to 58.5 s) therefore do not supply 30-60 s training examples, and the long-utterance gap they targeted stays open in training. | Any E04/D+ claim about long-speech robustness |
| major | open | `docs/evidence/lb_cpu_followup_20260924/metadata_shortcut_probe.json` | Each new source is single-class: LibriSpeech is all real voice, MLAAD all fake voice, MusicArena all fake music. On the raw data, format and duration alone separate the voice labels perfectly (AUC 1.0 in all 5 folds). The probe was never re-run on the normalized 15-s WAV derivatives, and source, language and content cues remain. | Any model trained on the roles-v5/v6 additions |
| major | open | `docs/evidence/lb_admission_20260924/rights_resolution.json; docs/evidence/lb_mlaad_admission_20260924/rights_decisions.json` | MLAAD rights decision was reversed. It was excluded at 18:10 ('excluded_research_scope_not_established'), then admitted at 18:49 (commit c951608) on the project's reading of DACON answer 417857. That answer concerns pretrained models, not data. explicit_publisher_approval=false, and the publisher scope is 'non-commercial academic research only' while the competition carries prizes. | Round-2 training-data handover if the 146 MLAAD clips are used |
| major | open | `origin/main:repository/data_v4/build_extra.py:1,179-185 (commit b6dfcb4)` | Cross-branch rights conflict. Main builds 'Jamendo real songs' rows from the same lb_collect_20260924 Jamendo payload that this branch excluded (unresolved Jamendo TDM prior-written-authorization terms). It also adds AI-Hub rows labelled 'training use only, no export abroad', which is risky for the round-2 requirement to hand over training data. Package DataV4_S2a_AIHubPhone trains on 800 AI-Hub 571 utterances. | Any main-branch S2 model or submission |
| minor | open | `docs/evidence/lb_cpu_followup_20260924/rights_review.json vs docs/evidence/lb_admission_20260924/rights_decisions.json` | The Music Arena rights review lists 'Downstream provenance/service terms and unknown preview generator identities remain unresolved', yet the admission decision marks it rights_review_closed_for_released_audio (CC BY 4.0 card only). Commercial generators' ToS (elevenlabs-music-v1 132, sonauto 159, riffusion 236 files in the collection) were not reviewed. | 882 MusicArena admitted clips |
| major | open | `docs/CPU_ADMISSION_2026-09-24.md table; docs/evidence/lb_admission_20260924/admission_receipt.json` | The real-music gap was not filled: Jamendo 0/1,000 admitted and FMA 0 downloaded (403). All 882 admitted music clips are fake, which shifts music class balance toward fake. The plan's E04 risk list names 'real music FP'. | E04 music EER and real-song false positives |
| major | open | `docs/CPU_PREPARATION_2026-09-24.md:46; plan W02` | The singing (real/AI) gap and the vocal/accompaniment mixed-authenticity gap had zero data collected or constructed. CtrSVDD was held over its BY-NC-ND license. | Song-type eval files (vocals + accompaniment) |
| minor | open | `repository/data_v4/prepare_mlaad_long.py:38` | New speech is English only (MLAAD filter fake/en; LibriSpeech English audiobooks), while the eval set is Korean voice-phishing including telephone channel. | Transfer of the voice additions to the eval domain |
| minor | unknown | `docs/PANNS_CNN_EXECUTION_2026-09-25.md:26,42` | Historical handover risk: 900 MUSAN noise processed files and 8 HAIM files are absent. The noise files were regenerated with different bytes, so the exact historical training bytes cannot be handed over if those rows fed the scored models. Whether 102411 used them is not established here. | Round-2 reproducibility of 102411 |
| major | fixed | `repository/data_v4/export_bank.py (diff, export(): training_admitted gate)` | Before b6f0b46, the exporter emitted rows with training_admitted=false whenever they had a split and labels. The exporter now skips them. |  |
| minor | fixed | `docs/ADDITIONAL_DATA_COLLECTION_2026-09-24.md:14,24` | The collection doc said MLAAD had 0 source-text group overlap with evaluation. Reference-speaker linkage later showed 247/400 linked to history, 216 of them to eval-role groups. The doc now carries a caveat and the 216 were excluded. |  |
| minor | open | `docs/evidence/lb_admission_20260924/integration_verification.json historical_cache_note` | Group identifiers of 72 historical bank rows changed, so any row-bound cache for them needs identity revalidation. |  |
| minor | open | `repository/data_v4/split.py:73-100` | The driver that produced the n=1000 metadata probe is not in the repo, and candidate rows carry duration_seconds and path suffix rather than the fields metadata_probe reads. The probe cannot be reproduced exactly from the repo. |  |
| minor | open | `docs/evidence/lb_mlaad_admission_20260924/admission_receipt.json prepared_audio_root` | The committed receipt embeds an absolute local user path (/Users/choiwuseck/Documents/ChatGPT/데이콘/...). |  |
| minor | unknown | `docs/ADDITIONAL_DATA_COLLECTION_2026-09-24.md:14` | The claim 'MLAAD 108 systems' has no backing count in any committed JSON (collection_summary/selection_audit lack a generator count). The admitted subset has new_generators=50. |  |

#### Lessons

- None of the 2,800 new originals or 1,477 admitted clips reached any trained or submitted model. 102411 predates the branch, E08 used a partial cache with 0 new rows, and PANNs 104224 used D0 with the new sources excluded. The value of the new data for the leaderboard is completely unmeasured.
- The admission pipeline is careful about provenance (hashes, lineage, role closure, masked unknown labels, notices). But it only screens for presence with AST at 0.5 and has 0 human listening.
- Single-class-per-source additions plus a metadata AUC of 1.0 are a strong warning: normalization to 15-s WAV is not enough, and source-balanced real/fake pairs are needed before any E04-style gain can be trusted.
- The data gaps the branch targeted are mostly still open: real music (0 admitted), singing (0), vocal/accompaniment authenticity combos (0), 30-60 s speech (cropped to 15 s), and the sealed eval set (none). Only fake AI music (882) and short English real/fake speech (449 + 146) were added.
- Rights decisions are inconsistent across branches: Jamendo is excluded on the branch but consumed by main's build_extra.py, and AI-Hub 'no export abroad' data sits in main's S2a. A single rights register is needed before any final submission, because round 2 requires handing over the actual training files.
- The admitted 1,477 clips are handover-ready: per-file source, revision and hash, THIRD_PARTY_NOTICES, archives with verified SHA-256 on VESSL (admitted-v2 9854dd53..., mlaad-admitted-v1 4f090b46...). The historical DataV4 bank's exact-byte handover is less certain (900 MUSAN regenerated, 8 HAIM missing).

#### Open opportunities (as proposed by the slice; ranked in the plan, not here)

- E04 needs features for only 1,477 x 15 s (~6.15 h audio). But no full-bank feature cache exists, so a clean E00-vs-E04 comparison needs full re-extraction or restoring the original DataV4 cache (not located). This is probably not feasible, and not worth the risk, inside 48 h with 3 submissions/day.
- Lowest-risk use of the new data inside 48 h, if any: add the 882 MusicArena AI-song clips as music-fake training for the music head (music EER weighs 0.3 in ADS). This needs a real-music counterweight such as the historical real music, and it cannot be validated against the sealed set because none exists.
- Resolve the rights register before the final pick. Drop Jamendo and AI-Hub rows from any candidate model unless written authorization exists. Otherwise keep the final model on data whose handover is already cleared (historical DataV4 plus LibriSpeech/MusicArena CC BY 4.0).
- Re-run the metadata probe on the normalized derivatives and on source-balanced subsets before trusting any gain from the additions.
- The 80-sample listening sheet (20 per source) is ready but empty, and filling it is cheap. The AST 'Singing' score is already computed per clip and could flag MusicArena vocal presence (all 882 currently have VOICE_PRESENT unknown).

<details><summary>Files this reader did not open (10)</summary>

- `branch:repository/data_v4/verify_acquired_audio.py`
- `branch:repository/data_v4/package_acquired_audio.py`
- `branch:repository/data_v4/stage_gpu_inputs.py`
- `branch:repository/data_v4/gpu_experiments.py`
- `branch:repository/test_candidate_admission.py`
- `branch:repository/test_candidate_review.py`
- `branch:repository/test_data_admission_comparison.py`
- `branch:docs/GPU_PREPARATION_2026-09-24.md (only grep lines)`
- `branch:docs/DATASETS.md`
- `VESSL-hosted artifacts (collection_plans.json, candidate_manifest.jsonl, ast_reviews.jsonl, roles-v5/v6 manifests, admitted-v2.tar.gz, mlaad-admitted-v1.tar.gz): not in repo`

</details>

### 3.3 A100 E00/E08 partial training, packaging, LB regression diagnosis

Branch cpu/lb-improvement-20260924, 2026-09-24. The CPU preflight and audio transfer (43,516 files) finished. A full E00/E04 A100 run started on wsp-nvxqw1jf9zdv, but the user stopped feature extraction at 5,952/43,516 files after 2,499.7 s. Two heads were then trained on that ordered partial cache (train 5,196 / selection 421 / calibration 335, seed 41): E08 (voice top-30% logit mean) and an E00 control (voice max). Neither is the planned full-data E00. On selection, which was also used for early stopping: voice EER 0.0627014702→0.0501611762 (401 rows, about 79 real, so a gap of about one real file), file EER 0.0726953842→0.0627458694 (421), music identical at 0.1715686275 (35). E08 was packaged as a legacy-decision ZIP: PANNs max presence, max FILE fusion, no fitted presence head or CDF ensemble. The user reported its LB ADS as 0.7064761905, versus 0.7683174603 for the incumbent (believed DataV4 102411). A CPU head-swap diagnostic on the same 756 rows (all of them in DataV4's old val split) put the regression mainly on the partial-data music head. On calibration, music EER was 0.1861 vs 0.0931, and swapping in the DataV4 music head recovered about 70% of the proxy-ADS gap. E08 also dropped to 1 seed from 3, dropped duration balancing and changed the decision layer, so its result says little about pooling itself. The only pooling test separate from the partial-data retrain was a post-hoc swap on DataV4's frozen max-trained heads. It made calibration voice EER worse (0.0236 → 0.0355 for top-30%, 0.0374 for LME) and left selection unchanged. E02 (voice local loss 0.25) and E09 (LME) exist only as configs. The full-data E00/E04/E08/E09 runs and F0 calibration never ran.

#### Verified claims

| Verdict | Claim | Corrected statement / evidence |
|---|---|---|
| **confirmed** | The E08 LB regression (ADS 0.7683174603 → 0.7064761905, user-reported) comes from a package that changed training data (ordered partial 5,196 train rows), seeds (3→1), duration balancing (off), the music head (from scratch on 373 rows) and the decision layer (legacy PANNs max presence and max fusion). Top-30% pooling was not isolated, so the regression is not evidence against pooling and E08 must not be resubmitted. | Confirmed as stated. Add two points: the voice head was also retrained from the DF-Arena initial classifier on the partial data (5,001 voice rows, 1 seed), and 0.7064761905 plus the 102411 = DataV4 link are user-reported or inferred, not verified in the repo. |
| **partially** | In the CPU head-swap on calibration (file 335 / voice 308 / music 43 rows), E08 music EER is 0.18614718614718614 vs DataV4 0.09307359307359307. Swapping in the DataV4 music head cuts file EER from 0.08955223880597016 to 0.07276119402985076; swapping in the DataV4 voice head leaves file EER at 0.08955223880597016. The music head is the main regression driver. These rows were in DataV4's own val split, so DataV4 numbers are optimistic. | The numbers are correct. On calibration, the DataV4 music head is the largest single-head effect: file EER 0.08955223880597016 → 0.07276119402985076, equal to 6+24 → 5+19 errors on 67 real / 268 fake. Music EER differs by only 2 real + 2 fake of 43 files (22 real / 21 fake). On selection, swapping either head gives the same file EER, 0.06274586939417784 → 0.05432074482035146. Swapping both heads plus the fitted decision layer is needed to reach DataV4's 0.055970149253731366. The rows were DataV4 val, used for its checkpoint selection and file-ensemble fit, so DataV4 is optimistic. Treat music as the largest ADS lever, not as a proven sole driver. |
| **confirmed** | No valid evidence that top-k or LME voice pooling helps. On the same partial data, E00 (max) vs E08 (top-30%) selection voice EER was 0.06270147024137118 vs 0.05016117619309693 on 401 rows (about 79 real), measured on the split used for early stopping, so the gap is about one real file. Post-hoc swapping DataV4's max-trained voice heads made calibration voice EER worse: 0.023565891472868243 → 0.03550387596899228 (top-30%) and 0.03744186046511626 (LME), on 308 rows. | No valid evidence either way. The E00→E08 selection gain (0.06270147024137118 → 0.05016117619309693 on 401 rows, 79 real / 322 fake) equals 1 real + 4 fake errors, on the split used for epoch selection. The post-hoc calibration loss (0.023565891472868243 → 0.03550387596899228 top-k, 0.03744186046511626 LME, on 308 rows: 50 real / 258 fake) equals about 1 real + 1–2 fake errors and is biased toward max. On selection, max, top-k and LME are identical at 0.03762088214482269. A proper test needs full-data retraining, which is CPU-feasible on the existing DataV4 caches if pool_odds is ported to main's trainer. |
| **partially** | Full-data E00/E04/E08/E09 never ran. Feature extraction stopped at 5,952/43,516 after 2,499.7 s on an A100-80GB (about 2.38 files/s, so about 5.1 h for the full union before head training). Only the partial cache (sha cbd0230645b3...) and the partial-run results (sha 96e5e1139b73...) are saved. | True that full-data E00/E04/E08/E09 never ran, and that only the partial cache (cbd0230645b3…) and results (96e5e1139b73…) came out of this A100 run. The extraction rate was 5952 files in 2499.7 s on an A100-SXM4-80GB (2.38 files/s). The 5.1 h full-union estimate is a lower bound because the per-file index rewrite grows quadratically. This floor applies only to re-extracting the 43,516-row union under the branch's new feature contract. Full-data head retraining on the existing DataV4 caches (42,039 rows) has already been done on CPU in main's P1Refit/S2a, with no GPU extraction. |

#### Experiments and their status

- **GPU preparation CPU preflight (job-aj1ufwbahfpk, job-lregdqqbkv9q)** — *completed, not evaluated*
  - Change: Staged 1,477 new clips and re-verified hashes of 43,516 audio files and 25 assets; E00 = 32,197 rows, E04 = 33,674, development = 9,842
  - Data/split: E00 train 27,788 / selection 2,386 / calibration 2,023; E04 adds 1,477 train rows
  - Supported conclusion: Infrastructure only. The first job ended 'failed' about 606 s in (timeout suspected). The retry succeeded after 317.25 s. No model quality was measured.
  - Evidence: `branch:docs/evidence/lb_gpu_preparation_20260924/cpu_retry_execution.json`
- **Full-data A100 E00/E04 run (wsp-nvxqw1jf9zdv)** — *partial*
  - Change: Planned: extract the full 43,516-file union, train E00 and E04 (seed 41, max 50 epochs, patience 5), then compare with a fixed F0
  - Protocol: gpu_experiments.py preflight→features→train-e00→train-e04→compare (repository/data_v4/gpu_experiments.py:235-248)
  - Supported conclusion: Only the features stage ran, stopped at 5,952/43,516 after 2,499.7 s (about 2.38 files/s, so the full union would take about 5.1 A100-hours). No full-data E00, no E04, no compare stage and no F0. Partial cache saved to objvol completed_cache.zip (sha cbd02306..., 162,285,344 bytes).
  - Evidence: `branch:docs/evidence/lb_a100_partial_train_20260924/final_export_verified.json`
- **E00_control (partial cache, voice max)** — *evaluated, invalid protocol*
  - Change: Default config: voice_pooling=max, music_local_loss_weight=0.25, voice_local_loss_weight=0, seed 41. Music WindowMIL trained from scratch; voice Linear(1280,2) initialised from the DF-Arena classifier.
  - Data/split: Ordered partial subset: train 5,196 (music 373 per epoch, voice 5,001: fake 4,325 / real 676), selection 421, calibration 335. All 5,952 IDs were in the old DataV4 train/val.
  - Protocol: The selection split was used both for early stopping (EER, then BCE) and for the reported metrics. E00 was never scored on calibration; not in repo.
  - Metrics: selection: file EER 0.0726953842 (421), voice EER 0.0627014702 (401, 5 unknown labels), music EER 0.1715686275 (35), voice presence AUC 0.998014888337469 (418), music presence AUC 0.9997779422649888 (421); score null. Voice checkpoint from epoch 7 of 12; music epoch 22 of 27.
  - Supported conclusion: A same-data control for E08 only. It is not the planned full-data E00 baseline and not comparable with DataV4.
  - Confounders: Same selection split for model choice and reporting; tiny music split; ordered partial data
  - Evidence: `branch:docs/evidence/lb_a100_partial_train_20260924/E00_control/validation_metrics.json`
- **E08_topk (partial cache, voice top-30% logit mean)** — *evaluated, invalid protocol*
  - Change: voice_pooling=topk, voice_topk_fraction=0.3 (ceil(0.3n) windows, minimum 1) for train, eval and inference; everything else as E00_control. Config repository/configs/lb_improvement/E08_voice_topk.json; pooling code full_bank_contract.py:48-54.
  - Data/split: Same as E00_control
  - Protocol: Same as E00_control. Selection is in-sample for the choice of epoch; E08 had 26 epochs to choose from vs 12 for E00.
  - Metrics: selection: file EER 0.06274586939417784 (421), voice EER 0.05016117619309693 (401), music EER 0.1715686274509804 (35, identical to E00), presence AUCs identical to E00; voice-route AUC 0.9919805 vs 0.9898970 for E00. Voice checkpoint from epoch 21 of 26. calibration (from comparison.json): file 0.08955223880597016 (335), voice 0.061007751937984495 (308), music 0.18614718614718614 (43).
  - Supported conclusion: Weak evidence that top-30% is no worse than max on this partial, selection-biased split. It shows no improvement over the incumbent.
  - Confounders: Top-30% equals max for clips with 3 or fewer windows (ceil(0.3n)=1), so the arms differ only through longer clips. The voice EER gap of 0.0125 is about one real file of 79. No held-out calibration comparison against E00.
  - Evidence: `branch:docs/evidence/lb_a100_partial_train_20260924/integration_audit.json`
- **E02 voice local loss** — *proposed*
  - Change: voice_local_loss_weight=0.25: BCE on window-occupancy targets confirmed by time alignment
  - Supported conclusion: Never run: config only, no run evidence in repo
  - Evidence: `branch:repository/configs/lb_improvement/E02_voice_local.json`
- **E09 voice LME pooling** — *proposed*
  - Change: voice_pooling=lme: logsumexp(odds) - log(n) (full_bank_contract.py:52-53); music LME unchanged
  - Supported conclusion: Never trained. It was renamed from E08_voice_lme in commit 95613b0. The only related evidence is the post-hoc swap diagnostic below.
  - Evidence: `branch:repository/configs/lb_improvement/E09_voice_lme.json`
- **Post-hoc voice pooling swap on frozen DataV4 heads (local_experiments_20260925)** — *evaluated, invalid protocol*
  - Change: Recomputed voice pooling on DataV4's 3-seed max-trained voice heads, with no retraining: max vs LME vs top-30%. Presence head and CDF ensemble kept fixed.
  - Data/split: Same 756 partial rows; all were in DataV4's old val split
  - Metrics: calibration voice EER (308 rows): max 0.023565891472868243, top-30% 0.03550387596899228, LME 0.03744186046511626. selection voice EER (401): 0.03762088214482269 for all three. File EER unchanged in both splits (0.055970149253731366 on 335; 0.05432074482035146 on 421).
  - Supported conclusion: No support for switching the pooling of already-trained heads. Pooling-aware training is untested on full data.
  - Confounders: Train/inference pooling mismatch (heads trained with max). Rows were in DataV4's val. File EER is exactly unchanged in both splits for both modes, which is not explained.
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **E08 packaging + CPU inference QA** — *completed, not evaluated*
  - Change: Took the DataV4 base archive (05ae7868...), swapped in E08 music_41.pt and voice_41.pt, and removed the fitted presence_head.json and file_ensemble.json (--legacy-decision-layer): legacy PANNs max presence and max FILE fusion
  - Metrics: 3 real files (WAV/MP3): max absolute difference from the saved A100 predictions 0.000491679. On a long clip (5 windows, 2 selected) top-k gave 0.1137824729 vs cached 0.1137860492; the max counterfactual is 0.2156186253.
  - Supported conclusion: The package runs and applies top-k pooling. Accuracy was not validated, nor was L4 runtime.
  - Evidence: `branch:docs/evidence/lb_e08_package_20260924/inference_qa.json`
- **E08 LB submission** — *evaluated, valid protocol*
  - Change: E08 legacy-decision package vs incumbent
  - Metrics: ADS 0.7064761905 vs 0.7683174603 (user-reported; 100% of the 1,200 eval files). Total and CPS not in repo.
  - Supported conclusion: E08 as packaged is worse on LB by 0.0618412698 ADS. The single submission confounds data, seeds, decision layer and pooling.
  - Evidence: `branch:docs/E08_REGRESSION_2026-09-24.md`
- **E08 regression CPU head-swap diagnostic** — *evaluated, invalid protocol*
  - Change: Scored DataV4 3-seed heads and E08 heads on the same cache, with music-only and voice-only swaps under the legacy decision layer
  - Data/split: The same 756 partial rows
  - Metrics: calibration (file 335 / voice 308 / music 43), file/voice/music EER. DataV4_original: 0.055970/0.023566/0.093074. DataV4 heads + legacy decision: 0.061567/0.023566/0.093074. E08: 0.089552/0.061008/0.186147. E08 with old music head: 0.072761/0.061008/0.093074. E08 with old voice head: 0.089552/0.023566/0.186147. selection (421/401/35): DataV4_original 0.054321/0.037621/0.114379; DataV4 legacy 0.044371/0.037621/0.114379; E08 0.062746/0.050161/0.171569; both swaps give file EER 0.054321. Derived proxy ADS on calibration: DataV4 0.93938, E08 0.88718, E08 with old music 0.92350, E08 with old voice 0.89467. Derived CPS proxy on calibration: 0.99975 fitted vs 0.98200 legacy.
  - Supported conclusion: The music head trained on 373 partial rows is the largest driver of the regression; the voice head and the legacy decision layer (which costs CPS) contribute. Pooling is not identified as a cause. Direction matches LB, but the proxy ADS (about 0.94) overstates LB (0.768) by about 0.17.
  - Confounders: All 756 rows were in DataV4's old val split, which was used for epoch choice and for fitting the presence/CDF layers, so DataV4 numbers are optimistic. Music splits are tiny. Some labels are unknown, so score is null.
  - Evidence: `branch:docs/evidence/lb_e08_regression_20260924/comparison.json`
- **A100 preparation audio export + existing A100 restart** — *completed, not evaluated*
  - Change: CPU job job-3votgz7mfrno exported portable-audio.tar (12,369,408,000 bytes). The restart of wsp-5i3knd3gxeox failed.
  - Supported conclusion: Infrastructure only. The restart returned 'workspace start failed' and a new workspace was created instead.
  - Evidence: `branch:docs/A100_PREPARATION_2026-09-24.md`

#### Defects

| Severity | Status | Location | Description | Affected results |
|---|---|---|---|---|
| major | open | `docs/E08_SUBMISSION_2026-09-24.md:3-14; docs/E08_REGRESSION_2026-09-24.md:7` | E08 was promoted to an official submission after beating only a same-data partial control (E00_control). It was never compared against the incumbent DataV4 package on the same rows. | E08 LB submission (ADS 0.7064761905); one daily submission used |
| major | open | `docs/evidence/lb_e08_package_20260924/package_receipt.json:20-21; repository/package_datav4_submission.py:56-58` | The E08 package switched the decision layer: the fitted presence head and CDF file ensemble were replaced by legacy PANNs max presence and max FILE fusion. This happened at the same time as the head swap, so the effects cannot be separated, and it probably cost CPS. | E08 LB total/CPS (not in repo); any future legacy-decision package |
| major | open | `repository/full_bank_train.py:239-262 (early stopping on selection) + docs/A100_PARTIAL_TRAINING_2026-09-24.md:9-13` | The E00-vs-E08 comparison is reported on the same selection split that chose each checkpoint (best of 26 epochs vs best of 12). E00 was never scored on calibration. | The claimed E08 voice gain (0.0627→0.0502 on 401 rows) |
| major | open | `docs/evidence/local_experiments_20260925/summary.json:32-48 (old_exposure)` | All 756 selection/calibration rows used to compare against DataV4 were in DataV4's old val split. That split chose DataV4's epochs and fitted its presence head and file_ensemble, so DataV4_original metrics are optimistic. E08_REGRESSION.md only says overlap was 'not excluded'. | All comparison.json DataV4 rows; local pooling diagnostic |
| major | open | `docs/evidence/lb_a100_partial_train_20260924/cache_saved.json (role_route_label_counts)` | The partial cache is the first 5,952 rows in extraction order, not a stratified sample. Music has 373 train / 35 selection / 43 calibration rows; voice training is fake-heavy (4,325 fake / 676 real). | E08 and E00_control music heads; all music EER conclusions (one file is roughly 0.03-0.06 EER) |
| minor | open | `docs/E08_REGRESSION_2026-09-24.md:17-27` | The markdown table leaves out the selection result for 'DataV4 heads + legacy max': file EER 0.04437 there, better than the fitted layer's 0.05432, while calibration shows the reverse (0.06157 vs 0.05597). The decision-layer effect flips sign between splits. | Any conclusion that the legacy decision layer is worse or better |
| minor | unknown | `docs/evidence/local_experiments_20260925/summary.json:1271 (pooling)` | In the post-hoc pooling swap, file EER is identical to max-pooling in both splits and selection voice EER is identical for all three modes. This is unexplained; the diagnostic may not have propagated the new pooling fully. | The pooling evidence based on frozen heads |
| minor | open | `docs/evidence/lb_a100_partial_train_20260924/*/validation_metrics.json` | Some voice labels are unknown (5 voice-EER and 3 presence labels in selection; 8 and 6 in calibration), so the evaluator's combined score is null throughout. | All partial-cache metrics |
| minor | open | `docs/evidence/lb_e08_package_20260924/inference_qa.json` | E08 was checked on 3 files only. L4 runtime and memory for the full 1,200-file test, and dependency install, were not validated before submission. | E08 LB runtime (not in repo) |

#### Lessons

- Before spending an LB submission, compare the candidate with the incumbent package (DataV4, 102411) on the same rows. Beating a control trained on the same data is not evidence of beating the incumbent.
- Change one thing per submission. E08 changed training data (5,196 partial rows), seeds (3→1), duration balancing (on→off), the music head (retrained on 373 rows), the decision layer (fitted→legacy max) and voice pooling at once, so the LB drop can't be pinned on pooling.
- In the head-swap diagnostic, restoring the DataV4 music head recovered about 70% of the calibration proxy-ADS gap (0.88718→0.92350 vs 0.93938). Restoring the voice head recovered about 0.0075. Music quality drives both music EER and file EER.
- Top-30% pooling equals max for clips with 3 or fewer windows (window 64,600 samples, hop 32,300 at 16 kHz). It only matters for longer clips, so expect small effects.
- Local selection/calibration proxy ADS (about 0.93-0.94 for DataV4) overstates LB ADS (0.768) by about 0.17, because those rows were in DataV4's own val split. Use them for direction only. For the one pair measured (DataV4 vs E08) the direction matched LB.
- An ordered partial feature cache gives unusable music splits (35/43 rows) and a skewed voice class balance. If a subset is needed, stratify by route, class and source.
- Checkpoint SHA differs even when weights are identical, because the payload embeds run_identity/config. The music heads have different hashes but identical EER/AUC across E00 and E08.
- Full-feature extraction ran at about 2.38 files/s on an A100-80GB (5,952 files in 2,499.7 s), so all 43,516 files would take about 5.1 h before any head training.

#### Open opportunities (as proposed by the slice; ranked in the plan, not here)

- Keep DataV4 (102411, total 0.7912534339) as the incumbent. Do not resubmit E08 or any head trained on the 5,952-row partial cache.
- If trying a pooling or local-loss variant (E08/E09/E02), train it on the full E00 train set with the DataV4 recipe (3 seeds, duration balance, fitted presence head and CDF ensemble). Change only the pooling, and compare against DataV4 on rows held out from DataV4's val. None of this exists yet.
- Decision layer: the legacy max fusion vs fitted CDF ensemble effect flips sign between splits (selection file EER 0.04437 vs 0.05432; calibration 0.06157 vs 0.05597), and both splits are in DataV4's val. It is not a reliable LB lever without clean held-out rows.
- The music head is the highest-leverage component (weight 0.3, and it feeds file EER through fusion). Evidence here says do not retrain it on less data. A music improvement would need full data or extra data (E04's 1,477 rows are not in the partial cache).
- The saved partial cache (objvol lb-improvement-20260924/a100-partial-v1/completed_cache.zip, sha cbd02306...) plus about 5.1 A100-hours of extraction would allow a full-data E00/E04 run if GPU time is approved before 2026-09-29 10:00 KST.
- Find the E08 LB total, CPS and runtime (not in repo) to measure how much CPS the legacy PANNs-max presence lost.

<details><summary>Files this reader did not open (5)</summary>

- `branch:docs/evidence/lb_gpu_preparation_20260924/ remaining files (cpu_retry_environment.json, cpu_retry_runtime.log, package_receipt.json, cpu_retry_readiness.json, staging_verification.json, startup_change.json, completion_transfers.json, transfers.json, cpu_verify_job_request.json, cpu_job_request.json, retry_transfers.json, original_worktree_verification.json, cpu_job_runtime.log, cpu_retry_preflight.json) - skim scope, infra only`
- `branch:docs/evidence/lb_a100_preparation_20260924/ (linked from A100_PREPARATION doc; not in assigned list)`
- `branch:docs/evidence/lb_a100_run_20260924_v1/workspace_logs.txt (pip install section skipped)`
- `E08/E00 validation_predictions.csv, *_history.json, candidate_lock.json, results.zip: not in repo (object volume only)`
- `DataV4 file_ensemble.json / presence_head.json / run_identity.json: not in repo`

</details>

### 3.4 Local head experiments, CPU score diagnostics, research notes (cpu/lb-improvement-20260924, commits 728bda5, 780692a, f3a984d, 0e5052a, 654488a)

Five small-head variants were trained on top of the frozen DataV4_DurationBalanced heads, using a partial A100 feature cache of 5,952 rows (train 5,196 / selection 421 / calibration 335): music residual with kernel 1 and kernel 3, voice linear retrain, voice linear with reweighting, and a voice residual MLP. Each ran with 3 seeds and a fixed final epoch. Every row had already been seen by the reference model: train rows were in its old train split, and selection/calibration rows were in its old val split, where it was selected and its CDF was fit. The comparisons therefore favour the reference, and none is independent. - **Voice variants:** worse than the reference on both roles. - **Music residuals:** better on the 35-row selection panel, worse on the 43-row calibration panel. The paired 95% intervals are about ±10pp wide. - **Pooling and singing-conditioned calibration:** LME, top-30% and the singing-conditioned calibration all made calibration Voice EER worse. - **Only positive signal:** a monotone Platt fit on the voice-fake logit, fit on the selection role, lowered calibration FILE EER from 0.05597 to 0.04478 (335 rows, no CI, not deployed). Result: no variant was adopted, and DataV4 was kept. The full-proxy recompute on the DurationBalanced test split (6,444 rows) found two error clusters: - Voice-fake EER is 0.2060 when music is present (n=1,544) against 0.0443 for voice only (n=3,595). - Music-fake EER is 0.2001 (n=2,849). - Substituting true music-fake labels cuts FILE EER from 0.1094 to 0.0445; substituting true voice-fake labels cuts it to 0.0607. The presence diagnostics (on an older, different DataV4 candidate) show little leverage on FILE: true presence gives −0.00366 (CI includes 0). PANNs v1 trains presence only. ads_cached_experiments.py (independent5 / shared5 / joint9 heads plus a FILE logistic) is implemented, but no results are in the repo. The research notes are proposals only.

#### Verified claims

| Verdict | Claim | Corrected statement / evidence |
|---|---|---|
| **partially** | No local head variant beat the preserved DataV4_DurationBalanced heads. Voice linear, reweight and MLP were worse on target Voice EER in both roles (sel 0.0407-0.0502 vs 0.0376 on n=401; cal 0.0413-0.0552 vs 0.0236 on n=308). The music residuals improved selection Music EER (0.0572 vs 0.1144, n=35) but worsened calibration (0.1396 vs 0.0931, n=43), with CIs spanning 0. LME and top-30% voice pooling were also worse. | No cached-feature head variant improved its target component EER in both exposed roles. The voice variants' target EER was worse on point estimates (sel 0.0407-0.0502 vs 0.0376, n=401; cal 0.0413-0.0552 vs 0.0236, n=308), and all their paired CIs span 0. Music residuals improved selection Music EER (0.0572 vs 0.1144, n=35) and worsened calibration (0.1396 vs 0.0931, n=43); those CIs also span 0. Two variants lowered FILE EER in one role only: voice_linear on selection (0.0444 vs 0.0543) and music_k1 on calibration (0.0448 vs 0.0560). LME and top-30% pooling were worse on calibration (0.0374 and 0.0355 vs 0.0236) and tied on selection. All rows were exposed to the reference, no new data was cached, and the panel has only 15-16 voice-with-music rows and 35-43 music rows, so the test is underpowered. |
| **confirmed** | On the DurationBalanced historical test proxy (6,444 rows), voice-fake EER is 0.20596 when music is present (n=1,544) vs 0.04425 voice-only (n=3,595), and music-fake EER is 0.20005 (n=2,849). Oracle music-fake labels cut FILE EER from 0.10940 to 0.04455; oracle voice-fake labels cut it to 0.06066. | summary.json full_proxy matches exactly: - predicted_rows 6444 (bank 6450, 6 rows missing predictions). - voice_with_music: rows 1544 (728 real/816 fake), EER 0.2059632, AUC 0.8743. - voice_only: rows 3595 (2287/1308), EER 0.0442526. - music: rows 2849 (1819/1030), EER 0.2000550. - Reference FILE EER 0.1093995. MUSIC oracle 0.0445455 (2,849 rows replaced). VOICE oracle 0.0606637 (5,139 rows replaced). The generating code (full_proxy.py:14-16) reads data_v4_phase_c_20260923T1653Z/train_duration_balanced/test_predictions_decision.csv. That is the same HEADS directory the DataV4_DurationBalanced packager uses (package_datav4_submission.py:26-28). The code also checks that the FILE combiner replays to within 1.1e-16. The reference metrics and input hashes (da46ff8b..., 6444/6450 rows, the same 6 missing IDs) are recorded independently in lb_cpu_20260924/baseline_lock.json, with identical file_eer, voice_eer and music_eer. The raw CSVs are not in the repo (they sit in the local results/ tree), so the recomputation is from the JSON, cross-checked across two JSON files. Caveats: - Status is evaluated-valid on a historical held-out test that had been previously inspected. It is not the 1,200-file eval distribution. - voice_with_music is not a verified singing label (per the limitation field). - The oracles give FILE-leverage diagnostics under a fixed CDF combiner, not attainable gains. - The decision gloss 'not from presence' rests on claim 3, not on this evidence. |
| **partially** | Presence has negligible FILE leverage. True-presence substitution changes FILE EER by -0.00366 (95% CI [-0.00636,+0.00013], 6,268 rows, older candidate). Fitted presence temperature/Platt gives +0 / +0.00424. PANNs v1 trains only presence losses and copies the fake outputs, so it cannot improve Voice/Music EER; this is consistent with the 104224 regression. | From markdown only (raw JSON not in repo), on an older candidate's 6,268-row panel: substituting true music presence changes FILE EER by -0.00366 (CI [-0.00636,+0.00013]); substituting both presences gives -0.00366 (CI [-0.00638,+0.00021]); voice presence gives 0. Presence temperature/Platt fitted on 2,021 rows and scored on 2,384 give 0 and +0.00424. These numbers are in cpu-score-diagnostics.md, not the cited audit doc. The code confirms that PANNs v1 trains only the two presence losses and keeps the fake heads frozen: fake outputs are recomputed with unchanged weights, so Voice/Music EER cannot improve. That fits the small 104224 regression (ADS -0.00086, CPS -0.0034), but the archive identity was not server-verified. |
| **partially** | The only positive local signal is post-hoc: a monotone Platt on the voice-fake logit (fit on selection, n=401) lowered calibration-role FILE EER from 0.05597 to 0.04478 (335 rows, exposed, no CI), with Voice EER unchanged by construction. Conditional transforms (fake x presence) worsened Voice/Music EER; FILE raw max / noisy-OR changes were within noise. | A monotone Platt on the seed-mean voice-fake logit (coefficient +4.25, fit on 401 exposed selection rows) lowered calibration FILE EER from 0.05597 to 0.04478 (335 rows) and in-sample selection FILE EER from 0.05432 to 0.04285, with Voice EER unchanged. No CI was computed, and both roles were used to fit the reference CDF. It is not the only positive FILE signal: music_k1 gave the same calibration FILE drop (to 0.04478), and voice_linear lowered selection FILE EER to 0.04437, although both hurt or failed to improve their target EER. From markdown only (older-candidate 6,183-row panel): fake×presence raised Voice/Music EER by +0.0015 and +0.0082, both CIs spanning 0; raw-max and noisy-OR FILE changes were within noise. This is a risky LB probe, not an expected gain. |

#### Experiments and their status

- **music_k1 (pointwise residual on frozen WindowMIL)** — *evaluated, invalid protocol*
  - Change: MusicResidual kernel=1 (80,705 params) adds delta to frozen window logits then LME; 3 seeds x 60 steps, lr 3e-4, fixed epoch 20; seed-mean probs
  - Data/split: train 373 music rows (old train); eval selection music n=35 (18R/17F), calibration n=43 (22R/21F); FILE n=421/335; all rows exposed to reference (old val)
  - Protocol: fixed final epoch, no early stopping; presence/CDF frozen; bootstrap over component_id
  - Metrics: Music EER sel 0.05719 vs ref 0.11438; cal 0.13961 vs ref 0.09307; FILE EER cal 0.04478 vs 0.05597, sel 0.05432 = ref. Paired component bootstrap (400) cal [-0.0238,+0.1396], sel [-0.1417,+0.0872]
  - Supported conclusion: No reliable gain; sign flips between roles. Rejected.
  - Confounders: tiny panels (1 error ~2.3-5.7pp); exposed roles; train rows already fit by base
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **music_k3 (temporal residual)** — *evaluated, invalid protocol*
  - Change: MusicResidual kernel=3, 3 temporal layers (105,281 params); same budget
  - Data/split: same as music_k1
  - Protocol: as music_k1
  - Metrics: Music EER sel 0.05719, cal 0.13961; FILE cal 0.05784, sel 0.05432. Paired CI vs ref cal [-0.0627,+0.1623], sel [-0.1417,+0.0005]. k3 minus k1 cal [-0.0475,+0.1152] (43 groups), sel [-0.0872,0.0] (35 groups)
  - Supported conclusion: No evidence that temporal context helps. Rejected.
  - Confounders: param count differs from k1, so temporal-context effect not isolated
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **voice_linear retrain** — *evaluated, invalid protocol*
  - Change: Continue training the preserved Linear(1280,2) voice head (2,562 params), 800 steps, lr 1e-4, max pooling
  - Data/split: train 5,001 voice rows; sel voice n=401 (79R/322F, 5 unknown); cal n=308 (50R/258F, 8 unknown)
  - Protocol: fixed final epoch
  - Metrics: Voice EER sel 0.04073 vs 0.03762; cal 0.04132 vs 0.02357. FILE sel 0.04437 vs 0.05432; cal 0.05970 vs 0.05597. Paired CI cal [-0.0043,+0.0420] (116 groups), sel [-0.0026,+0.0210] (157 groups)
  - Supported conclusion: Worse point estimates on the target metric in both roles. Rejected.
  - Confounders: exposed roles
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **voice_linear_reweight** — *evaluated, invalid protocol*
  - Change: Same as voice_linear but loss weights rebalanced across MUSIC_PRESENT cells per class (clip 0.25-4); local_head_experiments.py:163-182
  - Data/split: same as voice_linear; the 1,477 added rows have 0 cached features, so no new data was used
  - Protocol: fixed final epoch
  - Metrics: Voice EER sel 0.05016, cal 0.05519 (vs 0.03762/0.02357); FILE sel 0.05127, cal 0.05970; voice-with-music slice unchanged (cal 0.1909 n=16, sel 0.2614 n=15). CI cal [-0.0021,+0.0521]
  - Supported conclusion: Worst voice variant; domain reweighting did not fix the voice-with-music gap. Rejected.
  - Confounders: reweighting only; not new data
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **voice_mlp residual** — *evaluated, invalid protocol*
  - Change: VoiceResidual: frozen linear plus LayerNorm-MLP(128) correction to fake odds (166,657 params); run only because selection voice_with_music EER > 0
  - Data/split: same as voice_linear
  - Protocol: fixed final epoch
  - Metrics: Voice EER sel 0.04073, cal 0.04132; FILE sel 0.05432, cal 0.05784. CI cal [-0.0065,+0.0416], sel [-0.0014,+0.0210]
  - Supported conclusion: No gain. Rejected.
  - Confounders: exposed roles
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **Voice pooling: LME and top-30% (no fitting)** — *evaluated, invalid protocol*
  - Change: Replace window max of the frozen voice fake odds with LME or top-k 0.3; local_head_experiments.py:389-402
  - Data/split: cal voice n=308, sel n=401
  - Protocol: frozen heads, seed mean
  - Metrics: cal Voice EER: max 0.02357, LME 0.03744, top-30% 0.03550; sel unchanged at 0.03762; FILE unchanged (0.05597 / 0.05432)
  - Supported conclusion: Max pooling is at least as good. This contradicts the top-k voice pooling used in the E08 candidate. Rejected.
  - Confounders: exposed; small
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **Voice-fake logit recalibration (logit_only Platt)** — *evaluated, invalid protocol*
  - Change: StandardScaler + LogisticRegression on voice fake logit (coef 4.2536), fit on selection known voice rows (n=401), applied before the frozen FILE CDF combiner; local_head_experiments.py:404-419
  - Data/split: fit selection; held-out calibration role 335 rows (both exposed to the base)
  - Protocol: single fit, single apply
  - Metrics: FILE EER cal 0.05597 -> 0.04478 (-0.01119); sel (fit role) 0.05432 -> 0.04285; Voice EER unchanged (monotone): cal 0.02357, sel 0.03762. No CI computed
  - Supported conclusion: The only local signal with a FILE gain; not significant, not deployed. A candidate for a post-hoc-only submission, with high risk.
  - Confounders: exposed data; n=335; no bootstrap; CDF tables were fit on uncalibrated Fv
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **Voice logit + PANNs singing calibration** — *evaluated, invalid protocol*
  - Change: Adds tag_singing_max as a second feature (coefs 4.2492, 0.0959)
  - Data/split: fit on selection, eval on calibration
  - Protocol: as logit_only
  - Metrics: cal Voice EER 0.03744 (vs 0.02357), cal FILE 0.04478; sel Voice EER 0.05016 (worse even on the fit role), sel FILE 0.05432
  - Supported conclusion: Singing conditioning gives no gain. Rejected.
  - Confounders: exposed; small
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **Full-proxy slice recompute (DurationBalanced test)** — *evaluated, valid protocol*
  - Change: Recompute metrics from saved test_predictions_decision.csv of data_v4_phase_c_20260923T1653Z/train_duration_balanced; FILE replay error 1.11e-16
  - Data/split: bank_test 6,450 rows, 6,444 predicted (6 missing); historical, previously inspected in-distribution test, not an LB proxy
  - Protocol: descriptive, no fitting (full_proxy.py:32-40)
  - Metrics: FILE EER 0.10940 (n=6,444); Voice EER 0.11071 (n=5,139, 261 unknown); Music EER 0.20005 (n=2,849); presence AUC voice 0.99795 (n=6,268), music 0.99865 (n=6,444). Slices: voice_only EER 0.04425 AUC 0.99209 (n=3,595, 2,287R/1,308F); voice_with_music EER 0.20596 AUC 0.87425 (n=1,544, 728R/816F); music EER 0.20005 AUC 0.88543 (n=2,849, 1,819R/1,030F)
  - Supported conclusion: Voice-fake detection degrades about 4.7x in EER when music is present; music-fake EER is about 0.20. These are the two largest error pools.
  - Confounders: voice_with_music is not a verified singing label; in-distribution sources
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **Fake-label oracle substitution (full proxy and local cache)** — *evaluated, valid protocol*
  - Change: Replace known-present component fake prob with its 0/1 label; presence and combiner fixed
  - Data/split: full proxy 6,444 (music 2,849 / voice 5,139 replaced); local cache cal 335 / sel 421 (music 78 / voice 709 replaced)
  - Protocol: sensitivity only
  - Metrics: Full proxy FILE 0.10940 -> music oracle 0.04455, voice oracle 0.06066. Local cal FILE 0.05597 -> music 0.02985, voice 0.01493; sel 0.05432 -> music 0.04132, voice 0.01147
  - Supported conclusion: On the larger proxy, music-fake errors drive more of the FILE errors than voice-fake errors.
  - Confounders: not an attainable bound
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **CPU diag A: older DataV4 candidate metrics and source breakdown** — *evaluated, valid protocol*
  - Change: Recompute on data_v4_results_20260923T1639Z (fake ckpts and CDF differ from the preserved package)
  - Data/split: old val 4,927; old test 6,444 (now development)
  - Protocol: no fitting
  - Metrics: val FILE 0.073421, Voice 0.059922 (3,785), Music 0.124699 (1,748); test FILE 0.113911, Voice 0.122801 (5,139), Music 0.196189 (2,849); presence AUC test 0.997946 / 0.998654. Music EER by source: constructed_mix 0.225947 (518), constructed_song 0.248418 (400), haim 0.188345 (786). Zero (0,0) files in either panel
  - Supported conclusion: Constructed songs are the hardest music source. There is no noise/silence coverage for presence.
  - Confounders: different candidate from the submitted package; JSON only in _workspace (not in repo)
  - Evidence: `branch:docs/superpowers/research/2026-09-25-cpu-score-diagnostics.md`
- **CPU diag B: true-presence substitution** — *evaluated, valid protocol*
  - Change: Replace predicted presence with truth; fake and CDF fixed
  - Data/split: 6,268 test rows with known presence, 2,572 components
  - Protocol: predeclared
  - Metrics: FILE EER 0.10289136 baseline; voice truth delta 0 [-0.00066,+0.00082]; music truth -0.00365574 [-0.00636,+0.00013]; both -0.00365574 [-0.00638,+0.00021]; 1,000 paired bootstrap
  - Supported conclusion: Presence has negligible leverage on FILE EER in the dev proxy.
  - Confounders: older candidate; JSON not in repo
  - Evidence: `branch:docs/superpowers/research/2026-09-25-cpu-score-diagnostics.md`
- **CPU diag C: fixed presence temperature 0.5/1/2** — *evaluated, valid protocol*
  - Change: Scale presence logits (unfitted)
  - Data/split: same 6,268
  - Protocol: predeclared
  - Metrics: FILE EER T=0.5 0.10242605 (-0.00046531 [-0.00128,+0.00046]); T=2 0.10242605 (-0.00046531 [-0.00168,+0.00122]); AUC unchanged
  - Supported conclusion: No effect.
  - Confounders: older candidate
  - Evidence: `branch:docs/superpowers/research/2026-09-25-cpu-score-diagnostics.md`
- **CPU diag D: fitted presence temperature / Platt** — *evaluated, invalid protocol*
  - Change: Fit per-target T (voice 0.7229, music 0.8555) or Platt on calibration 2,021; apply to selection 2,384
  - Data/split: calibration 2,021 (1,193 components) / selection 2,384 (1,268 components); historical CDF was fit on this old val
  - Protocol: exploratory, recorded before running
  - Metrics: Selection FILE EER 0.06667462 raw; temperature +0 [-0.00156,+0.00170]; Platt 0.07091731 (+0.00424 [-0.00153,+0.00713]); AUC voice 0.99882332, music 0.99587812 for all
  - Supported conclusion: Presence calibration gives no FILE gain; Platt point estimate is worse. Rejected.
  - Confounders: CDF exposure
  - Evidence: `branch:docs/superpowers/research/2026-09-25-cpu-score-diagnostics.md`
- **Conditional-scoring transforms (five-output audit)** — *evaluated, valid protocol*
  - Change: (a) VOICE/MUSIC_FAKE := fake x presence; (b) FILE raw max; (c) FILE noisy-OR; (d) FILE max without presence
  - Data/split: complete-label dev panel 6,183 rows, 2,572 parent groups; Voice n=5,139, Music n=2,588; older candidate
  - Protocol: predeclared, 2,000 parent-group bootstrap
  - Metrics: baseline FILE 0.10480029 / Voice 0.12280055 / Music 0.17612018. fake x presence: Voice 0.12433595 (+0.00154 [-0.00113,+0.00209]), Music 0.18431885 (+0.00820 [-0.00242,+0.01149]). raw max FILE 0.10351641; noisy-OR 0.10364943 (CIs include 0; reversed sign on old val); max without presence 0.11433303
  - Supported conclusion: Do not multiply fake by presence. Combiner swaps are within noise.
  - Confounders: results JSON only in _workspace (not in repo); older candidate
  - Evidence: `branch:docs/superpowers/research/2026-09-25-five-output-proposal-audit.md`
- **DataV4 package real-audio CPU/MPS parity** — *evaluated, valid protocol*
  - Change: Unpack 53 receipt-verified members; infer 10 real files
  - Data/split: 10 existing real files (voice/music/mix, wav/flac/mp3)
  - Protocol: network denied, weights_only
  - Metrics: CPU CLI vs stored: max abs err 0 on all 5 columns; reversed 2-file subset err 0; MPS max err 2.60e-5 (fails 1e-5); CPU 95.08 s / 10 files
  - Supported conclusion: The preserved package reproduces deterministically on CPU.
  - Confounders: 10 files only
  - Evidence: `branch:docs/evidence/local_experiments_20260925/summary.json`
- **ads_cached_experiments: independent5 / shared5 / joint9 heads + FILE logistic fusion** — *implemented, not run*
  - Change: Mean-pooled voice+MERT+24 presence stats; 3 arms x 3 seeds; best epoch by selection BCE (ads_cached_experiments.py:459-467); FILE logistic on 4 components + 6 interactions trained on train role (359-396)
  - Data/split: current cache or legacy two-cache (cache_tv/cache_test) with current roles
  - Protocol: 30-min CPU cap; adoption_allowed=False
  - Metrics: none in repo
  - Supported conclusion: None; results not in repo.
  - Confounders: mean pooling drops temporal info; preserved heads saw train rows, so the FILE fit is not OOF
  - Evidence: `branch:repository/ads_cached_experiments.py`
- **legacy_cache_export** — *implemented, not run*
  - Change: Hash-verify and copy feature_cache_v4.tar (cache_tv + cache_test)
  - Data/split: n/a
  - Protocol: n/a
  - Metrics: no export receipt in repo
  - Supported conclusion: Infrastructure only.
  - Confounders: n/a
  - Evidence: `branch:repository/legacy_cache_export.py`
- **D+ 1,477 added rows / raw+stem / multi-layer MERT** — *proposed*
  - Change: Train with LibriSpeech 449, MusicArena 882, MLAAD 146
  - Data/split: added_rows_cached=0
  - Protocol: n/a
  - Metrics: none
  - Supported conclusion: Not done: no cached features.
  - Confounders: n/a
  - Evidence: `branch:docs/LOCAL_EXPERIMENTS_2026-09-25.md`
- **Opus xhigh follow-up arms A-I and new data sources** — *proposed*
  - Change: v1 R0-R5, D+, paired SNR/codec augmentation, label cleaning, MIL readout, presence calibration, distillation, raw/stem, generated data (MeloTTS/ACE-Step); ESC-50, BabySlakh, MUSDB18-HQ, MoisesDB, MedleyDB, FSD50K (on hold)
  - Data/split: none
  - Protocol: n/a
  - Metrics: none
  - Supported conclusion: Plans only; no data downloaded or trained.
  - Confounders: n/a
  - Evidence: `branch:docs/superpowers/research/2026-09-25-opus-xhigh-followup.md`

#### Defects

| Severity | Status | Location | Description | Affected results |
|---|---|---|---|---|
| major | open | `local_head_experiments.py roles / summary.json data.old_exposure` | All 5,952 cached rows were in the reference model's old train/val. selection (421) and calibration (335) were old val, used for head selection and CDF fit. Comparisons against the reference are therefore biased toward the reference and are not independent. | All five head variants, pooling and calibration diagnostics |
| major | open | `summary.json comparison.*.seed_metrics (music)` | Music panels are too small to resolve differences: selection 35 (18R/17F), calibration 43 (22R/21F). One error moves EER about 2.3-5.7pp, and the paired CIs span about 15-23pp. | music_k1, music_k3, temporal_vs_pointwise |
| minor | open | `cpu-score-diagnostics.md and five-output-proposal-audit.md vs full_proxy.py:14-17` | Two different historical DataV4 prediction sets are in use. The CPU diagnostics and conditional-scoring audit use data_v4_results_20260923T1639Z (test decision SHA b767d4ab..., file_ensemble 1df117a2...), whose fake checkpoints and CDF differ from the preserved package (FILE max diff 0.056). full_proxy uses train_duration_balanced (SHA da46ff8b..., file_ensemble 3e0056d1...). Metrics differ: FILE 0.113911 vs 0.109400, Voice 0.122801 vs 0.110710, Music 0.196189 vs 0.200055. They must not be pooled. | Presence sensitivity and conditional-scoring conclusions apply to the older candidate, not DurationBalanced |
| minor | open | `summary.json protocol` | The reference archive path and SHA used for the head run are not recorded in summary.json. Only test_local_head_experiments.py:22 shows submissions/DataV4_DurationBalanced/submit.zip. | Identity of the reference heads |
| minor | open | `_workspace/* (not in repo)` | The underlying results JSON for the CPU diagnostics (results.json, calibration_*.json), the conditional-scoring audit, the Opus rounds and the local head checkpoints/NPZ are not in the repo; only hashes and markdown are. Numbers cannot be re-verified here. | CPU diag A-D, conditional-scoring transforms |
| minor | unknown | `five-output-proposal-audit.md:3` | The note says there is no evidence yet that the preserved package is identical to the user's best LB submission (102411). Package-to-submission identity is unverified in these files. | Any claim that the local reference equals the LB-best model |
| minor | open | `ads_cached_experiments.py:459-467 vs local_head_experiments.py:214-255` | Protocols differ. The ADS arms pick the best epoch by selection BCE, while the local heads use a fixed final epoch. The ADS FILE logistic is fit on train-role scores from heads that already saw those rows (not OOF). | Any future ads_cached_experiments results |

#### Lessons

- No local cached panel is independent: all 5,952 rows were seen by the DataV4 reference. The LB is the only unbiased evaluation.
- Small-head changes on frozen features give no reliable gain over the DataV4 heads. Voice variants were worse on both roles, and music results flip sign between the 35-row and 43-row panels.
- Max pooling of voice window odds beats LME and top-30% (cal Voice EER 0.0236 vs 0.0374 / 0.0355). This is consistent with the E08 top-30 regression, though E08 also changed other things.
- Error mass sits in music fake (EER about 0.20, n=2,849) and voice fake when music is present (EER 0.206, n=1,544 vs 0.044 voice-only). Presence is near-saturated (AUC about 0.998).
- True presence changes FILE EER by only -0.0037 (CI includes 0). Presence-only work such as PANNs v1 cannot improve Voice/Music EER, since it copies the fake outputs.
- Monotone per-column transforms leave that column's EER/AUC unchanged but can shift FILE through the CDF combiner. The only positive local signal was of this kind (voice-logit Platt, FILE -1.12pp on 335 rows).
- Multiplying fake by presence worsens Voice/Music EER on point estimates; do not use it.
- Two historical DataV4 score sets exist with different CDFs; do not mix them.

#### Open opportunities (as proposed by the slice; ranked in the plan, not here)

- Singing / voice-with-music fake detection: voice EER 0.206 vs 0.044 voice-only on the 6,444-row proxy. Eval songs are vocals plus accompaniment, so this directly hits Voice EER (0.18 weight) and FILE.
- Music-fake path: EER about 0.20. The music oracle cuts proxy FILE EER from 0.1094 to 0.0445, the largest single lever for FILE (0.45) and Music (0.27). Constructed_song is the worst source (0.248, n=400; older candidate).
- Post-hoc voice-fake logit Platt before the FILE CDF combiner (coef 4.2536 on standardized logit). CPU-only, no retraining; FILE -0.01119 on 335 exposed rows, no CI. Would need an LB submission to test; per-column EERs are unchanged by construction.
- FILE combiner swap to raw max or noisy-OR: -0.0013 / -0.0012 FILE EER on the older candidate, CI includes 0 and reversed on val. Low-priority LB probe.
- ads_cached_experiments.py (shared5/joint9, FILE logistic with interactions) is implemented and CPU-bounded to 30 min, but has no results; it could be run if the caches exist (outside this read-only audit).
- The 1,477 added rows (LibriSpeech/MusicArena/MLAAD) have no cached features. Any use needs GPU feature extraction, which is unlikely within 48h.

<details><summary>Files this reader did not open (9)</summary>

- `_workspace/lb_local_20260925/ (head checkpoints, NPZ predictions, epoch logs): not in repo`
- `_workspace/panns_cnn_v1/cpu_score_diagnostics/{results.json,calibration_fit.json,calibration_results.json,run.py,calibration.py}: not in repo`
- `_workspace/panns_cnn_v1/proposal_audit/conditional_scores/: not in repo`
- `_workspace/panns_cnn_v1/opus_xhigh_followup/ (prompts/responses): not in repo`
- `_workspace/panns_cnn_v1/data_method_research/primary_receipts.json: not in repo`
- `ads_cached_experiments run outputs (summary.json, preserved_heads/report.json, file_logistic/report.json): not in repo`
- `branch:docs/E08_REGRESSION_2026-09-24.md (out of slice)`
- `branch:docs/superpowers/specs/2026-09-25-panns-cnn-expanded-design.md (out of slice)`
- `branch:repository/v5_fusion.py (joint9 implementation, out of slice)`

</details>

### 3.5 PANNs CNN adaptation design, execution, submission 104224

The PANNs work fine-tuned only the Cnn14 audio tagger that feeds the frozen DataV4 presence logistic. Arm R3 adds an SE gate (initialised as identity) after block5 and fine-tunes the block6 convs, 56,755,264 trainable params. The six fake heads, encoders, presence logistic and the FILE CDF ensemble tables (max, noisy_or, max_nopresence) all stayed frozen. So the change moves VOICE/MUSIC_PRESENT, and FILE only through Pv/Pm inside the max and noisy_or terms. By design it cannot change leaderboard Voice or Music EER. The locked design was 12 runs: R1/R2/R3/R5 x seeds 41/42/43. It never finished. The run that shipped was a 12-hour A100 workflow: R1 continued from a parent checkpoint, R3 started fresh, seed 41 only, early stopping (max 200 epochs, patience 15) on selection-role presence BCE. The same selection role then chose between R0/R1/R3 and was the packaging input. Its internal selection ADS was "near 0.923" per the markdown; chosen.json, terminal.json and the export's selection.json are not in the repo. The first R3 package crashed on the judge's NumPy 1.26.4. The fixed package (sha 6efcb018…, inferred to be 104224) scored total 0.7901426243, ADS 0.7674603175, CPS 0.9942833862, runtime 26m16s. That is below 102411 on every component: ADS -0.0008571428, CPS -0.0033938096, total -0.00111081. It was also 8m44s slower. Inferred mechanism: presence AUC sum fell 0.0067876. With fake outputs unchanged, the ADS drop implies FILE EER rose 0.0017142856. The CDF tables were never refit to the shifted Pv/Pm distribution. The architecture review's head variants (music residual k=1/k=3, voice linear, reweighted and MLP) were run locally on exposed data and none was adopted. Raw+stem or multi-layer MERT features, D+ data, CDF refit and R2/R4/R5/R6/D1/D2 were never run.

#### Verified claims

| Verdict | Claim | Corrected statement / evidence |
|---|---|---|
| **confirmed** | Submission 104224 (inferred to be the PANNs R3 package, sha 6efcb018…) scored below 102411 on every component: total 0.7901426243 vs 0.7912534339, ADS 0.7674603175 vs 0.7683174603 (-0.0008571428), CPS 0.9942833862 vs 0.9976771958 (-0.0033938096). Runtime 26m16s vs 17m32s. | Per transcribed screenshots (not in repo), 104224 has total 0.7901426243, ADS 0.7674603175 and CPS 0.9942833862 in 26m16s. That is below 102411 (0.7912534339 / 0.7683174603 / 0.9976771958, 17m32s) by -0.0011108096 / -0.0008571428 / -0.0033938096. The 104224 = 6efcb018 link and the 102411 = DataV4 base (05ae7868) link are both unverified. |
| **partially** | The PANNs package changes only the Cnn14 tagger feeding the frozen presence logistic. Fake heads, presence logistic and file_ensemble CDF tables are byte-preserved, so VOICE/MUSIC fake EER cannot change by design. FILE changes only through Pv/Pm in the max and noisy_or CDF terms. The ADS loss therefore implies (inferred) a FILE EER rise of about 0.0017142856, with CDF tables never refit. | By construction the PANNs package preserves fake heads, presence_head.json and file_ensemble.json (equal weights max/noisy_or/max_nopresence, sha 3e0056d1, never refit). VOICE/MUSIC fake probabilities do not depend on PANNs, apart from possible GPU numeric drift from the added deterministic-algorithms flag. FILE changes via Pv/Pm in 2 of 3 CDF terms. The ~0.0017142856 FILE EER rise holds only if the DataV4 base (05ae7868) equals the 102411 archive, which is unverified. Presence work affects FILE as well as CPS, so its upside is not capped at +0.00023. |
| **partially** | The internal known-label selection ADS 'near 0.923' did not transfer (leaderboard ADS 0.7675). The selection role was reused for early stopping, arm choice and packaging, and the workflow ships the best trained arm even when R0 wins (panns_submission_workflow.py:155). chosen.json is not in the repo. | The code reuses the 'selection' role for early stopping (selection_presence_bce), R0/R1/R3 arm choice and packaging. Workflow line 155 falls back to the best trained arm when R0 wins. chosen.json and the per-arm evaluation JSONs are not in the repo, so the selection ADS (~0.923 per markdown only) and whether R3 was chosen or a fallback are unverifiable. The leaderboard ADS was 0.7674603175. |
| **partially** | The judge runs Python 3.11.15, NumPy 1.26.4 and PyTorch 2.7.1+cu128 on an L4. A package that passed A100 QA under NumPy 2.2.6 / PyTorch 2.9.1 crashed on the judge and burned a slot on 2026-09-26 (20:37:55). The fix was verified only on macOS CPU with 3 files. | The judge environment is L4 / Python 3.11.15 / torch 2.7.1+cu128 / numpy 1.26.4 (per contract doc). The original PANNs R3 package (5717f10b) failed at Engine init with a numpy._core.multiarray AttributeError, a post-execution error that counts against the daily limit, per a user screenshot of 2026-09-26 20:37:55 that is not in the repo. The prior A100 QA ran torch 2.9.1 (JSON). Its NumPy 2.2.6 is md-only. The fix was pre-verified only on CPU with 3 files (NumPy 1.26.4 / torch 2.7.1). The subsequent scored 104224, if it is fix1, shows it later ran on the L4 in 26m16s. |

#### Experiments and their status

- **PANNs Cnn14 minimal adaptation review with Opus (R0-R3)** — *proposed*
  - Change: Proposal: SE after block5 (132,160 params), block6 conv fine-tune (56,623,104), arms R0/R1/R2/R3, masked file-level BCE on presence logits
  - Supported conclusion: Design only. Found 0 both-absent rows in the bank and 900 MUSAN noise rows missing from the portable archive.
  - Evidence: `branch:docs/PANNS_CNN_OPUS_REVIEW_2026-09-25.md`
- **Expanded design v1.0 lock** — *proposed*
  - Change: R3 is the primary arm. Required comparison R0/R1/R2/R3/R5 (R5 = temporal-weighted pooling + block6). Conditional follow-ups R4 (block4 gate, 19 params), R6, D1/D2, D+. Deployment seed 41. Adoption requires an independent holdout with G 95% lower bound > 0.
  - Supported conclusion: No quality claim: design_v1_lock.json records implementation_started false, quality_improvement_proven false.
  - Evidence: `branch:docs/evidence/panns_cnn_review_20260925/design_v1_lock.json`
- **W0-W3 engineering pilots (CPU/MPS, 5 optimizer steps per arm)** — *completed, not evaluated*
  - Change: Implemented AdaptedCnn14 with SE and TemporalPool, a differentiable 24-feature readout, and masked loss. Ran 5 steps for R1/R2/R3/R5 on a 32-file train-only panel (105 segments).
  - Data/split: 32 train-role files, 0.448-69.3 s
  - Metrics: initial tag max abs 0.0 on CPU; presence max error 1.119346799971499e-08; CPU/MPS gradient rel L2 max 1.3466006794746556e-06; R3 MPS mb4 0.303 s/step, 4.10 GB peak
  - Supported conclusion: Wiring, gradient flow, frozen-state invariance and resume are correct. Says nothing about quality.
  - Evidence: `branch:docs/evidence/panns_cnn_execution_20260925/engineering_validation.json`
- **Noise recovery, D0 admission, longest-file backward** — *completed, not evaluated*
  - Change: Regenerated 900 MUSAN both-absent rows with new derived IDs; D0 = 42,939 files, of which 28,315 are train (527 both-absent). Tested backward on the 69.3125 s file under MPS.
  - Data/split: presence_by_role: train 28,315; selection 2,435 (1633 V-only, 410 both, 288 M-only, 55 V-unknown, 49 neither); development 10,118; calibration 2,071
  - Metrics: interrupted vs continuous weights max abs 0.0; MPS peak 4.7375 GiB
  - Supported conclusion: Data coverage now includes the both-absent class, but only from one source (MUSAN).
  - Evidence: `branch:docs/evidence/panns_cnn_execution_20260925/review_fix_validation.json`
- **MPS Demucs stem probe** — *evaluated, valid protocol*
  - Change: Extract Demucs stems on MPS instead of CPU
  - Data/split: 43 train files (engineering panel)
  - Metrics: presence max abs 3.755445717534833e-05 vs 1e-5 tolerance; 1 of 43 files failed; 5.92x speedup
  - Supported conclusion: Rejected on the numerical gate. The CPU stem cache is kept.
  - Evidence: `branch:docs/evidence/panns_cnn_execution_20260925/review_fix_validation.json`
- **Historical-cache five-output head comparison (CPU notebook)** — *evaluated, invalid protocol*
  - Change: Independent five heads, shared five heads, and joint nine-state heads, each against the preserved heads; seeds 41/42/43, 20 epochs
  - Data/split: 42,039 historical NPZ rows under current roles. Development role was historically exposed; weighted diagnostic, not official.
  - Metrics: dev diagnostic ADS: preserved 0.892041 (Music EER 0.136377); FILE-combiner-only 0.891937; independent 0.855586 (0.226796); shared 0.857133 (0.224671); joint 0.850080 (0.245883)
  - Supported conclusion: None adopted. New heads made development music EER worse.
  - Confounders: Pooling, initialisation and architecture differ from the preserved heads; the real-voice/fake-music state has 3 train rows vs 113 dev rows
  - Evidence: `branch:docs/evidence/panns_notebook_20260925/verification.md`
- **A100 CUDA preflight parity** — *partial*
  - Change: Compare CUDA Cnn14 tags and presence against CPU gold on 43 rows
  - Metrics: Before the fix: 32 of 43 rows extracted, tag max diff 1.806020736694336e-5 > 1e-5, presence 2.123993032288052e-6. After the fix: with the same CUDA stems, presence max 9.665041580259048e-08 and tag 7.748603820800781e-07.
  - Supported conclusion: The CPU/CUDA mismatch came from Mac vs Linux 32 kHz resampling (6.7e-8). Fixed by running parity on the same runtime with shared frozen stems (commits c7bd38d, 7ae7101).
  - Evidence: `branch:docs/superpowers/specs/2026-09-25-panns-cnn-expanded-design.md`
- **Locked 12-condition v1 comparison (R1/R2/R3/R5 x 41/42/43)** — *partial*
  - Change: Full training with early stopping (max 200 epochs, patience 15, min_delta 1e-4, selection presence BCE); arms run in order R1, R2, R3, R5
  - Supported conclusion: Apparently only R1 seed 41 started: the submission trainer resumes an R1 'parent' checkpoint (panns_submission_train.py:110) and panns_cloud_prepare.py:73 fixes the arm order. No results in repo. R2, R5 and seeds 42/43 have no evidence of training.
  - Evidence: `branch:repository/panns_cloud_prepare.py`
- **12-hour A100 submission workflow (R1 continued, R3 fresh, seed 41)** — *evaluated, invalid protocol*
  - Change: Train R1/R3 under absolute deadlines. Evaluate R0/R1/R3 on selection using cached frozen fake scores. Choose by selection ADS subject to ADS>=R0 and CPS>=R0; if R0 wins, package the best trained arm anyway. Package on top of the DataV4 reference.
  - Data/split: train role 28,315 files (steps/epoch ceil(28315/4)=7,079); selection role 2,435 files used for early stopping, arm choice and packaging
  - Metrics: Internal selection ADS 'near 0.923' (markdown only; chosen.json, evaluation_R*_41.json, best_epoch and stop_reason not in repo)
  - Supported conclusion: Produced the R3 package. The internal metrics did not transfer to the leaderboard.
  - Confounders: Selection reused for stopping and choice. Rows historically exposed to the preserved model. stop_reason could have been time_budget.
  - Evidence: `branch:repository/panns_submission_workflow.py`
- **NumPy 1.26 checkpoint-loader fix** — *evaluated, valid protocol*
  - Change: safe_globals allowlist switched to np.core.multiarray.scalar; weights and algorithm unchanged; only 2 of 58 zip members changed
  - Data/split: 3 hash-verified audio files, macOS CPU, NumPy 1.26.4 / PyTorch 2.7.1
  - Metrics: max abs prob diff vs NumPy 2.4.6 run 1.1102230246251565e-16; 46.94 s; A100-vs-CPU outputs differ up to 0.001394
  - Supported conclusion: The crash is fixed. L4 runtime, memory and 1,200-file coverage were not validated.
  - Evidence: `branch:docs/evidence/panns_submission_numpy126_20260926/verification.json`
- **Leaderboard test of the PANNs R3 package (104224)** — *evaluated, valid protocol*
  - Change: Swap the Cnn14 tagger for R3 inside the DataV4 package. Presence logistic, CDF FILE ensemble and fake heads unchanged. Deterministic algorithms turned on.
  - Data/split: Official eval, 1,200 files, Public = 100%
  - Metrics: total 0.7901426243 vs 0.7912534339; ADS 0.7674603175 vs 0.7683174603 (-0.0008571428); CPS 0.9942833862 vs 0.9976771958 (-0.0033938096); runtime 26m16s vs 17m32s
  - Supported conclusion: Regression on both components. Inferred: presence AUC sum fell 0.0067876192; with fake outputs frozen, the ADS drop means FILE EER rose about 0.0017142856.
  - Confounders: Package-to-submission link inferred. 102411 not verified to be the DataV4 archive. Decoder switched to SoundFile for normalized WAV. CUDA determinism flag.
  - Evidence: `branch:docs/evidence/panns_leaderboard_20260926/result.json`
- **Architecture review, step 2: music residual on MERT windows, k=1 vs k=3** — *evaluated, invalid protocol*
  - Change: Residual correction over the frozen WindowMIL head (80,705 / 105,281 params)
  - Data/split: 5,952 cached rows (train 5,196 / selection 421 / calibration 335), all previously in the preserved model's train/val
  - Metrics: calibration Music EER: baseline 9.31%, k1 13.96%, k3 13.96%; selection 11.44% vs 5.72%; k3-k1 bootstrap 95% CI -4.75 to +11.52 pp
  - Supported conclusion: Not adopted. No evidence the temporal kernel helps.
  - Evidence: `branch:docs/LOCAL_EXPERIMENTS_2026-09-25.md`
- **Architecture review, step 3: voice head retrain, reweight, residual MLP** — *evaluated, invalid protocol*
  - Change: Linear retrain (2,562 params), reweighted linear, residual MLP (166,657)
  - Data/split: same exposed 5,952-row cache
  - Metrics: calibration Voice EER: baseline 2.36%, linear 4.13%, reweight 5.52%, MLP 4.13%
  - Supported conclusion: All worse than the baseline. Not adopted.
  - Evidence: `branch:docs/LOCAL_EXPERIMENTS_2026-09-25.md`
- **Monotone voice-logit calibration fitted on selection** — *evaluated, invalid protocol*
  - Change: Post-hoc monotone calibration of the voice logit
  - Metrics: calibration FILE EER 5.60% -> 4.48%, Voice EER unchanged
  - Supported conclusion: Not deployed: fitted on exposed data, and it changes calibration, not ranking.
  - Evidence: `branch:docs/LOCAL_EXPERIMENTS_2026-09-25.md`
- **PANNs arms R2 (SE only) and R5 (temporal pool + block6)** — *implemented, not run*
  - Change: Implemented in panns_adaptation.py (ChannelSE, TemporalPool) and exercised in pilots
  - Supported conclusion: No full training or selection results. The submission path admits only R1/R3 (panns_submission.py:22).
  - Evidence: `branch:repository/panns_adaptation.py`
- **Promising but never run (both reviews)** — *proposed*
  - Change: R4 block4 gate; R6 module combo; D1/D2 block5+6; D+ (1,477 extra rows: LibriSpeech 449, MLAAD 146, MusicArena 882); CDF/FILE-ensemble refit on the calibration role; raw+stem and multi-layer MERT features; domain calibration on unexposed data; train-parent synthetic mixing; independent holdout
  - Supported conclusion: None executed. Most need new feature extraction or GPU work.
  - Evidence: `branch:docs/ARCHITECTURE_OPUS_REVIEW_2026-09-25.md`

#### Defects

| Severity | Status | Location | Description | Affected results |
|---|---|---|---|---|
| major | open | `repository/panns_submission_workflow.py:154-160; repository/panns_submission_evaluate.py:27-34,132-140` | When choose_model returns R0 (no trained arm beats R0 on both selection ADS and CPS), selected_export is None and the workflow packages preferred_trial_export anyway. A trained arm is always shipped, so the 'R3' package name does not show that R3 passed the gate. | 104224 package choice |
| major | open | `repository/panns_submission_train.py:79-84,159-164; panns_submission_evaluate.py:54-140` | The same 2,435-row selection role drives early stopping every epoch, the R0/R1/R3 choice and the packaging decision. Those rows were also historically exposed to the preserved model. The internal ~0.923 selection ADS was treated as predictive; the leaderboard returned ADS 0.7675. | 104224 selection |
| major | open | `repository/decision_heads.py:88-99; spec line 195` | FILE_FAKE_PROB is the weighted mean of val-fitted CDF tables for max(Pv*Fv, Pm*Fm), noisy_or and max_nopresence. The tables were fitted on the old presence distribution and were not refit after PANNs shifted Pv/Pm. This is a plausible cause of the inferred FILE EER increase. | 104224 ADS (-0.0008571428) |
| major | fixed | `repository/panns_submission.py:100-101 (pre-5a7a1bf)` | The safe_globals allowlist used numpy._core.multiarray, which does not exist in the judge's NumPy 1.26.4. The first R3 submission crashed. The A100 QA ran NumPy 2.2.6 / PyTorch 2.9.1, not the judge's 1.26.4 / 2.7.1+cu128. | Failed submission 2026-09-26 20:37:55 (one daily slot lost) |
| minor | open | `repository/panns_submission.py:117-118,47` | The package turns on torch.use_deterministic_algorithms(True) and runs the Cnn14 suffix in 4-segment chunks. Runtime rose from 17m32s to 26m16s (+8m44s). The cause was never measured, and L4 runtime for 1,200 files was never validated before submitting. | 104224 runtime |
| minor | unknown | `docs/evidence/panns_leaderboard_20260926/result.json` | The link from package to submission is inferred: the screenshot shows filename 'sumbit.zip' with no server hash. The claim that 102411 is the DataV4 archive 05ae7868… is also unverified. The score delta cannot be attributed solely to the PANNs change. | 104224 vs 102411 comparison |
| minor | unknown | `not in repo (A100 run-v9 / application-v11 outputs)` | The selection artifacts (chosen.json, evaluation_R0/R1/R3_41.json, terminal.json, export selection.json with best_epoch and stop_reason, package_qa/verification.json) are missing, so there is no way to audit whether R3 converged by early stopping or was cut off by the deadline, or what its selection CPS was relative to R0. | 104224 provenance |
| minor | fixed | `docs/evidence/panns_notebook_20260925/verification.md:77` | The first A100 CUDA parity check failed: tag diff 1.806e-5 > 1e-5 on 3 rows. Traced to Mac vs Linux resampling and later fixed with same-runtime parity and shared CUDA stems. | Training gate only |

#### Lessons

- The PANNs adaptation touches only the presence outputs. FILE moves only through Pv/Pm in 2 of 3 CDF terms, and Voice/Music fake EER is frozen by design. So the ADS upside is capped at 0.5*FILE-EER effects, and CPS was already near its ceiling at 0.9977 (max remaining +0.00023 total).
- A known-label selection ADS of about 0.92 on in-distribution, historically exposed roles was about 0.155 above the leaderboard ADS of 0.7675. Internal selection numbers are not leaderboard predictors.
- Reusing one panel for early stopping, arm choice and packaging, plus a fallback that ships a trained arm when the baseline wins, bakes in selection optimism.
- Changing an upstream score while freezing CDF tables fitted on the old score distribution can hurt FILE EER even when presence BCE improves.
- Every package must be run under the judge's exact stack (Python 3.11.15, NumPy 1.26.4, PyTorch 2.7.1+cu128) before submitting. QA on the A100 stack (NumPy 2.2.6 / PyTorch 2.9.1) missed a crash that cost a slot.
- Enabling deterministic algorithms and changing suffix batching added about 50% runtime (17m32s to 26m16s). Runtime headroom should be checked for any new package.
- Several days of engineering (cloud control, parity gates, resume) produced a single seed-41 arm that regressed on the leaderboard. The planned multi-arm, multi-seed comparison never ran, so there is no controlled evidence on R1 vs R3 vs R0.

#### Open opportunities (as proposed by the slice; ranked in the plan, not here)

- Keep 102411 (0.7912534339) as the reference or final submission. Nothing in this slice supports resubmitting any PANNs variant.
- If presence is revisited: refitting the FILE CDF tables and ensemble weights on the calibration role for the existing presence outputs was proposed but never run. It is cheap (CPU only), but the calibration rows are historically exposed.
- Not executed from the architecture review: raw+stem or multi-layer MERT features, D+ training with the 1,477 admitted rows (LibriSpeech 449, MLAAD 146, MusicArena 882), and domain calibration on unexposed data. All need new extraction or GPU work and are unlikely to fit 48 hours without the prohibited GPU runs.
- R2 (SE only) and R5 (temporal pool) are implemented but never trained. Low expected value, given that R3 lowered CPS on the leaderboard.
- Remove the preferred_trial_export fallback (panns_submission_workflow.py:155) before any future automated packaging, so a baseline win yields no package.
- Recover chosen.json, terminal.json and the export's selection.json from the A100 volume (/shared/panns-cnn-v1-20260925/...) to learn R3's best_epoch, stop_reason and selection CPS relative to R0. They are not in the repo.

<details><summary>Files this reader did not open (11)</summary>

- `branch:repository/panns_adaptation_cache.py`
- `branch:repository/panns_adaptation_run.py`
- `branch:repository/panns_fast_step.py (only header)`
- `branch:repository/panns_training_state.py`
- `branch:repository/panns_training_budget.py`
- `branch:repository/panns_cloud_report.py`
- `branch:repository/panns_workspace_watch.py`
- `branch:repository/panns_jupyter_transport.mjs`
- `branch:repository/DACON_PANNs_ADS_CPU_A100.ipynb`
- `remaining test_panns_*.py files`
- `NOT IN REPO: A100 run outputs chosen.json, evaluation_R0/R1/R3_41.json, train_R1_41/train_R3_41 terminal.json and export/selection.json (best_epoch, stop_reason), candidate.json packaging receipt, package_qa/verification.json, fixed.json/parity_gate.json, the leaderboard screenshot, the submission ZIPs`

</details>

### 3.6 Core pipeline code diff audit main vs branch

I audited the shared training, feature, eval, packaging and decision code on main (b6dfcb4), on the merge-base c157c2f and on cpu/lb-improvement-20260924 (1c08b1b). I read the diffs and the current files, and ran git merge-tree. **Package identity.** The DataV4_DurationBalanced package has runtime hash f0b14ebe, which matches c157c2f's full_bank_submission.py with the candidate name substituted. That runtime uses bf16 heads and softmax-max voice pooling. decision_heads.py (f807e7a9) is identical on c157c2f, main and branch. **Train/inference consistency holds for the LB-scored package.** - Voice max over odds followed by sigmoid equals the runtime softmax-max, because a 2-class softmax is a monotone sigmoid of the odds. - Music log-mean-exp (LME) is the same in training, evaluation and runtime. - The peak cap at 0.6 applies to every data v4 row, because data v4 banks have no input_kind column. - The silence gate and presence features are computed the same way in training and at runtime. - On branch, pool_voice is shared by training, evaluation and runtime for max, lme and topk. **Defects found.** - Main's Echoes parser collapses 12 generators into Echoes/TTA and Echoes/ATA. As a result, 594 of 4,488 ingested rows were kept, and 59 of 273 reference songs span splits. The branch fixes the code but has not re-run it. - file_ensemble.py:72 applies the FILE mask to all five reporting outputs, not just FILE. - decision_heads.py (and full_bank_contract.py on branch) are part of the feature-cache identity. Editing them blocks head retraining on existing caches. On branch it also blocks packaging. - On branch, pair_components hands out the construction budget in alphabetical role order, so train rows get what is left. This is implemented but not run. - Main's fusion time budget makes each file's output depend on when it is processed. **Merge.** A merge of branch into main has 3 textual conflicts and several semantic breaks. After a merge, main's scripts could no longer train on, score or package the DataV4, P1Refit and S2a heads, and the CPU/MPS training path would be removed. **E08.** The E08 leaderboard drop is confounded: 1 seed, a partial cache, the legacy decision layer, and a music head retrained on 373 rows. It does not test top-k pooling.

#### Verified claims

| Verdict | Claim | Corrected statement / evidence |
|---|---|---|
| **partially** | The LB-best package (102411, inferred DataV4_DurationBalanced) runs c157c2f code. Its runtime full_bank_submission.py hash (f0b14ebe) matches c157c2f with the candidate substituted, and its decision_heads.py (f807e7a9) is unchanged on main and branch. It uses bf16 heads, voice softmax-max and the equal-weight CDF FILE ensemble. The branch packager cannot repackage these heads (no feature_contract in run_identity), so any rebuild must use main's packager. | All runtime modules in the DataV4 package are byte-identical to c157c2f; full_bank_submission.py is c157c2f's with the candidate substituted (f0b14ebe recomputed). decision_heads.py (f807e7a9) is unchanged at c157c2f, at main HEAD and on the branch. That runtime uses bf16 heads and voice softmax-max. 'Equal-weight' FILE fusion is asserted only by a hard-coded packager string; the fitted file_ensemble.json is not in the repo. 102411 = DataV4 is inferred, not proven. The branch packager rejects run_identity files without feature_contract, and no main trainer writes one, so a rebuild needs main's packager. Main HEAD's packager would ship the b6dfcb4 fp32 runtime (2b0a52d4). An exact bf16 reproduction must package c157c2f's full_bank_submission.py, with --output set to a new path. |
| **partially** | Merging cpu/lb-improvement-20260924 into main produces 3 textual conflicts (full_bank_submission.py, full_bank_train.py, package_datav4_submission.py) and semantic breaks. After the merge, score_rows would reject heads without feature_contract, every main cache would fail the implementation-hash check, and the CPU/MPS training path would be gone. Main's scripts could then no longer train, score, refit or package the DataV4/P1/S2a heads. | The merge yields exactly 3 textual conflicts (full_bank_train.py 8 hunks, package_datav4_submission.py 4, full_bank_submission.py 1). Whatever the resolution, these hold: branch score_rows rejects the DataV4/P1/S2a heads because they lack feature_contract; the merged full_bank_features.py hash breaks training on every existing H100 and local cache; and the merged runtime needs full_bank_contract.py, which main's packager does not ship. The CPU/MPS trainer path is lost only if the conflicts are resolved toward the branch, and even a manual resolution then fails at the post-train evaluate step. Cherry-pick instead of merging. |
| **confirmed** | Editing decision_heads.py blocks head retraining on all existing feature caches: full_bank_train.py:50-52 checks the implementation hashes recorded at full_bank_features.py:117-119. On main it does not block FILE-ensemble refitting or packaging, because score_rows skips the check. | Confirmed. Also: an edit blocks cache resume or extension, not only retraining. Editing the presence-feature functions in decision_heads.py would silently put cached features out of step with runtime-computed ones. New candidates should go in files outside the implementation tuple (for example full_bank_evaluate.py, full_bank_submission.py or a new module). |
| **partially** | E08's LB ADS of 0.7064761905 is not evidence against top-k voice pooling. It changed 1 seed, partial data, no duration balance, a music head trained on 373 rows, and the legacy decision layer all at once. On the calibration split, most of the loss comes from the music head: Music EER 0.18614719 vs 0.09307359 (n=43), and File EER 0.07276 once the DataV4 music head is swapped in. | E08's LB drop (user-reported ADS 0.7064761905) is confounded. It changed 1 seed, partial 5,196-row train data, no duration balance, a from-scratch music head and the legacy presence/max FILE layer together, so it cannot be charged to top-k. On the proxy splits the music head accounts for most of the gap. Calibration: Music EER 0.18614719 vs 0.09307359 (n=43), File EER 0.07276119 with the DataV4 music head (n=335), about 70% of the proxy-ADS gap. These rows were DataV4's own val rows, used for its checkpoint and CDF fitting, so the comparison is biased toward DataV4. Top-k voice pooling is not supported. Its only favourable result is a single-seed E00-vs-E08 gain on the early-stopping selection set. An inference-time top-30% swap on the DataV4 heads worsened calibration Voice EER from 0.023566 to 0.035504 (n=308) and was not adopted. |

#### Experiments and their status

- **DataV4_DurationBalanced package (LB 102411)** — *evaluated, valid protocol*
  - Change: data v4 bank, --duration-balance, 3 seeds, voice max-MIL, music LME WindowMIL, presence logistic heads (train split), equal-weight CDF FILE ensemble fitted on val
  - Data/split: LB = 100% of 1,200 eval files. Local: val 4,931 used for both checkpoint selection and FILE-table fit; held-out test 6,450
  - Metrics: LB ADS 0.7683174603, CPS 0.9976771958. Local held-out test (DB heads, n=6,450): File EER 0.1094, Voice 0.11071, Music 0.2, ADS 0.86316; val (4,931) ADS 0.91849
  - Supported conclusion: Current best LB. Local held-out proxy ADS overestimates LB ADS by about 0.095
  - Confounders: val reused for epoch selection and FILE tables. Equal weights were kept (grid gain 0.0008 < 0.002), so the fit has few degrees of freedom
  - Evidence: `repository/results/fusion_20260927/db_all_w1/fusion_report.json`
- **E00 max vs E08 top-30% voice pooling (partial A100)** — *evaluated, invalid protocol*
  - Change: voice_pooling topk 0.3 vs max; seed 41; partial cache of 5,952 files (train 5,196)
  - Data/split: selection 421 files (voice 401, music 35). The same selection set was used for epoch selection in both arms
  - Metrics: Voice EER 0.0627014702 -> 0.0501611762 (n=401); File EER 0.0726953842 -> 0.0627458694 (n=421); Music 0.1715686275 in both (n=35)
  - Supported conclusion: Weak same-data signal that top-30% pooling beats max for voice. It is not independent evidence
  - Confounders: selection reused, partial extraction-order data, 1 seed, small n
  - Evidence: `branch:docs/A100_PARTIAL_TRAINING_2026-09-24.md`
- **E08 LB submission** — *evaluated, invalid protocol*
  - Change: E08 heads + --legacy-decision-layer (PANNs max presence, max FILE)
  - Data/split: LB 1,200 files
  - Metrics: ADS 0.7064761905 (user-reported) vs 0.7683174603
  - Supported conclusion: The package regressed on LB. Top-k pooling is not isolated and is not refuted
  - Confounders: Several changes at once: 1 seed instead of 3, no duration balance, partial data, music head retrained on 373 rows, legacy decision layer
  - Evidence: `branch:docs/E08_REGRESSION_2026-09-24.md`
- **E08 regression CPU diagnostic** — *evaluated, invalid protocol*
  - Change: head swaps between DataV4 and E08, and legacy vs ensemble decision layer
  - Data/split: selection 421 / calibration 335 (voice 401/308, music 35/43). Overlap with DataV4 training and FILE-fit data not excluded
  - Metrics: Calibration File EER: DataV4 original 0.055970, DataV4+legacy 0.061567, E08 0.089552, E08+old music 0.072761. Music: E08 0.186147 vs DataV4 0.093074. Selection File EER: DataV4 original 0.05432074 vs DataV4+legacy 0.04437123 (legacy better)
  - Supported conclusion: Most of E08's drop comes from its music head. Whether the CDF ensemble beats the legacy max decision layer is unresolved: it wins on calibration and loses on selection
  - Confounders: possible overlap with DataV4 training data; tiny music n
  - Evidence: `branch:docs/evidence/lb_e08_regression_20260924/comparison.json`
- **PANNs R3 suffix adaptation (LB 104224)** — *evaluated, valid protocol*
  - Change: Cnn14 block6+SE trained through the frozen presence logistic heads; DataV4 heads, runtime and file_ensemble.json kept unchanged
  - Data/split: LB 1,200 files
  - Metrics: ADS 0.7674603175 (-0.0008571428), CPS 0.9942833862 (-0.0033938096), total 0.7901426243
  - Supported conclusion: PANNs adaptation did not improve LB; presence AUC got worse
  - Confounders: FILE CDF tables were not refit to the new Pv/Pm; torch.use_deterministic_algorithms(True); runtime 26m16s vs 17m32s
  - Evidence: `branch:docs/evidence/panns_leaderboard_20260926/result.json`
- **A1 bf16 vs fp32 head inference** — *evaluated, valid protocol*
  - Change: main b6dfcb4 removed bf16 autocast from the runtime heads
  - Data/split: P1 heads, 4,931 val rows
  - Metrics: max |d| voice 0.0014, music 0.005; Spearman 1.0; val EER identical (voice 0.05017, music 0.11074)
  - Supported conclusion: The bf16 (LB package) vs fp32 (main runtime) difference is negligible
  - Evidence: `repository/results/audit_20260927/EVIDENCE.md`
- **Branch voice local loss / LME pooling (E02, E09)** — *implemented, not run*
  - Change: voice_local_loss_weight 0.25 (E02); voice_pooling lme (E09)
  - Data/split: configs only
  - Metrics: none in repo
  - Supported conclusion: none
  - Evidence: `branch:repository/configs/lb_improvement/E02_voice_local.json`
- **Echoes parser fix + role-aware construction (branch)** — *implemented, not run*
  - Change: build_manifest.py ing_echoes <mode>/<gen> parsing, echoes_reference group key, per-role pairing
  - Data/split: roles-v3 assigned; construct and GPU feature extraction not executed
  - Metrics: none
  - Supported conclusion: none. Estimated about 3,670 Echoes rows would survive the per-generator caps (vs 594)
  - Evidence: `branch:docs/CPU_PREPARATION_2026-09-24.md`

#### Defects

| Severity | Status | Location | Description | Affected results |
|---|---|---|---|---|
| major | open | `main repository/data_v4/build_manifest.py:135-143 (self-test :852-854)` | ing_echoes assumes the layout <gen>/<mode>, but the real layout is Echoes/<mode>/<gen>/file, so generator becomes 'Echoes/TTA' or 'Echoes/ATA'. The group key echoes_item:{mode}:{stem} never joins the same reference song across generators. The self-test enshrines the wrong layout | DataV4 heads (LB 102411, 104224): Echoes volume, cap and weighting, plus minor val/test leakage |
| minor | open | `main repository/data_v4/file_ensemble.py:72` | Rows are filtered on FILE_FAKE_mask=='1' before scoring, so rows with an unknown FILE label drop out of all five val/test outputs and official metrics, not only the FILE fit | DataV4 local test/val reports (test_one_shot n, *_official_*); not the fitted tables |
| major | open | `main repository/full_bank_features.py:117-119, local_features.py:50-53, full_bank_train.py:50-52; branch full_bank_features.py:180-181, package_datav4_submission.py:112-120` | The feature-cache identity includes sha256 of decision_heads.py (and, on branch, full_bank_contract.py). read_items rejects any cache whose recorded hash differs from the current file. Editing the decision layer or pooling config therefore blocks head retraining on every existing cache. The branch packager also refuses runtime decision_heads.py or full_bank_contract.py that differ from the cache producer | Any plan that edits decision_heads.py and also retrains heads, or packages on branch |
| blocker | open | `merge origin/main <- origin/cpu/lb-improvement-20260924: full_bank_submission.py:118-125, full_bank_train.py:45-50/186-232/262-272, package_datav4_submission.py:33-51/84-107/146-171/204-227 (merge-tree coordinates)` | 3 textual conflicts. Semantic breaks after resolution: (a) branch score_rows -> validate_feature_contract (branch full_bank_contract.py:65-68) raises for run_identity without feature_contract, which breaks main's fusion_fit.py:80, dev_report.py:47, compare_runs.py:86, file_ensemble.py and train_backbone_heads.py on DataV4/P1/S2a heads. (b) The changed full_bank_features.py hash invalidates every main cache (H100 DataV4 and MPS S2) for training. (c) branch check_gpu (full_bank_features.py:33-44) removes main's CPU/MPS training path. (d) The branch packager requires feature_contract and has no FUSION path | Any merged codebase used to retrain, refit or repackage DataV4/P1Refit/S2a heads |
| major | open | `branch repository/data_v4/build_manifest.py:464-470 (pair_components)` | With several roles, the construction budget n goes to roles in sorted order (calibration, development, sealed, selection, train) with n - len(pairs). Earlier roles can use the whole budget and leave train with no constructed mixes or songs | future data v4 constructions only (implemented_not_run) |
| minor | open | `main repository/full_bank_features.py:173-176 vs full_bank_submission.py:83-89 (c157c2f :77-79)` | Training keeps the original gain for input_kind legacy*/constructed* rows, but runtime always caps the peak at 0.6. That is a train/runtime normalization mismatch for legacy crops with peak > 0.6 | older balanced7144/Existing9504-lineage heads only; not DataV4 LB packages |
| minor | open | `main package_datav4_submission.py:43-45; branch :60-64; branch panns_submission.py:110-111` | No hash binding between file_ensemble.json and the checkpoints and presence outputs it was fitted on; only fitted_on is checked. Stale CDF tables can be packaged silently, which is what happened in the PANNs package (Pv/Pm changed, FILE tables not refit) | LB 104224; any future package that reuses file_ensemble.json |
| minor | open | `main repository/full_bank_submission.py:127-133,161` | The fusion time budget drops AntiDeepfake/SONICS members for files processed after 48 min. A file's VOICE/MUSIC score then depends on processing order and time (per-file independence, rule 2-4), and two scoring functions get mixed in one EER ranking | none shipped |
| minor | open | `main full_bank_train.py:63-66,129; branch :75-78` | Voice-route rows whose separated stem is silent stay in the optimizer (silence embeddings, possibly with label 1), while predict, score_rows and runtime force Fv=0. SILENCE_RMS=1e-5 (baseline_original/script.py:42) almost never fires, so VOICE_FAKE on voiceless files comes from the HTDemucs residual | all heads; magnitude unknown |
| minor | open | `main repository/data_v4/file_ensemble.py:82` | The prev-candidate guard uses all(prev is None) and 'prev' in weights. With partial test prev coverage, candidate sets are mixed silently; a prev weight of 0 exits spuriously | none shipped (no prev in packages) |
| minor | open | `main repository/data_v4/export_bank.py:29-34` | route() drops rows with no known component fake label (for example VP=0/MP=0 noise, FILE-only rows), so the FILE ensemble and head evaluation never see 'none'-type files | FILE-table coverage for DataV4 packages |

#### Lessons

- The LB-best package runtime is byte-identical to c157c2f full_bank_submission.py (hash f0b14ebe with the candidate substituted). decision_heads.py f807e7a9 is the same on c157c2f, main and branch, so decision-layer code has not drifted.
- Voice pooling is consistent between training and runtime. Training uses max over the odds (logit difference) followed by sigmoid; runtime uses max of softmax[:, fake], which is identical for 2 classes. Music LME is also identical. On branch, pool_voice is shared by training, evaluation and runtime for max, lme and topk.
- DataV4 features and runtime both cap the peak at 0.6, because data v4 banks carry no input_kind. The gain mismatch exists only in the legacy full_bank banks.
- bf16 (LB package) vs fp32 (main runtime) head inference is negligible: max |d| 0.0014 voice, 0.005 music, identical val EER on 4,931 rows.
- The local held-out test proxy (6,450 rows) gives ADS 0.86316 for the DataV4 heads, while LB ADS is 0.7683174603. Use proxies only for relative ranking.
- E08's LB drop confounds five changes. The calibration diagnostic attributes most of it to the music head retrained on 373 rows (Music EER 0.186147 vs 0.093074, n=43).
- The legacy max vs CDF-ensemble decision layer is unresolved locally: the ensemble wins on calibration (0.055970 vs 0.061567, n=335) and loses on selection (0.054321 vs 0.044371, n=421), with possible overlap with the fitting data.
- On main, FILE-ensemble refit and packaging need no GPU and ignore implementation hashes. Retraining heads needs decision_heads.py byte-identical to the cache producer's.
- The branch packager cannot repackage DataV4, P1Refit or S2a heads: their run_identity has no feature_contract. Use main's packager for those heads.

#### Open opportunities (as proposed by the slice; ranked in the plan, not here)

- CPU-only FILE-decision tweaks on existing DataV4 caches using main's file_ensemble.py. The grid-best weights repeatedly favor noisy_or/max_nopresence (e.g. db_all_w1 grid_best {max 0, max_nopresence 3, noisy_or 7}, gain 0.0008 < min_gain 0.002). Put new candidates in a separate module so decision_heads.py stays byte-identical and caches remain valid for retraining.
- Refit file_ensemble tables on PANNs-adapted Pv/Pm before any further PANNs-based submission. It is cheap (cached features) but the gain is unverified.
- Test top-k voice pooling with the full DataV4 recipe (3 seeds, duration balance, presence + ensemble, full cache). The only evidence is partial and same-set: voice EER 0.0627 -> 0.0502 (n=401). This needs a GPU run.
- Cherry-pick the branch fixes onto main without merging: file_ensemble known_file_panel, Echoes parser, and the graph_scoring _label mask. Keep main's runtime and packager for the DataV4/P1/S2a heads.
- The Echoes re-ingest would add about 3,000 modern-generator music rows, but it needs feature extraction. It is probably not feasible before the 2026-09-29 deadline.

<details><summary>Files this reader did not open (6)</summary>

- `presence_head.json / file_ensemble.json / file_ensemble.report.json for any package (not in repo; hashes only)`
- `main repository/data_v4/fusion_fit.py, build_extra.py, score_members.py, train_backbone_heads.py, dev_report.py (only import lines grepped)`
- `main repository/antideepfake_runtime.py, eight_heads.py, full_bank_submission_smoke.py (grep only)`
- `branch panns_* modules other than panns_submission.py and panns_adaptation.py; ads_cached_experiments.py; local_head_experiments.py; legacy_cache_export.py`
- `branch data_v4/prepare_roles.py, gpu_experiments.py and other new data_v4 scripts`
- `localization_models.py beyond line 120; music_temporal.py outside the diff`

</details>

### 3.7 Official submission timeline and main late work reconciliation

I rebuilt the official ledger from both branches. There are 20 recorded submissions: 17 scored and 3 errors. The best recorded result is still 102411 (2026-09-24 02:09:50): total 0.7912534339, ADS 0.7683174603, CPS 0.9976771958, 17m32s. It is inferred to be DataV4_DurationBalanced (sha 05ae7868…), because packaging commit c157c2f landed 4.5 min before the submission. The branch cpu/lb-improvement-20260924 then used 3 slots that are not recorded on main: - E08: ADS 0.7064761905, user-reported. Its ID, total and CPS are not recorded. - The PANNs R3 upload on 09-26 20:37:55 failed with a NumPy 1.26 error. - 104224 (PANNs fix1): total 0.7901426243, ADS 0.7674603175, CPS 0.9942833862, 26m16s. Both scored attempts came in below 102411. Main has one late commit (b6dfcb4, 09-27 04:56 KST) that is not on the branch. It adds: - P1Refit (heads refit on train plus the held-out test) and S2a (P1 plus 800 AI-Hub phone utterances), both packaged but with no leaderboard record. - Zero-shot AntiDeepfake and SONICS fusion, rejected on held-out data. - A learning curve showing more groups help voice but not music. The two branches share no evidence but do not contradict each other. Both keep DataV4 as the baseline and both find that changing many things at once, or touching the presence path, hurts. Three code files were edited on both sides and would conflict on merge. By member hashes, P1 and S2a differ from DataV4 only in heads, file_ensemble and the runtime wrapper; presence_head.json and decision_heads.py are identical. CPS should therefore stay at 0.9976771958. Scores before 09-24 come from docs and from an audit ledger that quotes live_submissions.json, which is not in the repo. Some IDs and ADS values are derived by me and marked as inferred.

#### Verified claims

| Verdict | Claim | Corrected statement / evidence |
|---|---|---|
| **partially** | Best recorded official score is 102411 (2026-09-24 02:09:50): total 0.7912534339, ADS 0.7683174603, CPS 0.9976771958, 17m32s, inferred to be DataV4_DurationBalanced (sha 05ae7868…). Later scored attempts E08 (ADS 0.7064761905) and PANNs R3 fix1 104224 (total 0.7901426243, ADS 0.7674603175, CPS 0.9942833862) were both lower. | The repo records 102411 as the best official total: 0.7912534339 (ADS 0.7683174603, CPS 0.9976771958, 17m32s). The source is a transcribed user screenshot that is not in the repo, and the total is arithmetically consistent. It is bound to DataV4_DurationBalanced (05ae7868…) only by timing (packaging commit 4.5 min earlier), not by archive hash. Two later scored results are recorded, both lower: E08 (user-reported ADS 0.7064761905 only, submitted about 22:49-23:28 KST on 09-24, no ID, total or CPS) and 104224 (0.7901426243, whose link to the fix1 zip is inferred). |
| **partially** | DataV4_P1Refit and DataV4_S2a_AIHubPhone are packaged, CRC- and offline-verified (10 files, CPU), and never submitted. They differ from DataV4 only in heads, file_ensemble and the runtime wrapper. presence_head.json and decision_heads.py hashes are identical and QA presence columns match exactly, so CPS should stay 0.9976771958. P1 val ADS 0.92397 vs DB 0.91849 on 4,931 val (selection split). P1 test metrics are invalid because the heads trained on test. | P1 and S2a are CRC-checked and offline CPU-verified on 10 files, not L4-validated. The repo has no record of either being submitted. Apart from metadata, they differ from DB only in the 6 head checkpoints and file_ensemble.json. The runtime wrapper is byte-identical to DB's except for the CANDIDATE string, so it keeps DB's bf16-on-CUDA head inference. The presence assets, the code and the 10-file QA presence outputs are identical, so CPS should match 0.9976771958 if the official scorer is deterministic; this is not L4-verified. On the in-sample val split (4,931 rows), P1 ADS is 0.92397 vs DB 0.91849. P1 test metrics are invalid because P1 trained on the 6,450 test rows. DB→P1 is not strictly single-lever: data, training precision/device and the file_ensemble all change. |
| **confirmed** | Zero-shot AntiDeepfake/SONICS fusion is rejected under the pre-declared protocol. With the selected weights, all-rows held-out test ADS drops from 0.86316 to 0.85465 (6,450 test, DB heads never trained on test), and SONICS gets weight 0. | Confirmed as stated. Add: the AD-unseen selection subset actually improved slightly (test ADS 0.85739→0.85909 on 3,632 rows, with test used in selection). Rejection comes from the all-rows check, where FILE EER worsened 0.1094→0.12758 while voice EER improved 0.11071→0.1078. The 'pre-declared' ordering is asserted in fusion_runs.sh, not proven by git history. |
| **partially** | Slots are fragile. The PANNs R3 first upload (09-26 20:37:55) crashed under official NumPy 1.26.4 and used a slot; 2 of 20 earlier recorded submissions also failed on environment ABI issues. Recorded timestamps show 4 submissions on KST 09-13, so 3/day reset semantics are unverified. | The PANNs R3 first upload (displayed 09-26 20:37:55, screenshot not in repo) crashed on a numpy._core attribute under NumPy 1.26.4. It was a runtime error, so under the contract rule it was deducted. Before it, the repo records 17 submissions for 09-13..09-24 per the ledger (18 with E08), of which 2 failed on environment ABI issues. V5_B1 was an install-stage error, which the rule says is not deducted; V6 89989 failed at import and was deducted. Four scored uploads carry the displayed date 09-13, so the daily reset boundary is unverified; a 00:00 UTC reset would fit. KST 09-26 already has 2 recorded uploads. |

#### Experiments and their status

- **DataV4_DurationBalanced (LB 102411)** — *evaluated, valid protocol*
  - Change: DataV4 bank heads (train 30,658; duration balance; seeds 41/42/43), trained presence_head, val-fitted CDF file ensemble
  - Data/split: Official hidden eval, 100% of the data; internal val 4,931 / held-out test 6,450
  - Metrics: LB total 0.7912534339, ADS 0.7683174603, CPS 0.9976771958. Internal DB val ADS 0.91849 (F/V/M 0.07003/0.05096/0.121); test ADS 0.86316 (0.1094/0.11071/0.2)
  - Supported conclusion: Best recorded official score. Internal val and test ADS overstate LB by about 0.15 and 0.095.
  - Confounders: Screen-to-archive binding not hash-verified; many factors changed together vs 93093
  - Evidence: `lb_branch/repository/DACON_LB_IMPROVEMENT_AUDIT_2026-09-24.md:19; repository/results/fusion_20260927/db_all_eq/fusion_report.json`
- **E08 partial5952 top-30 voice (LB)** — *evaluated, valid protocol*
  - Change: New heads on a partial 5,952-row cache: 1 seed, no duration balance, legacy PANNs max presence plus max FILE, top-30% voice pooling
  - Data/split: Official LB. Internal: selection 421 rows (voice 401, music 35)
  - Metrics: LB ADS 0.7064761905 vs 0.7683174603 (−0.0618412698). Internal selection voice EER 0.05016 vs E00 0.06270; music 0.17157 for both
  - Supported conclusion: Large regression. Beating a partial-data control does not predict beating the submitted model.
  - Confounders: About 5 factors changed at once; E00 control was a new partial-data baseline, not the submitted model
  - Evidence: `lb_branch/docs/E08_REGRESSION_2026-09-24.md; lb_branch/docs/evidence/lb_a100_partial_train_20260924/E08_topk/results.json`
- **E08 vs DataV4 CPU cached diagnostic** — *evaluated, invalid protocol*
  - Change: Swapped music and voice heads and the decision layer between E08 and DataV4 on the same cache
  - Data/split: 756 cached rows: selection 421 and calibration 335 (voice 308, music 43). Overlap with DataV4 training not excluded
  - Metrics: Calibration F/V/M: DataV4 0.055970/0.023566/0.093074; E08 0.089552/0.061008/0.186147; E08 plus DataV4 music head 0.072761/0.061008/0.093074
  - Supported conclusion: E08 is worse on every component. The rows were exposed to DataV4, so this does not show generalization.
  - Confounders: Rows likely seen by DataV4 in training
  - Evidence: `lb_branch/docs/evidence/lb_e08_regression_20260924/comparison.json`
- **PANNs R3 Cnn14 SE+block6 adaptation (LB 104224)** — *evaluated, valid protocol*
  - Change: Fine-tuned the PANNs presence tagger's SE and block6 layers (seed 41); other heads DataV4
  - Data/split: Official LB; internal known-label selection ADS near 0.923 (underlying JSON not in repo)
  - Metrics: Total 0.7901426243 (−0.00111080948), ADS 0.7674603175 (−0.0008571428), CPS 0.9942833862 (−0.0033938096)
  - Supported conclusion: Presence-path adaptation lowered both CPS and ADS. It does not beat DataV4.
  - Confounders: Screen filename differs from the local zip; DataV4-to-102411 equivalence unverified
  - Evidence: `lb_branch/docs/evidence/panns_leaderboard_20260926/result.json`
- **DataV4_P1Refit** — *completed, not evaluated*
  - Change: DataV4 recipe refit on train 30,658 plus the held-out test 6,450 (CPU fp32, seeds 41-43); val 4,931 unchanged; file_ensemble refit
  - Data/split: val 4,931 (also used for checkpoint selection and file-ensemble fitting); test is now training data
  - Metrics: Val ADS 0.92397 vs DB 0.91849 (F 0.06554 vs 0.07003, V 0.05017 vs 0.05096, M 0.11074 vs 0.121). AD-unseen val (2,368) ADS 0.91493 vs 0.9085. Test ADS 0.94459 is INVALID because the heads trained on test
  - Supported conclusion: Packaged and QA'd (sha 1b9ba441…, 10 files CPU, independence 0.0, presence columns identical to DB). There is no LB result. The internal gain is small.
  - Confounders: Val is the selection split, so the gain is selection-biased; no independent holdout left
  - Evidence: `repository/submissions/DataV4_P1Refit/submit.json; repository/results/fusion_20260927/p1_all_w1/fusion_report.json`
- **DataV4_S2a_AIHubPhone** — *completed, not evaluated*
  - Change: P1 plus 800 AI-Hub 571 real call utterances (434 calls) as voice training data
  - Data/split: val 4,931; 400 held-out AI-Hub calls exist, but no evaluation of them is in the repo
  - Metrics: val_route_eer voice 0.050964484809966785 (identical to DB's value; P1 0.05016798882146502); music 0.1107406779382887 (= P1). QA MUSIC_FAKE_PROB identical to P1 on all 10 files
  - Supported conclusion: A voice-only change on top of P1. No val gain over P1, and main's own note calls it refuted.
  - Confounders: A script comment says 'A parts are dropped (S2a refuted)', but no supporting evidence is in the repo
  - Evidence: `repository/submissions/DataV4_S2a_AIHubPhone/submit.json; repository/results/vessl_runs_20260927/s2b_features.sh:2`
- **Zero-shot member fusion (AntiDeepfake xlsr1b/2b, df0, SONICS) on DB heads** — *evaluated, valid protocol*
  - Change: Val-CDF fusion of voice members with our heads; SONICS for music
  - Data/split: Pre-declared: select on AD-unseen rows (val 2,368 / test 3,632), then no-harm check on all rows (val 4,931 / test 6,450); DB heads never trained on test
  - Metrics: Unseen test ADS 0.85739→0.85909. All-rows test ADS 0.86316→0.85465 and val 0.91849→0.90972 with the selected weights {ours 1, xlsr1b_max 0.25, df0 0.25}. SONICS weight 0 (music-alone val EER 0.498). xlsr2b variant: unseen test 0.85898
  - Supported conclusion: Rejected: fusion fails the no-harm check on all rows.
  - Confounders: AntiDeepfake training data overlaps ASVspoof, MLAAD and FLEURS sources
  - Evidence: `repository/results/fusion_20260927/db_all_w1/fusion_report.json; db_unseen_1b/fusion_report.json; fusion_runs.sh`
- **Fusion weights transferred to P1** — *evaluated, invalid protocol*
  - Change: Applied the db_unseen_1b weights to P1 heads
  - Data/split: val 4,931 / test 6,450 (P1 trained on test)
  - Metrics: Val ADS 0.92397→0.91449 on all rows; unseen val 0.91493→0.91462; test numbers invalid
  - Supported conclusion: On val, fusion also hurts P1.
  - Evidence: `repository/results/fusion_20260927/p1_all_w1/fusion_report.json`
- **Nested-group learning curve** — *evaluated, valid protocol*
  - Change: 12.5/25/50/100% of train groups, DataV4 recipe
  - Data/split: One seed (41), CPU fp32; val for early stopping; held-out test never trained on
  - Metrics: Voice test EER 0.12901/0.12844/0.1228/0.10974 (24,623 rows at 100%). Music test EER 0.25817/0.23294/0.19821/0.20593 (9,363 rows at 100%)
  - Supported conclusion: More groups still help voice. Music has plateaued or is noisy between 50% and 100%. This weakly supports P1's extra data mainly for voice.
  - Confounders: Single seed; the number of updates changes with data size
  - Evidence: `repository/results/audit_20260927/learning_curve.json`
- **Audit checks A1-A17 (main)** — *evaluated, valid protocol*
  - Change: bf16 vs fp32 heads, MPS/CPU vs H100 parity (10 files), split integrity, one-step optimizer, metric recomputation, P1/S2a zip traceability
  - Data/split: 4,931 val (A1); 10 files (A2); 42,039-row split (A17)
  - Metrics: A1 val EER identical (voice 0.05017, music 0.11074; max |d| 0.0014/0.005). A8 |d|<1e-9. A17 zero overlap
  - Supported conclusion: No leakage or metric errors found in the P1/S2a packages.
  - Evidence: `repository/results/audit_20260927/EVIDENCE.md`
- **S2 extra-data feature extraction (main, MPS)** — *partial*
  - Change: AI-Hub, MLAAD-ko, Zeroth, Jamendo and Music Arena extra banks A/B
  - Data/split: bank_A_train 5,341 items
  - Metrics: 3,182/5,341 extracted, then paused with SIGSTOP at 03:17 KST; later script keeps only the B parts
  - Supported conclusion: No heads trained; no evidence yet.
  - Evidence: `repository/results/audit_20260927/EVIDENCE.md; results/vessl_runs_20260927/s2b_features.sh`
- **New-backbone heads (extract_backbones.py / train_backbone_heads.py)** — *implemented, not run*
  - Change: AntiDeepfake-1B window embeddings plus multi-layer MERT features for trained heads
  - Data/split: Data v4 rows excluding AI-Hub
  - Metrics: No results in repo (only run.sh)
  - Supported conclusion: None.
  - Evidence: `repository/results/vessl_runs_20260927/backbones/run.sh`
- **Existing9504_MIL package** — *completed, not evaluated*
  - Change: Previous 9,504-row MERT330 WindowMIL plus DF fc5 heads
  - Data/split: train 6,652 / val 1,426 / test 1,426 (test has 1,001 historically exposed rows)
  - Metrics: Local val file EER 7.9944%, test 7.4334%. On the v4 proxy test F/V/M 0.1735/0.1685/0.2690 vs DataV4 0.1094/0.1107/0.2001 (ledger)
  - Supported conclusion: Never submitted. Clearly worse than DataV4 on the v4 proxy, so not worth a slot.
  - Evidence: `repository/submissions/Existing9504_MIL/README.md; lb_branch/docs/evidence/lb_cpu_20260924/audit_findings_ledger.json`
- **B_full1696 package** — *evaluated, valid protocol*
  - Change: Frozen experts plus B head, noisy-OR FILE, retrained on all 1,696 originals
  - Data/split: No independent validation (val and sealed were used for training)
  - Metrics: Probable LB 91610 = 0.6838 (4 dp; binding by file mtime only)
  - Supported conclusion: Probably submitted and below M330. It is not an unsubmitted candidate.
  - Confounders: Package-to-submission binding inferred
  - Evidence: `docs/EXPERIMENTS_ALL_2026-09-17.md:503-515; lb_branch/docs/evidence/lb_cpu_20260924/audit_findings_ledger.json:1012-1014`
- **Branch local residual/linear head comparisons** — *evaluated, invalid protocol*
  - Change: Music residual k=1/3, voice linear refit and reweighting, voice residual MLP; LME and top-30 pooling; monotone calibration
  - Data/split: 5,952 cached rows, all previously in DataV4 train/val; calibration music 43, voice 308
  - Metrics: Baseline calibration music EER 9.31% and voice 2.36% were best; LME voice 3.74%, top-30% 3.55%
  - Supported conclusion: Nothing adopted; DataV4 kept.
  - Evidence: `lb_branch/docs/LOCAL_EXPERIMENTS_2026-09-25.md`
- **Balanced7144 vs Expanded (LB pair)** — *evaluated, valid protocol*
  - Change: ASVspoof-only fc5 retrain, peak cap, new data
  - Data/split: Official LB
  - Metrics: 93093 total 0.7559347884 vs 92644 0.7579036931 (ADS −0.0021904762, CPS +0.0000252381)
  - Supported conclusion: Small regression.
  - Confounders: 92644 package identity unconfirmed; several changes at once
  - Evidence: `repository/results/balanced7144_regression_20260919/analyze.py:150-157`

#### Defects

| Severity | Status | Location | Description | Affected results |
|---|---|---|---|---|
| major | fixed | `lb_branch repository/panns_submission.py:101 (restricted checkpoint loader allowlist)` | The PANNs R3 zip used numpy._core.multiarray, which does not exist in the official NumPy 1.26.4. The first upload crashed and used a slot. Earlier A100 QA ran NumPy 2.2.6, so it did not catch this. | 09-26 20:37:55 submission (error) |
| major | open | `All post-09-24 LB records (102411, E08, 104224)` | The leaderboard screen shows only 'submit.zip' or 'sumbit.zip' and no archive hash. Package identity rests on timing and conversation context, and E08's ID, total and CPS were never recorded. | Attribution of 0.7912534339 to DataV4_DurationBalanced; attribution of 104224 to PANNs fix1 |
| major | open | `repository/results/fusion_20260927/p1_all_w1/fusion_report.json and p1_unseen_w1/fusion_report.json (baseline_test, fused_test)` | The reports include test metrics for P1 heads, which were trained on those 6,450 test rows. For example, test file EER 0.05151 and 0.02729. These numbers are contaminated and must not be cited. | Any P1-vs-DB test comparison |
| minor | open | `Branch vs main: repository/full_bank_submission.py, full_bank_train.py, package_datav4_submission.py` | Both lines of work modified these files since c157c2f. The branch adds config-driven pool_voice and full_bank_contract; main adds the fusion path and removes bf16 autocast from the heads. They will conflict on merge, and rebuilding the P1/S2a zips from main HEAD would change the runtime from the QA'd, bf16-autocast version. | Reproducibility of the packaged P1/S2a runtime |
| minor | open | `docs/EXPERIMENTS_ALL_2026-09-17.md:69` | This table lists B_full1696 as '미확인' (unconfirmed). The branch audit ledger infers it was submitted as 91610 with 0.6838. The doc also omits 91610, 92294, 92644 and 93093. | Submission count and candidate lists |
| minor | open | `Main vs branch evidence stores` | Main has no record of E08, the PANNs error or 104224, and the branch has no record of P1, S2a or fusion. The primary submission logs (live_submissions.json, live_submissions_20260918.txt, DataV4_DurationBalanced/README.md, compare_report.json) and the PANNs selection JSON are not in the repo. | The ledger's pre-09-24 IDs and ADS values come from quotations in the ledger |
| minor | unknown | `repository/results/vessl_runs_20260927/s2b_features.sh:2` | The comment says 'S2a refuted', yet S2a is still packaged, and no refutation evidence is in the repo. | Whether S2a is worth a slot |
| minor | unknown | `data_v4/file_ensemble.py:72 (as cited by the branch LB audit)` | The FILE_FAKE_mask filter drops 6 HAIM rows from all outputs, so the proxy decision CSV has 6,444 rows instead of 6,450. | Proxy test denominators |

#### Lessons

- Internal ADS does not predict leaderboard ADS well. DataV4: val 0.91849 (4,931 rows) → held-out test 0.86316 (6,450) → LB 0.7683174603. PANNs: selection ADS ~0.923 → LB 0.7674603175. The only proxy-to-LB pair (Balanced7144→DataV4) transferred 0.63 of the proxy gain (n=1).
- Changing many things at once has hurt: E08 changed data, seeds, balance, decision layer and pooling together and lost 0.0618412698 ADS. Only C08 and S2 were clean single-lever LB comparisons.
- Changes to the PANNs/presence path have lowered CPS every time (AST 0.8620483598 and 0.8818198413; V5B1 ~0.9706; PANNs R3 0.9942833862 vs 0.9976771958). CPS headroom is at most +0.00023 total.
- Scoring is deterministic: 88418 and 88809 match to 10 dp. P1 and S2a keep presence_head.json and decision_heads.py byte-identical to DataV4, so their CPS should stay 0.9976771958 and any change will be ADS-only.
- Submission errors use up slots. Recorded errors: V5_B1 (torch ABI), V6 (NumPy ABI), PANNs R3 (NumPy 1.26). Test against the official NumPy 1.26.4 / torch 2.7.1 before uploading.
- Recorded usage per KST display date: 09-13: 4 (limit is 3/day, so the reset time or timezone semantics are unverified); 09-14: 2 (1 error); 09-15: 3 (1 error); 09-16: 3; 09-17: ≥1; 92294/92644/93093 fall between 09-17 and 09-19 14:11 (dates not in repo); 09-20..23: 0 (inferred); 09-24: 2; 09-25: 0 recorded; 09-26: 2 (1 error); 09-27: unknown.
- Leaderboard ADS values lie exactly on a k/126000 grid (DataV4 96808, PANNs 96700, E08 89016), so ADS differences resolve to 1/126000 ≈ 7.9e-6.
- Zero-shot AntiDeepfake and SONICS members do not help DataV4 heads on held-out rows; SONICS gets weight 0.
- More training groups still lower voice test EER (0.1228 at 50% → 0.10974 at 100%, single seed), while music does not improve (0.19821 → 0.20593).

#### Open opportunities (as proposed by the slice; ranked in the plan, not here)

- Submit DataV4_P1Refit (sha 1b9ba4411b87…): a single lever (+6,450 rows) that reuses DataV4's presence and decision code. Expected CPS is unchanged, and the val ADS gain is +0.00548 (selection-biased).
- If P1 does not lose, S2a (sha c7c8b1c83d24…) is a clean voice-only follow-up on top of P1, since its music predictions equal P1's. Its val shows no gain over P1, and main labels it 'refuted' without evidence in the repo, so it has low priority.
- Before planning slots for the final ~53 h (deadline 2026-09-29 10:00 KST), check the remaining daily slots and reset time on the account page. The 09-13 records show 4 submissions in one KST day.
- For every new submission, record the ID, displayed time, total/ADS/CPS and local zip sha. E08's ID, total and CPS are missing now.
- Before uploading, re-verify P1/S2a zips under NumPy 1.26.4. They share the DataV4 runtime lineage, which ran successfully on the server, so the risk is lower than for PANNs.
- Column-constant diagnostic submissions (E = 0.5 + (Ac−A0)/w) could decompose LB ADS into components, but they use slots and DACON discourages LB probing (talkboard 417702).
- The branch's 1,477 admitted supplemental rows (LibriSpeech 449, MusicArena 882, MLAAD 146) were never trained into any head. Main's S2-B music extra data is only partly extracted.

<details><summary>Files this reader did not open (8)</summary>

- `live_submissions.json / live_submissions_20260918.txt / live_leaderboard_20260918.json (not in repo)`
- `submissions/DataV4_DurationBalanced/README.md (not in repo)`
- `results/data_v4_20260923/.../compare_report.json and file_ensemble.report.json (not in repo)`
- `PANNs R3 selection/evaluation JSON behind 'selection ADS near 0.923' (not in repo)`
- `AI-Hub 400-call held-out evaluation for S2a (not in repo)`
- `lb_branch/docs/PANNS_A100_PREPARATION_2026-09-25.md, PANNS_CNN_OPUS_REVIEW_2026-09-25.md, ARCHITECTURE_OPUS_REVIEW_2026-09-25.md (only grepped)`
- `lb_branch/docs/A100_PARTIAL_TRAINING_2026-09-24.md and other 09-24 data-collection docs (outside slice)`
- `main repository/data_v4/fusion_fit.py, component_fusion.py, score_members.py source (not audited line by line)`

</details>

## 4. Completeness critique and resolved questions

### 4.1 Gaps found by the critic

- Branch coverage (checked with git for-each-ref and merge-base --is-ancestor): there are 5 refs. main is b6dfcb4 (2026-09-27 04:56 +0900). claude/amazing-archimedes-a92agr is also b6dfcb4, identical to main. origin/data-v4 is 435f153; it is an ancestor of main, 0 commits ahead and 4 behind. origin/pr-1 (pull/1) is 13a957e; it is an ancestor of main, 0 ahead and 12 behind. cpu/lb-improvement-20260924 is 1c08b1b, 59 ahead and 1 behind. data-v4 and pr-1 have no content of their own, so reading main's history covers them. The visible slice outputs do not show that main's history before c157c2f was read (results/ dirs dated 09-12 to 09-20, EXPERIMENTS_ALL).
- The relayed slice input is cut off partway through slice 3, verdict 2. Slices 4-7 and the 2 research-agent outputs are not visible. I cannot confirm from the input that PANNs (panns_* evidence), main b6dfcb4 or the pre-09-23 history were covered.
- Main b6dfcb4 has two CRC-verified packages that no visible slice evaluates. (1) DataV4_P1Refit, sha 1b9ba441…: train 30,658 plus the 6,450 held-out test rows used as train, val 4,931, 3 seeds, CPU fp32. (2) DataV4_S2a_AIHubPhone, sha c7c8b1c8…: P1Refit plus 800 AI-Hub 571 utterances. Neither has an LB result in the repo. Both have official_l4_validated=false and a QA of 10 files on CPU only (repository/submissions/*/submit.json, qa/verification.json). P1Refit's test metrics are evaluated-invalid-protocol because it trained on test: results/fusion_20260927/p1_all_w1 baseline_test ADS 0.94459, music EER 0.03191, n=6,450. Its only comparable evidence is val 4,931, which was used for checkpoint selection and CDF fitting in both models: ADS 0.92397 vs DataV4 (DB) 0.91849, +0.00548. By component, File 0.06554 vs 0.07003, Voice 0.05017 vs 0.05096, Music 0.11074 vs 0.121. On AD-unseen val (n=2,368): 0.91493 vs 0.9085. The gap is about the size of the pair-specific bootstrap SD of ~0.0046 quoted in slice 1.
- Main results/audit_20260927/learning_curve.json (seed 41, val = early-stopping split, test = held-out 6,450 route subset; completed-unevaluated) is not covered in the visible slices. Voice test EER by data fraction: 0.12901 at 12.5% (2,462 rows), 0.12844 at 25%, 0.1228 at 50%, 0.10974 at 100% (24,623 rows). Music test EER: 0.25817 (1,071), 0.23294, 0.19821, then 0.20593 at 100% (9,363), so it is non-monotone. At 100% the voice run stopped at best_epoch 50 = epochs_run 50, which equals the full_bank_train.py:307 --max-epochs default of 50. Val was still falling (0.05096→0.05077 at the last epoch), so the voice head is probably under-converged.
- Main results/fusion_20260927 (AntiDeepfake XLS-R and SONICS zero-shot members; evaluated, rejected) is not covered in the visible slices. On all rows, DB test ADS went 0.86316→0.85465 (db_all_w1) and →0.84153 (db_all_eq). On AD-unseen test, 0.85739→0.85909 (db_unseen_1b), but fusion_runs.sh selects with '--select valtest', so this unseen-test gain is in-sample. SONICS was never chosen (chosen_music {'ours':1} in all 9 reports). The rejection verdict in main's commit message is consistent with this.
- cpu branch 654488a repository/ads_cached_experiments.py plus repository/DACON_PANNs_ADS_CPU_A100.ipynb: implemented but not run. The notebook has 7 cells and 0 outputs, and no result JSON or doc exists.
- The cpu branch docs/superpowers/research/2026-09-25-{cpu-score-diagnostics,data-method-evidence,opus-xhigh-followup,five-output-proposal-audit}.md are not cited in the visible slices. cpu-score-diagnostics §B: substituting true presence on test (n=6,268, 2,572 components) changes File EER by only -0.00365574, 95% CI [-0.00637977,+0.00021478]. Presence temperature 0.5/2 gives -0.00046531. This bounds the presence→FILE lever at about 0.0037 File EER on that proxy, and on a different candidate (see contradictions).
- Merge divergence: both lines modify repository/full_bank_train.py, full_bank_submission.py and package_datav4_submission.py (diff stat 160+/67-). main's P1Refit and S2a were built from main code, which lacks the branch fixes. The FILE_FAKE_mask row drop is in data_v4/file_ensemble.py:71. The Echoes generator parser bug is in build_manifest.py:140-141. Main also keeps the construction-closure train distribution. The P1Refit file_ensemble.json (c44d9973…) was refit with main's code.
- Runtime: the limit is 60 min inference, 10 min install, 10 GB ZIP (repository/COMPETITION_CONTRACT.md:51). DataV4 took 17m32s and PANNs 26m16s on LB. P1Refit and S2a L4 runtimes are unmeasured, and their ZIP sizes are 6,351,609,758 and 6,351,699,756 bytes. The 'runtime fusion path (48-minute member budget)' in full_bank_submission.py on main was never LB-tested.
- Not in repo (either branch): the user's DACON submission history after 104224 (2026-09-26 21:13:14). There is therefore no evidence whether P1Refit or S2a were submitted on 09-27, or how many daily slots remain. Also missing: the DataV4 package's per-seed best epochs (model/trained/validation_metrics.json sha e264f1d1…, only hashed) and _workspace/cache_v4 (referenced by main learning_curve.py:31-32).

### 4.2 Contradictions between slices or branches

- Two different 'DataV4 test' proxies exist on the same 6,444 rows. baseline_lock.json uses the phase-C duration-balanced candidate (candidate_lock ea7f9c56…, file_ensemble 3e0056d1…, test_predictions da46ff8b…): File 0.10939954 / Voice 0.11070987 (n=5,139) / Music 0.20005498 (n=2,849). This matches main fusion_report baseline_test 0.1094/0.11071/0.2, ADS 0.86316. lb_branch docs/superpowers/research/2026-09-25-cpu-score-diagnostics.md:11-13,97-100 uses the phase-B candidate data_v4_results_20260923T1639Z (candidate_lock 02364a6a…, file_ensemble 1df117a2…, test_predictions b767d4ab…): File 0.113911 / Voice 0.122801 / Music 0.196189 on the same denominators. The doc itself says the two share only presence-head bytes. The five-output audit (6,183-row panel) also uses the phase-B scores. Any synthesized 'DataV4 proxy' number must name its candidate.
- The MLAAD rights-reversal interval is internally inconsistent in slice 2. Its summary says '39 minutes later'; its own verification computes 19m45s (6edb00e 18:29:36 → c951608 18:49:21 KST).
- Slice 2's claim 'no complete DataV4 feature cache located' (branch cache_inventory_summary.json, 09-24 17:49) conflicts with main. Main's learning_curve.py:31-32 and audit scripts read _workspace/cache_v4/cache_tv and cache_test, and main trained P1Refit/S2a on CPU from them. Slice 2's verification already downgraded the claim; the synthesizer should treat it as superseded by main.
- Slice 1 corrected 'presence is worthless' to 'presence feeds FILE fusion (0.45 weight)'. The cpu-score-diagnostics §B oracle-presence test bounds that indirect effect at -0.00365574 File EER (CI includes 0) on the phase-B candidate. The direction agrees, but the plausible gain is small (<~0.0017 total score).
- S2a receipt anomaly (main submissions/DataV4_S2a_AIHubPhone/submit.json). choice.val_route_eer voice 0.050964484809966785 exactly equals DataV4's voice_ensemble value, although the S2a voice_4x.pt hashes differ and its 10-file QA VOICE_FAKE differs (asvspoof row 0.96932 vs 0.98657). Its music val_route_eer 0.1107406779382887 and 10-file MUSIC_FAKE values (0.3301340192556381, 0.01762334785113732) exactly equal P1Refit's, despite different music_4x.pt hashes. The metadata may be copied, or it may be a coincidence; this is unverified.
- Slice 1 and slice 3 use 'the regression is not attributable to pooling' and 'the music head is the main driver' on 756 rows, all of which were in DataV4's own val split. Main's learning curve shows music held-out EER does not improve from 50% to 100% data (0.19821→0.20593, single seed). So attributing E08's music loss to 'only 373 train rows' is plausible, but more music data is not shown to help at the full-data scale.
- Competition mechanics and probing: COMPETITION_CONTRACT.md:60 permits diagnostic submissions, and the FAQ (contract line ~111) says the best score counts, with ties going to the earlier entry. Main results/audit_20260927/EVIDENCE.md records DACON ruling 417702 'LB probing discouraged'. A constant-0.5 column probe therefore carries review risk beyond the slot cost.

### 4.3 Decision-critical questions, resolved from repository evidence

#### Q1. Have DataV4_P1Refit (sha 1b9ba441…) and/or DataV4_S2a_AIHubPhone (sha c7c8b1c8…) already been submitted, and with what total/ADS/CPS? If not, is P1Refit's +0.00548 val-ADS gain over DB (0.92397 vs 0.91849, n=4,931, both selection-contaminated) enough to spend a slot?

Confidence: **medium**

1) Were they submitted? The repo can't say. Nothing in it records a leaderboard (LB) result for either package, on main or on cpu/lb-improvement-20260924.
- The last LB record is PANNs 104224 (2026-09-26 21:13:14 KST: total 0.7901426243, ADS 0.7674603175, CPS 0.9942833862), committed in 1c08b1b at 21:48:55 KST.
- The P1 and S2a receipts came later, in main b6dfcb4 at 2026-09-27 04:56 KST. Their submit.json files have no LB fields and say "official_l4_validated": false.
- The user's DACON submission history after 104224 is needed. That history also gives the remaining daily slots.
- The only 9/26 KST submissions on record are the failed PANNs R3 (20:37:55, NumPy error) and 104224. If nothing has been used on 9/27 KST yet, at most 9 slots remain (3 on 9/27, 3 on 9/28, 3 on 9/29 before 10:00), assuming the quota resets at midnight KST. Not verifiable from the repo.

2) Status of each package:
- P1Refit: packaged and hash-verified, not yet scored on LB. Its val metrics exist but were used for checkpoint selection and file-ensemble fitting, so they are biased upward. Its test-split metrics (ADS 0.94459) are meaningless because it trained on the test rows.
- S2a: packaged and hash-verified, not yet scored on LB. It has val metrics only. Its evaluation on held-out AI-Hub phone calls (dev_report.py output) is not in the repo.

3) Is P1 worth a slot? Yes, one slot, before S2a.
- **Only the heads differ from DB.** The packaged full_bank_submission.py in DB, P1 and S2a is the c157c2f file with only the CANDIDATE string swapped (I recomputed all three hashes and they match). decision_heads.py (f807e7a9), presence_head.json (4c54797f), script.py and the base archive are the same in all three. P1 changes only the 6 head checkpoints, file_ensemble.json and the lock/identity/metrics/manifest files.
- **CPS should not move.** The presence columns are byte-identical across DB, P1 and S2a on the 10-file QA. So if 102411 really is the DB package, P1's CPS should equal 0.9976771958 and its total change is 0.9 × its ADS change. To beat 102411, P1 needs LB ADS above 0.7683174603. That 102411 = DB (sha 05ae7868) is still unconfirmed.
- **The val gain is paired on the same rows.** Recomputed from the fusion reports' baseline_val (raw head scores through each package's own file_ensemble), P1 minus DB is +0.00548 ADS on n=4,931:
  - File EER −0.00449, worth +0.00224 ADS
  - Voice EER −0.00079, worth +0.00016 ADS
  - Music EER −0.01026, worth +0.00308 ADS
- **It holds on the smaller subset too.** On the 2,368 val rows unseen by the zero-shot fusion models (the AntiDeepfake/SONICS scorers), P1 vs DB is 0.91493 vs 0.9085, +0.00643. Both models picked checkpoints on the same val rows, so the selection bias is roughly shared.
- **Rough uncertainty.** My own binomial approximation (per-file val predictions are not in the repo): paired SE of the ADS difference is about 0.0015–0.0025, so the gain is roughly 2–3.5 SE.
- **Small caveat on the DB baseline.** DB's music EER is 0.11980 at route level in its submit.json but 0.121 file-level in the fusion recompute; P1 shows 0.11074 in both. So the recompute may understate DB by about 0.0004 ADS, making the true gap about +0.005.
- **Val gains shrink on LB.** Val error (1−ADS) is 0.0815 against 0.2317 on LB. The only measured proxy-to-LB pair (B7144→DataV4, measured on the test split) kept about 63% of the gain (proxy +0.06107, LB +0.03833; n=1). And 104224 shows a val-selected change can come out slightly negative on LB (ADS −0.00086).
- **Rough expectation, not a forecast.** At 63% transfer, P1 would gain about +0.0035 ADS, for a total of about 0.7944.
- **P1 adds exactly the domains val can't measure.** The 6,450 test rows include:
  - real: FLEURS-ko 1,500, MUSAN music 645, constructed_song 400/400 (every one of them), constructed_enhanced 276/300
  - fake: Lyria-3 Pro 300, fx_encoder 300, OpenVoice 600, VoxCPM 350

  These are the held-out domains that cost DB most (DB test ADS 0.86316 vs val 0.91849), and only an LB submission can show their effect.
- **Learning-curve audit (one seed each).** With 50%→100% of the training groups, voice held-out-test EER fell from 0.1228 to 0.1097, while music held-out-test EER did not improve (0.1982→0.2059).
- **Nothing is lost by trying.** Public is 100% of the eval set and Private equals the final Public, so an LB score is the exact final score. There is no public-overfitting penalty; the only cost is the slot.

4) S2a: do not submit it before P1 is scored.
- On val it adds nothing: voice route EER 0.050964, the same as DB and worse than P1's 0.050168.
- Its music output is identical to P1's: same music route EER (0.1107406779382887) and identical MUSIC_FAKE and presence columns on all 10 QA files.
- It adds 800 real-only AI-Hub phone utterances (434 calls) with no phone-channel fakes. That is the "real call channel = real" shortcut which build_extra.py:195-196 was designed to test, so it carries that risk.
- Its evaluation on the held-out phone calls is not in the repo.
- Once P1 is scored, S2a becomes a clean one-change LB test (only the voice heads differ from P1).

5) Decision rule:
- If P1's LB total is above 0.7912534339, refit is confirmed. Remaining slots can go to S2a or to more seeds or epochs.
- If P1 is at or below 102411, stop refit variants and move slots to decision-layer and other changes.

Operational note: P1's local QA took 606.9 s for 10 files, against 66.7 s for DB and 82.4 s for S2a. EVIDENCE.md:33-34 records memory pressure on the local machine at that time. Since the runtime code and head architecture match DB, judge runtime should be close to 102411's 17m32s, but this has not been checked on the L4.

Evidence: Receipts, no LB fields: repository/submissions/DataV4_P1Refit/submit.json (sha 1b9ba441…, val_route_eer voice 0.05016798882146502 music 0.1107406779382887, official_l4_validated false); repository/submissions/DataV4_S2a_AIHubPhone/submit.json (sha c7c8b1c8…, voice 0.050964484809966785 music 0.1107406779382887); DataV4_DurationBalanced/submit.json (sha 05ae7868…, voice 0.050964484809966785 music 0.11980428730055412). Last LB record: origin/cpu/lb-improvement-20260924:docs/evidence/panns_leaderboard_20260926/result.json (104224, 2026-09-26 21:13:14); failed PANNs submission at 20:37:55: docs/evidence/panns_submission_numpy126_20260926/verification.md:3. No later LB commits (branch head 1c08b1b 21:48:55 KST; main head b6dfcb4 2026-09-27 04:56 KST, commit message 'Submission receipts … (no archives…)'). Val ADS recomputed from repository/results/fusion_20260927/{db_all_w1,p1_all_w1,db_unseen_1b,p1_unseen_w1}/fusion_report.json baseline_val. Baseline definition: repository/data_v4/fusion_fit.py:115-118 plus repository/component_fusion.py:19-20 (ours-only passes through, heads' own file_ensemble.json). Runtime identity: package_datav4_submission.py:46-52 (CANDIDATE substitution); recomputed sha256 of c157c2f:repository/full_bank_submission.py with the substitution matches the f0b14ebe/458c50cb/8dbd000e package hashes; b6dfcb4 version does not. QA: submissions/*/qa/verification.json (presence columns identical in all 3; MUSIC_FAKE P1==S2a; max |P1−DB| MUSIC_FAKE 0.4822, FILE 0.0856; seconds 66.66/606.86/82.36). Audit: repository/results/audit_20260927/EVIDENCE.md:33-34 (memory pressure), :37-38 (A1 P1 val EER voice 0.05017 music 0.11074), :47-50 (A7 traceability, A8 metric recompute). Learning curve: repository/results/audit_20260927/learning_curve.json. Test-split composition and transfer ratio: origin/cpu/lb-improvement-20260924:docs/evidence/lb_cpu_20260924/audit_findings_ledger.json (DATA_DISTRIBUTION-3 ~l.264-275; ERROR_ANALYSIS-5 ~l.117-135; VALIDATION_GAP-4 ~l.735-742: proxy Δ 0.06107174 vs LB Δ 0.03832540, ratio 0.62755). S2a data and shortcut: submit.json 'heads' field; repository/data_v4/build_extra.py:195-196. 102411 = DB unconfirmed: origin/cpu/lb-improvement-20260924:docs/PANNS_CNN_EXECUTION_2026-09-25.md (W0 line). Not in repo: any LB result for P1/S2a, per-file val predictions (so paired SE is estimated, not recomputed), S2a held-out-call dev report, the user's daily slot usage.

#### Q2. Is the voice head under-converged at the 50-epoch cap, so that a CPU-only refit with a higher --max-epochs (and/or more seeds) is the cheapest real gain? Did the three DataV4/P1Refit seeds also stop at the cap?

Confidence: **medium**

Short answer: the voice head does hit the epoch cap and is still improving slowly on val (the in-distribution validation split). The remaining val headroom is small, about 0.001-0.002 EER. Nothing in the repo shows that more epochs help held-out or LB voice EER. More seeds do nothing for voice. So this is a cheap lever with low expected value, not "the cheapest real gain."

1) Seeds and the cap
- DataV4 duration-balanced (DB) run on H100 (phase_c.sh:24 --max-epochs 50 --patience 5):
  - voice_41: best epoch 50, completed 50. Stopped by the cap.
  - voice_43: best epoch 50, completed 50. Stopped by the cap.
  - voice_42: best epoch 44, stopped at 49 by patience.
  - Music heads: best epochs 9-12, far from the cap.
  - Source: branch ledger RUNTIME_BUDGET-2 (status partially_confirmed). The underlying results.json, *_history.json and audit_receipt.json are not in repo.
- P1Refit (CPU fp32): per-seed epochs are not in repo anywhere. Status unknown.
- Correction to the premise: validation_metrics.json holds official val metrics (full_bank_evaluate.py:82), not per-seed histories. Each packaged voice_{41,42,43}.pt stores the best epoch, validation_eer and validation_bce (full_bank_train.py:261-264). So torch.load(...)['epoch'] on the local submit.zip members settles it offline:
  - Epochs completed = min(best+5, 50).
  - A seed was truncated by the cap exactly when best >= 46.

2) Learning curve (evaluated-valid: 1 seed (41), CPU fp32, val used for selection, held-out test never trained on)
- At 100% data, voice ran 50/50 epochs with best epoch 50. Val EER over the last 10 epochs went 0.05156 -> 0.05077, a change of -0.00099. The smallest val step is 0.00019, about one file.
- Fitting a + b*exp(-e/tau) to epochs 5-50 (my recomputation) gives an asymptote of 0.04859 and tau of 24.2 epochs. Predicted val EER is 0.04938 at 75 epochs and 0.04887 at 100 epochs, so only 0.0014-0.0019 headroom (about 7-10 val files). The power-law fit was degenerate.
- The other fractions: 50% stopped at the cap with best 47 (stale 3); 25% and 12.5% stopped by patience at epochs 40 and 42.
- Voice val EER depends on the number of optimizer updates, not on data size. At matched update counts the full-data run matches each subset's best:

| Subset (fraction) | Updates | Subset best val | 100% run at matched updates | 100% run val |
|---|---|---|---|---|
| 12.5% | ~370 | 0.06311 | ~388 | 0.06390 |
| 25% | ~770 | 0.06132 | ~776 | 0.06052 |
| 50% | ~2115 | 0.05574 | ~2134 | 0.05574 |

- So the "voice improves with data" held-out trend (test 0.12901 / 0.12844 / 0.1228 / 0.10974 at 420 / 880 / 2250 / 4850 updates) is confounded with update count.
- Test EER was computed only at the best checkpoint (learning_curve.py:77-79). Whether held-out voice EER keeps falling after epoch 50 is not in repo.
- Music at 100% stopped by patience (best 12, 17 run). Its test EER got worse, 0.19821 -> 0.20593, so music is not cap-limited.

3) Signs that the voice head is limited by capacity or domain, not epochs
- P1 trained on the 6,450 test rows. Its voice EER on those same rows is still 0.10039, against 0.11071 for DB, which never trained on them (fusion_20260927/p1_all_w1 vs db_all_eq baseline_test). This is not a valid held-out number; it is a training-fit check, and it shows the head cannot fit those domains well. By contrast, P1's music EER on those rows fell to 0.03191.
- Val vs held-out gap: 0.05096 vs 0.11071 for DB.
- Rows whose separated voice stem is silent are forced to 0 (full_bank_train.py:129), so no amount of training can fix those errors.
- Candidate (not independently verified) test error breakdown: Lyria-3 Pro voice miss rate 0.631 (n=222); fleurs_ko real false-positive rate 0.099.

4) More seeds: no gain for voice (code-verified)
- Every seed loads the same fc5 initialisation (full_bank_train.py:220-222). The seed changes only the batch order (full_bank_batches.py:24). The model is a Linear(1280,2) probe, a convex problem.
- DB 3-seed ensemble val EER 0.050964 vs the single CPU seed at 0.05077.
- Candidate evidence (ARCHITECTURE-5, not verified): per-seed 0.0508 / 0.0510 / 0.0508, ensemble 0.0510, Spearman between seeds 0.9999-1.0.

5) Cost and practical limits
- Compute is trivial: the 100% voice learning-curve run took about 0.2 min on CPU (cumulative time 1.1 -> 1.3 min).
- An existing run cannot be resumed with a larger cap, because max_epochs is part of run_identity (full_bank_train.py:191, 199-201). It has to be a fresh retrain.
- full_bank_train retrains both routes, so a CPU refit also changes the music heads from H100 bf16 to CPU fp32. That confounds any LB comparison unless only the voice .pt files are swapped in.
- The branch trainer is GPU-only (--gpu-name A100/H100); only main b6dfcb4 has --device cpu.

6) Expected value
- If the val headroom of 0.0014-0.0019 carried over 1:1 to the LB, the direct score gain would be 0.9 × 0.2 × Δ ≈ 0.0003. Even using the confounded 2.6x val-to-test ratio seen across fractions, it is about 0.0009.
- For scale: the LB gap between DB and PANNs was 0.0011. On 1,200 eval files, a gain this size is probably about one file flip.
- An epoch-extension run is only proposed: no run with max_epochs above 50 exists on any branch.
- A cheap, decisive check that has not been run: retrain voice on the train split for 150 epochs with patience 15, logging held-out test EER every epoch.

Evidence: Checked directly (this pass): - repository/results/audit_20260927/learning_curve.json: all 8 results rows. Voice at 100%: best_epoch 50, epochs_run 50, updates 4850, val 0.05077, test 0.10974, rows 24623. Voice at 50%: best 47, run 50. Voice at 25% and 12.5%: best 35/37, run 40/42. Music at 100%: best 12, run 17, test 0.20593. I recomputed the fits and the matched-update comparison in pure Python. - repository/results/audit_20260927/learning_curve.py:59-81: test EER is computed only at best_state; selection is lexicographic (EER, BCE). - repository/full_bank_train.py (main b6dfcb4): - 191: max_epochs is part of run_identity. - 199-201: identity mismatch raises an error. - 220-222: every seed loads the same fc5 init. - 232: learning rate 1e-4 for voice. - 243-245: epoch loop and patience check. - 261-264: the .pt payload stores the epoch and validation EER/BCE. - 283-284: results.json stores epoch and epochs_completed. - 129 (predict): silent voice stem forced to 0. - 307-308: defaults are 50 epochs and patience 5. - repository/full_bank_batches.py:24: the seed only permutes batch order. - repository/full_bank_evaluate.py:82: validation_metrics.json = official_metrics output. - repository/package_datav4_submission.py:29-30, 49: the raw voice_*.pt bytes are packaged. - repository/data_v4/phase_c.sh:24: DB run used --max-epochs 50 --patience 5 --duration-balance. - DB trainer sha d7cd7401 = c157c2f version. Submissions and fusion reports: - repository/submissions/DataV4_DurationBalanced/submit.json: voice_ensemble 0.050964484809966785. - repository/submissions/DataV4_P1Refit/submit.json: voice_ensemble 0.05016798882146502. - repository/results/fusion_20260927/{db_all_eq,p1_all_w1}/fusion_report.json: test voice EER 0.11071 (DB) vs 0.10039 (P1, in-sample); test music EER 0.2 vs 0.03191. Branch evidence (origin/cpu/lb-improvement-20260924:docs/evidence/lb_cpu_20260924/audit_findings_ledger.json): - Line 283 (DATA_DISTRIBUTION-3): voice 41/43 best epoch 50 of 50; music best 9-12. - Line 1082ff (RUNTIME_BUDGET-2, partially_confirmed): 41/43 select and complete 50; 42 selects 44 and stops at 49. "Does not prove underfitting." - Line 565ff (ARCHITECTURE-5, not independently verified): per-seed EER and Spearman. - Line 794ff (VALIDATION_GAP-8): voice_41 epoch 1 0.0784 -> epoch 46 0.0510. - The branch's full_bank_train.py is GPU-only (--gpu-name). Other runs: - E00/E08 partial-cache runs (branch docs/evidence/lb_a100_partial_train_20260924/*/results.json): voice stopped by patience at 12 and 26 epochs on a 5,952-file bank. - There is no run with more than 50 epochs on main, cpu/lb-improvement, data-v4 or pr-1. Not in repo: - results/data_v4_20260923 (results.json, voice_*_history.json, voice_validation.csv), audit_receipt.json, _workspace/p1_refit_cpu, _keep_best_20260926, and every P1 per-seed epoch or history. - The LB voice-present count is unknown, so the per-file LB granularity is an estimate.

#### Q3. Which DataV4 candidate is submission 102411: the phase-C duration-balanced package (zip 05ae7868…, candidate_lock ea7f9c56…, proxy test File 0.10940) or the phase-B original (candidate_lock 02364a6a…, proxy test File 0.113911)?

Confidence: **medium**

Answer: most likely the phase-C duration-balanced package. The repo has no DACON-side record that ties submission 102411 to any archive, so this is inferred, not verified. Status of the 102411→archive binding: not in repo (it needs the user's DACON submission history or a local upload log). Status of "DataV4_DurationBalanced zip = phase-C heads": evaluated-valid, because the hash receipts are in the repo. Status of a phase-B DataV4 package: not in repo. No packager, receipt or zip for phase-B heads exists on any branch, so nothing shows one was ever built.

Why phase-C:
(1) The only DataV4 packager hard-codes the phase-C heads. Its commit message names the duration-balanced submission.
(2) The receipt of the only packaged DataV4 archive before P1 has the phase-C hashes: candidate_lock ea7f9c56, file_ensemble 3e0056d1, and the --duration-balance recipe.
(3) The lb-branch diagnostics found that the preserved package and the phase-B candidate share only the presence-head bytes. Their fake-checkpoint lists and CDFs differ.
(4) Timeline:
- phase-B results zip: 16:39Z (01:39 KST)
- phase-C train_balanced 16:44:06Z → file_ensemble 16:51:41Z; results zip 16:53Z (01:53 KST)
- packaging commit c157c2f: 02:05:19 +0900
- 102411 displayed time: 02:09:50 (timezone not verified)

This matches the PANNs analogue. Submission 104224 shows 21:13:14, which is 9m41s after the package-fix commit 5a7a1bf (21:03:33 +0900), with a 26m16s runtime and the record committed at 21:48:55. That suggests DACON's displayed time is the upload time. On that reading, 102411 was uploaded 4m31s after the phase-C package was committed.
(5) The user's own local layout names the phase-C heads as "keep best": DB=_keep_best_20260926/train_duration_balanced.

Runtime and CPS cannot tell the two apart. Both candidates share presence_head 4c54797f and the same architecture. Proxy presence AUCs are identical: 0.997946 / 0.998654.

Caveats:
- The lb-branch docs themselves mark the binding as unverified.
- If 02:09:50 were the completion time, the upload would have been about 01:52:18 KST. That is before the phase-C zip existed, but it would also need a phase-B packager that is not in the repo.

Impact on the anchor (recomputed): proxy ADS for phase-C is 0.8631417627 (File n=6,444, Voice n=5,139 known with 261 unknown, Music n=2,849; historical test split, now "development"). Proxy ADS for phase-B is 0.8596276 (from the diagnostics' 6-decimal values 0.113911 / 0.122801 / 0.196189, same denominators). The implied Balanced7144 proxy ADS is 0.8631417627 − 0.06107174 = 0.8020700227; the ledger's rounded 0.1733/0.1601/0.2641 give 0.8021. LB/proxy ratio: 0.62755 if phase-C was submitted, or 0.0383253968 / 0.0575576 = 0.66586 if phase-B. The E08 comparison.json label "DataV4_original" refers to the phase-C archive 05ae7868, not phase-B. P1Refit (1b9ba441, lock 2e9bc36e, ens c44d9973) reuses the phase-C --duration-balance recipe.

Evidence: Package identity: - repository/submissions/DataV4_DurationBalanced/submit.json:3 has sha256 05ae786864dc…; :62 has candidate_lock ea7f9c56c638…; :63 has file_ensemble 3e0056d110d8…. choice.heads reads "--duration-balance"; val_route_eer is music 0.11980428730055412 and voice 0.050964484809966785. - qa/verification.json:2 has the same zip sha. It was written by verify_datav4_submission.py:89. Packager: - git show c157c2f:repository/package_datav4_submission.py:22 sets HEADS = data_v4_phase_c_20260923T1653Z/train_duration_balanced; :23 sets PRESENCE = the phase-B presence_head.json; :24 sets OUTPUT = submissions/DataV4_DurationBalanced. Main's version has the same lines 22-24. - Commit c157c2f is dated 2026-09-24 02:05:19 +0900 with the message "Package and verify the data v4 duration-balanced submission". - git log --all over repository/package_* and repository/submissions/*: no phase-B DataV4 package on any branch (main, cpu/lb-improvement-20260924, data-v4, pr-1, claude/amazing-archimedes). Baseline lock: - lb_branch/docs/evidence/lb_cpu_20260924/baseline_lock.json has candidate_lock ea7f9c56… and file_eer 0.10939954188481675 (n=6,444). Its reference field reads "historical DataV4; not independent or official score". Diagnostics: - lb_branch/docs/superpowers/research/2026-09-25-cpu-score-diagnostics.md:12 says the preserved package and the historical candidate share only presence-head bytes, and the package CDF shifts FILE by up to 0.056572735 (val) and 0.055985468 (test). :24 gives test 6,444: FILE 0.113911, Voice 0.122801, Music 0.196189. :97-100 give phase-B hashes: candidate_lock 02364a6a…, file_ensemble 1df117a2…, test decision b767d4ab…. - lb_branch/docs/ARCHITECTURE_OPUS_REVIEW_2026-09-25.md:41 gives the phase-C file_ensemble sha 3e0056d1… (max / noisy_or / max_nopresence, no prev). Timeline: - repository/data_v4/phase_b.sh:47 and phase_c.sh:33 name the results zips with UTC timestamps (T1639Z and T1653Z). - audit_findings_ledger.json:1162 has phase_b train_heads at 16:34:17Z; the phase_c.steps line in the same file has train_balanced 16:44:06 → file_ensemble 16:51:41. - lb_branch/repository/DACON_LB_IMPROVEMENT_AUDIT_2026-09-24.md:19-20 records the screenshot (102411, 02:09:50) and states the hash binding needs separate evidence. Binding explicitly unverified: - ledger:1118 says "do not assume exact local archive binding". - ledger:742 calls the 0.62755 ratio "context-attributed". - docs/PANNS_CNN_EXECUTION_2026-09-25.md:25 says "Equivalence to the user's leaderboard-best submission remains unconfirmed". - docs/evidence/panns_leaderboard_20260926/result.json:36-37 has reference_submission_id null and reference_archive_equivalence_verified false. Timestamp analogue: - result.json has displayed_timestamp 2026-09-26 21:13:14 and duration 1576 s. - git: 5a7a1bf is dated 21:03:33 +0900 and 1c08b1b is dated 21:48:55 +0900. User layout: - repository/results/fusion_20260927/fusion_runs.sh:6 sets DB=_keep_best_20260926/train_duration_balanced. Recompute: - Python ADS formula gives phase-C 0.8631417627 and phase-B 0.8596276, implying Balanced7144 0.8020700227. Ratios: C 0.6275471568, B 0.6658618833. Not in repo: results/data_v4_20260923/*, compare_report.json, the DataV4_DurationBalanced README.md, live_submissions.json, DATA_PLAN_236749_2026-09-19.md, and the 102411 screenshot or any server-side archive hash.

## 5. External research

Egress blocked arxiv.org, isca-archive.org, ieeexplore and openreview for the research agents; GitHub code and READMEs were opened. "Verified" means the agent opened the source; unverified rows rest on search-engine snippets.

### 5.1 Singing voice, cross-condition calibration

| Source | Venue | Verified | Relevance | Finding | Transfer limits |
|---|---|---|---|---|---|
| [SingFake: Singing Voice Deepfake Detection (ICASSP 2024)](https://arxiv.org/abs/2309.07525) | ICASSP 2024 (arXiv 2309.07525) | no | direct | Taken from search-result snippets only; the paper was not opened. Four speech countermeasures (CMs) were trained on ASVspoof2019 LA: AASIST, Spectrogram-ResNet, LFCC-ResNet and wav2vec2+AASIST. On SingFake song mixtures their EER was about 50%. On Demucs-separated vocals, the spectrogram and raw-waveform systems (Spectrogram-ResNet, AASIST) reached about 38% EER, while LFCC-ResNet and wav2vec2+AASIST stayed near 50%, which the paper reads as speech features overfitting to speech. Training on SingFake, on either mixtures or separated vocals, gave substantial gains. Gaps remained for unseen singers, communication codecs (T03), and languages and musical contexts (T04). Data: 28.93 h bona fide and 29.40 h deepfake, 40 singers, 5 languages. | The egress proxy blocked arxiv.org, isca-archive.org, ieeexplore, openreview, huggingface.co, asvspoof.org, singfake.org and research.nvidia.com, so every paper-level number in this list comes from search-engine snippets. The CMs here were trained only on ASVspoof2019 speech. That is a weaker starting point than DF-Arena-1B, which is XLS-R-based and trained on multiple datasets, so the absolute ~50%/~38% figures overstate the gap for our model. The direction still holds: separated vocals beat mixtures for speech CMs, and in-domain singing training data is what closes the gap. |
| [SingFake official repo: dataset/separate.py](https://raw.githubusercontent.com/yongyizang/SingFake/main/dataset/separate.py) | GitHub (official) | yes | direct | The separation pipeline runs `demucs --two-stems=vocals -n mdx_extra` and then pyannote/segmentation VAD on the vocal stem. The resulting VAD timecodes are used to cut vocal-active clips. The T03 codec subset comes from simulate_codec.py. Model checkpoints were not released, for copyright reasons. | Shows that SingFake scores only vocal-active segments of the separated stem. Our pipeline takes the max over all non-overlapping 4.04 s vocal-stem segments (audit_findings_ledger.json:475). The separator also differs: mdx_extra here versus HTDemucs in ours. |
| [SVDD 2024: The Inaugural Singing Voice Deepfake Detection Challenge](https://arxiv.org/abs/2408.16132) | IEEE SLT 2024 | no | domain difference | Snippets only, not opened. CtrSVDD track: 47 teams submitted and 37 beat both baselines. The baselines scored 12.03% and 11.16% EER, and the top team reached 1.65% EER. WildSVDD, the in-the-wild mixture track, received no submissions. Winning systems used SSL front-ends, ensembles and adversarial training. | CtrSVDD is clean a cappella vocals from public singing corpora with 14 SVS/SVC generators. It has no accompaniment, no separation residue, and no Suno, Udio or Lyria end-to-end songs. Repo evidence: Existing9504, which trained with CtrSVDD, still had singing-subset EER 0.2559 on the proxy (audit_findings_ledger.json:228). |
| [SingGraph: Singing Voice Graph Modeling for SingFake Detection (official repo README)](https://github.com/xjchenGit/SingGraph) | Interspeech 2024 | yes | domain difference | Combines MERT (pitch and rhythm) with wav2vec2 (lyrics), plus RawBoost and beat-matching augmentation. The README reports relative EER improvements on SingFake over the previous SOTA of 13.2% for seen singers, 24.3% for unseen singers, and 37.1% for unseen singers with different codecs. The README contains no absolute per-subset EER table. | Requires training a dual-backbone graph model, which is outside the <48h frozen-head constraint. Its value here is that it supports adding music-FM features computed on the vocal stem. |
| [How Does Instrumental Music Help SingFake Detection?](https://arxiv.org/abs/2509.14675) | arXiv 2025 (ICASSP 2026 per NVIDIA page title) | no | plausible unverified | Snippets only. Instrumental accompaniment acts mainly as data augmentation, not as an intrinsic cue such as rhythm or harmony. Fine-tuning increases reliance on shallow speaker features and reduces sensitivity to content, paralinguistic and semantic information. | Not opened, and no numbers were extracted. It suggests a mixture-level shortcut is unlikely to carry real signal, which agrees with the repo's rejection of the music-to-voice max shortcut (audit_findings_ledger.json:1378). |
| [Comprehensive Layer-wise Analysis of SSL Models for Audio Deepfake Detection (official repo)](https://github.com/Yaselley/SSL_Layerwise_Deepfake) | NAACL Findings 2025 (arXiv 2502.03559) | yes | direct | Frozen Wav2Vec2, HuBERT and WavLM (small and large) with learnable per-layer weights, feeding an FFN or AASIST back-end. Datasets include LA19, LA21/DF21, ADD23, HABLA, PartialSpoof, SceneFake and CtrSVDD. Lower layers (1-6 for small models, 1-12 for large) give the most discriminative features, and reduced-layer models stay competitive at 2-3x lower cost. The README gives no EER numbers. | Uses raw SSL checkpoints. DF-Arena-1B is already anti-spoof fine-tuned, so its layer profile may differ. Using intermediate layers would mean re-extracting hidden states for the train and val cache: inference only, no backbone training. |
| [Are Music Foundation Models Better at Singing Voice Deepfake Detection? Far-Better Fuse them with Speech Foundation Models (FIONA)](https://arxiv.org/abs/2409.14131) | ICASSP 2025 (arXiv 2409.14131) | no | domain difference | Snippets only. Frozen music FMs (MERT variants, music2vec) were compared with speech FMs, including speaker-recognition models, using simple downstream heads. Speaker-recognition FM embeddings (x-vector) were the best single representation for singing voice deepfake detection (SVDD). Fusing x-vector with MERT-v1-330M gave 13.74% EER on CtrSVDD. | 13.74% EER on CtrSVDD is far worse than the fine-tuned top systems (1.65%). CtrSVDD is a cappella. The result only supports the view that pitch, timbre and speaker cues on the vocal stem complement a speech CM. |
| [Towards generalisable and calibrated audio deepfake detection with self-supervised representations](https://arxiv.org/abs/2309.05384) | Interspeech 2024 | no | direct | Snippets only. Large frozen SSL representations with a logistic-regression head of fewer than 2k parameters cut the average EER over eight deepfake datasets from 30.9% to 8.8%, and gave better-calibrated probabilities. The official code repo github.com/danoneata/aletheia was opened: it uses wav2vec2-xls-r-2b features plus LR, trained on ASVspoof19 and In-the-Wild with 1k-8k training subsets, and its README gives no numbers. | Speech only, no singing. It supports frozen features plus a tiny head, but says nothing about singing transfer. |
| [DCA-PLDA: Discriminative Condition-Aware PLDA (official repo)](https://github.com/luferrer/DCA-PLDA) | Computer Speech & Language vol. 71, 2021 (plus ICASSP 2020 and Odyssey 2020 papers) | yes | domain difference | The calibration parameters depend on an estimated condition, represented as a very low-dimensional vector derived from the embeddings, and on speech duration. The whole backend is trained discriminatively with BCE. The authors claim good out-of-the-box calibration across a very wide range of conditions when trained on diverse data. | This is speaker verification, where the problem is calibration drift across conditions under a single threshold. The same logic applies to pooled Voice EER over speech and singing. However, the repo's own oracle test (audit_findings_ledger.json:229) caps the pooled-shift gain at about 0.005 Voice EER on the proxy. |
| [WeSpeaker VoxCeleb v2 recipe README (AS-Norm + QMF quality-aware calibration)](https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/README.md) | GitHub (official), QMF added 2024.05 | yes | domain difference | ResNet34 EER on vox1-O/E/H: 0.867/1.049/1.959% baseline, 0.787/0.964/1.726% with AS-Norm, and 0.659/0.821/1.437% with AS-Norm+QMF. ECAPA-c1024: 0.856/1.072/2.059% baseline, 0.808/0.990/1.874% with AS-Norm, 0.707/0.894/1.615% with AS-Norm+QMF. The QMF uses quality side-information (duration, embedding magnitude, imposter statistics) inside a logistic calibration. That improves pooled EER because different trials get different effective shifts. | The gains come from speaker trials with a per-trial cohort. Anti-spoofing has no enrollment side, and the only condition variable available here is voice+music versus voice-only. |
| [When EER Hides Deployment Failure: Auditing Threshold Transfer and Unlabeled Score Calibration for Speech Deepfake Detectors](https://arxiv.org/abs/2606.21584) | arXiv 2026 | no | plausible unverified | Snippets only. Any strictly increasing global score transform leaves EER unchanged; this covers z-norm, temperature/shift, and embedding mean alignment under a frozen linear head. Among seven audited corrections, AS-norm with an unlabeled target cohort collapsed EER from 11.2% to 60.2%. Pseudo-label calibration that cut HTER by 38% relative on In-the-Wild degenerated to 50% HTER on DF21, where the spoof prior is 96%. | Only condition-dependent transforms can change pooled EER. Normalizing on unlabeled hidden-test statistics is risky, because the class prior of the 1,200 eval files is unknown. |
| [CompSpoof: component-level audio anti-spoofing (official repo)](https://github.com/XuepingZhang/CompSpoof) | arXiv 2509.15804, 2025 | yes | domain difference | Four classes: bona fide or spoofed speech crossed with bona fide or spoofed environmental sound. Separating the components and then running per-component anti-spoofing with joint learning raised F1 from 0.840 to 0.912 (dev) and from 0.827 to 0.908 (eval). | The second component is environmental sound, not music. It supports scoring on separated stems and the need for hybrid-labelled data, for example real vocal with AI backing, which must be VOICE_FAKE=0. |

**Actionable ideas proposed by the research agent** (screened in the plan against repository evidence):

- **[proposed; not implemented] Add AI singing examples to the voice head's training and retrain only the frozen-feature linear head fc5 (2,562 params). Add the vocal stems of fully AI-generated songs as VOICE_FAKE=1 rows. These are the legacy fake-song rows (V1M1) that currently have VOICE_FAKE=null: ex:SONICS 751, ex:suno 18, ex:udio 14. Leave FakeMusicCaps (400) and HAIM/Echoes excluded until their vocal labels are checked. Match them with equal cell weight against real singing with accompaniment and real a cappella.**
  - Mechanism: The measured failure is ranking inside singing, not only an offset. Within-singing AUC for Lyria versus real singing is 0.748-0.782 for every head. Fake singing gets only 1.5% of the fake-class weight (3.5% after duration-balance), and V1M0 fake singing has 0 rows. So the linear head has almost no singing-fake direction to learn. SingFake reports that in-domain singing training substantially fixes speech-CM failure on vocals and mixtures.
  - Cost under 48 h: CPU minutes, if the cache already holds DF-Arena features for the vocal stems of those rows. The full_bank_train.py:35-36 docstring says the cache holds both stems for mixed rows, but this was not checked for these IDs. 15 small-head trainings took about 65.1 s on CPU (docs/LOCAL_EXPERIMENTS_2026-09-25.md:29). The code change is a label override in the manifest; the mask is at repository/full_bank_train.py:30-41. A valid evaluation still needs a generator-held-out split (Lyria excluded from training) and rows the preserved model never saw. All 5,952 cached local-eval rows were previously exposed (LOCAL_EXPERIMENTS:9).
  - Evidence: docs/evidence/lb_cpu_20260924/audit_findings_ledger.json:218 (DATA_DISTRIBUTION-1, weights) and :228 (within-singing AUCs), :233 (the null-labelled legacy rows). Loss cell weighting: repository/full_bank_batches.py:28-35. SingFake (arXiv 2309.07525, unverified). SONICS says Suno/Udio songs are end-to-end generated, including vocals (search snippet only). Upper bound if Lyria were detected perfectly: proxy Voice EER 0.1107 -> 0.0763 (ledger:231); evaluated-invalid-protocol oracle.
  - Main risk: The head could relearn the old 'music context -> fake' shortcut. The previous head had real MUSAN median 0.84 and MUSAN FPR 0.668 (ledger:76, :683). That would hurt real-song and real-vocal-with-AI-backing files, which must stay VOICE_FAKE=0 (ledger:1378). SONICS/Suno vocals may not generalize to Lyria or other unseen generators in the hidden test.
- **[proposed] Add a condition-routed singing head on the same frozen 1280-d DF-Arena vocal-stem embedding. Either use two linear/LR heads soft-gated by the predicted voice+music condition (MUSIC_PRESENT_PROB, optionally a PANNs singing tag), or one LR on [e, g*e, g], where g is the gate.**
  - Mechanism: Speech and singing get separate decision directions and intercepts. A single fc5 fitted mostly on speech then no longer has to serve both. Pooled Voice EER then depends on each condition's own ranking plus alignment across conditions. Routing by audio type is also what strong all-type systems do: the AT-ADD summary names 'structured fusion or routing' (snippet, unverified).
  - Cost under 48 h: CPU-hours or less on cached embeddings. Presence AUCs are about 0.998 (CPS 0.9976771958), so the gate is reliable for music presence. It is not reliable for singing versus speech over background music.
  - Evidence: Val singing-only Voice EER 0.1884 versus speech-only 0.0343; val fake singing versus real singing AUC 0.907 (ledger:230). Proxy singing-subset EER 0.2262 versus speech 0.0689 (ledger:228). Proxy voice-with-music 20.60% (n=1,544) versus voice without music 4.43% (n=3,595) (docs/LOCAL_EXPERIMENTS_2026-09-25.md:40-41). Frozen feature plus LR with fewer than 2k params generalizes (Pascu et al., unverified). The Kheir et al. repo shows singing (CtrSVDD) is handled with frozen SSL plus a small back-end (verified).
  - Main risk: There are few fake-singing generators, so the singing head may overfit to them; the val-to-proxy drop in singing AUC (0.907 -> ~0.78) shows generator shift. Speech over background music, common in voice-phishing audio, would be routed to the singing head. Sample sizes are small: music-EER rows numbered 35 in selection and 43 in calibration (docs/evidence/local_experiments_20260925/summary.json).
- **[partial; evaluated-invalid-protocol] Condition-aware calibration of VOICE_FAKE logits. Option (a): a per-condition affine map a_c*s+b_c, fitted by log-loss on labelled held-out rows, with an interaction term rather than the additive-only term already tried. Option (b): Z-norm per condition anchored on bona fide cohorts only, using the mean/std of real-speech and real-singing scores.**
  - Mechanism: A global monotone transform cannot change EER. A condition-dependent one keeps each condition's own EER but realigns thresholds across conditions, so pooled EER can move. Option (b) needs only bona fide data, which is plentiful and does not depend on the generator; it mirrors Z-norm, where the impostor side is the bona fide class.
  - Cost under 48 h: Under 1 CPU-hour on stored scores. Existing code at repository/local_head_experiments.py:404-418 already fits LR on [odds, tag_singing_max], additive only.
  - Evidence: The gain ceiling is small. An oracle constant offset on singing rows moved proxy Voice EER from 0.1107 to at best 0.1059, and made val worse (0.0510 -> 0.0542) (ledger:229; the script scratchpad/audit/dd/sing_chk2.py is not in the repo). An offset on all V1M1 rows gave test 0.0965, but it was tuned on test. The additive PANNs-singing LR moved calibration Voice EER from 0.02357 to 0.03744 (n=308, exposed rows; singing coefficient 0.0959) (summary.json 'calibration' block; LOCAL_EXPERIMENTS:35). External support: DCA-PLDA and WeSpeaker QMF (verified), plus 'When EER Hides' (unverified).
  - Main risk: The val and proxy offsets point in opposite directions, so a shift fitted on dev may not transfer to the hidden speech/singing mix and generators. A 0.005 Voice EER gain is worth about 0.0009 total score. Normalizing on the unlabeled test cohort can collapse EER (AS-norm 11.2% -> 60.2% in 2606.21584, unverified). Use it only as a guarded add-on after idea 1 or 2.
- **[proposed; the related pooling variants were evaluated-invalid-protocol] Vocal-activity-gated segment pooling for the voice-with-music condition: drop 4.04 s vocal-stem segments below an energy/VAD threshold before taking the max. Speech-only pooling stays unchanged.**
  - Mechanism: In songs, near-silent vocal-stem segments hold mostly separation residue, and a frozen speech CM scores them unreliably. SingFake scores only VAD-positive clips of the separated vocal stem. Max pooling over residue segments can inflate real-song scores or add noise to fake ones.
  - Cost under 48 h: CPU only, if per-segment voice scores and per-segment stem RMS are stored. The known cache feature 'voice_silent' is file-level only (experimental/stacking_v2/FEATURE_SPEC.csv:58), so per-segment RMS may need a Demucs re-run over the eval-like set (hours). The inference change is small.
  - Evidence: SingFake repo dataset/separate.py: mdx_extra two-stem separation plus pyannote VAD (verified). Repo: switching max to LME or top-30% worsened calibration Voice EER from 2.36% to 3.74% / 3.55% (LOCAL_EXPERIMENTS:33; exposed rows, n=308, evaluated-invalid-protocol). Gating itself was not tested. The max-pooling length effect was judged medium-low (ledger:475).
  - Main risk: Fake vocals with low energy (breathy passages, low-mix vocals in Lyria/Suno) could be discarded. It changes an input to the fitted FILE CDF fusion, so FILE could shift. The gain is unmeasured.
- **[proposed] Intermediate-layer features for the singing head: take a learnable weighted sum of DF-Arena/XLS-R hidden layers from the same frozen forward pass on the vocal stem, followed by LR.**
  - Mechanism: Lower and middle SSL layers carry more generic acoustic and artifact information and are less speech-content-specific. The final DF-Arena embedding is tuned for speech TTS artifacts, which fits the poor singing transfer.
  - Cost under 48 h: Needs re-extraction of hidden states for train/val. That is a GPU inference pass over ~30k rows (bank train 30,658), so it may not fit inside 48h together with packaging. Submission runtime would barely change (same forward pass), but the package must be modified and re-validated.
  - Evidence: Kheir et al. repo (verified): frozen SSL layers, with lower layers (1-12 in large models) most discriminative, including on CtrSVDD song data. The SingFake snippet says wav2vec2-based CMs stayed near 50% on vocals, which suggests top-layer speech features overfit to speech (unverified).
  - Main risk: DF-Arena-1B is fine-tuned, so the layer finding for raw SSL may not hold. The extraction cost and the package change carry deadline risk (2026-09-29 10:00 KST, 3 submissions/day). There is no leakage-free singing holdout to confirm the gain before submitting.
- **[proposed] Add pitch- and timbre-aware features for singing, computed on the vocal stem: MERT (already in the package for the accompaniment stem) and/or a speaker-embedding model. Fuse them in a tiny LR with the DF-Arena singing score, used only when the gate says voice+music.**
  - Mechanism: Singing artifacts from SVS/SVC and end-to-end song generators show up in pitch contour, vibrato and timbre consistency. Speech anti-spoof embeddings capture these poorly, while speaker-recognition and music FMs capture them better.
  - Cost under 48 h: One extra MERT forward pass on the vocal stem per file at inference; the current runtime is 17m32s. Feature extraction on train/val is needed, as GPU inference. Head training is CPU-cheap.
  - Evidence: FIONA (x-vector + MERT-v1-330M, 13.74% EER on CtrSVDD; unverified). SingGraph MERT+wav2vec2: relative EER gains of 13.2%, 24.3% and 37.1% (verified README). The repo stacking plan already lists raw/stem MERT comparisons as requiring new GPU telemetry (experimental/stacking_v2/FEATURE_SPEC.csv:50-53).
  - Main risk: Existing9504, trained with CtrSVDD data, still had singing EER 0.2559 (ledger:228), so the limiting factor may be training data rather than features. The 13.74% figure is weak and measured on a cappella data. There is runtime and packaging risk before the deadline. Lowest priority of the six.

### 5.2 AI-music detection and generalisation

| Source | Venue | Verified | Relevance | Finding | Transfer limits |
|---|---|---|---|---|---|
| [Comprehensive Layer-wise Analysis of SSL Models for Audio Deepfake Detection (Kheir et al.)](https://github.com/Yaselley/SSL_Layerwise_Deepfake) | NAACL Findings 2025 (arXiv:2502.03559) | yes | domain difference | Only the official GitHub README was opened; arXiv and ACL Anthology are blocked by egress. The README covers Wav2Vec2, HuBERT and WavLM (small and large) on ASVspoof19/21, ADD23, HABLA, PartialSpoof, SceneFake and CtrSVDD (song). It quotes: 'Lower layers (1-6 for small models, 1-12 for large models) provide the most discriminative features', and 'Reducing layers (e.g., 4-6 for small models) achieves competitive performance while improving inference speed by 2-3x'. | The models are speech SSL, not MERT, and the task is speech or singing deepfake detection, not decoder artifacts in an accompaniment stem. Our MERT-v1-330M has 24 layers, so the analogous 'lower half' is L1-12; the current L12 sits at that boundary. No per-dataset EERs could be verified. |
| [Probing-Guided Layer Selection from Self-Supervised Speech Models for Generalizable Audio Deepfake Detection](https://arxiv.org/abs/2606.30791) | arXiv:2606.30791, 2026 | no | domain difference | From the search snippet only; not opened. Stage 1 fits a lightweight XGBoost probe per layer to rank layers by cross-domain discriminability. Stage 2 attention-pools and projects the selected layers, then feeds them to one classifier. On XLS-R-300M, 4 selected layers give 4.94 +/- 0.32% EER on In-The-Wild and a 5.07% cross-domain average, a 28% relative improvement over a 25-layer frozen baseline. | Evaluated on speech only. It is a method template: per-layer probes are CPU-cheap on frozen features. The number of layers needed for music or generator artifacts is unknown. |
| [Layer-wise Investigation of Large-Scale Self-Supervised Music Representation Models](https://arxiv.org/abs/2505.16306) | arXiv:2505.16306, 2025 | no | plausible unverified | From the search snippet only; not opened. MusicFM and MuQ show an acoustic-to-semantic trend with depth, and a single task-appropriate layer often beats trainable weighted fusion of all layers. A separate snippet about the MERT paper (arXiv:2306.00107, also not opened) says lower layers hold pitch and low-level acoustics, middle layers suit key, genre and emotion, and top layers may overfit the pretext task. The MERT GitHub README (github.com/yizhilll/MERT, opened) confirms MERT-v1 expects 24 kHz input and gives no layer guidance. | No deepfake or artifact task was tested, and no paper found gives MERT layer-wise results for AI-music detection. Our pipeline upsamples 16 kHz to 24 kHz before MERT (repository/full_bank_features.py:217-220; mert12_submission.py:142-145), so MERT's 8-12 kHz band is always empty. |
| [SONICS: Synthetic Or Not - Identifying Counterfeit Songs](https://github.com/awsaf49/sonics) | ICLR 2025 (arXiv:2408.14080) | yes | direct | Opened README: 5-second models reach F1 0.78/0.78/0.76 (sensitivity 0.69/0.69/0.66, specificity 0.94/0.94/0.92) for SpecTTTra-alpha/beta/gamma. 120-second models reach F1 0.97/0.92/0.88 (sensitivity 0.96/0.86/0.79). Search snippets, unverified: audio is at 16 kHz; unseen algorithms (Suno v2, Suno v3, Udio-32) are harder than seen ones (Suno v3.5, Udio-130), with Udio-32 the hardest. | The data is English Suno/Udio songs only. Short context sharply lowers sensitivity, and our 4.04 s windows are in that short-context regime. SONICS zero-shot was already about chance on our data. |
| [A Fourier Explanation of AI-music Artifacts (fakeprint detector)](https://github.com/deezer/ismir25-ai-music-detector) | ISMIR 2025 (arXiv:2506.19108) | yes | domain difference | Opened code. compute_fakeprints.py defaults are sr=44100, N_FFT=16384, fmin=5000 Hz, fmax=16000 Hz, a sliding-minimum lower hull with area=10 subtracted and clipped at 0, clipping to max_dB=5 then max-normalising, and averaging over time. train_test_regressor.py uses LogisticRegression(class_weight='balanced') with an 80/20 split and seed 123. From the paper abstract via snippet, unverified: transposed-conv zero-upsampling gives peaks fixed by the architecture, with >99% accuracy on DAC, Encodec, Suno and Udio. The companion ICASSP 2025 repo README (github.com/deezer/deepfake-detector, opened) names robustness to manipulation and generalisation to other models as open challenges. | The default band of 5-16 kHz is mostly above our 8 kHz Nyquist. A community 16 kHz port (github.com/lofcz/ai-music-detector, README opened) uses the 1-8 kHz band and claims 99.88% accuracy. It also says the method is sample-rate dependent, needs retraining for each architecture, and was not tested on MP3. Peaks are specific to each architecture, and pitch or speed changes move them (arXiv:2607.27454, unverified). |
| [MusicDET: Zero-Shot AI-Generated Music Detection](https://github.com/Chaolei98/MusicDET) | ICML 2026 (arXiv:2605.18072) | yes | plausible unverified | Opened README: frequency-guided normalizing flows are trained only on real music and fakes are flagged as out-of-distribution. Backbones are XLSR, MERT, SpecTTTra, ViT and ConvNeXt; datasets are FakeMusicCaps and SONICS. From snippets, unverified: the real-only model reaches an average cross-generator EER of 4.51% on FakeMusicCaps (AASIST: 32.73%) and 2.89% on SONICS. The class-conditional variant reaches 0.89% on FakeMusicCaps. | In FakeMusicCaps and SONICS, real and fake audio come from different corpora and resampling chains, so the OOD score may partly measure corpus shift. Our real training music is MUSAN, while eval real music is Korean phishing-context background music, sometimes telephone-channel or MP3. |
| [Echoes: A semantically-aligned music deepfake detection dataset](https://arxiv.org/abs/2603.23667) | arXiv:2603.23667, 2026 | no | domain difference | From the search snippet only; not opened. The dataset has 4,468 AI tracks (131 h) from 12 generators, paired with 300 bona-fide references. Fakes are conditioned on bona-fide waveforms or descriptors. With XLS-R 2B features, detectors trained on existing datasets transfer poorly to Echoes, and training on Echoes transfers best. Held-out generators remain difficult; provider diversity and semantic alignment help. | Our fx_encoder_style_transfer fakes are also reference-conditioned. The results support generator and provider diversity over more data from the same generators, which fits our flat learning curve. No numbers were verified. |
| [ArtifactNet: Detecting AI-Generated Music via Forensic Residual Physics](https://arxiv.org/abs/2604.16254) | arXiv:2604.16254, 2026 | no | plausible unverified | From the search snippet only; not opened. A 3.6M-parameter UNet extracts codec residuals, which are split by HPSS into 7 channels and classified by a 0.4M-parameter CNN. ArtifactBench has 6,183 tracks (4,383 AI tracks from 22 generators, 1,800 real). On the unseen partition (n=2,263) it reports F1 0.9829 and FPR 1.49%, against CLAM at F1 0.7576 / FPR 69.26% and SpecTTTra at F1 0.7713 / FPR 19.43%. Codec-aware training with 4-way WAV/MP3/AAC/Opus augmentation cuts cross-codec probability drift by 83% (0.95 to 0.16). | The authors evaluated on their own benchmark, and competitors were run from published checkpoints. The approach needs a trained UNet, which is not a frozen-feature CPU head. The codec-augmentation result is the transferable part. |
| [Detection of AI-generated stems within hybrid human-AI music](https://arxiv.org/abs/2607.26874) | ISMIR 2026 (arXiv:2607.26874) | no | direct | From the search snippet only; not opened. Setup: MUSDB18-HQ vocals plus accompaniment, with individual stems autoencoded through a neural codec. Running generic source separation and then detecting on the separated stem confirms that 'artifacts associated with an AI-generated stem are not reliably recovered by generic source separation systems'. The proposed parallel design uses separation only to estimate each stem's relative energy, combined with a mix-level detector's prediction, feeding stem-specific classifiers on short chunks averaged per track. The code repo github.com/deezer/ismir26-hybrid-human-ai-music-detection was opened; its README says 'Coming soon'. | Fakes here are codec reconstructions, not commercial generators. It directly questions detecting on the HTDemucs accompaniment stem, but no numeric EERs were verified. |
| [How Much AI Is in This Track? Quantifying the Proportion of AI-Generated Stems in Hybrid Music Mixtures](https://arxiv.org/abs/2608.07285) | arXiv:2608.07285, 2026 | no | domain difference | From the search snippet only; not opened. Regression on the AI energy ratio reaches MAE 0.076 and R^2 0.85 on held-out mixtures from the same pipeline. Sensitivity depends on the instrument. Drums and guitar carry strong codec-artifact signatures, with drums separable from about 2500 Hz upward. Bass shows essentially no real/AI separation, and vocals only very subtle differences. | Uses a codec-reconstruction proxy. This supports the accompaniment (drums and guitar) as the artifact-rich part. However, our Lyria instrumental-segment FNR of about 0.75 shows the L12 time-mean feature does not capture it for unseen generators. |
| [How Does Instrumental Music Help SingFake Detection? (with SingFake, arXiv:2309.07525)](https://arxiv.org/abs/2509.14675) | arXiv:2509.14675, 2025; SingFake at ICASSP 2024 | no | domain difference | From search snippets only; not opened. SingFake: systems trained on Demucs-separated vocals generally beat those trained on mixtures, except W2V2-AASIST, which learned well from mixtures. 2509.14675: the instrumental track acts mainly as data augmentation. Models rely on low-frequency information in the singing voice and are largely insensitive to instrumental cues. Fine-tuning increases reliance on shallow speaker features. | This is about vocal deepfakes, not music-generator detection. It suggests SSL front-ends can work on mixtures and that separation is not always necessary. |
| [AI-generated music detection in broadcast monitoring (AI-OpenBMAT) and Assessing AI-generated music detection in real-world broadcast monitoring (BAMM)](https://arxiv.org/abs/2602.06823) | ICASSP 2026 (arXiv:2602.06823); arXiv:2608.07359, 2026 | no | direct | From search snippets only; not opened. AI-OpenBMAT has 3,294 one-minute excerpts (54.9 h), with Suno v3.5 continuations mixed at broadcast SNRs. SpecTTTra and a CNN baseline drop below 60% F1 when the music is in the background or short. BAMM (arXiv:2608.07359) has 40 h of real TV: detection is near-perfect on clean foreground music, degrades under synthetic broadcast, and degrades further on real TV with substantial score overlap. Speech masking and short music length are named as critical open problems. | This is the closest analogue to voice-phishing files with background music under speech. The generator was Suno v3.5 only, and it evaluates end-to-end detectors, not frozen MERT features. |

**Actionable ideas proposed by the research agent** (screened in the plan against repository evidence):

- **Sweep MERT layers with CPU probes on a generator-held-out split. Extract all 25 hidden states in the existing single forward pass, fit per-layer logistic or XGBoost probes, select 2-4 layers (lower/middle expected), and feed their concatenation or attention pool into WindowMIL in place of L12-only.**
  - Mechanism: Decoder and codec artifacts are low-level acoustic features. L12 of 24 sits at the acoustic-to-semantic boundary, and time-mean pooling of higher layers abstracts the artifacts away. Probes pick layers whose discrimination transfers across generators.
  - Cost under 48 h: The forward pass already returns every layer (output_hidden_states=True at repository/full_bank_features.py:220 and mert12_submission.py:144-145), so extraction compute equals the original L12 pass. Storage grows to about 25x1024 per window instead of 1x1024. Probes and head retraining take CPU minutes. Submission runtime stays near the current 17m32s. Repo status: implemented-not-run or unevaluated. eight_features.py:28 has LAYERS=(6,12,24); results/eight_h100_20260917 contains only runtime_sources/*.sh with no metrics (results not in repo); docs/superpowers/plans/2026-09-24-dacon-lb-improvement-workflow.md:280 lists E12 as proposed.
  - Evidence: Kheir et al. (README verified): lower layers 1-12 of large SSL models are most discriminative, including on CtrSVDD song. Pan et al., Interspeech 2024 (arXiv:2406.10283, unverified): WavLM-large layers up to about 12 contribute most. Probing-guided selection (arXiv:2606.30791, unverified): 4 XLS-R layers give 4.94% ITW EER, 28% relative better than a 25-layer frozen baseline. Serrano et al. (arXiv:2509.12003, unverified): the best single layer is strong for OOD. Music SSL acoustic-to-semantic trend (arXiv:2505.16306, unverified).
  - Main risk: All layer evidence is from speech SSL; none covers MERT on music-generator artifacts. Lower layers are more sensitive to channel, codec and corpus shortcuts (MUSAN real versus generated fakes), so held-out gains may not reach the leaderboard. With only 2 held-out fake generators, layer selection can overfit; use leave-one-generator-out.
- **Add a mixture-branch music feature: MERT windows computed on the original mixture, plus the HTDemucs accompaniment-to-mixture energy ratio. Late-fuse it with the accompaniment-stem branch and compare on the held-out set, including Lyria instrumental segments.**
  - Mechanism: HTDemucs re-synthesises the stem and may suppress or overwrite generator artifacts. The Deezer result is that separating first and then detecting does not recover stem artifacts; it uses separation only to estimate energy and detects on the mix.
  - Cost under 48 h: One extra MERT pass over mixture windows costs the same as the stem pass. Fusion and head training take CPU minutes. Submission runtime grows by roughly the current MERT share. Status: proposed; no mixture-branch music feature was found in the repo.
  - Evidence: arXiv:2607.26874 (Deezer, ISMIR 2026, unverified). arXiv:2608.07285 (unverified): drums and guitar carry strong artifacts, bass and vocals weak. SingFake (unverified): W2V2-AASIST learns well from mixtures. arXiv:2509.14675 (unverified): instrumental acts as augmentation.
  - Main risk: Voice-phishing mixtures are dominated by speech, so mixture features can confound MUSIC_FAKE with voice. AI-OpenBMAT and BAMM (unverified) report F1 below 60% under speech masking. Gate the mixture branch by stem energy ratio or low vocal energy. On instrumental Lyria windows the mixture is nearly the stem, so an FNR of about 0.75 there may simply reflect generator shift.
- **Add Afchar-style fakeprints as a CPU feature: time-averaged log-magnitude STFT, minus a sliding-minimum lower hull, clipped and normalised over the 1-8 kHz band at 16 kHz, computed on accompaniment or music-present windows. Use it in a logistic regression or as an extra input to the music fusion.**
  - Mechanism: Transposed-conv zero-upsampling in neural decoders produces spectral peaks fixed by the architecture. These peaks are independent of the SSL embedding, so the feature is complementary to MERT.
  - Cost under 48 h: CPU only. N_FFT=16384 at 16 kHz is about 1.02 s per frame; all training files plus the 1,200 eval files take roughly 1-2 h. Logistic regression takes seconds. Status: proposed; no fakeprint code was found in the repo.
  - Evidence: deezer/ismir25-ai-music-detector code (verified): defaults fmin=5000, fmax=16000 at sr=44100; lower_hull area=10; max_dB=5; LogisticRegression(class_weight='balanced'). lofcz/ai-music-detector README (verified): 16 kHz port with the 1-8 kHz band claims 99.88% accuracy. arXiv:2607.25530 (unverified): one-class 'Finding the noise' variant on fakeprints.
  - Main risk: Deezer's default band lies mostly above our 8 kHz Nyquist. MP3 encoding, resampling or pitch/speed changes erase or move peaks (arXiv:2607.27454, unverified). Peaks are architecture-specific, so an unseen decoder (Lyria-3 Pro) or FX-style-transfer fakes may show none. Validate per generator before fusing.
- **Add a real-only density score: a Gaussian/Mahalanobis model or a small normalizing flow fit on frozen MERT window features of real music, with its NLL as an extra fusion feature or a fallback music score.**
  - Mechanism: It models real music rather than known fakes, so unseen generators score as out-of-distribution instead of needing to resemble the training fakes.
  - Cost under 48 h: CPU minutes on cached L12 or selected-layer features. Status: proposed.
  - Evidence: MusicDET (README verified; numbers unverified): real-only cross-generator EER 4.51% on FakeMusicCaps and 2.89% on SONICS. 'Finding the noise' (arXiv:2607.25530, unverified).
  - Main risk: Eval real music (Korean background music, telephone channel, MP3) is itself out-of-distribution relative to MUSAN, which inflates false positives. MusicDET's benchmarks have real/fake corpus confounds. Validate with a second real corpus: repository/new_panel_data_jamendo_bank.py describes an unassigned pre-2016 Jamendo bank, but whether it is populated or featurised was not verified.
- **Late-fuse scores from heterogeneous frozen encoders already in the repo (MERT-stem, PANNs, wav2vec/anti-spoof features). Learn the fusion weights on leave-one-generator-out folds. Add codec augmentation (MP3/AAC/Opus, telephone band) only if the re-extraction budget allows.**
  - Mechanism: Encoders with different pretraining catch different generator cues, and score fusion averages out generator-specific failures. Codec augmentation removes channel shortcuts.
  - Cost under 48 h: Fusion on cached scores takes CPU minutes. Augmentation requires re-extracting features, about the cost of one full extraction.
  - Evidence: Serrano et al. (arXiv:2509.12003, unverified): score fusion improves OOD. Phukan FIONA (arXiv:2409.14131, unverified): x-vector plus MERT-v1-330M fusion is best at EER 13.74% for SVDD. CLAM (arXiv:2512.00621, unverified): MERT plus wav2vec2 reaches F1 0.925 against 0.869 prior SOTA on MoM. SOFIA README (github.com/homura23/SOFIA, verified): 5 encoders including FXEncoder++, MuQ and MERT-95M. ArtifactNet (unverified): codec-aware training cuts drift by 83%. AT-ADD summary (arXiv:2608.14249, unverified): top systems combine large SSL, augmentation, multi-crop and fusion.
  - Main risk: The PANNs submission 104224 scored ADS 0.7674603175 against 102411 at ADS 0.7683174603, so a previous encoder addition did not improve the leaderboard. The flat learning curve means augmentation adds invariance but not generator coverage. Weights tuned on 2 fake generators can overfit.
- **Fix the evaluation protocol before trusting music gains: leave-one-generator-out crossed with a second, non-MUSAN real-music corpus. Report the held-out music EER on both real sources.**
  - Mechanism: When every real music file is from MUSAN, a detector can learn corpus identity. The held-out EER of about 0.20 then partly measures MUSAN versus not-MUSAN. The flat learning curve from 50% to 100% training data points to a generator-coverage limit rather than a data-quantity limit.
  - Cost under 48 h: CPU hours to rescore cached features on new splits, plus extraction for any new real corpus. Status: proposed.
  - Evidence: 'Probing Token Spaces under Generator Shift' (arXiv:2606.08663, ICML 2026, unverified): standard and real-source-restricted splits are nearly saturated, while restricting the fake source exposes large differences. X-Codec tokens are best when training on Udio only; MERT tokens are stronger when training on Suno-v3.5 only. Echoes (unverified): held-out generators remain hard and provider diversity helps. An unverified search snippet (arXiv:2507.10447 or 2601.13647) warns that resampling chains such as 44.1 kHz to 16 kHz to 24 kHz for MERT create exploitable confounds.
  - Main risk: A small second real corpus gives noisy EER. Protocol changes alone do not raise the leaderboard score; they only stop false-positive decisions.

## 6. Direct checks in this session

### 6.1 Incumbent pipeline trace (runtime `c157c2f`)

Input handling: 16 kHz mono, peak capped at 0.6, then HTDemucs splits it into a vocals stem and the sum of the other
stems (backing).

| Output | How it is computed | Training/inference match |
|---|---|---|
| Voice | DF-Arena-1B on non-overlapping 4.04 s vocal-stem segments; fc5-input 1280-d embedding → `Linear(1280,2)` initialised from fc5; file score = max over segments of softmax[fake]; mean of 3 seeds | same as training (max odds → sigmoid) |
| Music | MERT-v1-330M hidden state 12, time-mean of 4.04 s backing windows (hop 2.02 s, resampled 16→24 kHz) → WindowMIL (MLP per window, log-mean-exp); mean of 3 seeds | log-mean-exp identical in training, evaluation and runtime |
| Presence | logistic heads on PANNs class-group statistics + stem-energy features | same features both sides |
| FILE | equal-weight mean of val-CDFs of max(Pv·Fv, Pm·Fm), noisy-OR and max(Fv, Fm) | fixed per file, no cross-file statistics |

- Gain: data v4 banks have no `input_kind`, so every DataV4 row is peak-capped in training exactly as at runtime.
  The train/runtime gain mismatch exists only for legacy full_bank rows.
- Precision: bf16 vs fp32 head inference moves the heads by at most 0.0014 (voice) / 0.005 (music), with identical
  val EER. [V]

### 6.2 Package identity (recomputed from `repository/submissions/*/submit.json`)

| Package | Runtime = c157c2f + candidate string | decision_heads.py = c157c2f | presence_head.json | file_ensemble.json | Members differing from DB |
|---|---|---|---|---|---|
| DataV4_DurationBalanced (DB) | yes | yes | 4c54797f | 3e0056d1 | — |
| DataV4_P1Refit | yes | yes | 4c54797f | c44d9973 | 6 heads, file_ensemble, lock/identity/metrics/manifest, candidate string |
| DataV4_S2a_AIHubPhone | yes | yes | 4c54797f | 99e28a97 | same set as P1 |

### 6.3 Main-branch measurements (tables generated from repo JSON)

#### Zero-shot member fusion on DataV4 heads (`repository/results/fusion_20260927/*/fusion_report.json`)

| Run | Rows (val/test) | Baseline test File / Voice / Music, ADS | Fused test File / Voice / Music, ADS | Voice weights | Music weights |
|---|---|---|---|---|---|
| db_all_eq | 4931/6450 | 0.1094 / 0.1107 / 0.2000, 0.8632 | 0.1474 / 0.1238 / 0.2000, 0.8415 | {'ours': 1, 'xlsr1b_max': 1} | {'ours': 1} |
| db_all_eqdf | 4931/6450 | 0.1094 / 0.1107 / 0.2000, 0.8632 | 0.1456 / 0.1253 / 0.2000, 0.8422 | {'ours': 1, 'xlsr1b_max': 1, 'df0': 0.5} | {'ours': 1} |
| db_all_w1 | 4931/6450 | 0.1094 / 0.1107 / 0.2000, 0.8632 | 0.1276 / 0.1078 / 0.2000, 0.8547 | {'ours': 1, 'xlsr1b_max': 0.25, 'df0': 0.25} | {'ours': 1} |
| db_unseen_1b (unseen) | 2368/3632 | 0.1216 / 0.1186 / 0.1936, 0.8574 | 0.1200 / 0.1143 / 0.1936, 0.8591 | {'ours': 1, 'xlsr1b_max': 0.25, 'df0': 0.25} | {'ours': 1} |
| db_unseen_all (unseen) | 2368/3632 | 0.1216 / 0.1186 / 0.1936, 0.8574 | 0.1205 / 0.1134 / 0.1936, 0.8590 | {'ours': 1, 'xlsr2b_max': 0.25} | {'ours': 1} |
| db_unseen_eq (unseen) | 2368/3632 | 0.1216 / 0.1186 / 0.1936, 0.8574 | 0.1216 / 0.1315 / 0.1936, 0.8548 | {'ours': 1, 'xlsr1b_max': 1} | {'ours': 1} |
| db_unseen_eqdf (unseen) | 2368/3632 | 0.1216 / 0.1186 / 0.1936, 0.8574 | 0.1197 / 0.1203 / 0.1936, 0.8580 | {'ours': 1, 'xlsr1b_max': 1, 'df0': 0.5} | {'ours': 1} |
| p1_all_w1 | 4931/6450 | 0.0515 / 0.1004 / 0.0319, 0.9446 | 0.0739 / 0.1045 / 0.0319, 0.9326 | {'ours': 1, 'xlsr1b_max': 0.25, 'df0': 0.25} | {'ours': 1} |
| p1_unseen_w1 (unseen) | 2368/3632 | 0.0273 / 0.1122 / 0.0290, 0.9552 | 0.0284 / 0.1130 / 0.0290, 0.9545 | {'ours': 1, 'xlsr1b_max': 0.25, 'df0': 0.25} | {'ours': 1} |

`p1_*` test rows were trained on by P1; those test numbers are contaminated and listed only for completeness.

#### Learning curve, DataV4 objective, seed 41 (`repository/results/audit_20260927/learning_curve.json`)

| Route | Train fraction | Rows | Groups | Updates | Best epoch / run | Val EER | Held-out test EER |
|---|---:|---:|---:|---:|---|---:|---:|
| music | 0.125 | 1071 | 719 | 110 | 17 / 22 | 0.20549 | 0.25817 |
| music | 0.25 | 2149 | 1438 | 198 | 17 / 22 | 0.16778 | 0.23294 |
| music | 0.5 | 4364 | 2875 | 576 | 27 / 32 | 0.13986 | 0.19821 |
| music | 1.0 | 9363 | 5750 | 629 | 12 / 17 | 0.12220 | 0.20593 |
| voice | 0.125 | 2462 | 1323 | 420 | 37 / 42 | 0.06311 | 0.12901 |
| voice | 0.25 | 5451 | 2646 | 880 | 35 / 40 | 0.06132 | 0.12844 |
| voice | 0.5 | 11275 | 5292 | 2250 | 47 / 50 | 0.05574 | 0.12280 |
| voice | 1.0 | 24623 | 10583 | 4850 | 50 / 50 | 0.05077 | 0.10974 |

### 6.4 Implied official error levels

| Submission | ADS | Weighted EER (1 − ADS) | Note |
|---|---:|---:|---|
| G0 baseline | 0.65775 | 0.342 | CPS 0.98932 |
| M330 (music swap only) | 0.71190 | 0.288 | ADS derived assuming G0's CPS |
| DataV4 (102411) | 0.76832 | 0.232 | CPS 0.99768 |
| DataV4 held-out proxy (internal) | 0.86316 | 0.137 | not an official score |

The internal held-out proxy understates official error by about 0.095 weighted EER.

### 6.5 V-SING script wiring check (`repository/data_v4/sing_relabel_screen.py`)

The script was run on a synthetic cache written in the exact on-disk schema (index identity, npz fields, manifests),
under torch 2.7.1, NumPy 1.26.4, transformers 4.57.6 and sklearn 1.8.0. Fake singing was placed on a direction that
the speech fakes do not use.

| Arm | Pooled voice EER | Voice-only | With music | Real-with-music FPR | ΔVoice vs incumbent (95% CI) | Gate |
|---|---:|---:|---:|---:|---|---|
| incumbent (fc5 init) | 0.5125 | 0.550 | 0.525 | 0.575 | — | — |
| control (`--no-completion`) | 0.425 | 0.225 | 0.550 | 0.375 | −0.0875 [−0.206, +0.018] | reject |
| completion | 0.3125 | 0.375 | 0.275 | 0.300 | −0.200 [−0.302, −0.104] | reject once the control is compared (voice-only worse than control) |

- **What this establishes:** the wiring works, and the gate catches a speech-for-singing trade.
- **What it does not establish:** anything about real data.
- **Refit mode:** the output dir passed the packager's checks (lock hashes, val-fitted ensemble with no `prev`, results
  keys, music heads copied byte-for-byte).

## 7. Evidence limitations

- dacon.io is blocked from this environment (egress 403), so there is no live leaderboard, rank-15 cutoff or
  remaining-slot count.
- Post-09-17 leaderboard numbers come from transcribed screenshots that are not in the repo.
  - The 102411 → DataV4 and 104224 → PANNs fix1 bindings are timing inferences.
  - E08's ID, total and CPS were never recorded.
- No feature caches, prediction CSVs or audio are in the repo. Recomputations use repo JSON only; several branch
  diagnostics exist only as markdown plus hashes.
- The research agents could not open arxiv/ISCA/IEEE/OpenReview; unverified rows in section 5 rest on search snippets.
- Status of the 09-27 VESSL multi-layer backbone extraction and the local MPS S2 feature chain is unknown.
- P1Refit's L4 runtime, NumPy 1.26.4 checkpoint load and per-seed best epochs are unmeasured.

## 8. Update — main `59c2318` (S3 package), checked 2026-09-27 13:23 KST

### 8.1 What changed on main

| Commit | Content |
|---|---|
| 3753e10 | S3 package: MERT layers 6+12 music heads; AntiDeepfake-1B embedding voice heads fused with the DF-Arena (P1) heads at w=0.5; FILE grid refit; runtime `music_layers.json` + `voice_ad_*.pt` |
| 0836799 | Results bundle: `data/data_v4_bank` manifests (no audio, no AI-Hub rows), member scores, head checkpoints for DB / P1 / S2a / S3, screen and comparison JSON |
| 59c2318 | AI-Hub usage record (metadata only); S3 uses no AI-Hub data |

### 8.2 Held-out evidence for S3 (DB protocol; `results/audit_20260927/ads_compare.json`)

| Configuration | Val File / Voice / Music, ADS | Held-out File / Voice / Music, ADS |
|---|---|---|
| baseline (DF-Arena voice, MERT-12 music) | 0.0691 / 0.0509 / 0.1252, 0.91773 | 0.1089 / 0.1195 / 0.1908, 0.86438 |
| music MERT-6+12 | 0.0680 / 0.0509 / 0.1119, 0.92222 | 0.1064 / 0.1195 / 0.1847, 0.86748 |
| + voice AD-head fusion w=0.5 (**S3**) | 0.0596 / 0.0376 / 0.1119, 0.92912 | 0.0996 / 0.1106 / 0.1847, **0.87267** |
| + voice AD-head fusion w=1 | 0.0585 / 0.0366 / 0.1119, 0.92984 | 0.1000 / 0.1125 / 0.1847, 0.87209 |

Music layer seeds (`music_layer_seeds.json`), val / held-out means over seeds 41–43:
- [12]: 0.1289 / 0.1946
- [6]: 0.1293 / 0.1849
- [6,12]: 0.1167 / 0.1810
- [3,6,9]: 0.1156 / 0.1843

[6,12] is below [12] for every seed in both splits. The single-seed layer screen
(`backbone_heads_screen/screen_report.json`) puts the lowest held-out route EER at L6 (0.188) and L3 (0.189), versus L12
(0.206) and L24 (0.212).

### 8.3 Checks run in this session

**Backbone feature coverage.** IDs come from `results/backbone_heads_screen/predictions.json`; sources come from
`data/data_v4_bank`.

| Split | Covered | Range across sources with ≥ 50 rows | Range across manifest-position deciles |
|---|---:|---|---|
| val | 3,680 / 4,931 (0.746) | 0.66–0.78 | 0.72–0.76 |
| held-out | 4,822 / 6,450 (0.748) | 0.72–0.78 | 0.72–0.77 |

Extraction used strided shards stopped at a deadline, but coverage is flat, so the backbone manifest was shuffled. The
missing ~25% is a random subsample, not a source-skewed slice.

**Judge-stack checkpoint load** (NumPy 1.26.4, torch 2.7.1, transformers 4.57.6):

| Package | Heads loaded with `weights_only=True` | Repo copy = zip member | Music input width |
|---|---:|---|---:|
| DataV4_DurationBalanced | 6 | 6/6 | 1024 |
| DataV4_P1Refit | 6 | 6/6 | 1024 |
| DataV4_S3_Backbones | 9 (incl. `voice_ad_*`) | 9/9 | 2048 |

`antideepfake_runtime.py` and `component_fusion.py` import cleanly, and the `component_fusion` self-check passes. The
AntiDeepfake `model.safetensors` is not in the repo and was not loaded.

**Best epoch / epochs run** (from `*_history.json`):

| Package | Voice seeds 41 / 42 / 43 | Music seeds 41 / 42 / 43 |
|---|---|---|
| DB | 50/50, 44/49, 50/50 | 12/17, 11/16, 9/14 |
| P1 | 39/44, 24/29, 50/50 | 15/20, 16/21, 12/17 |
| S3 | no histories in repo | no histories in repo |

**Package identity.** S3's presence head, `decision_heads.py`, `script.py` and `requirements.txt` equal DB's. The ZIP is
8,286,541,720 bytes (< 10 GB). CPU QA: 10 files in 84.9 s (DB: 66.7 s).

### 8.4 Open risks

- **Order-dependent fallback.** The runtime keeps a 48-minute fallback that drops the AntiDeepfake member for files
  scored after 48 min. This makes VOICE depend on processing order (rule 2-4) if the L4 run is slow.
- **Mild test-informed selection.** Val preferred w=1 and [3,6,9]; test preferred the chosen w=0.5 and [6,12]. Both gaps
  are ≤ 0.0011.
- **Not validated on L4.**
- **Bundled change.** S3 includes P1, which has no leaderboard result yet.
