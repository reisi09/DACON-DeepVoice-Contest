# Round 16 — cross-examination (2026-09-28 ~17:15 KST). Both reviewers get the same file.

Your two round-15 answers: codex_round15.md (long-running reviewer) and fresh15_r1.md (fresh reviewer). Read both.

## 1. Corrections to ROUND15 (accepted)
- AT-ADD singing FILE EER: W_new 0.094 / W_newB 0.094 / s7_new 0.151 / s7_new_ob 0.130 (not "0.13-0.15 for all").
- New rows: DB training adds the train split only (23,955; 8 failed dumps drop out). P1 finals used all 40,214 rows with features
  (train + newval + newtest, NEW_VAL=0), so E2 is NOT held-out for any P1 package. DB results validate the recipe only.
- Korean fakes: HAIM B6 "kr" songs (ACE-Step, VF=1, MF=1) exist in the data; Korean fake SPEECH is what is missing.
- Parity: code_r14/w_runtime_parity.py is now the fixed version (compares FILE <= 0.01, checks receipt sha256, ID count,
  finite values, exits nonzero on FAIL). s7_new V/M/F 0.00018/0.00089/0.00011, W_new 0.00064/0.00102/0.0002,
  W_newB 0.00055/0.00101/0.00021; all three passed the VESSL CUDA offline check (sha of the rebuilt tree = receipt).
- Rule L leak into WavLM (long reviewer): does NOT apply. VERIFIED from run_identity.json: stage A (final_*L) trains DF/AD with
  label_completion and no WavLM; stage B (final_*) has label_completion False, trains WavLM + music, reuses A's voice heads.

## 2. New evidence (CPU only; code_r14/virtual_configs.py builds members from saved DB every-row predictions)
Deltas vs s7_ref, same tables / fusion 1-0.5-1 / FILE 0-9-1; E2 = all 12,426 rows (unlinked-only numbers differ by < 0.002).
| candidate | old-val ADS | E1 | E2 | E1+E2 | E3 473 real FPR |
|---|---|---|---|---|---|
| s7_new | 0.96469 | +0.0064 [.0013,.0111] | +0.0285 | +0.0210 | 0.027 |
| W_newB | 0.96347 | +0.0032 | +0.0470 | +0.0334 | 0.040 |
| ens s7_new+W_newB (all members, 6 seeds) | 0.96407 | +0.0061 | +0.0417 | +0.0305 | 0.027 |
| ens s7_new+W_new | 0.96426 | +0.0057 | +0.0398 | +0.0286 | 0.027 |
| mix voice s7_new / music W_newB | 0.96180 | +0.0103 [.0036,.0162] | +0.0389 | +0.0298 | 0.013 |
| mix voice W_newB / music s7_new | 0.96636 | +0.0006 | +0.0366 | +0.0244 | 0.040 |
| mix voice s7_new / music s7_new_ob | — | +0.0050 | +0.0376 | +0.0285 | — |
(mix voice s7_new / music W_new — both components never saw newval — is being computed.)
Observation: old-val ranks the candidates almost in REVERSE of E1 (the highest old-val candidate is the worst on E1).
Old val was used for early stopping of every head and for every CDF table; W_newB music was early-stopped on val+newval.

## 3. Proposal from the main agent (challenge it)
P-a. Integrated candidate = voice heads of s7_new (P1 final_s7_newL + final_s7_new WavLM) + music heads of W_new or W_newB
     (P1), FILE 0/9/1, fusion 1/0.5/1. Build it from EXISTING P1 heads with the trainer's --reuse-voice/--reuse-wavlm/
     --reuse-music (no training; val CDF and FILE tables recomputed on the old val, CPU on the Mac), package, CPU verify,
     parity, CUDA check (~0.2 H100-hours, ~$0.6). No runtime code change (single music source; no 6-head ensemble).
P-b. Selection among {s7_new, mix s7_new/W_new, mix s7_new/W_newB, W_newB} by a pre-registered rule on data not used for
     training or early stopping of the compared heads: newval (3,841) scored by the saved DB models (CPU), plus E1/E2 as
     already-seen confirmation; winner's-curse guard: adopt a mix only if it beats s7_new on BOTH E1 and E2-unlinked with the
     bootstrap CI lower bound > 0 on E1+E2, and not worse by > 0.002 on E2-open.
P-c. Submission order: (1) the integrated candidate if it passes, else s7_new; (2) W_newB; (3) the remaining one.
Budget: GPU ~0.2 h (CUDA check) + CPU workspace ~1 h ($0.3); VESSL credit now $19.00 (all GPUs paused).

## 4. Questions (Korean, VERIFIED / INFERRED / UNKNOWN, short, numbers)
(a) BLOCKING objections to P-a / P-b / P-c, with the minimal fix. In particular: is selecting on E1/E2 after they were seen
    acceptable with the guard in P-b, or must newval be the ONLY selection data? Is packaging P1 heads (trained on E1/E2)
    acceptable when the DB protocol validated the recipe (as for S3-S8)? The fresh reviewer proposed packaging DB heads and
    re-splitting val with 2.5 GPU-hours of retraining: keep, drop, or shrink — given $19 and the 06:00 deadline?
(b) Anything essential missing in the verification of a hybrid package (reused heads from two different P1 finals).
(c) Final recommended plan with GPU-hours, dollars and timeline.
