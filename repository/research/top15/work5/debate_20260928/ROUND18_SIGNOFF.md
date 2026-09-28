# Round 18 — sign-off on the consolidated plan (2026-09-28 17:22 KST). Both reviewers get the same file.

Your round-17 answers: codex_round17.md (long) and fresh17_r3.md (fresh). The consolidated plan below merges them.
Reply only with: BLOCKING objections (with the minimal fix), your choice on the two open points in section 5, and corrections
of any number that is wrong. Korean, VERIFIED / INFERRED / UNKNOWN, short.

## 1. Integrated model decision
- Design kept for the competition: S7 encoders and structure unchanged; voice = s7_new heads (DF MLP, AD ResMLP with rule L,
  WavLM h8 MLP without L), fusion 1/0.5/1; music = WindowMIL 3840-d; FILE 0/9/1 fixed; presence unchanged; seeds 41/42/43.
- The only integrated variant tested (voice s7_new + music W_new, built from existing P1 heads, no retraining, no runtime change)
  is REJECTED under the pre-registered rule on the available newval rows (+0.0013, CI [-0.0020, +0.0057]).
- Optional completion (both reviewers accepted it as a completion, not a second look): back-fill presence for the 1,521
  missing newval rows (HTDemucs + PANNs on the same package Engine), keep IDs / candidates / CDF / thresholds / bootstrap /
  gates fixed, open nothing until complete, apply the same rule ONCE; report the partial result too. Cost 0.1-0.2 H100-hours
  ($0.30-0.60) + the mix package verification (~0.2 H100-hours, $0.6). Needs the user's GPU approval (GPUs are paused).

## 2. Submissions (2026-09-29; LB results do not change recipes)
1. s7_new (sha fcdd8691...): vs s7_ref E1 +0.0064 [.0013,.0111], E2-unlinked +0.0285; E3 real FPR 0.027 (W_newB 0.040).
2. W_newB (sha 8d3b62d3...): vs s7_ref E1 +0.0032 [-.0040,.0093], E2-unlinked +0.0467; E2 singing FILE EER 0.094 (all rows,
   s7_new 0.151); a different training recipe.
3. open point A (section 5). A mix that passes the completed test and its package verification replaces slot 3.

## 3. Design record for after the competition (hypotheses, not a confirmed optimum)
Components: voice fixed as above; music compare s7 / W / 50:50 probability average of two recipes (needs a music-only
checkpoint list loader); FILE compare 0/9/1, 1/1/1, 0/1/0 after head selection; presence fixed unless broadcast-mix presence
errors are measured. Data: Korean fake SPEECH (real 2,000 + fake 2,000, >= 4 TTS/VC families), broadcast mixes and commercial
music; splits: early-stopping val, a clean selection set >= 5,000 rows (including unseen generators), an untouched final test
>= 5,000 rows opened once, lineage (speaker / song / generator / source track) kept within one split; 3 group splits x
3 seeds for stability. Estimate 6-10 H100-hours plus data acquisition (UNKNOWN).

## 4. Budget
VESSL $19.00 at 16:42 (all GPUs paused); the CPU workspace ($0.30/h) runs since ~16:55 and is paused after this round. Plan with no further GPU: $0. With the optional
completion + mix package: <= 0.5 H100-hours ($1.49 cap). Elice credit KRW 10,000 unused.

## 5. Open points (choose one each, one-line reason)
A. Slot 3 when no mix: W_new (verified now; fresh reviewer) or s7_new_ob (needs its CUDA check, ~$0.5; long reviewer)?
B. Is the optional completion worth running at all tonight, given it can only change slot 3?
