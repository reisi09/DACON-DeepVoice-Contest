# Final plan after the 2026-09-28 debate (08:40-09:40 KST)
Participants: long-running Codex session (rounds 10-12), fresh independent Codex session (r1-r3), fresh independent
Claude verifier (r1-r3), fresh Claude dataset scout (r1), main Claude. Records: research/top15/work5/debate_20260928/.

## Conclusion
1. Data imbalance is real, and its most damaging form is the TRAINING WEIGHTS, not the raw counts (unanimous, VERIFIED):
   the (label, source, generator) cell weighting plus global duration balancing gives
   - voice heads: singing 41% of the real mass but 15% of the fake mass; Korean 5.7% of the fake mass (weighted fake share
     0.24 vs row share 0.57); MLAAD+ASVspoof 83% of the fake mass; Kish-effective rows real 2,113 / fake 6,891;
   - music head: one legacy lineage component (artist:FMA:1041) 22% of the real and 29% of the fake mass; Kish-effective
     lineage groups real 19 / fake 12.
   This encourages "singing voice => real", "Korean speech => real" and memorising a few lineages; it was never ablated.
2. Coverage gaps (VERIFIED) remain and cannot be fixed by weighting: fake a cappella singing / SVS / SVC / AI covers 0,
   real Korean accompanied songs 0 (by metadata), natural realV+fakeM 0 (P1's 112 are all constructed), commercially
   produced real music absent. The DB held-out cannot measure these domains (it predicted S8 wrongly: DB +0.0043, LB -0.0034).
3. Not established: that weighting or gaps explain the whole 0.07 DB-LB gap; S8's failure cause (raw-mix features vs FILE
   change vs interaction) stays confounded.

## Pre-registered candidates
- W (primary): voice strata KS/XS/SM/SG with target masses 0.25/0.45/0.05/0.25 in each class (soft targets), music global
  50:50; base max(n_cell,20)^-1/2 inside (class, stratum); duration balance inside strata, clip [0.5,2] (falls back to
  [0.1,10] for a stratum whose weighted duration-only AUC exceeds max(0.60, S7's value) - logged before training);
  final caps simultaneously: class mass exact, row <= 5x class mean, lineage component <= 2% of class mass; non-convergence
  = failure. Implemented: data_v4/weighting.py (self-check passes; dry run on the P1 rows: voice Kish rows real 6,962 /
  fake 15,311, strata balanced; music Kish groups real 229 / fake 616, max group 2%).
- C (control): S7 weights + the same caps.
- All heads retrained (DF voice + AD voice with rule L, WavLM-8 member, music on the S7 features); encoders, inputs, seeds,
  presence and voice fusion weights fixed; FILE coefficients fixed at 0/9/1 with the candidate CDF tables refit on val.
- D1 (data, prepared in parallel, trained only on top of the chosen weighting): 2x2 vocal/accompaniment remixes, donors
  assigned by lineage to train only, accompaniment donors only from no-vocal rows (VP=0), fake singing from clean SingMOS-Pro
  svs/svc with SingMOS gt in the same ratio, licence whitelist (no BY-SA with NC, no blank-licence legacy rows, no ND),
  -12/-6/0/+6 dB, every mix through the real runtime separation + features, WAVs and recipes kept for round 2.
- Conditional data: CtrSVDD Chinese sources with bona fide pairs only (ND: unmixed); AI-Hub 465/473/71544 only on deneb-kr
  (kr-west) AND only after the organisers confirm round-2 submission is compatible with AI-Hub's third-party restriction.

## Gates
Train-only (DB protocol) models decide first, then P1. Must pass: package integrity, zip <= 10 GB, runtime parity
(V 0.02 / M 0.01 / F 0.02), CUDA offline run, peak GPU memory <= 22.4 GiB absolute and <= S7 + 0.3 GiB, L4 estimate <= 45 min;
natural-source subset (no constructed/legacy remix/HAIM edits) dADS >= -0.005 vs S7, no component EER worse by > 0.02 there,
no natural real-source FPR up by > 5 pp. Superiority on the DB is reported, not required (best-ever scoring; the DB cannot
see the LB domains). New-domain generalisation stays UNVERIFIED unless held-out new data exist.

## Slots 2026-09-29 (B = the weighting actually submitted in slot 1)
1 (00:00): W (fallback C). 2: slot-1 LB - S7 >= +0.001 -> B + D1; in (-0.003, +0.001) -> the other of W/C + D1;
<= -0.003 -> S7 weights + D1. 3: fixed list - {W or C not yet used, AI-Hub variant if cleared}, last resort S8 with FILE 0/9/1.
Never S9. Slot-1 and all slot-2 branch packages verified by 23:30; the last upload starts before 09:00.
