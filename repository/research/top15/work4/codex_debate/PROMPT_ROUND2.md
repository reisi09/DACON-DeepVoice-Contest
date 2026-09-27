Round 2. Claude's response to your round 1 (new evidence in results/claude_round2_evidence.md):

ACCEPTED (report being corrected): S7 runtime not implemented (SSLAM/WavLM absent from full_bank_submission.py) and
fuse() silently drops missing members (latent defect; fix = raise when a weighted member is missing); "1,135 unseen
generators" -> 6 generators / 1,135 fake rows, DB rows = 4,822 of 6,450; HuBERT was in the B grid; E1 is a packaging
check, not a quality claim; the 4-of-5-seeds idea would be a post-hoc criterion (diagnostic only); FILE mean comparison
was a scale error (rank check A confirms no defect); the 0.09 gap cause ("LB more unfamiliar") is a hypothesis.

NEW: your leave-one-generator-out check (B): S6-S5 flips sign without lyria-3 pro (-0.00081) and is ~0 without
fx_encoder, while S6+both-S6 stays >= +0.0074 in every exclusion. Seed check (C): seed 42 genuinely worse.

REMAINING QUESTIONS. Give a final position on each (short, decisive, with the one fact that would change your mind):
Q1. Daily quota: unused 09-28 slots expire. Given B, is the 09-28 order S6 -> S5 -> S4 still right, or S5 first
    (S5-S4 is the robust +0.012; S6's L gain is fragile)? Should S4 be used at all when S5 dominates it on held-out?
Q2. S7 needs new runtime code tonight (EAT-family second model + WavLM member + fuse assertion), P1 training, packaging
    (~9.73 GB zip, lossless fp16 weights), CPU verification and train/runtime parity. The gain is robust across
    generators (B) but not across seeds (C) and fails the locked rule. Build it for the 09-29 slots, or stop?
Q3. If built, should the voice side use S6 heads (label completion L, fragile) or S5 heads (no L) under WavLM?
    Note: the RT numbers exist for both bases: S5+both vs S5 +0.00934, S6+both vs S6 +0.01044 (rt_s6enc_report.json).
Q4. Anything in Claude's round-2 evidence that is itself flawed?
Answer in Korean, under ~600 words.
