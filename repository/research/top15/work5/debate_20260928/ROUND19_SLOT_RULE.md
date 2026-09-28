# Round 19 — slot 3 and the conditional rule (2026-09-28 17:27 KST). Both reviewers get the same file.

Round-18 answers: codex_round18.md (long), fresh18_r4.md (fresh). Accepted corrections:
- The pre-registered test compared DB-protocol head predictions (virtual_configs.py); P1 heads are what gets packaged.
- The presence back-fill is "a completion of the evaluation after seeing the partial result", not equal to a test that was
  complete from the start. The report will say so.
- "$0" means no additional GPU cost; CPU ~27 min = ~$0.135 so far (real bill UNKNOWN).

## 1. Remaining disagreement: slot 3 without a mix
- long: s7_new_ob if its CUDA check fits in the 0.5 H100-hour cap, else W_new.
- fresh: W_new (verified now, no extra cost or completion risk).
Facts (DB protocol, vs s7_ref; SUBMISSION_PLAN_20260929.md in this folder, section 3):
| | E1 | E2 unlinked | E2 open unlinked | E2 AT-ADD | E3 real FPR |
|---|---|---|---|---|---|
| s7_new | +0.0064 | +0.0285 | +0.0058 | +0.0349 | 0.027 |
| W_new | +0.0033 | +0.0442 | +0.0058 | +0.0562 | 0.040 |
| W_newB | +0.0032 | +0.0467 | +0.0071 | +0.0583 | 0.040 |
| s7_new_ob | +0.0049 | +0.0363 | +0.0049 | +0.0469 | not measured |
s7_new_ob: CPU verify ok, runtime parity PASS (V/M/F 0.00018/0.00028/0.0001), CUDA check not run (x1 was paused).
User requirement: each submission must give a meaningful number even if it scores low. W_new and W_newB differ only in
early stopping (offline gaps <= 0.0025), so W_new after W_newB measures almost nothing new; s7_new_ob vs s7_new isolates the
400 AI-OpenBMAT rows (part of its OpenBMAT gain was leakage; its E2-open is lower than s7_new's).

## 2. Conflict with the written plan (section 4.1 of SUBMISSION_PLAN)
The plan chooses slots 2 and 3 from the s7_new LB result (vs S7 ADS 0.8374):
>= 0.845 -> W_new, W_newB | 0.838-0.845 -> W_new, "winner's next step" | 0.834-0.838 -> W_newB, "decide later" |
< 0.834 -> "re-examine". Rounds 17-18 instead fixed s7_new -> W_newB -> (W_new | s7_new_ob | mix).
LB results must not change any recipe; choosing among already built and verified normal packages is allowed.

## 3. Questions (Korean, VERIFIED / INFERRED / UNKNOWN, short)
(a) Final slot-3 choice after reading the other reviewer's argument and the table. One line reason.
(b) One conditional table to replace section 4.1: for each s7_new LB band, slots 2 and 3 from
    {W_newB, W_new, s7_new_ob (if CUDA ok), mix (if it passes the completed test and verification)}. No new training.
(c) Anything in section 4.0 ("the single grader property") that the new order invalidates.
