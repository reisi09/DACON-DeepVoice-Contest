# Round 4 — verify the two rejections (2026-09-28 23:34 KST). Same reviewer session.

Results of the pre-registered gates (x2_gates.py in this folder = the final version after your round-3b requests: N2
feasibility filter, T1 per-class/max-share HOLD, regrouped N1):
- X1 (s7_new heads, FILE rule B = noisy_or only): x1_gates.json -> target FAIL (FPR 0.0179 -> 0.0136, CI [-0.016, +0.0018]),
  N1 FAIL, low-Pm (0.1-0.5) fake FILE FNR +0.155 FAIL. (X1 ran before the N2 feasibility fix; its N2 is not meaningful.)
- X2 (s7_new recipe + 1,200 broadcast rows, DB protocol, 4 members exit 0 - db_status.json): x2_gates.json -> T1 FAIL
  (dJ +0.058 worse, CI [-0.042, +0.094], 72 track components), T2 PASS (point -0.042), N1 FAIL (CI lower -0.0063), N2 FAIL
  (32 feasible points, min -0.008), S FAIL (all seeds worse), realV+fakeM FILE FNR +0.0116 FAIL; E3 (Mac, aggregate,
  e3_s7_new_vs_s7_bc_aggregate.json) unchanged.
- Decision: no candidate passes; slot 3 stays W_new (already verified); the locked set (LOCK_v2.json) is NOT scored, kept.

Questions (Korean, short): (1) Is any computation error visible that could flip either rejection? (2) Is keeping the
locked set unscored and reporting "no improvement" the correct reading? (3) Anything else to record for the report.
