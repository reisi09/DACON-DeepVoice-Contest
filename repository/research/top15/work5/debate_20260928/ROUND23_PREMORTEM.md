# Round 23 — pre-mortem for the 2026-09-29 submissions (2026-09-28 17:43 KST). Both reviewers get the same file.

The plan is signed off (FINAL_REPORT_DRAFT.md v3). Round 17 left one item UNKNOWN: L4 60 min / 22.4 GiB compliance.
Facts (from the local receipts, 17:42 KST):
| package | zip bytes | CUDA offline check (H100) |
|---|---|---|
| S7 (scored on the LB on 09-28, same encoders and runtime code) | 9,735,068,819 | n/a (LB scored) |
| s7_new | 9,745,265,234 | PASS |
| W_newB | 9,744,331,906 | PASS |
| W_new | 9,745,263,696 | PASS |
| s7_new_ob | 9,745,305,856 | not run |
All four new zips were re-hashed on disk at 17:35 and equal their receipts. Only head weights / tables differ from S7
(19 changed members per the package chain logs); encoders and Engine code are S7's. Mac CPU verification of s7_new took 273 s (QA set).
Schedule: slot 1 upload at 00:00, slot 2 after slot 1 is scored, slot 3 upload starts before 09:00; deadline 10:00.

Questions (Korean, VERIFIED / INFERRED / UNKNOWN, short):
(a) The top 3 failure modes that would waste a slot or block a submission tomorrow, each with the cheapest check or
    mitigation that needs no GPU and no new model. Include whether the L4 time / memory limit is still a real risk
    for heads-only changes, given S7 was scored.
(b) Is anything BLOCKING before the report goes to the user?
