# Round 2 — consolidated review (2026-09-28 21:40 KST). Same reviewer session.

New evidence in this folder (round2/): code_audit.md (a separate read-only Claude coding session: findings F1-F13),
_scale_check.py (reproduced by the coordinator: E1 ADS 0.90921; no-music real files Fm q90 0.469 vs fake 0.035; real 36.1 %
of max_nopresence decided by the music head; oracle Fm=0 on no-music files: E1 FILE EER 0.0741 -> 0.0716 at 0/9/1),
e1_slices.txt (E1 weakness map), ledger.md, runtime_parity_*_rerun.txt, and DESIGN_KO.md (the coordinator's draft plan).
You may now read the coordinator's plan. Answer ONLY material issues that could change the design, the acceptance decision,
or execution safety. Korean, VERIFIED / REPORTED / HYPOTHESIS / ESTIMATE / UNKNOWN / NOT RUN, file:line.

1. F2 (code_audit.md F2; decision_heads.py:70-98): defect or intended behaviour? Are the three pre-registered FILE rules of
   DESIGN_KO.md R3/X1 the right set? Fitting/selection on the DB-protocol newval (not used by s7_new DB models): valid?
   A Pm-gated rule may hide quiet music in broadcast audio (the weakest slice): which guard slices and thresholds must be
   fixed before running X1?
2. Experiment ranking X1 > X2 and gates G1-G7 (DESIGN_KO.md 5.1): change anything BEFORE experiments? Is any gate
   unmeasurable with the current data (e.g. realV+fakeM has 0 rows in newval and E2)?
3. Locked set (DESIGN_KO.md 4.4 / 9): unused AI-OpenBMAT excerpts filtered by every used music track + unused archive
   rows. Adequate for G6 (non-inferiority vs the S7 and s7_new packages)? Any flaw?
4. Any BLOCKING safety or validity issue in the plan as written.
