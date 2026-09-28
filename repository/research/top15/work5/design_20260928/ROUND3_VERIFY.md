# Round 3 — independent verification during implementation (2026-09-28 22:25 KST). Same reviewer session.

The user approved option B of DESIGN_KO.md (round2/): R1-R4, X1 diagnosis, locked set, X2 (broadcast low-SNR rows), one final
candidate. Files in this folder are the implementation. No X1/X2 metric has been computed yet. GPU (x1) is booting for the
feature dump only; DB training of the candidate starts after the dump.

Implemented:
- R1: fit_ft_fusion.diff (eer NaN on one class; Boot.delta drops undefined resamples, reports None instead of p_gt_0 = 0),
  test_boot_undefined.py (passes locally and on VESSL); S7-recipe E1 ADS 0.90921 still reproduces.
- R2: integrity_check.py -> integrity_summary.json (content sha256 of all 40,222 new rows, exact duplicates across splits,
  full lineage components with every OpenBMAT music track and every remix donor).
- X2 data: bcast_rows.py (1,200 train rows + 360 'bpanel' rows; donors: train split for train rows, bank test / newtest for
  the panel; temporal human+AI mixes 'concat'/'crossfade' removed from fake-music donors because a short excerpt may be human).
- Locked set: locked_rows.py (unused AT-ADD eval-pool files minus every used speaker/source group; unused AI-OpenBMAT
  excerpts minus every used music track). Change made before any metric: speech sources are NOT used for the OpenBMAT
  exclusion (they recur across AI and human excerpts; with them only 10 excerpts per class survive).
- X2 gates: x2_gates.py (pre-registered T-type gates of DESIGN_KO.md 5.2). One clarification made now, before results:
  a protected slice with no rows in any development set ('sequential') is reported N/A and does not block; a slice with rows
  but fewer than 20 groups per class is HOLD (blocks).
- GPU plan: run_x2_gpu.sh + jobs_x2.txt (DB protocol, same as the s7_new jobs + the broadcast rows); P1 afterwards with
  NEW_ROWS=new_rows.csv,bcast_rows.csv only if the gates pass.

Questions (Korean, VERIFIED / HYPOTHESIS / UNKNOWN, file:line; BLOCKING first):
1. Any defect that would make X2 labels wrong, leak evaluation rows into training (E2-OpenBMAT clean, newval, bpanel,
   locked set), or make x2_gates.py compute something other than the pre-registered gates?
2. Are the two clarifications (N/A rule; OpenBMAT exclusion by music track only) acceptable, or do they weaken the design?
3. Anything that must change BEFORE the DB training starts (about 30 minutes from now)?
