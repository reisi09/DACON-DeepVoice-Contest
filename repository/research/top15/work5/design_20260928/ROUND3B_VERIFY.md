# Round 3b — fixes for your round-3 BLOCKING items (2026-09-28 22:40 KST). Still no X1/X2 metric seen.

1. Panel lineage: bcast_rows.py v3 — clips use only their own speaker group (a cross-speaker top-up merged whole sources);
   speakers need >= 12 s of speech; panel donors skip every group present in any train row and every leak-flagged ID
   (integrity.json); parent_ids list every utterance used + the music donor; group_id = connected component over all speech
   groups and the music donor group; asserts: no component spans train and panel, no donor row is used by both; mix()
   returns None when an RMS is 0 and the pair is rejected. Result: 1,560 rows, all 12 cells full, BUT components: train 9,
   panel 3 — shared music donors (panel fake music: 96 files in few groups) link the pairs.
2. Consequence, decided now (before results): T2 (synthetic panel) becomes a point-estimate sanity check (dJ <= -0.02),
   because no group CI exists with 3 components. T1 (natural E2 OpenBMAT clean rows) is the only CI gate; its bootstrap
   unit is now the connected component of music tracks (< 20 components -> HOLD).
3. S gate implemented (per-seed paired delta, >= 2 of 3 seeds); N2 uses raking (IPF) with a marginal-error check (< 0.01);
   a slice that vanishes after filtering is HOLD; only 'sequential' (confirmed absent) is N/A; 'adoption' is pending the
   Mac E3 check.
4. Locked v2 (locked_v2.py, LOCK_v2_summary.json): 30 rows dropped whose audio sha256 equals a bank or new row; OpenBMAT
   regrouped by track components (159); OpenBMAT part labelled conditional (music-track separated only).
5. GPU runner v2 (run_x2_db.sh): per-ID npz validation (loadable, finite presence), worker exit codes with a 45-min
   limit, DB jobs under a 90-min limit, set -euo pipefail; stale bcast npz deleted (audio changed).

Questions (Korean, short, BLOCKING first): does anything above still block the DB training? Is the T2 point-only change
acceptable or should X2 then rest on T1 alone (and would you require anything else for T1)?
