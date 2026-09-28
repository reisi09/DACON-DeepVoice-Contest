# Round 5c — locked-set scorer + X3 chain, BEFORE any use (2026-09-29 00:35 KST). Same reviewer session. BLOCKING-only.

The locked set (LOCK_v2, 770 rows: AT-ADD speech 150/150, singing fake-only 100, music 100/100, OpenBMAT 100/100 with
159 track components; minus 30 hash duplicates) has never been scored. If X3 passes its gates and E3, the design
(design_locked_excerpt.md: 4.4, 5.1, 5.3 G6, 9.4) scores it ONCE with the P1 heads of X3, S7 and s7_new.

Files (round5c/):
- score_locked.py: verifies ids/labels sha256 against LOCK_v2.json, refuses to run if the output exists, applies each
  package's P1 heads to the packaged-Engine npz (dump_features.py) exactly as w_runtime_parity.py does (fused DF/AD/WavLM
  voice with voice_silent -> 0, WindowMIL music over mert+eat+sslam, file_score on the npz runtime presence). G6 per
  reference: paired group bootstrap (2000, seed 0) dADS CI lower >= -0.002 and every source x component layer with two
  classes EER worsening CI upper <= 0.015; a layer with < 20 groups in a class, or any undefined CI -> 'regression check
  only'. Tested only on 132 E2 dev rows (LOCK NONE; sanity_run_E2_132rows.txt): s7_new P1 was trained on those rows, S7 not.
- after_db_x3.sh (x1): after the DB jobs, P1 s7_sn starts at once (killed on rejection) and the x3 gates run alongside.
- x3_mac_chain.sh (Mac): gates -> E3 on the Mac (aggregate FPR increase <= 0.01 over sources, FILE and VOICE) -> P1 ->
  locked set once -> package (pkg_chain.sh) -> CUDA check -> pause x1; any rejection pauses x1.

Questions (Korean, BLOCKING first, concise, file:line):
1. Any code error in score_locked.py that would make locked-set predictions or the G6 verdict wrong (vs
   w_runtime_parity.py / the design)? Does anything allow scoring the locked set more than once or before the candidate
   is fixed?
2. Any ordering/stop error in the two chain scripts (e.g. a wait that never ends, a rejection that does not stop x1)?
