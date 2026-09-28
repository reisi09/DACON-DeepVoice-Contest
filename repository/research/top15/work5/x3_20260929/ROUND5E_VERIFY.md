# Round 5e — fixes for your round-5d BLOCKING items (2026-09-29 00:43 KST). Same reviewer session. BLOCKING-only.

1. PGID: after_db_x3.sh now runs `setsid bash -c 'echo $$ > /root/x3/p1.pgid; export NEW_ROWS=...; exec /root/final_new.sh ...' &`,
   so the session leader writes its own PID (= new PGID) and execs P1; the log line prints the recorded PGID and the
   ps-reported PGID. This version replaced the previous one while it was still waiting (x1: one after_db process,
   pgid 35707; DB jobs not yet finished; /root/final_s7_sn.out did not exist at the swap).
2. Budget: bal() tries the VESSL billing API 3 times and otherwise returns failure; budget() then fails (fail-closed).
   Tested on this Mac: balance 12.23 -> budget 0 ok, budget 100 fails, missing CLI -> "balance unavailable" and fails.
   wait_for checks the budget before its first poll; E3, the locked scorer, the package chain and the CUDA check run
   under guarded(): background process, budget and deadline checked every 30 s, killed + x1 stopped on violation,
   exit code returned. The floor is the cumulative $14 cap (+$0.10); the CUDA check also needs $0.75 headroom.
3. Locked inputs: score_locked.py now requires LOCKED_ROWS to be byte-identical to the manifest committed and pushed
   before any metric (research/top15/work5/x2_20260928/locked_locked_rows.csv, commit 39542f1 2026-09-28 23:36 KST,
   sha256 dcae0c85...; copy here as locked_manifest_committed.csv). That manifest pins ID, source, labels, audio_sha256
   and the npz path of every row; each audio file must hash to its manifest value, and each npz sample count (16 kHz) must
   match its audio duration (checked alone, no metric: 770 rows, max offset 0.000 s).
   Parity: agreed — the old branch could have hidden a small difference. The fixed w_runtime_parity.py will be rerun for
   the already packaged candidates before the report; results will be stated as rerun evidence.

Question (Korean, BLOCKING first, concise, file:line): are the three round-5d BLOCKING items resolved? Any new error?
