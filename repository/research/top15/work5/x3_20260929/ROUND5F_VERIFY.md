# Round 5f — fixes for your round-5e BLOCKING items (2026-09-29 00:55 KST). Same reviewer session. BLOCKING-only.

State: DB jobs done (all exit 0); P1 s7_sn running since 00:46 KST (recorded PGID 35941 = ps PGID); x3 gates running;
the Mac chain (x3_mac_chain.sh, run from a copy) is waiting for EVAL_OK. The locked set has not been scored.

1. guarded(): deadline AND budget are checked before the start and every 30 s; CMD runs in its own session through a
   python os.setsid() shim (macOS has no setsid(1)); a violation sends TERM then KILL to the whole process group.
   Local tests: exit code 3 propagates; a deadline hit during `bash -c 'sleep 377 & sleep 377'` leaves 0 sleep processes;
   an expired deadline refuses to start the command.
2. Downloads (scp runs/models) now run under guarded() with ConnectTimeout; the billing query has a 30 s subprocess
   timeout (3 tries, then failure = stop). The locked scorer, package chain and CUDA check are guarded; the scorer
   cannot start after its deadline, so it cannot create the one-time record late.
3. Feature provenance: locked_recheck.py re-extracted a seeded sample of 30 locked rows with the same packaged Engine
   (dump_features.py, fresh output dir) and compared every stored npz key: exact keys equal, all float keys cosine
   1.000000, 0 mismatches (locked_recheck.json; no labels read, no metric). score_locked.py requires that record with
   result PASS and the manifest sha256. Remaining limit stated in the report: provenance is proven for the 30-row sample,
   the other 740 npz rest on the same extraction run and the duration check.

Question (Korean, BLOCKING first, concise, file:line): are the three round-5e BLOCKING items resolved? Any new error?
