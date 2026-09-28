# Round 5d — fixes for your round-5c BLOCKING items (2026-09-29 00:37 KST). Same reviewer session. BLOCKING-only.
(Correction: the round-5c brief said 00:35 KST; it was written at 00:24 KST.)

State: X3 rows validated (1,500 npz, all keys finite), relabel left 16 crops voice-unknown; the 4 DB jobs started
00:35 KST. Nothing has been scored; the locked set is untouched. after_db_x3.sh v2 replaced v1 while v1 was still waiting.

1. G6 (score_locked.py verdict(), main): a defined failure -> FAIL; else any undefined CI or layer with < 20 groups per
   class -> HOLD; PASS only when all criteria are defined and met; overall res['G6'] = FAIL > HOLD > PASS over references.
   x3_mac_chain.sh proceeds only on the exact string PASS (an empty read also stops).
2. Once only: /root/x3/LOCKED_SET_USED.json is created with O_CREAT|O_EXCL BEFORE any prediction, independent of the
   output name, never removed; it records candidate, per-heads-dir sha256 (all files), rows/lock sha256, the npz content
   hash chain and the output path. Each locked row's audio must match its locked audio_sha256. LOCK NONE refuses any
   locked ID or any npz under /feats/locked/.
3. Failures: after_db v2 prints DB_FAILED / EVAL_OK / EVAL_FAILED; run_x3_eval.sh deletes old gate outputs first; the Mac
   chain requires EVAL_OK, checks every exit code (downloads, score_e3, new_eval, E3 check, locked scorer, pkg_chain,
   CUDA verify + TREE_OK). pkg_chain.sh: the CPU verify no longer sits in an && list (set -e now stops on it).
4. Waits: 60 s polls with deadlines (gates 02:30, P1 03:05 KST) and the budget cap (VESSL billing balance must stay
   >= $4.85, i.e. cumulative <= $14); P1 success = exact '^done s7_sn HH:MM:SS$' + every head file present; P1 runs in its
   own session (setsid) and only its recorded PGID is killed; pause is retried and confirmed by workspace state 'stopped'.
5. VOICE fusion: score_locked.py and w_runtime_parity.py now skip fusion only when voice_silent (as Engine.predict).
   The earlier parity PASS results stand: the old branch could only have produced a large mismatch (FAIL), never a PASS.

Questions (Korean, BLOCKING first, concise, file:line):
1. Are the five round-5c BLOCKING items resolved? Any new error in these files?
