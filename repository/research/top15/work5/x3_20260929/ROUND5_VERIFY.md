# Round 5 — X3 (SONICS songs) code verification BEFORE running (2026-09-28 23:55 KST). Same reviewer session.

User asked for a new experiment: add real AI songs from SONICS (HF awsaf49/sonics, CC BY-NC 4.0, SONICS train split,
Suno chirp-v3.5 and Udio udio-120s, all with vocals) to the s7_new recipe, to improve FILE EER on songs/mixed audio.
Only the download has started (fake_songs.csv, part_01.zip, part_02.zip). Nothing else has run; no X3 result exists.

Files: sonics_rows.py (rows: 750 per algorithm, one song per SONICS lyrics id, no id already in the bank, 20 s crop at 30 %
via open_rows.crop, labels voice+music present and fake as the bank's 'sonics' rows), jobs_x3.txt (DB protocol = the
s7_new jobs + these rows as pool 'train'), run_x3_db.sh (the X2 v2 runner re-pointed), x2_gates.py MODE=x3 (added now,
before results): primary target T3 = FILE EER on E1 files with voice and music present, decrease >= 0.01 with group CI upper
< 0 (< 20 groups per class -> HOLD); T1 broadcast as a guard (dJ <= +0.02); N1, N2 (feasible grid), P (protected slices),
S (per-seed dFILE EER on the T3 slice, >= 2 of 3 negative), E3 on the Mac as before.

Questions (Korean, BLOCKING first, concise, file:line):
1. Any code error in sonics_rows.py / jobs_x3.txt / run_x3_db.sh / the x3 branches of x2_gates.py that would make labels,
   lineage, features or the gates wrong? (e.g. the bank id exclusion regex, crop, zip member lookup, the T3 slice)
2. Does anything leak evaluation rows (E1 songs, E2) into the X3 training rows?
3. Is the T3 gate a fair pre-registered target for this intervention? Anything to fix before running?
