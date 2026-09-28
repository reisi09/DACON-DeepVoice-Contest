# Round 26 — decide tonight with one more piece of evidence (2026-09-28 18:22 KST). Both reviewers get the same file.

Round 25: gaps agreed (real voice + fake music: train 5 rows; sequential speech<->music: 0; partial fakes: 121).
Tonight: long = conditional YES (one W_newB-recipe candidate + 1,600 contrast mixes, <= 4 H100-h, $11.92 cap, DB decision
by 23:00, verified package by 06:30); fresh = 0 GPU.

## New evidence (VERIFIED from research/top15/work5/newdata/new_eval_newtest.json, E2_by_source/openbmat, 161 rows)
AI-OpenBMAT = broadcast audio: real speech with human or AI music (the "real voice + fake music" and sequential cases).
| model | FILE EER | FILE real FPR | MUSIC EER | MUSIC real FPR |
|---|---|---|---|---|
| s7_ref | 0.392 | 0.378 | 0.366 | 0.333 |
| s7_new (600 OpenBMAT rows in training) | 0.336 | 0.200 | 0.336 | 0.144 |
| W_new | 0.354 | 0.300 | 0.311 | 0.122 |
| W_newB | 0.354 | 0.311 | 0.311 | 0.156 |
newval OpenBMAT (94 rows): MUSIC EER 0.291 for both s7_new and W_new. 65 of the 161 E2 rows share music tracks with training.
Every other E2 slice is far better (E2 all: FILE EER 0.067-0.084). This is the weakest domain we can measure, and 600
natural rows did not fix it. The LB share of such files is UNKNOWN; the official types include mixed audio "동시에 또는
순차적으로".
Raw audio: new rows live on the VESSL volume (/root/newdata/...); S7 bank audio is on VESSL too (paths audio/xx/<hash>).

## Questions (Korean, VERIFIED / INFERRED / UNKNOWN, short)
(a) With this evidence, final call for tonight: YES (one candidate) or NO. One-line reason tied to the top-15 objective.
(b) If YES, the single candidate, fully specified: base recipe (s7_new or W_newB), data added (contrast mixes realV+fakeM
    with matched realV+realM controls, sequential speech<->music, more AI-OpenBMAT crops with track-level lineage; sizes;
    donors that are instrumental-only; labels; split rules), the pre-registered test (which clean rows: E2-OpenBMAT unlinked,
    newval OpenBMAT, a held-out-donor contrast panel; thresholds incl. vetoes on E1 / E2-unlinked), timeline with
    checkpoints, GPU cap, and stop rules. No second candidate.
(c) If NO, what exactly the report to the user should recommend for after the competition.
