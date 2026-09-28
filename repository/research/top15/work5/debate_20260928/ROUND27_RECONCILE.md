# Round 27 — reconcile into one plan, or stop (2026-09-28 18:25 KST). Both reviewers get the same file.

Round 26: both said YES to one candidate (long: s7_new base, <= 1,320 rows; fresh: W_newB base, 1,600 rows, 20% loss share).
Before the user is asked to approve GPU, check the YES against these facts (VERIFIED by the main agent from
research/top15/work5/newdata/rows/new_rows.csv and new_eval_newtest.json):

1. The "train 5 rows" gap is already partly filled in today's packages: D1 remixes (636, all train, in every P1 and DB run
   with NEW_TRAIN) = realV+fakeM 156 | realV+realM 121 | fakeV+realM 140 | fakeV+fakeM 219.
2. AI-OpenBMAT in DB training: train 345 (real music 164 / AI music 181), newval 94, newtest 161; VOICE labels are empty
   (masked). s7_new_ob (= s7_new + 400 more OpenBMAT rows) got E2-open-unlinked +0.0049 vs s7_new +0.0058 (vs s7_ref), and
   part of its OpenBMAT gain came from shared tracks (leakage). Its CUDA check is not run.
3. So with 156 realV+fakeM remixes + 345 OpenBMAT rows, OpenBMAT FILE EER is still 0.336-0.354 (MUSIC 0.311-0.336).
   The AI-OpenBMAT paper itself says detectors fail when short music excerpts are masked by dominant speech (low SNR).
   Hypothesis: the failure is the broadcast condition (music level far below speech, short excerpts), not row count.
4. AI-OpenBMAT has 3,294 one-minute excerpts in total (CC BY 4.0, Zenodo 20031232, the zip is on the VESSL volume);
   we used 600 (+400 in s7_new_ob). FakeMusicCaps audio is CC BY-NC 4.0 (Zenodo 15063698), 27,605 x 10 s.

## Questions (Korean, VERIFIED / INFERRED / UNKNOWN, short)
(a) Does your YES survive these facts? If the lever is the broadcast condition, which single data change targets it
    (e.g. mixes at music-to-speech ratios of -20/-15/-10 dB with short music excerpts; more AI-OpenBMAT with track-level
    lineage), and is there evidence it would move OpenBMAT EER when 156 remixes + 345/745 OpenBMAT rows did not?
(b) If still YES: ONE merged spec (base recipe, rows, SNR/duration grid, lineage, labels, tests + thresholds, timeline,
    GPU cap). If NO: say NO and what the report should recommend. Do not give a second candidate.
