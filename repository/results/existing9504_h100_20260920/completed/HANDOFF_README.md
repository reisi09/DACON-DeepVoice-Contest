# DACON_Existing9504_H100 H100 handoff

- status: **complete**
- built_at: 2026-09-19T21:36:51+00:00
- workspace: wsp-9683432r72m7; deployment: /shared/existing-only-9504-h100-20260920

## Stages (measured wall time)

| stage | status | seconds (all attempts) |
|---|---|---|
| environment | complete | 226 |
| features | complete | 192 |
| pilot | complete | 128 |
| preflight | complete | 13 |
| stage_inputs | complete | 1969 |
| train | complete | 235 |
| test_eval | complete | 36 |
| parity | complete | 39 |
| handoff | running | 0 |

Total measured stage time: 0h47m17s (target 16.0 h; reported, not enforced).

## Pilot (H100)

- MERT batch: 256
- projected feature extraction: 5.23 h over 9504 rows (154.7 h audio)
- pilot peak allocated VRAM: 25.1GiB of 79.2GiB

## val (masked known-label metrics; bootstrap CI over components)

| head | EER [95% CI] | AUC | n |
|---|---|---|---|
| file | 0.0799 [0.0383, 0.1761] | 0.9645 | 1426 |
| voice | 0.0080 n/a | 0.9998 | 746 |
| music | 0.0912 [0.0651, 0.1213] | 0.9651 | 680 |

## test (masked known-label metrics; bootstrap CI over components)

| head | EER [95% CI] | AUC | n |
|---|---|---|---|
| file | 0.0743 [0.0330, 0.1751] | 0.9668 | 1426 |
| voice | 0.0000 n/a | 1.0000 | 744 |
| music | 0.0909 [0.0634, 0.1113] | 0.9700 | 682 |

## Raw-audio parity: bitwise

## Caveats (carried from the dataset README)

- Not an independent test: historical development exposure val 997/1426, test 1001/1426.
- No official/L4 score claim; unknown labels are masked, never imputed.
- Singing (CtrSVDD, 2034 rows) is train-only; singing generalisation is not evaluated.
- Music source/length bias remains (metadata source-only AUC ~0.97 on music val/test).
- Checkpoints were selected on val only; test is scored only after candidate_lock.json (re-scoring is deterministic).
- No submit.zip is included; offline packaging parity must be proven later with parity/ samples.

## Layout

model/ checkpoints+lock+training evidence; predictions/ and metrics/ val/test; dataset/ manifest, deployment and staging receipts; code/ author_code.zip, source.zip, notebook; environment/ lock, env receipt, static ffmpeg; assets/assets.zip (frozen encoders); measurements/ pilot, VRAM, timings; logs/; parity/ audio+features+recomputed samples; SHA256SUMS.txt covers every member.
