# V5 measured comparison

Local development; incomplete labels; not a leaderboard score

Execution: partial_preparation_timeout. Missing arms are not experimental results.

| Arm | Status | FILE EER | VOICE EER | MUSIC EER | VP AUC | MP AUC |
|---|---|---:|---:|---:|---:|---:|
| B0 | evaluated | 0.378175 | 0.150000 | 0.542222 | 0.997917 | 0.985132 |
| B1 | evaluated_budget_censored | 0.321825 | 0.170833 | 0.368333 | 0.998333 | 0.985855 |
| B2 | evaluated_budget_censored | 0.378175 | 0.150000 | 0.457778 | 0.969583 | 0.593553 |
| B3 | not_run_preparation_timeout | — | — | — | — | — |
| B4 | not_run_preparation_timeout | — | — | — | — | — |
| B4_joint | not_recorded | — | — | — | — | — |

EER: lower is better. AUC: higher is better.

The 230-row dev panel contains 210 unique audio files. Duplicate splice rows retain their original selection weight; analysis.json includes unique-audio metrics and identical-input prediction deltas.

Expert targets were 60 epochs. Budget-censored results describe only the actual completed epochs below. Checkpoints were selected among evaluated epochs, not every completed epoch. B1/B2 have different exposure counts, so their difference is not an isolated augmentation effect.

| Job | Completed epochs | Updates | Selected epoch |
|---|---:|---:|---:|
| B1_music | 8 | 552 | 5 |
| B1_presence | 9 | 846 | 5 |
| B1_voice | 5 | 250 | 4 |
| B2_music | 5 | 345 | 1 |
| B2_presence | 4 | 376 | 1 |
| B2_voice | 6 | 300 | 4 |
