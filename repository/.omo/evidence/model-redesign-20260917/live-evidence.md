# DACON 236749 live evidence — 2026-09-17

Read-only capture around 02:01 KST. Sources are official pages; submissions were read in the authenticated in-app browser. The original user tab stayed on `mysubmission`; a separate research tab was used. No upload, training, or external screen manipulation was performed.

## Target and measured gap

The user set the ultimate goal to **official leaderboard rank 1**. This document supports a design milestone; it does not claim that rank has been achieved or authorize new compute expenditure.

| Item | Score | ADS | CPS |
|---|---:|---:|---:|
| Current leader, team `55`, rank 1 | 0.88450 | 0.87192 | 0.99765 |
| Our team `killerpjw`, rank 175 | 0.73964 displayed | 0.71189 | 0.98939 |

[Official leaderboard](https://dacon.io/competitions/official/236749/leaderboard). Captures: `leaderboard-all-dom.txt`, `leaderboard-dom.txt`, `leaderboard.png`. Displayed values are rounded/truncated; use authenticated submission values for exact personal scores. Rank and leader score can change.

[Authenticated submission history](https://dacon.io/competitions/official/236749/mysubmission), capture `submissions-dom.txt`:

| ZIP | Official score |
|---|---:|
| M330_MLP_full1206_submit.zip | 0.7396465503 |
| C08_submit.zip | 0.7314822646 |
| baseline_submit.zip | 0.6909110688 |
| C3_full_clean_submission.zip | 0.6843536931 |
| V5_B1_fixed.zip | 0.6776485873 |
| V6_final_submission_fixed.zip | 0.6727822646 |
| DACON_236749_AST_epoch17.zip | 0.6453405503 |
| submit.zip (2026-09-16 23:06:28) — user identifies it as S2 | 0.6378679788 |
| DACON_Sumbit_1.zip | 0.6174819841 |

Arithmetic is recorded in `gap-calculation.json`. Best-to-leader gap is 0.1448534497 using the displayed leader score. Perfecting current CPS alone adds at most 0.001061. The displayed ADS difference contributes approximately 0.144027, and CPS difference 0.000826; approximately 99.4% of the displayed gap is ADS. Matching the present leader with unchanged CPS requires ADS approximately 0.872846. Relative to the leader's displayed ADS, weighted EER must fall about 55.5%. This is an ambition calculation, not a predicted improvement. ADS does not identify the individual File/Voice/Music EERs; those leaderboard components are unavailable here.

## Official metric and runtime constraints

[Evaluation specification](https://dacon.io/competitions/official/236749/overview/evaluation), capture `evaluation-dom.txt`:

`Score = .45(1−FileEER) + .18(1−VoiceEER) + .27(1−MusicEER) + .05 VoicePresenceAUC + .05 MusicPresenceAUC`.

FAKE is positive. EER uses full ROC points, nearest absolute FPR/FNR difference, then their mean. Component EER uses ground-truth presence. Public covers 100% of test and final Private equals closing Public. Top 15 advance to a separate review; leaderboard rank 1 and final award rank are distinct.

Inference: L4 22.4 GiB VRAM, 6 vCPU, 28 GB RAM, Python 3.11.15, CUDA 12.8; offline after installation. ZIP ≤10 GB, unpacked ≤32 GB, installation ≤10 minutes, inference ≤60 minutes. Across 1,200 files the hard average budget is 3 seconds/file; proposed 50-minute gate gives 2.5 seconds/file, both including overhead. These are arithmetic bounds, not measured throughput. Use the official preinstalled package versions where possible. The page inconsistently mentions `open/` once after documenting `data/`; retain the already working input-discovery path and verify packaged inference rather than treating that stray wording as a new path requirement.

## Evaluation distribution and legal method boundaries

[Task description](https://dacon.io/competitions/official/236749/overview/description), capture `description-dom.txt`: 1,200 files, 4–60 seconds, all 16 kHz, mono or stereo, multiple containers/codecs and some telephone audio. Voice includes speech and isolated vocals. Music means accompaniment without vocals; songs with singing are mixed. Mixed can be simultaneous or sequential. A generated component makes the file FAKE; ordinary non-generative enhancement, denoising or gain adjustment does not.

Therefore no evidence supports blaming S2's score on removal of evaluator frequencies above 8 kHz. Native-separator resampling and stereo handling may still change predictions, but their causal score effects are unmeasured. Whole-song authenticity is not automatically trustworthy component authenticity when generation provenance is unclear.

[Rules](https://dacon.io/competitions/official/236749/overview/rules), capture `rules-dom.txt`: qualifying public external data, models and APIs are permitted, with participant-verified licenses/terms and source disclosure. Evaluation-data training, pseudo-labeling and model updates are prohibited. Each file's prediction must be independent of all other evaluation files; segment aggregation inside the same file is permitted. This excludes cross-file score normalization, transductive adaptation and cohort-based calibration. Fixed calibration learned from allowed development data remains a candidate.

Maximum 3 submissions/day. Python. Second-stage deliverables include reproducible training code, two HWP reports and all training data files; preserve provenance during development. The page lists leaderboard close September 29 and second-stage materials October 5; exact closing time has not been audited here.

## Unknowns that constrain the plan

- Hidden test labels, component EER breakdown, generating systems, exact mixture proportions and duration distribution.
- Rank-1 model architecture and training corpus. No claim that our proposed components reproduce the leader.
- End-to-end L4 runtime for the proposed redesign; H100 extraction time cannot establish it.
- S2 submission association is the user's explicit statement; the submission page itself shows `submit.zip` at 2026-09-16 23:06:28.

These unknowns require fully labeled, source-disjoint local validation and bounded ablations. They must not be filled with inferred evaluator labels or cross-file statistics.
