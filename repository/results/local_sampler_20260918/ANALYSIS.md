# Local sampler comparison, 2026-09-18

The corrected sampler repairs a real coverage defect. It does **not** establish better held-out EER in this comparison. Continue using the correction for sampling correctness; keep any claim of better predictive quality separate.

## Fixed comparison

- Frozen MERT layer 12, mean of all valid feature windows; existing 1,322 NPZ records, with audio identity and frozen-asset identity checked.
- Original split retained: train 831, dev 164, calibration 200, test 127. Connected provenance components do not cross these splits.
- Mean/std normalization fitted only on train 831. No checkpoint, threshold, hyperparameter, or layer selection was performed on test (or any other split) by this comparison.
- Each arm: seeds 41/42/43; 500 updates; batch 32; 16 examples per class per batch; 1024→128 GELU→1 MLP; AdamW lr 0.001, weight decay 0.01; gradient norm clip 1.0. CPU, 2 threads, PyTorch 2.7.1.
- Legacy behavior comes from the preserved original source. The candidate uses the corrected shared `eight_heads.balanced_draws` API. Only draw scheduling differs between the two local training arms.
- The dev/test sets were already inspected in previous work. These are historical held-out diagnostics, not a fresh independent validation set and not official leaderboard scores.

## Exposure and training diagnostic

For every seed, legacy training visited 789/831 examples and permanently omitted 42 FMA REAL examples. Corrected training visited all 831 examples in each of five consecutive 100-update phases. Both arms retained exactly 1,600 REAL and 1,600 FAKE draws per phase.

Across all 500 updates, each FMA REAL example received 0 or 95–96 legacy draws versus 47–48 corrected draws. FMA/FMA examples received 72–73 draws in both arms. This confirms the change distributes the same source exposure over all eligible parents.

At the diagnostic threshold 0.5, the probability ensemble's training REAL false positives fell from 18/139 to 0/139. All 18 original mistakes belonged to the 42 omitted examples. This is an in-sample effect and is not evidence of generalization.

## EER results

High score means FAKE. EER uses the existing full-ROC nearest-point implementation. Lower is better.

| Evaluation | Legacy | Corrected | Corrected minus legacy |
|---|---:|---:|---:|
| Dev probability-ensemble EER | 0.15913669064748198 | 0.20071942446043162 | +0.041582733812949635 |
| Test probability-ensemble EER | 0.09155518394648826 | 0.09155518394648826 | 0 |
| Dev mean of seed EERs | 0.16153477218225415 | 0.21338129496402874 | +0.05184652278177459 |
| Test mean of seed EERs | 0.08995261984392416 | 0.10200668896321068 | +0.01205406911928652 |

Seed EERs (41, 42, 43):

- Dev legacy: 0.15913669064748198, 0.16273381294964023, 0.16273381294964023.
- Dev corrected: 0.238705035971223, 0.20071942446043162, 0.20071942446043162.
- Test legacy: 0.09636287625418055, 0.08674749163879597, 0.08674749163879597.
- Test corrected: 0.08674749163879597, 0.09636287625418055, 0.12290969899665549.

Probability averaging and averaging seed EERs are different operations. The ensemble EER rows are computed directly from mean probabilities.

## Uncertainty and REAL diagnostics

The paired bootstrap samples provenance-connected groups for the **fixed trained probability ensembles**; it does not retrain models and excludes training-seed and historical layer-selection uncertainty.

| Split | Components | Observed EER difference | Percentile 95% interval | Invalid resamples |
|---|---:|---:|---|---:|
| Dev | 27 | +0.041582733812949635 | [-0.0055555555555555376, 0.07857998720031481] | 0/1,000 |
| Test | 24 | 0 | [-0.09945820951331354, 0.1008583902310924] | 0/1,000 |

Both intervals include zero. The point estimate is worse on dev and unchanged on test; the available group sample does not establish a clear improvement or degradation of ensemble EER.

REAL false positives at the uncalibrated diagnostic threshold 0.5:

| Split/source | REAL examples | Legacy errors | Corrected errors |
|---|---:|---:|---:|
| Dev, FMA | 10 | 6 | 5 |
| Dev, FMA/FMA | 15 | 7 | 5 |
| Test, FMA | 14 | 2 | 2 |
| Test, FMA/FMA | 9 | 4 | 3 |

Fewer false positives at one fixed threshold can coexist with a worse EER because EER depends on both classes across score thresholds. Source slices containing only one class have null EER; their denominators and threshold diagnostics remain available in `comparison.json`. Calibration outcomes were only recorded diagnostically and were not used to tune either arm.

## Fidelity checks and artifacts

- All three legacy draw hashes exactly match their original saved E6 checkpoints.
- Instrumented legacy training at seed 41 gives bitwise-identical probabilities to the original `legacy.fit_head` CPU implementation.
- Replaying the original saved E6 checkpoints on CPU differs from stored test predictions by at most 5.364418029785156e-7.
- Local legacy retraining differs from saved-model CPU replay by at most 1.2516975402832031e-6 across all examples and seeds. This small historical runtime variation does not explain the corrected arm's dev change.
- `comparison_lock.json`: exact input/source hashes, optimizer and normalization contract, runtime.
- `input_lineage.json`: every feature NPZ hash, audio hash, split, and connected component.
- `normalization.npz`: train-only mean/std and fitted IDs.
- `exposure.csv`: draw count for every train example and every 100-update phase, including zeros.
- `losses.csv`: BCE and gradient norm for every update of every arm/seed.
- `predictions.csv`: all row-aligned probabilities, ensembles, and historical replay probabilities.
- `*_schedule.npz`, `*_seed*.pt`: exact draw indices and trained models.
- `comparison.json`: per-seed and ensemble metrics, source slices, phase summaries, and bootstrap validity.
- `legacy_eight_heads.py`, `corrected_eight_heads.py`, `runner_source.py`: source snapshots matching the locked source hashes.

No GPU was started for this experiment. No official score or production-ready quality improvement is claimed.
