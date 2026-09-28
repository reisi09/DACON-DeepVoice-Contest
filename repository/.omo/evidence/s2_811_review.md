# S2 8:1:1 split review

## Scope and evidence inspected

- New split and evaluation sources: `prepare_s2_811.py`, `s2_evaluation_report.py`.
- Result artifacts: `results/s2_811_h100_20260916/split_manifest.json` and `README.md`.
- Sealed comparison source: `results/fullft8_h100x1_20260916/downloaded/batch_worker.py`, limited to `make_splits`, lineage checks, and `train_trial`.

No training, model inference, GPU action, UI interaction, or expanded628 artifact was used.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Verified methodology and code path

- `prepare_s2_811.py:24-35` pins the sealed worker SHA-256 and requires exactly one `len(ordered) * .2` occurrence before replacing it with `.1`. This is a single split-ratio change, not a copied or hand-reimplemented training path.
- The patched worker retains its connected-parent construction and `require_disjoint` enforcement. The manifest's actual partitions independently contain no shared ID, audio SHA-256, PCM SHA-256, or parent group: train/tune/test are 630/88/90 files and 49/6/6 parent groups.
- Source-pool accounting is complete: the original S0 train pool has 918 files, and `630 + 88 + 90 + 110 = 918`. The 110 excluded files are exactly AudioLDM records removed only from S2 train/tune, as required by sealed `make_splits`; the S2 test retains 12 AudioLDM files.
- The new test is a subset of the old S2 test: all 90 new-test IDs exist in the sealed prior `07_S2_EAT_fullft/test_predictions.json`. The evaluation comparison therefore uses the same fixed rows for old and new scores (`s2_evaluation_report.py:65-69`). The README correctly labels this as an exploratory post-inspection rerun, not an untouched final test.
- A fresh model is created by the unchanged worker at `worker.py:570`; its distinct output root/trial name is set in `prepare_s2_811.py:62-65`. No pre-existing `best.pt`, `last.pt`, or `complete.json` was present under the new local result artifact at review time. The worker can only resume from a checkpoint in its new output folder after validating its identity (`worker.py:541-552,583-593`).
- Tune selection stays separate from test evaluation: the sealed `train_trial` selects checkpoint epochs using tune Music EER/balanced BCE (`worker.py:619-626`) and seals the checkpoint before test prediction (`worker.py:639-656`).
- Bootstrap samples connected original-parent components, not individual clips. `s2_evaluation_report.py:13-27` unions compound `group_id` parents before resampling; the actual 90-file test has six resulting components (sizes 20, 20, 20, 11, 10, 9), matching the stated six-group limitation. The paired delta resamples these same components for both models (`s2_evaluation_report.py:73-86`).

## Evidence boundary

This review validates split construction and evaluation-method intent only. It does not validate the still-running H100 training, model quality, GPU reproducibility, official score, or the L4 1,200-file limit. The requested split itself has only six independent test groups and six real songs, so any completed estimate must retain the README's uncertainty and exploratory-status limits.

## Skill-perspective check

Ran: `omo:programming` (including the Python reference) and `omo:remove-ai-slops`, with the active Ponytail review perspective.

- No deletion-only, tautological, prompt-text, or implementation-mirroring tests were introduced.
- No needless production parsing, normalization, abstraction, untyped escape hatch, or validation layer was added to this bounded change. Reusing the sealed worker with one guarded ratio replacement is proportionate to the goal.

## Recommendation

`codeQualityStatus: CLEAR`  
`recommendation: APPROVE`  
`blockers: []`
