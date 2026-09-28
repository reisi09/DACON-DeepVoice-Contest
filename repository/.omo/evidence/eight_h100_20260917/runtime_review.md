# Runtime and correctness review: Eight H100 experiment bundle

## Review basis

- Scope reviewed: all current untracked `eight_*.py`, `build_eight_h100_notebook.py`, generated `DACON_Eight_H100.ipynb`, and `eight_h100_run.sh`.
- The working tree has no baseline diff for these new files, so review is against the live source, not a claimed executor diff.
- Source hashes at review: `eight_lora_train.py` `843d130052eaf6857b48524f9e9a73c8b654e9a78603a3e32a2cd3c76cfa4940`; `eight_heads.py` `4437eac384ed62aaa37ee409f30a58afdcb845383ada4f1f7071293bed370074`; `eight_h100_run.sh` `e7b2e1e15e19cc91a39d34235d7b74faa9823e36dda62ab235fbcf2f3ea736e7`.
- Contract reviewed: `.omo/plans/eight-h100-20260917.md` and this attempt's design/orchestration evidence. The plan has no active ULW goal, so this report uses the required fallback evidence directory.
- Checks run after the fixes: `.venv/bin/python -m pytest -q test_eight_features.py test_eight_fixed.py test_eight_lora.py` (`17 passed`); `bash -n eight_h100_run.sh`; notebook build plus `nbformat.validate` (`18` cells); bounded sampler repro with 20 source strata/class (all strata reached). These are fixture/static checks only. They do not prove H100 forward passes, actual features, labels, metrics, upload/download, stop, or bootstrap restoration.

## Skill-perspective check

The `omo:programming` and `omo:remove-ai-slops` skill instructions were loaded before judging test relevance and maintainability. The focused pass found no deletion-only test, requested-removal test, tautology, hard-coded implementation-constant-only test, brittle prompt test, untyped escape hatch, or unnecessary production parsing/normalization introduced solely for the tests. The tests do contain useful numerical, provenance, train-only-density, selection-lock, and LoRA-gradient/checkpoint seams. They do not cover the release-blocking state transitions below; that is a missing meaningful integration check, not a reason to add shallow tests. The source does not violate either skill perspective in a release-blocking way.

## Findings

### CRITICAL

None.

### HIGH / P1

None remaining. The three initial P1 findings were rechecked and fixed in the current source:

1. [`eight_lora_train.py`](../../../eight_lora_train.py#L128) captures an explicit pristine adapter-plus-classifier state, restores and checks it before frozen calibration/dev (lines 144-147) and again before frozen test (lines 192-197). The updated LoRA test mutates the classifier and requires exact restored logits.
2. [`eight_heads.py`](../../../eight_heads.py#L46) advances a global ordinal across batches. The added source-strata test covers the prior >16-strata starvation case; the bounded 20-strata/class repro reached every source.
3. [`eight_h100_run.sh`](../../../eight_h100_run.sh#L34) now checks partial-archive CRC, member set, manifest identity, and every member digest before setting `archive_verified: true`.

### MEDIUM / P2

None remaining. The sampler now emits `class_<label>|variant:<name>` (and corresponding source/generator) counts at [`eight_heads.py`](../../../eight_heads.py#L172), and E3 writes those receipts alongside the shared draw SHA at [`eight_ablation.py`](../../../eight_ablation.py#L166). The observed combined 5,334/5,332/5,334 view totals are expected when the same class-balanced ordinal is used for both labels; per-class balance and arm identity are the relevant controls.

### LOW

None.

## Verified non-findings

- The AntiDeepfake path uses the model's native `forward`, which applies its layer normalization; E7's teacher cache binding records source audio SHA, start, common zero-padded window, model SHA, and preprocessing. The head-only control restores adapter-plus-classifier state per seed, so it is not a stale-head reuse inside its training loop.
- `RealOnlyGaussian` rejects fake labels on fit, and E8 fits from train REAL rows before calibration thresholding; its high-distance-is-fake polarity is explicit.
- The notebook has test-after-lock ordering at the orchestration layer, and the ordinary archive path validates ZIP CRC and declared member hashes.
- E3 reports a shared CE draw SHA and per-class × variant/source/generator counts for each arm/seed; both arms use the same schedule.
- The 384-bin E4 residual subset is treated as an intentional predeclared primary feature in the evidence ledger, not reported here as a defect.

## Decision

- `codeQualityStatus`: **WATCH**
- `recommendation`: **APPROVE**
- `blockers`: None remaining in the reviewed code. Do not treat this as H100 execution evidence; require the planned real preflight and archive/stop receipts before starting or declaring the run complete.
