# V2 code-quality preflight review

## Scope and evidence inspected

- Goal reviewed: ready the AST V2 train/select/infer/package path for the bounded A100 run while preserving file-independent inference, development-only selection, true component masks, and an executable offline package.
- Changed runtime files: `ast_multitask.py`, `train_v2.py`, `v2_scoring.py`, `v2_inference.py`, `package_v2_submission.py`, `run_v2.py`, and `build_v2_notebook.py`.
- Meaningful tests inspected: `test_ast_multitask.py`, `test_train_v2.py`, `test_v2_scoring.py`, `test_package_v2_submission.py`, and `test_run_v2.py`.
- Direct validation run: `.venv/bin/python -m pytest -q test_ast_multitask.py test_train_v2.py test_v2_scoring.py test_package_v2_submission.py test_run_v2.py` completed with `19 passed in 7.42s`.
- Existing evidence inspected: `.omo/evidence/v2/task-3-ast-gradients.txt`, `.omo/evidence/v2/task-5-evidence.json`, `.omo/evidence/v2/task-5-resume-parity.txt`, and `.omo/evidence/v2/task6_offline_package.txt`.
- Data files and actual A100 outputs were intentionally outside this preflight review because data acquisition is still in progress.

## Skill-perspective check

The `programming` Python reference and `remove-ai-slops` skill were loaded and applied before judging maintainability and test relevance. The meaningful gradient, mask, lock, resume, and offline-package tests are not deletion-only or tautological, and the production code has no material slop finding for needless parsing/normalization beyond the issue below. The diff does violate the programming perspective's coherent-boundary expectation: decoding/resampling is independently implemented in two production paths with different algorithms. No brittle prose/prompt test, implementation-mirroring test, untyped escape hatch, or needless abstraction was found in the reviewed scope.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Positive checks

- The five-head eligibility mask uses only true `VOICE_PRESENT==1` / `MUSIC_PRESENT==1` to expose component FAKE loss (`ast_multitask.py:111-120`), and the gradient test changes excluded labels while verifying their gradients are zero (`test_ast_multitask.py:47-71`).
- Trainability is limited to heads in frozen mode and heads plus final encoder block/final layer norm in last-block mode (`ast_multitask.py:75-100`); the real-small-AST test checks selected gradients and updates (`test_ast_multitask.py:22-45`).
- File-local tail coverage is deterministic and uses probability aggregation only within each file (`ast_multitask.py:160-200`), with a meaningful full-coverage/tail test (`test_ast_multitask.py:74-90`).
- Selection refuses any non-`dev_select` row (`v2_scoring.py:224-254`), then `run_v2.py:179-187` writes and verifies the candidate lock before reading the locked-test manifest. The lock test exercises altered weights/manifest handling (`test_run_v2.py:11-31`).
- The archive includes weights and runtime code, runs offline from a fresh extraction, preserves evaluator CSV order, and checks cross-file independence (`test_package_v2_submission.py:63-127`). This is useful evidence, but it does not cover sample-rate parity.
- The runtime mismatch is explicitly recorded by `run_v2.py:160-163` (environment receipt); the declared CPU `torch 2.7.1` / `transformers 4.57.6` versus Colab's pinned `transformers==4.57.6` is acceptable for this preflight, provided the final A100 receipt retains those versions. No runtime result was reviewed here.

## Decision

- **Re-review (2026-09-13): resolved.** `train_v2.py:19` now imports the packaged `v2_inference.load_audio` boundary rather than retaining an independent decoder/resampler. A direct 44.1 kHz WAV check now yielded equal `(16000,)` PCM arrays and maximum absolute difference `0.0`. `test_train_v2.py:85-98` uses a tiny real AST on a 44.1 kHz fixture and asserts train prediction equals deployment prediction to `1e-5`.
- **Re-review (2026-09-13): resolved.** `_trainable_state` now clones detached CPU tensors (`train_v2.py:126-128`), preventing the retained best epoch from aliasing later optimizer updates. `test_train_v2.py:73-82` verifies the snapshot stays unchanged after a real SGD step.
- Direct final validation: `.venv/bin/python -m pytest -q test_ast_multitask.py test_train_v2.py test_v2_scoring.py test_package_v2_submission.py test_run_v2.py` completed with `21 passed in 7.12s`.
- `codeQualityStatus`: `CLEAR`
- `recommendation`: `APPROVE`
- `blockers`: none in this preflight scope. Data acquisition and the actual A100 result remain outside this code-only review.
