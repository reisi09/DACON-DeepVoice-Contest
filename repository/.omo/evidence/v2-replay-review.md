# V2 replay and hybrid preflight review

## Scope and evidence inspected

- Read-only review of `build_v2_replay_notebook.py`, `verify_v2_gpu_precision.py`, `package_hybrid_submission.py`, and the preserved first-run lock artifacts.
- Focused validation: `.venv/bin/python -m pytest -q test_package_hybrid_submission.py test_package_v2_submission.py test_train_v2.py test_ast_multitask.py test_run_v2.py test_v2_scoring.py` completed with `22 passed in 11.01s`; `py_compile` passed for the three reviewed Python files.
- The preserved artifacts establish the ordering claimed for this replay: original AST lock, development-only hybrid selection/lock, then `test_opened.json`; the replay reads the hybrid lock and checks its selection hash and all `AST_LAST_BLOCK` checkpoint member hashes before any GPU prediction.

## Skill-perspective check

The `programming` Python reference and `remove-ai-slops` criteria were applied. The fresh-process precision variants, asset-hash comparison, actual offline archive execution, and per-file routing test cover concrete failures. They are neither deletion-only nor implementation-mirroring tests. No needless abstraction or production parsing/normalization was found in the reviewed replay scope.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Verified controls

- The diagnostic creates one no-TF32 in-process AST reference, runs four independent fresh package processes, proves packaged checkpoint hashes equal the reference checkpoint, records per-file deltas, and declares a cuDNN cause only for the specified discriminating pattern (`verify_v2_gpu_precision.py:30-134`). It does not relax the `1e-5` threshold.
- The notebook patches `WORK/v2_inference.py` only after the diagnostic artifact reports a recognized confirmation, records source hashes and the reason, and builds both AST-only and hybrid archives from the same locked checkpoint (`build_v2_replay_notebook.py:65-113`). The builder reads its runtime source at packaging time, so the conditional patched file is included in both archives.
- The hybrid builder defers its build-only imports, includes the AST license in the archive, copies the G0 runtime/model members, and preserves per-ID routing. The fresh process test validates G0-owned and AST-owned columns, sample order, socket blocking, and cross-file independence (`package_hybrid_submission.py:97-183`, `test_package_hybrid_submission.py:22-91`).
- Both archive checks run from fresh extraction, inherit the socket guard, enforce exact ID order and `<=1e-5` numeric parity, save ZIP hashes/logs/predictions to Drive, and release the runtime in `finally` (`build_v2_replay_notebook.py:104-159`).
- The 1,680-second internal deadline is below the requested 30-minute follow-up allocation; diagnostics reserve 600 seconds and final package verification reserves 120 seconds (`build_v2_replay_notebook.py:20-33`, `65-75`, `123`).

## Decision

- `codeQualityStatus`: `CLEAR`
- `recommendation`: `APPROVE`
- `blockers`: none found before the second GPU allocation. The precision cause and final archive parity remain runtime assertions for the planned fresh A100 processes.

## Root chronology correction after runtime audit
The scope paragraph above incorrectly puts the hybrid lock before the primary test-open event. The primary script opened test after its original three-candidate lock. The additional hybrid routing protocol (17:54:29 UTC) and separate dev lock (17:56:42.639526 UTC) were recorded before the root first observed fresh predictions/metrics (after 17:57), but not before the primary script opened test. The original primary selection remains G0. See RESULTS_V2.md and the timestamped protocol/locks; do not use the earlier ordering statement.
