# V2 orchestration preflight review

## Scope and evidence inspected

- Read-only review of `run_v2.py` and `build_v2_notebook.py`, including their direct package/lock/baseline APIs and the local Drive-input archives.
- `v2_validation/drive_inputs.json` names `v2_ast_assets.zip` and `v2_validation.zip`; every listed local part hash and byte count matched. Reconstructed archive SHA-256 values also match the JSON: AST `54c227b873cc3c424579fa06e8a06c442b91663de5b740b182f7cac30a3fac75`, validation `f08eef8f309fab3a176a89fe34fffeaf687fc99ffe4f16c60157addaf9796b9e`.
- Actual ZIP layouts match notebook destinations: AST archive root is `ast/` with `config.json`, `preprocessor_config.json`, and `model.safetensors`; validation archive root is `v2_validation/` with development and locked-test manifests plus PCM index. These resolve to `WORK/ast` and `WORK/v2_validation`, which are the paths passed to `run_v2.py`.
- The baseline SHA in `build_v2_notebook.py:92` matches local `baseline_original/baseline_submit.zip`: `cc86f143915c95fd958a0066cbf14b0db9f4c41fd213310c3650b9fe870d0c70`.
- Validation run: `.venv/bin/python -m pytest -q test_run_v2.py test_package_v2_submission.py test_train_v2.py test_ast_multitask.py test_v2_scoring.py` completed with `21 passed in 7.84s`; `py_compile` passed for both reviewed files.

## Skill-perspective check

The `programming` Python reference and `remove-ai-slops` criteria were applied. The lock, fresh-package execution, file-local prediction, and non-16 kHz parity tests protect concrete failure modes; they are not deletion-only, tautological, or implementation-mirroring tests. No untyped escape hatch, prompt test, unnecessary production normalization, or needless abstraction was found in the reviewed orchestration path.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Verified orchestration properties

- The notebook sets `HARD_DEADLINE = SESSION_START + 7080` (`build_v2_notebook.py:30-47`), below the 7,200-second allocation cap, starts a watchdog that releases the runtime, and has an independent `finally` release path (`build_v2_notebook.py:120-128`). The subprocess is terminated 60 seconds before that deadline (`build_v2_notebook.py:103-115`).
- The runner computes development predictions and trains candidates before selection (`run_v2.py:165-178`). It writes `candidate_lock.json` before `open_locked_test()` parses the locked manifest (`run_v2.py:179-186`). The lock test covers absent lock, changed selection record, and changed manifest (`test_run_v2.py:11-31`).
- Fresh comparison runs after the lock; the comparison JSON is written before package replay (`run_v2.py:187-203`). Package replay builds a new ZIP, extracts it to a fresh directory, invokes the packaged script through `verify_packaged`, checks exact IDs and `1e-5` probability parity, and records the receipt (`run_v2.py:128-146`).
- The notebook verifies each Drive part before concatenation, verifies the final archive before extraction, verifies the official baseline hash, and passes only the resolved archive layouts to the runner (`build_v2_notebook.py:73-102`).

## Decision

- `codeQualityStatus`: `CLEAR`
- `recommendation`: `APPROVE`
- `blockers`: none in the requested code-only orchestration preflight. The actual A100 runtime and newly finalized data quality are outside this review.
