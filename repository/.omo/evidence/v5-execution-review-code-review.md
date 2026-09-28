# V5 execution pre-GPU code review

## Scope and evidence inspected

Goal reviewed: execute the five V5 comparison arms under one two-hour Colab allocation, retain only best weights in Drive receipts, preserve unknown-label and whole-source boundaries, and record deadline censoring without overstating completion. Scope inspected: `v5_*.py`, `build_v5_notebook.py`, `test_v5_*.py`, and `.omo/evidence/v5-*` artifacts.

`omo ulw-loop status --json` returned `ULW_LOOP_PLAN_MISSING`, so this report uses the required fallback path rather than an attempt directory. There is no GPU execution artifact for V5; this review makes no GPU-run claim.

## Findings

### CRITICAL

None.

### HIGH

None after recheck. The prior temporal completion-reporting blocker is fixed: `v5_compare.py:48-57` records zero-epoch/zero-update/no-best temporal work as `budget_censored` without metrics, and only calls it evaluated when a trained best epoch exists. `v5_compare.py:211-212` consequently leaves `all_five_evaluated` false for that status.

### MEDIUM

1. **The temporal completion-reporting seam is now covered.**
   - `test_v5_compare.py:13-22` executes real expired `train_fusion` output through `record_temporal` and proves that B4 becomes `budget_censored`, has no metrics, and prevents the all-arms evaluated predicate. This is a practical regression test, not a constant or implementation mirror.

2. **Static quality gates are not clean for the executable closure.**
   - The focused V5 test suite passed: `.venv/bin/pytest -q test_v5_dataset.py test_v5_experts.py test_v5_training.py test_v5_features.py test_v5_fusion.py` -> `20 passed in 7.27s`.
   - `uvx ruff check` reports five issues in scope, including unused `json` and unsorted imports in `v5_compare.py:4-28`, a needless tuple literal in `v5_features.py:174`, and non-executable shebangs in `v5_compare.py:1` and `build_v5_notebook.py:1`.
   - `uvx basedpyright --level error` reports 65 errors across the closure, including untyped dynamic model access and runner/reporting type mismatches. The errors do not by themselves demonstrate a runtime GPU defect, but they mean the claimed typecheck gate is not established before a costly Colab attempt.
   - Minimal fix: clear the lint errors and either give the dynamic third-party model seams accurate narrow types/protocols or explicitly scope/report the typecheck limitation. Do not describe the closure as typecheck-green meanwhile.

### LOW

None.

## Checks that passed or were sufficient for this review

- Whole-file tail coverage and cross-expert time alignment have meaningful tests in `test_v5_features.py:12-28`.
- The partial-label joint likelihood, unknown-state marginalization, padding isolation, per-file batching, temporal-order sensitivity, and a connected cached-fusion update are meaningfully tested in `test_v5_fusion.py:15-194`.
- The data split test exercises the actual 1496/230 manifests and verifies the five group / 20-row NSynth holdout with zero group, path, and SHA-256 overlap (`test_v5_dataset.py:22-36`).
- Expert training tests demonstrate encoder gradient/update and tail-anchored full-file expert evaluation (`test_v5_training.py:51-108`).
- The notebook builder generated a 29-cell notebook. After excluding intentional IPython `%%writefile` headers, all 20 embedded Python source bodies compile successfully. The earlier direct full-cell compile failure was expected because IPython magics are not Python syntax.

## Skill-perspective check

Ran: yes. I loaded and applied `omo:programming` and `omo:remove-ai-slops` before judging test relevance and maintainability.

- `remove-ai-slops`: no deletion-only tests, requested-removal tests, tautological tests, prompt-text tests, or unnecessary production parsing/normalization were found in the inspected V5 tests and production code. The missed runner-status regression is a missing behavioral seam, not a reason to add shallow coverage.
- `programming`: the diff violates the type-strict perspective because the supplied basedpyright invocation has 65 errors, and it has a real reporting-contract gap described in the HIGH finding. It does not introduce a brittle prompt test or an implementation-mirroring test.

## Recommendation

- `codeQualityStatus`: **WATCH**
- `recommendation`: **APPROVE**
- `blockers`: none.

Recheck result: the focused V5 suite is now 21 passed; all 20 embedded source bodies compile after notebook regeneration. No further runtime-fatal integration defect was found in the reviewed path. A GPU run remains necessary to establish timing, Drive receipt delivery, and actual metrics.
