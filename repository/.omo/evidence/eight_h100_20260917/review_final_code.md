# Corrective code review — eight H100 aggregation

**Reviewed commit:** `3970207e336bae7b72c2c261a4c54129f57ad79c`
**Baseline:** `2a0697edef92903f604cdb50f77ef807a940c5d0`
**Scope:** the aggregation correction in `build_eight_h100_notebook.py`, its targeted regression, and the hash-addressed corrected-summary evidence. This is a read-only review; no GPU, API, archive, or source mutation was performed.

## Decision

- **codeQualityStatus:** `CLEAR`
- **recommendation:** `APPROVE`
- **reportPath:** `.omo/evidence/eight_h100_20260917/review_final_code.md`
- **blockers:** none

## Verification performed

- The exact corrective delta replaces the notebook-local DSU with the established `eight_fixed_io.connected_components` at [build_eight_h100_notebook.py:143](../../../build_eight_h100_notebook.py#L143), preserving singleton provenance atoms. It also reads each experiment's native `component_count`, label denominators, population scope, update count, class exposure, seeds, and lock metadata at [build_eight_h100_notebook.py:194](../../../build_eight_h100_notebook.py#L194), rather than copying global values or falling back to a 500-step head budget.
- Fresh checks passed from this checkout: `.venv/bin/python -m pytest -q test_eight_corrected_aggregation.py` (`1 passed`) and `.venv/bin/python .omo/evidence/eight_h100_20260917/verify_corrected_aggregation.py` (`VALID global=256 sealed=178 E8=null archive_and_notebook_immutable=true`). `py_compile`, JSON parsing of both corrected artifacts, and `git diff --check` also passed.
- The canonical corrected JSON hashes back to `correction_receipt.json`, explicitly supersedes the immutable sealed aggregation by SHA-256, reports global components `256`, retains native E2/E5=`4` and E7=`10`, and records E8 `fixed_updates: null` plus its non-iterative fit kind. It marks E1/E3/E4/E6/E8's raw top-level population metadata as generic and provides their branch denominators; E2/E5/E7 remain native-scope entries.
- The correction receipt records 77 saved-prediction EER comparisons, zero mismatches, and maximum absolute delta zero. It also binds the unchanged archive and executed-notebook hashes. The errata names the correction as a post-run supersession and states that no score, prediction, checkpoint, fitted model, archive member, or GPU run changed.

## Skill-perspective check

This check ran. I loaded and applied `omo:programming` (including its Python reference) and `omo:remove-ai-slops` before judging maintainability and test relevance.

The diff violates neither perspective. The regression at [test_eight_corrected_aggregation.py:25](../../../test_eight_corrected_aggregation.py#L25) is an artifact-integrity test for the concrete prior regression: it independently recomputes global connected components from the manifest, hashes immutable reports, verifies per-experiment native fields, checks the branch-versus-generic metadata boundary, and proves the archive and numerical evidence remain unchanged. It is not deletion-only, prompt-text, tautological, shape-only, or an implementation-constant test. The production change reuses the existing shared helper, adds no parsing/normalization layer, abstraction, dependency, or untyped escape hatch, and is limited to the aggregation seam.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Residual boundary

The immutable archive still contains its historical `final_aggregation.json` with `178` and the E8 fallback. That is intentional and is no longer presented as authoritative: the separately hashed canonical correction and errata explicitly supersede it. This review approves the metadata correction only; it does not turn the 77 preserved EER comparisons into an official score or a fresh-holdout claim.
