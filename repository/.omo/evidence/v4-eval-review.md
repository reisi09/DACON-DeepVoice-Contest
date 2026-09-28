# V4 evaluation/selection implementation re-review

**Scope:** `v4_evaluation.py`, `v4_selection.py`, `test_v4_selection.py`, and the V4 selection contract in `.omo/plans/trainable-recovery-v4.md`.

**Boundary:** implementation readiness only. No GPU training, sealed-panel scoring, or submission has run; their artifacts are pending execution and are not code-review defects.

## Verdict

- `codeQualityStatus`: **WATCH**
- `recommendation`: **APPROVE**
- `blockers`: none for implementation readiness.

The promotion path now enforces both development panels, partial-metric nullness, protected G0 heads and the derived FILE formula, connected-component bootstrap, 1.25x refit reserve, recipe-to-winner binding, full eligible-ID exposure coverage, weight/code locking, sealed-open chronology, hashed prediction receipts, and frozen-candidate-only sealed acceptance.

## Findings

### CRITICAL

None.

### HIGH

None. The earlier recipe-lock bypass, partial-refit-manifest bypass, and unbound sealed-prediction path are fixed and covered by meaningful regression tests.

### MEDIUM

None.

### LOW

1. `Report` and lock payloads still use broad nested `dict` aliases rather than typed value objects ([v4_evaluation.py](/Users/choiwuseck/Documents/ChatGPT/데이콘/v4_evaluation.py:15), [v4_selection.py](/Users/choiwuseck/Documents/ChatGPT/데이콘/v4_selection.py:36)). This is a `programming`-perspective type-discipline violation, but it does not create a demonstrated incorrect promotion path. Do not block the experiment for a broad typing rewrite.

## Reproduced controls

- `test_failed_or_resource_ineligible_candidate_cannot_be_recipe_locked`: rejects a G0-selected/P2-ineligible recipe lock.
- `test_refit_manifest_must_cover_the_recipe_locked_eligible_union`: rejects a manifest omitting a recipe-locked eligible ID.
- `test_sealed_gate_cannot_promote_a_runner_up`: rejects P2 after P1 is locked.
- `test_sealed_predictions_must_match_the_locked_inference_receipt`: rejects prediction-file substitution after the inference receipt.
- `test_actual_rounded_graph_g0_is_within_file_formula_tolerance`: accepts the real rounded G0 path while retaining a narrow formula tolerance.

## Checks

- `.venv/bin/python -m pytest -q test_v4_selection.py -vv`: **15 passed in 1.39s**.
- Focused five regression tests above: **5 passed in 0.21s**.
- The test suite changed while the first review ran. This report reviews the stable hashes below; earlier test counts and findings are superseded.

## Skill-perspective check

`omo:programming` and `omo:remove-ai-slops` were loaded and applied. The code has the low type-shape concern above. The tests are decision-valued: each covers a concrete invalid-promotion or numerical-rounding failure. No deletion-only, tautological, prompt-text, implementation-constant, or needless-parsing/data-normalization slop was found.

## Reviewed SHA-256

```
a3a8e307d1024f089748a1777265c1830ae22609c9df81013679499500172144  v4_evaluation.py
602f46332acdab20453e4f3e9272a21c15f6edcb5487465e39137a879d64e72c  v4_selection.py
6e429c30f47ed198a5b884fd23e92a5e2e8589e436024eafe4a75ea64d0394f3  test_v4_selection.py
06208ce573350d777f3884912a48bb98f998fb183434af7fed97fdbee2b78bc2  .omo/plans/trainable-recovery-v4.md
```

## Execution prerequisite (not a code blocker)

Before any submission decision, run the locked workflow and preserve the plan-required `selection.json`, `recipe_lock.json`, eligible/refit manifests, `weight_lock.json`, `sealed_opened.json`, sealed inference receipt, and final selection report. A passing synthetic suite is not evidence that a model passed sealed acceptance.
