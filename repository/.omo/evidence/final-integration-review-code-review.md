# Final integration code review: B1--B4 full-data refit and packaging

## Scope and evidence reviewed

- Goal reviewed: prepare the unchanged B1/B2 max-pooled and B3/B4 raw/Demucs temporal architectures for one all-permitted-local-data refit, four offline ZIPs, and no DACON submission.
- Files reviewed: `run_v5_final.py`, `test_v5_final_pool.py`, `v5_final_refit.py`, `test_v5_final_refit.py`, `v5_submission.py`, `test_v5_submission.py`, `package_v5_submission.py`, and `build_v5_final_notebook.py`.
- Evidence inspected: `.omo/evidence/v5-final-refit-training/validation.md`, `.omo/evidence/v5-final-notebook-builder/validation.md`, `.omo/v5-final-submission-notepad.md`, and the current source.
- Verification performed in this review: `pytest -q test_v5_final_pool.py test_v5_final_refit.py test_v5_submission.py` passed (11 tests, 2.18s); `py_compile` passed; `git diff --check` passed. These are static/small-fixture checks. No Colab run, full-pool refit, archive inference, L4 1,200-file timing, or measured RAM evidence was present.

## Skill-perspective check

Ran the `omo:programming`, `omo:remove-ai-slops`, and `ponytail:ponytail` review perspectives before judging maintainability and test relevance.

- `programming`: the test suite does not use brittle prose assertions, untyped escape hatches, or implementation-only prompt pins in the reviewed paths. The staged completion contract below is a correctness defect.
- `remove-ai-slops`: `test_v5_final_pool.py` protects source/provenance/unknown-label behavior; the refit tests exercise resume equivalence, optimizer/scheduler restoration, full-cohort exposures, and gradient connectivity; the packaging tests protect incomplete/tampered artifacts and weight exclusion. They are not deletion-only or tautological tests. The test suite misses the cross-module receipt-schema contract that failed this review.
- `ponytail`: no needless parsing/normalization or new abstraction is needed to fix the blocker; producer and consumer must agree on the existing receipt shape.

## Re-review result

The prior HIGH finding is fixed. The notebook now requires the runner's actual
`offline_validation` object with `status == "passed"`
(`build_v5_final_notebook.py:257-263`), accepts an absolute or result-relative
archive path only after containment verification (`340-349`), and preserves the
complete package receipts in `completion_verified.json` (`354`). This matches
the object emitted by `run_v5_final.py:231-235`.

The refit output-root convention is internally consistent: expert jobs use
`config.output / "jobs" / <arm_kind>` (`v5_final_refit.py:169`) and temporal
jobs use `config.output / <arm>` (`287`); the runner supplies `work / "trials"`
and copies each resulting artifact from those exact locations
(`run_v5_final.py:279-300`, `322-333`). Final state serialization retains
non-floating buffers while moving tensors to CPU (`v5_final_refit.py:256-260`,
`363-367`), and the focused test checks an integer buffer survives
(`test_v5_final_refit.py:92-96`).

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

1. **The required L4 60-minute maximum-duration evidence is explicitly absent.** The runtime receipt records `L4_full_1200_file_runtime: "not measured; A100 measurements do not prove the L4 limit"` in `run_v5_final.py:231-234`. This is correctly disclosed. It is a release/submission verification requirement, not a source preflight blocker.

2. **The corrected receipt handoff is source-reviewed but lacks a direct consumer/producer integration test.** Current small-fixture tests cover the refit, archive layout, decoder parity, and incomplete-artifact rejection, but do not execute the notebook's embedded `package_receipt_shape` against an actual runner receipt. The source contract is now consistent; add a narrowly scoped test only if the notebook builder changes again.

### LOW

None.

## Positive checks

- The final pool loader merges train/dev/sealed, requires licensed provenance, prefers `full_path`, validates source containment/audio/labels, and records data receipts (`run_v5_final.py:31-75`).
- Expert eligibility applies the same derived-example fake-head masking as `training_crop` (`v5_final_refit.py:114-125`); temporal rows retain their full derived views (`run_v5_final.py:302-321`).
- Fixed expert epochs are B1 `4/5/5`, B2 `4/1/1`, with the requested 60-epoch cosine horizon; B3/B4 use 20 epochs and preserve non-joint/joint selection (`v5_final_refit.py:44-49`, `280-305`).
- Checkpoint state includes model, optimizer, scheduler, completed epoch, exposures, and audits; final-state digests are checked before packaging (`v5_final_refit.py:144-158`, `256-263`, `357-369`; `package_v5_submission.py:67-92`).
- The package QA is designed to check parity, silence, multiple audio formats/durations, order/companion independence, offline flags, archive size, and RSS. It remains unexecuted evidence, not a green runtime result (`run_v5_final.py:84-238`).

## Decision

- `codeQualityStatus`: **CLEAR**
- `recommendation`: **APPROVE** for the GPU preflight.
- `blockers`: None for starting the full-data GPU run. Before any final evaluator-readiness or submission claim, require artifact-backed full 1,200-file L4 duration and RAM/parity/offline QA.
