# V4 recheck manifest-guard rereview

**Scope.** Read-only rereview of the prior manifest-binding blocker only. Inspected `recheck_v4_matched.py`, the current source-hash receipt, pinned manifests, targeted tests, and their cited evidence. No GPU/UI run, refit, notebook execution, source edit, or commit was performed.

**Verdict:** `CLEAR` / `APPROVE`.

## Independently verified evidence

- Current `recheck_v4_matched.py` SHA-256 is `811534002492c462e3411106919bb823e1ba924637cc9fd2fad74671da01f349`, equal to `.omo/evidence/v4-training/recheck-source-sha256.txt` line 1.
- The wrapper now rejects a mismatched G0 receipt before model loading (`recheck_v4_matched.py:56-57`), then rejects train/dev CSV digests before manifest parsing or trial output (`:58-61`). The pinned values equal the current `v4_validation/train.csv` and `v4_validation/dev.csv` hashes.
- It then parses the manifests and rejects anything other than 1,116 eligible train rows and 210 dev rows (`:62-66`). The current audited data parses as 1,516 train rows, 1,116 eligible music-supervision rows, and 210 dev rows.
- The receipt records the two manifest digests/counts and backend flags, and explicitly limits its CUDA statement to shared seed/RNG/initialization pairing rather than bitwise CUDA equivalence (`:75-82`). This resolves the prior wording concern without asserting unsupported GPU determinism.
- Fresh targeted execution passed: `2 passed in 3.63s` for `test_matched_recheck_rejects_changed_real_train_manifest_before_training` and `test_seed_receipt_reproduces_dropout_across_checkpoint_resume`. The first is a meaningful adversarial guard test: one-byte train-manifest drift must fail before a receipt/model/trial is created. The second exercises stochastic checkpoint-resume behavior; it is neither a deletion-only nor a constant-mirroring test.

## Findings

### CRITICAL

None.

### HIGH

None. The prior high-severity unbound-manifest condition is fixed by the pre-load SHA-256 and count guards.

### MEDIUM

None.

### LOW

None.

## Skill-perspective check

`omo:programming` (including its Python guidance) and `omo:remove-ai-slops` were loaded and applied before judging maintainability and test relevance. The delta does not violate either perspective: digest checks occur at the executable input boundary, use the standard library, add no needless abstraction or normalization, and the focused tests cover practical failure modes without brittle prompt assertions, tautology, or implementation-constant mirroring.

## Approval boundary

This approval covers the manifest/data-binding repair and its source receipt only. It does not certify a future GPU execution, a notebook result, or bitwise CUDA determinism.

**codeQualityStatus:** `CLEAR`  
**recommendation:** `APPROVE`  
**blockers:** None.
