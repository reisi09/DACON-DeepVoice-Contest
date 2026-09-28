# V4 matched recheck delta review

**Scope.** Read-only review of `a01a50e0c774f181c20238109127c5b81cc95dc1..working tree` for `train_v4.py`, `v4_training_trials.py`, `recheck_v4_matched.py`, and the related test/evidence files. No GPU, notebook, UI, refit, promotion, sealing, or source change was performed.

**Verdict:** `BLOCK` / `REQUEST_CHANGES`.

## Evidence independently checked

- Current `recheck_v4_matched.py` SHA-256 is `5c85f91add3fde1113d97087e1783e84b71a820e600e0fb479c75a2ff5dbe934`, exactly matching `.omo/evidence/v4-training/recheck-source-sha256.txt`. The earlier `de6f...` receipt is stale and must not be used.
- The CLI contract pins allocation start `1789283305`, deadline `1789290385`, 15 epochs, seed `20260913`, and raw G0 SHA-256 `bd1c27227829f36e9ae6191db7600920bf6649810f0896a791730b10555f0854`.
- The wrapper imports only SONICS-related training components; it neither imports AST nor invokes G0 generation, selection, refit, sealing, or submission. It passes `p3` only to P3, holds P1/P3 at 15 epochs, and rejects a digest-mismatched G0 file.
- The seed path resets Python, NumPy, CPU Torch, and CUDA Torch RNG before model construction and again at trial entry; trial contracts/receipts bind the seed-RNG hash, model fingerprint, source contract, and any existing resume-checkpoint digest. Checkpoints persist and restore all four RNG states (`v4_training.py:94-157`). P1/P3 comparison rejects mismatched initial fingerprints/RNG receipts and update/window-exposure counts (`recheck_v4_matched.py:88-98`).
- Independently ran the three targeted CPU checks: `3 passed in 5.66s` for dropout resume, real-SONICS terminal resume, and optimizer resume. These are meaningful behavioral checks: the dropout test compares the resumed optimization trajectory with uninterrupted training, and the SONICS test uses real assets/audio and a disk checkpoint. The reviewed test is not deletion-only, tautological, prompt-based, or a constant mirror.
- Existing eligibility evidence reports 1,116 eligible train rows and 210 dev rows, but this artifact does not constrain the wrapper at run time.

## Findings

### CRITICAL

None.

### HIGH

1. **The recheck can run with a manifest other than the required 1,116 eligible training rows and 210 development rows.** `recheck_v4_matched.py:56-70` records `len(train_rows)`, `len(eligible)`, `len(dev_rows)`, and the two CSV hashes only after accepting arbitrary `--data`; the only live data check is G0 ID/order alignment at `:60-61`. Thus a wrong directory (or a modified manifest paired with a matching G0 file) produces a self-consistent new receipt and still runs 15 epochs, violating the fixed-data causal contract. The static evidence file `eligibility-counts.txt` is not enforced by the executable. Before launch, bind the expected train/dev SHA-256 values and exact 1,116/210 counts in the wrapper and fail before writing/running a trial.

### MEDIUM

1. **GPU reproducibility remains unproved.** `reset_training_seed` seeds CUDA (`v4_training_trials.py:56-67`) and checkpoint resume restores CUDA RNG, but it does not select deterministic algorithms/backends. The new dropout-resume test (`test_v4_training.py:424-466`) is CPU-only. This does not invalidate the seed/receipt mechanism, but it cannot establish bitwise matched behavior for the planned A100 execution if a CUDA kernel is nondeterministic. Add a small GPU smoke only if the launch claim requires bitwise trajectory identity; otherwise report the scope as same initialization/RNG state rather than deterministic GPU equality.

### LOW

None.

## Skill-perspective check

`omo:programming` and `omo:remove-ai-slops` were loaded and applied before judging maintainability/test relevance. The diff does **not** violate either perspective: the new receipt/fingerprint code has a concrete contract purpose, the dropout test protects a plausible checkpoint/RNG regression, and no brittle prose/prompt test, untyped escape hatch, needless abstraction, unnecessary parsing/normalization, or implementation-constant test was found. The production additions are narrowly scoped to the reproducibility gap.

## Required blockers before approval

1. Enforce the exact expected train/dev manifest hashes and the required eligible-train/dev counts in `recheck_v4_matched.py` before either arm begins; then refresh the wrapper hash receipt and rerun the three targeted CPU checks.
2. Decide whether the GPU claim is only seeded/resumable or bitwise deterministic. If it is bitwise deterministic, add and pass a minimal A100-relevant deterministic/resume check before describing it that way.

No notebook result is approved by this review: at review time the source wrapper was current, but no new matched P1/P3 GPU receipt existed.
