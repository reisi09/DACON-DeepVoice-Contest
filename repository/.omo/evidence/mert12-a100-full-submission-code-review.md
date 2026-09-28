# MERT12 A100 prelaunch code review

## Scope and evidence

- Goal: authorize the bounded MERT12 full-data refit only on `wsp-5i3knd3gxeox`, with offline packaged inference and verified export/stop handling.
- Reviewed source: `mert12_full_refit.py`, `mert12_submission.py`, `package_mert12_submission.py`, `mert12_a100_launch.py`, `mert12_a100_watch.py`, `test_mert12_submission.py`, `DACON_MERT12_Full_A100.ipynb`, and staged `bootstrap.sh`.
- Attempt status: `omo ulw-loop status --json` reported `ULW_LOOP_PLAN_MISSING`; this is the required fallback report path.
- Skill-perspective check: ran `omo:programming` (including its Python guidance) and `omo:remove-ai-slops`. No violation found. The tests are decision-value checks for sampling/coverage, E6 window semantics, serialized checkpoint/file-mean parity, and cached-reference agreement; they are neither deletion-only nor tautological or implementation-mirroring. The production parsing and validation occur at external asset/archive boundaries required for integrity, rather than as needless normalization or abstraction.

## Verification performed

- `git diff --check` and `.venv/bin/python -m py_compile` over all five reviewed Python modules: pass.
- `.venv/bin/python -m pytest -q test_mert12_submission.py`: `5 passed in 9.57s`.
- `bash -n` passed for staged `bootstrap.sh`; staged and readback bootstrap files are byte-identical with SHA-256 `155aedeb2b28311ce6f06ecaba8ecc96fdc7f39c4e0644d2b9fa0e71436fd0e2`.
- `storage_preparation.json` was read and independently cross-checked against the staged upload directory: all nine objects have matching full-readback bytes and digests. This includes `source.zip` SHA-256 `e44a1e6ba053b7f783c0928ed076046992613fcbde2cbc55b86e1fc5b825eea0`, `features.zip` SHA-256 `188c016268af2ebc76fa42e5066d7f11e295b65ae214454a9b09a5b7420fdc4f`, and the final bootstrap digest above.
- The source ZIP manifest covers 20 source members; staged copies of the refit, submission, package, and notebook files exactly match the current reviewed worktree.
- The manifest SHA-256 is `f1013101c5198295405b4094f5820a235c1ba89da55483710bce4b93454f7ff1`; the audit has 1,322 unique ordered eligible IDs. `mert12_full_refit.py` gates the expected count, manifest digest, and exact audit IDs before training; `package_mert12_submission.py` repeats those audit/lineage checks before packaging.
- Refit code uses `np.load(..., allow_pickle=False)`, source/feature/frozen-asset hash binding, TF32-disabled CUDA, all-row balanced sampler exposure, nonzero gradient evidence, and parameter-update evidence. The fixed-head setup preserves E6 preprocessing/window mean/std behavior.
- Submission code loads weights with `weights_only=True`, makes per-file offline predictions, validates input/CSV membership and basename uniqueness, uses deterministic HTDemucs `shifts=0`, and checks cached-reference agreement. Launch is bound to the requested stopped A100 workspace and a two-hour deadline. Bootstrap moves the single large package to `results.zip` before building `evidence.zip`; watcher streaming-verifies submission members and independently verifies/extracts evidence before the stop record. Its failure branch delegates raw partial-export verification to the shared verifier.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Decision

- `codeQualityStatus`: CLEAR
- `recommendation`: APPROVE
- `blockers`: none.

This is a static/prelaunch approval only. It does not claim that GPU execution, output quality, an official score, or final download verification has happened; the launched run must produce and the watcher must validate those separate receipts.

## Addendum: post-review operational blockers

Reviewed the two release-blocker corrections after the initial decision.

- `mert12_a100_launch.py:68-83, 212-231` no longer depends on an SSH key. It takes the sole HTTPS Jupyter endpoint from the live workspace response, validates it against the fixed workspace, and passes it over stdin to the local Node helper. `jupyter_terminal.mjs:6-48` obtains the token only from that endpoint, authenticates terminal creation/WebSocket dispatch, returns only terminal name and the verified bootstrap PID, and does not persist or print the token or full workspace response. The launch still starts its watcher first and pauses the fixed workspace if dispatch fails.
- `bootstrap.sh:32-37` now guards the monitor kill with a nonempty PID test. It cannot issue `kill 0` on an early failure.
- Re-ran `git diff --check`, Python compilation, Node syntax validation, and Bash syntax validation for this changed path: pass. The staged/readback bootstrap SHA remains `155aedeb2b28311ce6f06ecaba8ecc96fdc7f39c4e0644d2b9fa0e71436fd0e2`.

No CRITICAL, HIGH, MEDIUM, or LOW finding remains from these two corrections. The prior CLEAR/APPROVE decision stands.

## Addendum: dispatch marker completion gate

A final concrete dispatch race was identified after the prior addendum: a PTY may echo `MERT12_DISPATCHED_PID=$pid` before shell expansion. The helper now resolves only after `/MERT12_DISPATCHED_PID=\d+/` is present (`jupyter_terminal.mjs:35`), while the final receipt still extracts the same numeric PID at `:46`. Thus command echo cannot be accepted as a completed dispatch. `node --check` passes. No blocker remains.
