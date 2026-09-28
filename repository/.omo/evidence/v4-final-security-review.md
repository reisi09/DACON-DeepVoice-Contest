# V4 final security and integrity review

## Decision

- `codeQualityStatus`: `BLOCK`
- `recommendation`: `REQUEST_CHANGES`
- `workflow`: `revise`
- `blockers`:
  1. Make the Colab runtime dependency closure immutable before executing installation code with the Drive mount available. At minimum, remove the two unversioned runtime requirements; preferably install a reviewed, fully resolved lock with artifact hashes before `drive.mount()`.
  2. Bind `DRIVE / "open.zip"` to a checked size and SHA-256 receipt before extracting it. The current inner `baseline_submit.zip` hash is checked too late to protect the extraction step.

This is a source/integrity review of the authorized private-Colab/private-Drive run only. It does not assert an A100 result, submit to DACON, access sealed data, or treat a user-authorized trusted model/checkpoint as hostile input.

## Immutable review target

| Item | Value |
| --- | --- |
| Repository / source | `/tmp/dacon-v4-review-a01a50e`, committed range |
| Base | `f69f4bf` |
| Head | `a01a50e0c774f181c20238109127c5b81cc95dc1` |
| Exact binary diff SHA-256 | `8ee3a53d8389a9655950fb55258c560ab780a07054e428dbddc76659f270c885` |
| Head checkout status | clean at inspection |

The generated notebook has 28 cells, 19 embedded source files, no code-cell outputs, and all 19 embedded source bodies exactly match the reviewed files. The checked 19-source SHA-256 receipt also matches the reviewed files.

## Findings

### CRITICAL

None.

### HIGH

1. **The mounted-Drive runtime installs two mutable package selections.** [build_v4_notebook.py:215](/tmp/dacon-v4-review-a01a50e/build_v4_notebook.py:215) declares `imageio-ffmpeg` and `safetensors` without versions, then [build_v4_notebook.py:216](/tmp/dacon-v4-review-a01a50e/build_v4_notebook.py:216)-[build_v4_notebook.py:218](/tmp/dacon-v4-review-a01a50e/build_v4_notebook.py:218) installs them after [build_v4_notebook.py:87](/tmp/dacon-v4-review-a01a50e/build_v4_notebook.py:87) mounts Drive. A future index resolution can execute a different package build while the private Drive is available; `source_hashes.json`, archive receipts, and model hashes do not cover installed package bytes. This also breaks the claimed reproducible runtime closure. Use an immutable reviewed dependency lock (including the two unpinned packages and transitive resolution) and install/verify it before mounting Drive.

### MEDIUM

1. **`open.zip` is extracted without an outer receipt.** [build_v4_notebook.py:240](/tmp/dacon-v4-review-a01a50e/build_v4_notebook.py:240)-[build_v4_notebook.py:246](/tmp/dacon-v4-review-a01a50e/build_v4_notebook.py:246) copies the private-Drive file and calls `safe_extract` before checking the nested `baseline_submit.zip` SHA-256. `safe_extract` rejects absolute and `..` member names at [build_v4_notebook.py:135](/tmp/dacon-v4-review-a01a50e/build_v4_notebook.py:135)-[build_v4_notebook.py:140](/tmp/dacon-v4-review-a01a50e/build_v4_notebook.py:140), but it has no pre-extraction size/hash binding for this outer Drive input. A changed or malformed `open.zip` can consume the bounded runtime/disk before the inner hash check fails. Add the expected outer size/SHA-256 to the embedded receipt and verify it before extraction. No general archive hardening is required once that known private input is bound.

### LOW

None.

## Boundary analysis

| Boundary | Counterexample considered | Control verified | Result |
| --- | --- | --- | --- |
| V4 Drive archive parts | Replaced part or reconstructed archive | Every part and assembled archive has size/SHA-256 checks before extraction at [build_v4_notebook.py:224](/tmp/dacon-v4-review-a01a50e/build_v4_notebook.py:224)-[build_v4_notebook.py:238](/tmp/dacon-v4-review-a01a50e/build_v4_notebook.py:238) | Controlled |
| Archive path traversal | Archive member is absolute or contains `..` | Rejected before `extractall` | Controlled for this Linux Colab scope |
| Official baseline outer archive | Altered/oversized Drive `open.zip` | Inner baseline hash only, after outer extraction | **Not controlled** |
| Baseline/model assets | Altered inner baseline or V4 asset archive | Fixed inner baseline hash; V4 assets are part/archive receipt checked | Controlled after outer-archive blocker is fixed |
| Checkpoint deserialization | A crafted `.pt` is loaded with pickle enabled | `torch.load(..., weights_only=False)` at [v4_training.py:151](/tmp/dacon-v4-review-a01a50e/v4_training.py:151) is used only for self-produced resume state in the authorized private output folder and requires the run contract to match | Accepted trusted-model use; not an untrusted-input exploit in the stated scope |
| Output/sealed decision integrity | Weight, code, predictions, or recipe changes after selection | Refits create recipe/weight/inference receipts and rehash each input before sealed acceptance in [v4_selection.py:157](/tmp/dacon-v4-review-a01a50e/v4_selection.py:157)-[v4_selection.py:294](/tmp/dacon-v4-review-a01a50e/v4_selection.py:294) | Controlled |
| Allocation / runtime release | Setup delay or long runner overruns the original allocation | Deadline is clamped to original allocation plus 7,080 seconds in [train_v4.py:53](/tmp/dacon-v4-review-a01a50e/train_v4.py:53)-[train_v4.py:56](/tmp/dacon-v4-review-a01a50e/train_v4.py:56); watchdog and final release call `runtime.unassign()` | Controlled, subject to the unchecked outer archive consuming budget first |
| Commands / secrets / public exposure | Shell injection, credentials in the diff, or automatic external submission | `subprocess` uses argument arrays; no secret/private-key pattern found; no DACON submission route appears | Controlled in reviewed scope |

## Verification and evidence assessment

- Independently verified the exact patch SHA, clean subject checkout, source receipt, and byte-for-byte notebook embedding. The notebook’s declared source receipt and generated notebook hash are current: `build_v4_notebook.py` `25a3be622d9115f9a9d7318f5ee3cc5f3b02a596fb7ba8c6ce5d62958a572c2d`, notebook `9e40c7c58bdc262f98699e0fef351ed15a01a8b8be7b296760bdfeead51a413c`, input receipt `5200d0e6310913d2ca455f0ba54fdb0800ed45a0cf2f25eea06e764aa22d0944`.
- Inspected the supplied exact-head test evidence: [pytest-v4-final.log](/tmp/dacon-v4-review-a01a50e/.omo/evidence/v4-training/pytest-v4-final.log) records 34 passing focused model/training/selection tests; [verification.md](/tmp/dacon-v4-review-a01a50e/.omo/evidence/v4-training/verification.md) accurately limits this to CPU validation and leaves A100 execution pending. A fresh run was attempted with network-disabled model settings and no cache provider, but the isolated `uv --no-project` interpreter has no `pytest`; it was therefore not represented as fresh test evidence.
- The tests have practical decision value: checkpoint/resume, deadline, recipe/weight locks, sealed receipt integrity, and real model-update paths are exercised. I found no deletion-only, requested-removal, tautological, prompt-text, or pure implementation-constant test.

## Skill-perspective check

This review loaded and applied `codex-security:assess-patch-risk`, `codex-security:security-diff-scan`, `omo:programming`, `omo:remove-ai-slops`, and Ponytail. The programming perspective found no newly introduced untyped escape hatch, brittle prompt test, or unjustified internal parsing at the reviewed trust boundaries. The remove-ai-slops pass found no useless test or needless production data extraction/normalization in the security-critical flow. The diff does violate the minimal/dependency-integrity perspective through the two unpinned runtime dependencies; that issue is material and is the HIGH blocker above.

## Risk assessment

- Impact if wrong: high, because installation code executes in a live Colab session with the user’s private Drive mounted.
- Regression likelihood: moderate for mutable package resolution and low-to-moderate for accidental/mutated `open.zip`.
- Regression protection: strong for V4 archives, model/result locks, sealed acceptance, and deadline logic; absent for the two blockers.
- Recoverability: a failed run releases the runtime and writes receipts, but it cannot undo a dependency compromise and consumes a scarce allocation.
- Confidence: high for source-bound findings; no claim is made about a live GPU result or external Drive state.
