# MERT12 final security and operational gate review

- recommendation: **PASS**
- reviewedCommit: `e31f1f6679b7fb23d1f490b4fceb10813c22a80b`
- baseCommit: `83190ff31b89c096f1dfa13742ca546c09b3c5a9`
- reviewType: bounded read-only static security/operational review
- blockers: `[]`

## Original intent

Use the already authorized A100 workspace to refit the selected MERT layer-12 head on all 1,322 eligible original music files, produce the fixed offline submission ZIP, verify it within the unchanged absolute deadline, stop compute after verified downloads, and submit it to DACON. This review is limited to token handling, ZIP trust and paths, offline/cross-file behavior, deadline enforcement, and stop/upload controls.

## Desired outcome

The artifact at the reviewed commit should provide a fixed, offline, file-independent inference package and bounded A100 operating path that does not expose credentials, does not trust unsafe ZIP paths, cannot silently extend the original deadline, binds uploads/downloads to hashes, and pauses only the intended A100 after artifact verification or failure.

## User outcome review

The reviewed code and locally available artifacts satisfy the requested static security and operational boundaries. The final submission ZIP was independently read from the original repository and reproduced as 6,268,222,843 bytes with SHA-256 `e7bebbded6fef06517db0d42421a575a0440849248de99856796ffc3e5eb5ee5`; it has 45 unique members, no absolute/traversal/backslash paths, and `ZipFile.testzip()` returned `None`.

This PASS is **not** evidence that attempt 6 runtime QA, verified workspace shutdown, DACON upload, or official submission completed. The inspected live receipts still show attempt 6 `status: running`, `archive_ready: false`; the existing `resource_stop.json` belongs to the earlier attempt-5 failure. Final QA and submission must be judged from fresh attempt-6-bound receipts when they exist.

## Criterion findings

### SEC-1 — token and endpoint handling: PASS

- No embedded API key, bearer credential, password, or secret assignment was found in the reviewed Python, shell, JavaScript, JSON, or notebook changes.
- `mert12_a100_launch.py` obtains the Jupyter URL from the authenticated CLI, validates HTTPS and the fixed workspace hostname, and passes it to the helper over stdin rather than a command-line argument.
- `_workspace/mert12_a100_20260918/jupyter_terminal.mjs` extracts the token only in memory, uses it for HTTP/WebSocket authorization, and emits only terminal name and PID. CLI failures report command class and exit code without stderr/full workspace responses.

### SEC-2 — ZIP trust, member paths, and integrity: PASS

- Upload preparation records size and SHA-256 for every staged input, downloads the object prefix, and rejects any file-set, size, or digest mismatch before compute.
- The primary bootstrap checks every uploaded file against `upload_manifest.json`, checks CRC and rejects absolute or `..` member paths before extraction.
- The final package builder creates members from a fixed map, verifies every member digest and CRC, enforces archive size limits, and emits a complete member-hash receipt.
- Attempt-6 reconstruction is accepted only after the exact final whole-archive SHA and byte size match; the watcher then checks unique member names, canonical safe paths, CRC, and every receipt-declared member hash.
- `redesign_h100_watch.verify_archive` rejects traversal, duplicate, directory, symlink, and special members before extraction and verifies the archive/member manifests.
- Independent artifact check: final ZIP SHA/bytes/path/CRC passed. The staged smoke ZIP also matched its storage-preparation SHA (`7c45eabe8e3bce76598ac58eb2c2577d8c0ef6d0a88755e248595ffed7d1e2c0`) and bytes (`5124430`), with 14 safe members and no CRC failure.

### SEC-3 — offline inference and cross-file state: PASS

- The runtime sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and disables telemetry before Transformers imports; model/feature-extractor loading uses local asset paths and `local_files_only=True`.
- Package verification resolves every manifest path and rejects paths outside the package root before hashing.
- Per-file music features are computed from the current file and pooled only within that file. `compare()` requires a full permutation and strict subset and rejects any probability difference above `1e-7`; cached/raw music parity retains the fixed `1e-5` gate.
- The decoder is explicitly pinned to `torchaudio.load + demucs.audio.convert_audio`, avoiding host FFmpeg branch state. The code disables both TF32 flags and cuDNN benchmarking for the runtime.

### OPS-1 — fixed deadline and bounded execution: PASS

- Attempt 6 requires the exact absolute deadline `1789695277.734545`, rejects launch after it, writes the same deadline to run state, reserves 180 seconds for finalization, and wraps notebook execution in `timeout`.
- Earlier full/recovery execution similarly derives remaining time from the passed absolute deadline; it does not reset it between stages.
- The watcher loads the original `workspace_run.json`, rejects non-finite or over-budget deadlines, caps downloads by remaining time, and enters a retrying pause path when the deadline is reached.

### OPS-2 — upload, target, download, and stop controls: PASS

- Launch is fixed to `wsp-5i3knd3gxeox`, expected name `jolly-routine`, resource `resourcespec-a100x1`, object volume `objvol-a5gxdg7er70i`, and prefix `mert12-a100-20260918/`; it refuses active compute, stale init scripts, duplicate run metadata, or changed staged inputs.
- Upload/readback completes before `workspace start`. Dispatch refuses a duplicate refit process and uses a lock inside the workspace.
- Watch downloads only a single named top-level object at a time, verifies whole/member/evidence hashes, and pauses only the fixed A100. MERT12's pause verifier requires `state == stopped` and `phase == standby`; transient deadline-stop failures are retried.

## Direct programming and remove-ai-slops pass

The direct diff pass found no success-criterion violation from speculative abstractions, parsing/normalization layers, or test overfit. The added tests cover practical risks: the decoder regression on real smoke audio, 1,322-row coverage logic/class balance, serialized-head numerical parity, order/subset independence, notebook/CLI integration, and bounded supplemental QA provenance. They do not merely assert deletion, constants, file existence, or implementation-shaped snapshots. The large operational scripts and compact embedded Python reduce readability, but this is a maintenance note rather than a failure of a stated criterion.

No separate final code-review report with explicit `programming` and `remove-ai-slops` coverage was found in the checked evidence directory. Per gate instructions, this is not a blocker because this direct review inspected the diff, production paths, tests, and referenced artifacts and found no criterion failure.

## Checked artifact paths

- `/tmp/mert12-final-review-e31f1f6` at `e31f1f6679b7fb23d1f490b4fceb10813c22a80b`
- `/Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/mert12_submission_build/final_review_context.md`
- `/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/MERT12_full1322_submit.zip`
- `/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/MERT12_full1322_submit.json`
- `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/{workspace_run.json,run_state.json,resource_stop.json,download_verification.json,REPORT.md}`
- `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/storage_preparation.json`
- `/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/mert12_a100_20260918/upload/smoke_inputs.zip`
- Reviewed source: `mert12_a100_launch.py`, `mert12_a100_watch.py`, `redesign_h100_watch.py`, `mert12_submission.py`, `package_mert12_submission.py`, `mert12_supplemental_qa.py`, `test_mert12_submission.py`, `test_mert12_supplemental_qa.py`, `_workspace/mert12_a100_20260918/upload/bootstrap.sh`, `_workspace/mert12_a100_20260918/final_qa_bootstrap.sh`, `_workspace/mert12_a100_20260918/resume_preflight.py`, and `_workspace/mert12_a100_20260918/jupyter_terminal.mjs`.

## Exact evidence gaps (non-blocking for this static review)

1. No fresh attempt-6 final QA receipt, executed QA notebook, final export/download verification, or attempt-6 `state=stopped` plus `phase=standby` receipt was available at review time.
2. No DACON submission receipt or official evaluation result was available.
3. No independent validation remains after merging all former local splits; this review therefore does not establish quality improvement, generalization, official score, or L4 runtime compliance.
4. The current human `REPORT.md` correctly states that final GPU QA and official submission remain outstanding; it must be updated only from the later attempt-6/submission artifacts.
