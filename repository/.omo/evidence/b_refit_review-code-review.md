# Full-data B refit and submission integration review

## Scope and evidence inspected

- `redesign_fit.py`, `redesign_full_refit_run.py`, `redesign_submission.py`, `redesign_submission_smoke.py`, and `package_redesign_submission.py`.
- `test_redesign_full_refit.py` and `test_redesign_submission.py`.
- `results/redesign_full_20260917/{input_audit.json,training_checks.log,packaging_checks.log,source_bundle.json}`.
- `COMPETITION_CONTRACT.md:42-58` for the submission environment contract.

`omo ulw-loop status --json` returned `ULW_LOOP_PLAN_MISSING`; this is the required fallback path. This is an interim report: real H100 refit, extracted-package audio parity, downloaded artifact, and shutdown evidence were not yet available at this review point.

## Findings

### CRITICAL

None.

### HIGH

None in the static diff. Runtime evidence is still required before approval.

### MEDIUM

None.

### LOW

None.

## Verified points

- The refit path uses `fixed_epochs=True` with empty validation rows and seed/epoch pairs 41/17, 42/17, and 43/8 (`redesign_full_refit_run.py:64-68`). `fit_music` retains early stopping for its existing callers and selects the final state only for fixed-epoch runs (`redesign_fit.py:147-259`).
- The runner fails closed on 1,696 unique rows, 1,216 eligible music labels, 480 speech rows with unknown music labels, former split counts 1469/129/98, and the preflight source-manifest SHA (`redesign_full_refit_run.py:44-57,104-128`).
- The package rejects a receipt/checkpoint mismatch, includes no input audio or cache, verifies member digests and CRC, and enforces the 10 GB ZIP / 32 GB expanded limits (`package_redesign_submission.py:87-169`).
- The packaged inference performs per-file extraction only and uses frozen voice/presence scores with the required noisy-OR file formula (`redesign_submission.py:82-99,124-143`).
- The smoke scenario is meaningful: five real representative audio files, silence, 60-second stereo audio, source-to-package parity at `1e-5`, network audit blocking, and a changed-order/subset independence check (`redesign_submission_smoke.py:42-125`). It remains unexecuted evidence until its produced artifacts are inspected.
- Focused tests were rerun from the repository environment: `./.venv/bin/python -m pytest -q test_redesign_full_refit.py test_redesign_submission.py` reported `9 passed in 3.19s`. The committed preparation artifacts additionally report 13 refit checks and 3 package checks. `git diff --check` and Python compilation were clean; Ruff was unavailable in `.venv`.
- `COMPETITION_CONTRACT.md:46-56` documents the official preinstalled dependency set and recommends omitting its duplicates from `requirements.txt`; the reduced requirements list is therefore not a defect.

## Skill-perspective check

Ran `omo:programming` (Python reference), `omo:remove-ai-slops`, and Ponytail perspective.

- `remove-ai-slops`: the tests protect fixed-epoch optimization, masking, deadline behavior, early-stop restoration, archive tampering, score parity, and ID/file independence. No deletion-only, removal-only, tautological, prompt-text, or implementation-constant tests were found. The synthetic archive test does not replace the required actual-audio smoke run.
- `programming`: no untyped escape hatch or needless abstraction was added in the reviewed path. The broad exception at `redesign_full_refit_run.py:124` is a documented top-level evidence-export boundary. The embedded-manifest completeness gap above is the only integrity concern identified.
- Ponytail: the implementation reuses the existing feature pipeline, checkpoint format, baseline I/O, and asset validation rather than creating a second inference stack. The new orchestration is scoped to the authorized B refit.

## Recommendation

`codeQualityStatus: CLEAR`  
`recommendation: APPROVE`  
`blockers: []`

## OOM recovery addendum

Attempt 1 is a verified partial success, not a completed smoke check. Its extraction receipt records 1,696 rows (1,598 cache hits and 98 new records) in 316.38 seconds; its full-refit receipt records the required seed/epoch checkpoint hashes; and its package report records CRC/member verification for a 6,267,071,302-byte archive. The packaged smoke then failed while its child process loaded DF-Arena: `output/smoke.log` records parent PID 93 using 76.14 GiB and only 2.75 MiB free on the H100.

The pending retry changes `redesign_full_refit_run.py:95-103` to drop the last CUDA-resident training result, collect garbage, empty the CUDA allocator cache, and write `pre_smoke_gpu_memory.json` before starting the child. The existing `best.pt` files are identity-bound, so rerunning `fit_music` loads them rather than retraining. `redesign_full_refit_run.sh:7-10` only detaches the worker from the notebook launcher; its worker branch still derives `timeout` from the unchanged absolute deadline. No defect was found in this recovery delta.

## Final execution evidence

- All three downloaded training receipts are complete fixed-epoch refits over 1,696 IDs: seed 41/17 and seed 42/17 have 459 updates each; seed 43/8 has 216. They have no early-stop IDs or validation loss. The three downloaded checkpoint SHA-256 values match `fullrefit_receipt.json` exactly.
- The receipt records 1,216 audible eligible music labels, the authorized 1,469/129/98 former split counts, former sealed membership as training, frozen voice/presence, noisy-OR fusion, and `official_score: null`.
- The OOM recovery receipt reports `allocated_bytes: 0` and allocator reservation reduced from 2 MiB to zero before the independent process. The actual child smoke succeeded.
- The actual H100 smoke exercised seven files (five real files, silence, and 60-second stereo audio). Both source/package parity and changed-order/subset independence have max absolute error `4.869993297518249e-11`, below `1e-5`. Both execution logs contain `NETWORK_AUDIT_GUARD_ACTIVE`.
- Runtime receipt records 120 windows, 35.419 seconds for the seven-file package execution, 20,652,236,800 peak allocated GPU bytes, 22,594,715,648 peak reserved GPU bytes, and 5,623,701,504 peak RSS bytes. It correctly reports `official_l4_validated: false`.
- The 6,267,071,302-byte package and its 6,267,061,930-byte expanded form are under the size limits. The cloud package report and local `submissions/B_full1696/submit.zip` have identical SHA-256 `1fda295b44925d5b255cc1235f9bee57141ace7469370ae3245af113572bfb00`; local CRC readback succeeds and `submit.json` records `exact_cloud_archive_match: true`.
- The exported evidence archive has 22 verified members and SHA-256 `7daf7ef2b709a6e51bf043f67b0e7f13dbcc78fbc46de923073a3dd0fdb10348`. `resource_stop.json` records verified standby pause of the existing workspace after export.

These are H100 smoke and package-integrity results. They do not establish L4 full-evaluation runtime, independent model quality, an official score, or a DACON submission.
