# H100 live launch evidence

Scope: same authorized H100, diagnostic A/B only; no submission or sealed-panel opening.

- Notebook: `781d89c868773c12e539f935dca31194cae37c47d94a291a87f77854ec501fab` (reviewed, unchanged).
- Workspace: `wsp-0z70yb8gv8zn`, original start `1789603030`, hard stop `1789617430` (2026-09-17 12:57:10 KST). Retrying never resets the budget.
- Actual device/runtime: `NVIDIA H100 80GB HBM3`, `torch 2.7.1+cu128`; full declared dependency import plus `torch.cuda.is_available()` returned 0 after missing `diffq==0.2.4` was installed with `--no-deps`.
- Input ZIP was uploaded before GPU creation: 3,563,625,498 bytes, SHA-256 `7e678e61728a9dad98108d7b74f53ebaccc2189cbf4e7b6b4d5f80fdd2365bd5`.
- All 27 frozen assets passed cloud verification. Actual remote data preparation completed before the import gate.
- First bootstrap failure: pip missing in the existing target venv. System pip `--python` installed only missing pinned notebook dependencies. Second preflight failure: missing diffq. Both before/after logs and full CUDA success are saved in `results/redesign_h100_20260917/`.
- Failed attempt 01 was downloaded without pausing compute during repair. Its 6,863-byte partial result archive and all 12 declared members were hash-verified, under `failed_attempt_01/`.
- Corrected bootstrap SHA: `6c90b5624b2d2bf4a8c7095905bed0133280a75b832eabd59e01bff0fc52234a`. GPU telemetry now writes outside exported roots, preventing append/hash races.
- Corrected watcher SHA: `ac61434e2a008c408f008cb895eb7a964cdb41e19966d0ac938040f3decc974e`. Actual CLI download directory semantics are verified; polling excludes the large input archive and unrelated result objects.
- Retry launched 2026-09-17 00:12:51 UTC on the same workspace. Feature extraction/training are not yet claimed by this initial receipt.
- Existing heartbeat `h100-15` ACTIVE every 15 minutes; no user screen manipulation.

Corrected watcher PID93517 replaced PID88310 only after fresh shared run_state returned running/archive_ready=false. Actual runner PID835, original hard deadline unchanged.

## Actual CUDA feature progress

At observation 1789604323.942091, the real 100-row CUDA pilot had completed 33/100 verified feature records. Sampled GPU utilization reached 99.0%, memory 11457.0 MiB. Training has not started at this observation. Full CUDA import check returncode0; frozen assets27/27 existing matches; storage ready with no missing sources. See live_execution_proof.json, feature_progress_at_launch.json, and gpu_telemetry_at_launch.csv.

Debug cleanup: permanent source fixes and real failure/repair evidence retained; no instrumentation, debugger process, or debug port was introduced. Temporary root .debug-journal.md and its session-only git exclude line were removed. Prior unrelated v4-training/source-sha256.txt edits were preserved.

## Completed real CUDA pilot

100/100 files, 1,644 windows completed in 272.330s. Pilot extrapolation for 1,598 train+validation files is 4351.829s (72.53min), not including downstream training/evaluation. The runner accepted the original-deadline budget gate and created notebook_execution.log; full notebook stage is underway, not yet a completed training claim. Source: results/redesign_h100_20260917/pilot.json.

Actual notebook extraction confirmed beyond the pilot: 111/1,598 records at UNIX 1789604543.0942442, with the first100 legitimately reused from the identity-checked pilot cache. This progress ETA includes cached records and is not a reliable full-run estimate; use the separate pilot extrapolation only as a rough estimate.
