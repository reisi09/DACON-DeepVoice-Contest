# Existing9504 A100 runtime repair — 2026-09-19

User authorized Codex to directly repair runtime errors. Original Claude v3 remains in `../claude_v3`.

## Confirmed cause and minimal fix
VESSL notebook inherited `FORCE_COLOR=1` and `CLICOLOR_FORCE=1`. `uv pip freeze` exited successfully but emitted ANSI-decorated package names; the strict pin parser returned zero instead of 83 packages. `uv_env` now removes those two variables and sets `NO_COLOR=1`. Toggling color reproduced and removed the failure on the live host. Package pins, model architecture, data, and optimizer logic are unchanged.

Regression: run `code/tests/test_environment_color.py` with the target venv Python and real uv on PATH. It compares parsed freeze output against installed distribution metadata under forced color. Passed locally and on the actual Linux environment. The regenerated notebook and ZIP are SHA-bound; cloud uploads were read back and checked.

## Actual runtime evidence
- All 9,504 selected audio files had already passed extraction/hash checks.
- 83 pinned packages and18 required imports passed; actual torchaudio/torchcodec decode passed after built-in FFmpeg installation. A100 80GB visible to PyTorch2.9.1/CUDA12.8.
- All24 train-only pilot samples completed. MERT batch256, peak allocated26,930,737,664 bytes. Feature-only projection7.797h, not a total-run guarantee.
- Remaining Run All stages resumed in the existing notebook kernel: features, train, test_eval, parity, handoff. At13:36UTC,163/9504 full-feature rows were complete and exactly one GPU compute process was present. Optimizer training had not started.
- Existing heartbeat now checks this A100 job every5minutes, stays quiet on unchanged progress, handles actual failures, and downloads/verifies the final handoff before pausing compute. No duplicate launch, different experiment, or DACON submission is authorized.

## Files
`DACON_Existing9504_A100.ipynb` and reproducible `author_code.zip` are deployed at `/root-1/existing-only-9504-20260919/`. Build ZIP with `code/tools/build_author_zip.py`, then bind it with `code/tools/make_notebook.py`. Original failed-run files are preserved locally under `before/` (large ZIPs are Git-ignored). See `runtime_recovery.json`, `environment_receipt.json`, `pilot_receipt.json`, `mert_batch_benchmark.json`, and `initial_features_health.log`.

Future successful training, final parity, and handoff are not claimed here. Test has1001 historical-exposure rows; singing is train-only.
