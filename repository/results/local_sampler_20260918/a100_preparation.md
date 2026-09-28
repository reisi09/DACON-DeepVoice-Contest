# Existing A100 preparation

The live CLI read on 2026-09-18 confirmed `wsp-5i3knd3gxeox` (`jolly-routine`) is stopped/standby, uses one A100 SXM 80 GB, and has an empty `initScript`. Reuse this workspace. No workspace was created, started, paused, patched, or terminated during this preparation, and no upload was performed.

`wsp-9683432r72m7` remains stopped/standby. Its old `initScript` still calls `bash /shared/redesign-full-20260917/redesign_full_refit_run.sh`; preserve this B workspace in the stopped state.

## Control and authentication

- CLI: `_workspace/bin/vesslctl --org qwdkw --team Default`, version `2026.09.14-02`. `.venv` has no `vessl` Python package.
- CLI config and credentials are already configured under `/Users/choiwuseck/Library/Application Support/vesslctl/`; no login is needed. Never print `credentials.json` or the raw `workspace show` output, which includes Jupyter endpoint tokens.
- Read: `workspace show wsp-5i3knd3gxeox -o json`; whitelist workspace/billing/volume fields before logging.
- Start: `workspace start wsp-5i3knd3gxeox`; stop compute while preserving data: `workspace pause wsp-5i3knd3gxeox`.
- Installed CLI has no workspace update/patch command. No patch is needed because the reused A100 has an empty init script. An authenticated HTTP PATCH method was not independently established and must not be guessed.
- Runtime dispatch: `_workspace/mert12_a100_20260918/jupyter_terminal.mjs` obtains a Jupyter terminal through the API and sends the command over its WebSocket. It accepts the credential-bearing endpoint only on stdin and returns terminal/PID only.
- The [official CLI reference](https://docs.cloud.vessl.ai/cli/cheatsheet) documents the same show/start/pause workflow.

## Storage and reuse

| Item | Confirmed value |
|---|---|
| Object volume | `objvol-a5gxdg7er70i`, `dacon-v6-training` |
| A100 object mount | `/root-1` |
| Object used bytes | `48,949,369,672` at initial inspection |
| A100 persistent root | `clustervol-4jnjmzdw5rik`, mounted `/root`, name `main` |
| Persistent root used bytes | `61,834,130,686` at initial inspection |
| Backing cluster-storage capacity | `10,240 GiB` |
| Ephemeral storage allocation | `200,000,000,000` bytes |
| A100 compute rate | `1.48` credits/hour; four hours is at most `5.92` listed compute credits before any separate charges |

Capacity and used bytes are API metadata, not a live `df` measurement. The CLI cannot list cluster-volume file contents while stopped. Check actual free space immediately after release and stop on insufficient capacity. The full sanitized API snapshot is in `a100_preparation.json`.

The new object prefix `localization-a100-20260918/` was empty at inspection. Use `/root/localization_a100_20260918` for extracted inputs and working outputs; keep `/root-1/localization-a100-20260918/` flat. Existing `mert12-a100-20260918/` holds many old bootstraps and results and must not be reused for this experiment.

Previously verified persistent assets are at `/root/mert12_assets`: MERT-330 plus DF-Arena, PANNs, HTDemucs, and baseline code (27 hashes verified by the previous run). This is historical evidence; recheck required hashes after startup. Live object listing confirms pretrained MERT/EAT/HTDemucs directories under `/root-1/fullft7/h100x1-20260916/models/`. The EAT `model.safetensors` object is `1,235,498,104` bytes.

No original AST weights were found in the bounded local/cache search or the inspected model prefixes. `v2_validation/drive_inputs.json` records original `MIT/ast-finetuned-audioset-10-10-0.4593` assets at revision `f826b80d28226b62986cc218e5cec390b1096902`, archive size `346,433,653`, SHA-256 `54c227b873cc3c424579fa06e8a06c442b91663de5b740b182f7cac30a3fac75`. The old `_workspace/v2_discovery/ast_assets_receipt.json` and original assets are absent locally. This does not establish that an AST copy exists in the stopped A100 filesystem.

## Existing runtime evidence

The latest successful A100 QA used `/root/V6_L40S/.venv/bin/python`. The recorded environment in `results/mert12_a100_20260918/attempt5/downloaded/unpacked/env311_probe.json` is:

- Python `3.11.15`; PyTorch `2.7.1+cu128`; CUDA runtime `12.8`.
- torchaudio `2.7.1+cu128`; torchvision `0.22.1+cu128`; imageio-ffmpeg `0.6.0`.
- The probe's FFmpeg binary was `7.0.2-static` under the prior run's extracted wheel.
- Transformers/NumPy/SciPy/safetensors exact versions are not present in this receipt. Collect them on startup instead of inferring them from the local environment.
- Older training used `/root/mert330-a100-venv/bin/python` and PyTorch `2.9.1+cu130`; do not silently substitute it for the Python 3.11 runtime.

The prior hardware receipt reports `NVIDIA A100-SXM4-80GB`, `85,094,825,984` CUDA memory bytes. Benchmark batch 32/64/128 in the new workload before choosing throughput settings; this preparation does not claim those batches fit.

## Implemented ops and release contract

`localization_a100_ops.py` owns preparation, launch, watch, and pause. It reuses `redesign_h100_watch` for CLI calls, ZIP/member SHA-256 verification, bounded downloads, and five-minute monitoring. An independent deadline-guard subprocess pauses the A100 even if ZIP verification blocks the main monitor. `test_localization_a100_ops.py` exercises meaningful paid-compute gates offline.

Before release, the root agent must provide exactly:

```text
_workspace/localization_a100_20260918/upload/bootstrap.sh
_workspace/localization_a100_20260918/upload/input.zip
_workspace/localization_a100_20260918/upload/upload_manifest.json
```

The manifest maps `bootstrap.sh` and `input.zip` directly to `{ "bytes": N, "sha256": "..." }`. Preparation uploads the folder, downloads all three objects into a temporary local directory, and compares file sets, byte counts, and complete SHA-256 hashes before writing `storage_preparation.json`.

```sh
.venv/bin/python localization_a100_ops.py prepare
.venv/bin/python localization_a100_ops.py launch --release
.venv/bin/python localization_a100_ops.py watch
.venv/bin/python localization_a100_ops.py pause
```

Launch is not authorized by this document. It requires root release after local analysis, data, and GPU code are ready. The launcher writes `workspace_run.json` exclusively, uploads it, starts the local detached monitor and its deadline guard, then requests A100 start and dispatches only the new prefix's bootstrap. An existing local run file or remote non-bundle object rejects launch. The deadline is recorded once as start + `14,400` seconds and cannot be extended by monitor restart. A failed prepare retry invalidates any previous preparation receipt before remote upload.

Bootstrap receives positional `STARTED_EPOCH` and `PAUSE_DEADLINE_EPOCH`. Export must reserve time for download and verification before the deadline. `run_state.json` uses running/completed/exported/failed with `archive_ready` and `archive_sha256`. `results.zip` must include `EXPORT_SHA256.json` covering every payload member. The sidecar and export receipt must bind the same digest; the receipt also records archive size, payload member count, status exported/partial_exported, `archive_ready=true`, and `archive_verified=true`.

`localization_bootstrap.sh` uses the recorded Python 3.11 executable for both preflight/export and ML work. It checks at least 30 GB free, all 27 original model hashes, pinned AST hashes, and one A100 with at least 79 GiB CUDA memory. Feature/training deadlines stop 600 seconds before the original pause deadline. A background publisher mirrors feature/training progress, MERT/AST batch benchmarks, and a short current log tail to flat `progress.json` every 300 seconds.

The root-prepared offline wheels are installed with `--no-deps --no-index --target /root/localization_a100_20260918/runtime`, and that overlay precedes the source directory in `PYTHONPATH`. Preflight requires Transformers `4.57.6`, tokenizers `0.22.2`, huggingface-hub `0.36.2`, safetensors `0.8.0`, and the recorded PyTorch/torchaudio `2.7.1+cu128`. It imports the actual feature/training modules and records wheel hashes. The original V6 environment is not modified. Optional distribution metadata is recorded as null when absent; the unused imageio-ffmpeg package is not a launch requirement.

`localization_export.py` exports any available evidence after successful, failed, or timed-out work. A completed export requires the exact primary nine checkpoints (mean/MIL/AST × seeds 41/42/43) plus the three added `original831_mean` diagnostic controls, matching lock hashes, matching selected updates, and completed evaluation receipts. Missing or changed checkpoints are exported with failed/partial status. Full export copies code, data/model manifests, runtime checks, progress logs, model checkpoints, and predictions; it does not copy the large intermediate feature cache.

Offline checks passed: wrong target, stale automatic bootstrap, running target, changed verified bundle, corrupted upload readback after a successful prepare, ambiguous start-failure cleanup, immutable restart deadline, independent deadline guard, unexpected monitor failure cleanup, invalid deadline bounds, completed primary/control export, and changed-control rejection with verified partial evidence. Commands and exact source hashes are recorded in `results/localization_a100_20260918/ops_verification.json`. This is control-path and package-integrity evidence, not model-quality or live GPU execution evidence.

An independent final review found no concrete launch blocker after correcting the preparation-retry, exceptional-startup cleanup, blocked-verification deadline, and unused-runtime-dependency findings. Live container preflight and the actual GPU experiment remain pending root release.
