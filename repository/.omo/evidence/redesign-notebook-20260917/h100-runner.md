# H100 operational runner evidence

Scope: local contract and failure-path checks for the bounded VESSL bootstrap. No cloud workspace was created and no H100 claim is made here; live CUDA, the 100-row pilot, full feature extraction, and training remain runtime gates.

## Frozen interface

- Runner: `redesign_h100_run.py`
- Bootstrap: `_workspace/redesign_20260917/cloud/redesign_h100_run.sh`
- Remote root: `/root/redesign_h100_20260917`
- Shared prefix: `/shared/redesign-h100-20260917`
- Input ZIP: root runner/notebook/runtime modules plus `input/sources.csv`, `input/audio/*`, `input/asset_hashes.json`; flat `INPUT_SHA256.json` covers every other regular member.
- Shared receipt: `upload_receipt.json` with `sha256`, `bytes`, `files`, `prefix`.
- Final shared objects: `run_state.json`, `results.zip`, `results.zip.sha256`, `export_receipt.json`.

## Meaningful checks

1. Input integrity and copied-notebook injection
   - Invocation: `python3 -m py_compile redesign_h100_run.py`, `bash -n .../redesign_h100_run.sh`, then the Python contract scenario captured in `h100-runner-smoke.log`.
   - Scenario: accept an exactly covered safe ZIP, reject a member with a false declared digest, reject `../` traversal, patch the actual released notebook to A/B and `B_ensemble`, disable compare-and-lock/open-sealed, and replace training knobs with injected values.
   - Binary observable: command exit 0; every boolean in `h100-runner-smoke.json` is true.
   - Artifacts: `h100-runner-smoke.json`, `h100-runner-smoke.log`.

2. Failure still produces verifiable partial results
   - Invocation: run the real shell wrapper against a hash-valid deliberately incomplete bundle, with synthetic `nvidia-smi`/`timeout` commands only for this local failure-path scenario.
   - Scenario: runner fails before notebook execution because required bundle members are absent; wrapper must retain failed status and create a CRC/readback-verified partial archive rather than losing evidence.
   - Binary observable: wrapper exits nonzero; `run_state.status=failed`; `archive_ready=true`; `archive_verified=true`; whole archive digest agrees across the archive, run state, and export receipt.
   - Artifacts: `h100-wrapper-failure.json`, `h100-wrapper-failure.log`.

## Runtime boundaries

The wrapper uses the caller-supplied `REDESIGN_WORKSPACE_CREATED_UNIX` unchanged for the four-hour hard deadline and 15-minute export reserve. It samples GPU utilization, memory, and power every 10 seconds. It checks the frozen runtime first and installs only missing notebook packages at `nbformat==5.10.4`, `nbclient==0.10.2`, and `ipykernel==6.29.5`, preferring the known `uv pip --python` path. The runner verifies all 27 materialized assets against the pinned manifest before CUDA, prepares historical data with `diagnostic=True`, runs only A/B with `B_ensemble`, records C/D as unavailable because there are no paired slots, forbids lock/sealed paths, marks comparison reports diagnostic-only, and always packages success or failure evidence.

`redesign_h100_run.py` is a single cohesive bootstrap boundary because the verified ZIP must extract one root runner before any project module is trusted. Its 250-line exception avoids a second pre-verification helper that would itself require a bootstrap trust mechanism.

## Live bootstrap repair

- Observed failure: the actual target `/root/fullft8/.venv/bin/python` reported `No module named pip`; `/root/.local/bin/uv` was also absent. No torch package was changed.
- Patch: both frozen shell copies now fall back to `python -m pip --python "$PYTHON" install ...`, using VESSL system pip to manage the pip-less target environment. The existing exact pins remain `nbformat==5.10.4`, `nbclient==0.10.2`, and `ipykernel==6.29.5`, and only missing modules are selected.
- Reproduction: created a real `venv --without-pip`, confirmed its interpreter cannot import pip, then successfully used system pip's `--python <target>` mode to inspect that environment.
- Binary observable: reproduction exit 0, both shell copies byte-identical, both pass `bash -n`.
- Artifacts: `h100-pipless-repair.json`, `h100-pipless-target-list.json`.

## Live export-race and diffq repair

- Both shell copies now stream the active 10-second sampler to `ROOT/gpu_telemetry.live.csv`, outside the runner's `output` and `evidence` export roots. The existing EXIT trap is unchanged.
- A separate missing-`diffq` guard installs exactly `diffq==0.2.4` with `--no-deps`, using the same uv-then-system-pip target selection. No torch install or upgrade command is present.
- Binary observable: both copies pass `bash -n`, are byte-identical, and static contract assertions confirm the telemetry destination, exact diffq pin, `--no-deps`, and absence of torch installation.
- Artifact: `h100-shell-race-diffq-fix.json`.
