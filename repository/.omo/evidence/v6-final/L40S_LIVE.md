# Live L40S run — 2026-09-15 KST

## COMPLETE — downloaded, verified, Paused

The trial is complete. Do not restart it automatically.

- Runner completed at **2026-09-14 18:54:31 UTC / 03:54:31 KST**.
  All 1806 feature records processed; music head finished 300/300 epochs on
  823 music files, fusion used 459 files, dev evaluation used 210 files.
- File EER 0.3857142857142857; voice EER 0.4291666666666667 on only 100 known-label
  files; music EER 0.4828571428571429 on 170 files; music presence AUC
  0.9954411764705886. Composite score is unavailable because voice presence labels
  lack both classes. The reused dev set is not a fresh untouched holdout.
  These results do not establish a performance improvement or leaderboard rank.
- Local deliverables: `results/v6_l40s_20260915/`. Submission ZIP is 6005189474 bytes,
  SHA256 `d7f41cea68f04940fa815a4baa4d81378ad0bc0b87ce51468bd95587dabbf731`.
  Small results ZIP and full submission ZIP both matched the remote SHA256 manifest.
  The submission embeds the exact downloaded heads, training metadata and metrics.
- Executed notebook: all 6 code cells executed, no error outputs. Metrics recomputed
  from saved predictions and dev labels match exactly. Final heads hash matches
  training metadata. L4 submission inference runtime remains unverified.
- Results also preserved in Object volume `objvol-a5gxdg7er70i`, prefix
  `results/v6_l40s_20260915/`. Feature cache and original files remain in `/root`
  on persistent cluster storage. No cloud data was deleted and no leaderboard
  submission was made.
- After local download verification, CLI Pause was requested and **state=stopped**,
  phase=standby were confirmed. Final balance **15.39291 credits**;
  reported burn rate **0/hour**. Total GPU runtime including initial upload/setup:
  7985.152639 seconds. This is not a claim that retained storage is permanently free.
- Heartbeat `vessl-l40s-pause` is now **PAUSED**. No further automatic runs needed.
- Final checks and cloud state are saved as `verification.json` and
  `cloud_final_state.json` in the local results directory.

## Current override: upload verified, GPU resumed for server-side download and training

- 02:53 KST heartbeat found the first notebook run failed during PANNs import:
  inherited `MPLBACKEND=module://matplotlib_inline.backend_inline` was rejected by
  matplotlib 3.9.2 inside the isolated Python3.11 environment. No trained heads yet.
  Direct remote reproduction with that backend exited 1; changing only MPLBACKEND
  to Agg made the same `import panns_inference` exit 0 and report backend `agg`.
- Fixed the runner's environment to `MPLBACKEND=Agg` before starting nbclient.
  Runner now skips download when the complete-sized bundle is already present,
  but always checks the full sealed SHA256 before notebook execution.
- Relaunched runner at about 17:54 UTC, shell background PID 699 (check current
  process rather than relying on this PID). Original compute deadline is unchanged.
  Failure diagnostics were preserved remotely as `/root/v6_backend_failure.log`
  and `/root/V6_L40S_BackendFailure.ipynb`; reproduction logs are
  `/root/v6_backend_bad.log` and `/root/v6_backend_good.log`.
- Reproduce the exact dependency failure mode on the remote host:
  `MPLBACKEND=module://matplotlib_inline.backend_inline /root/V6_L40S/.venv/bin/python -c 'import panns_inference'`
  (fails), then
  `MPLBACKEND=Agg /root/V6_L40S/.venv/bin/python -c 'import panns_inference,matplotlib; print(matplotlib.get_backend())'`
  (passes). Actual notebook execution remains the end-to-end check.

- At the 02:32 KST heartbeat, the local CLI reported `Uploaded 1 file(s), 0 failed`.
  Remote file length is exactly 9247576070 bytes. Local 5MiB-part multipart MD5
  matched the remote ETag `1afe9a460c3319442f97fe0a7d25b2e8-1764`; local full SHA256
  also matched the sealed bundle. Evidence: `object-upload-verified.json`.
- `vesslctl billing show` before resume: balance 18.014171, burn rate 0.
- GPU resumed after verification under the existing training authorization.
  CLI reported Running and actual `startedDt=2026-09-14T17:35:53.688312Z`.
- Updated `l40s_remote_runner.py` was pasted into `/root/v6_remote_runner.py`
  through Jupyter. It uses preinstalled `vesslctl` and workload authentication to
  download the object volume into `/root/v6_object_download`, checks size and full
  SHA256, then replaces the old partial `/root/V6_H100_RunBundle.zip`.
- The existing `/root/V6_L40S/.venv/bin/python` survived Pause. System Python
  successfully imported nbclient/nbformat before the runner launch.
- Launch environment: `V6_RESUMED_UTC=2026-09-14T17:35:53.688312+00:00`,
  `V6_AVAILABLE_CREDIT=18.014171`. Remote runner PID 164.
- Confirmed `/root/v6_run_state.json` stage `downloading_object_storage` at
  2026-09-14T17:36:35Z. No runner error in the log at that check.
- New notebook execution cap: 4 hours from actual resume, with 15 minutes reserved
  for downloading and Pause. Notebook process deadline **21:20:53.688312 UTC**;
  final Pause cutoff **21:35:53.688312 UTC / 06:35:53 KST**.
  The old 21:27:43 UTC cutoff below is superseded.
- Budget validation executed the actual substituted notebook parameter cell:
  verified four-hour cap minus 15-minute reserve; $2 available credit was rejected.
- The 15-minute heartbeat now monitors the remote runner/training again. Never
  restart the old browser upload or the already completed local upload.
- Download/verify final results then Pause. On irrecoverable errors, preserve
  diagnostics and Pause promptly. CLI Pause is available locally as a fallback:
  `_workspace/bin/vesslctl workspace pause wsp-tmcepusnv1cc`.

## Previous override: GPU paused, direct Object storage upload

The user requested: pause the workspace and upload first. This section supersedes
the old browser-upload and running-GPU status below.

- Pause was confirmed in Aside at about 02:13 KST. CLI subsequently confirmed
  `state=stopped`, `phase=standby`, total runtime 2736.303563 seconds.
- Last UI balance: 18.01 credits. No GPU compute is running in this workspace.
- `/root` is the mounted persistent cluster volume `clustervol-4jnjmzdw5rik` (`main`).
  Its partial Jupyter ZIP remains; do not use it as a complete bundle.
- Local official CLI: `_workspace/bin/vesslctl`, version `2026.09.14-02`.
  Installed from VESSL's official installer with checksum verification; browser
  OAuth succeeded for organization `qwdkw`, team `Default`.
- Object volume: `dacon-v6-training`, slug `objvol-a5gxdg7er70i`.
  Backend: `object_storage_e322branyapw`. No public sharing was configured.
- Upload command: `_workspace/bin/vesslctl volume upload objvol-a5gxdg7er70i
  _workspace/V6_H100_RunBundle.zip --remote-prefix v6/`.
- Active upload PID at launch: `17705`; unified exec session `6211`.
  Log: `_workspace/v6_object_upload.log` (CLI may buffer output until completion).
  An earlier detached launch PID 17664 exited without transferring; it is not active.
  `_workspace/v6_object_upload.pid` has been corrected to the active PID.
- Actual transfer was observed using `nettop`: 736687513 outbound bytes for PID
  17705. This proves transfer started, not that the remote object is complete.
- Expected remote key: `v6/V6_H100_RunBundle.zip`; expected bytes `9247576070`;
  SHA-256 `55981049ccf5c15c7e85a4c6f32bf53e40bb4c0bef58033224962186e06c6ac2`.
- Heartbeat `vessl-l40s-pause` now checks this local upload every 15 minutes.
  Keep quiet for ordinary progress. Check CLI log/process and remote listing;
  do not infer upload failure from an empty log or a missing incomplete multipart object.
- Keep GPU Paused until the upload and integrity checks finish. Before resuming
  the previously authorized trial, adapt the runner to read from Object storage
  and update its old absolute deadline/budget for the new compute session.
  Do not restart the old Jupyter upload or assume its paused process survives.
  Preserve the original requirement: download and verify completed training results,
  then Pause and confirm. Do not buy credits or submit to DACON.

## Historical setup (before the pause request)

User requires 15-minute spaced checks, result download on completion, then confirmed Pause.
All GPU operation, notebook/data upload, training, downloading and Pause are authorized.
Do not request redundant authorization. Do not buy credits or submit to DACON.

- Active workspace: `canine-funny`, `wsp-tmcepusnv1cc`
- Console: https://cloud.vessl.ai/qwdkw/Default/workspaces/wsp-tmcepusnv1cc/connect
- Jupyter: https://jupyter-wsp-tmcepusnv1cc.betelgeuse.cloud.vessl.ai/lab
- GPU: L40S x1, 48GB; 10vCPU / 97GB RAM, $1.80/hour.
- Created/compute-start reference: 2026-09-14 16:27:43 UTC (01:27:43 KST).
- Initial displayed balance: 19.38 credits. Compute ceiling $15; absolute operational
  cutoff **2026-09-14 21:27:43 UTC / 2026-09-15 06:27:43 KST** (five hours, ~$9).
  Preserve/download outputs and Pause by this time. Never leave running unattended.
- Old H100 workspace `hospitable-object` was still Paused after failed Start attempts;
  it is not the target. Do not retry H100.
- UI image says Python3.13/CUDA13, but real remote `python --version` is **3.14.6**.
  Isolated `/root/V6_L40S/.venv` Python3.11 was bootstrapped with uv0.11.19.
- Dependencies installing via `/root/v6_requirements.txt`, matching bundle pins
  torch2.7.1+cu128/torchaudio2.7.1+cu128/torchvision0.22.1+cu128.
  Log `/root/v6_setup.log`; Jupyter Terminal 1 shell ran install background job [1].

## Transfer and automatic execution

`V6_L40S_Experiment.ipynb` uploaded into `/root` and opened in Jupyter. `/root` is
persistent cluster storage, shown as 10T capacity in df (not an assertion about quota).
`V6_H100_RunBundle.zip` (same model/data used on L40S) is **uploading through Jupyter UI**.
At ~01:35 KST it was 309MB / 3.52%. Do not refresh/close the Jupyter tab while upload
runs. This tab owns the browser upload. Initial large-file warning was confirmed.
Local source `_workspace/V6_H100_RunBundle.zip`, expected bytes **9247576070**,
SHA-256 **55981049ccf5c15c7e85a4c6f32bf53e40bb4c0bef58033224962186e06c6ac2**.

`/root/v6_remote_runner.py` was pasted via the Jupyter terminal. Command sequence:
wait for setup background job, then nohup python runner > v6_notebook_runner.log.
Runner waits on bundle size every 60 seconds (server-side, no model tokens), verifies
SHA-256, uses nbclient to execute the complete notebook and saves outputs even on error.
It adjusts the notebook's internal 3-hour bound to **5 hours** for L40S/upload overhead,
still below compute and trial budgets. The saved executed notebook contains this change.

Check **`/root/v6_run_state.json`** first for waiting_upload/checking_bundle/
executing_notebook/complete/error. Also:
- `/root/v6_notebook_runner.log`
- `/root/V6_L40S/v6_run/progress.json` during feature extraction
- `/root/V6_L40S/v6_run/training.json` on successful training
- `/root/V6_L40S/v6_run/dev_metrics.json`
- `/root/V6_L40S/V6_final_submission.zip`
- `/root/V6_L40S_Executed.ipynb` (saved after execution or failure)

Kernel shown in original notebook is idle while nbclient uses its own kernel;
that idle indicator is not evidence that training stopped. The runner/progress files
are authoritative. GPU is currently already running: ensure completion/error/cutoff
causes a Pause, not just a kernel stop.

## Completion

Download final submission ZIP, executed notebook and the small result files
(heads.npz, training.json, dev_metrics.json, dev_predictions.json, losses.json,
training_loss.png) into local `results/v6_l40s_20260915/`. Prefer a small result ZIP
created through the Jupyter terminal plus separate large submission ZIP.
Verify complete downloads/local hashes, then use VESSL UI Pause and confirm Paused.
On unrecoverable failure, preserve diagnostic outputs and Pause promptly.

Heartbeat automation ID `vessl-l40s-pause` is ACTIVE every 15 minutes. Stay quiet
for normal unchanged status. Disable after downloads and confirmed Pause.

Computer Use: persistent node_repl `sky=(await import('@oai/sky')).sky`; use
Aside bundle id `at.studio.AsideBrowser`. Get fresh AX state after UI actions.
AX terminal content is absent; emit screenshot for terminal output, or open the
small progress JSON in Jupyter editor. Filter huge logs before returning AX text.

Verified at 16:36:53 UTC: setup completed successfully, nohup runner PID 1128,
`v6_run_state.json` stage=waiting_upload, uploaded_bytes=416284672.
Server-side runner is active. Remote notebook execution will start after SHA-256 verification.
