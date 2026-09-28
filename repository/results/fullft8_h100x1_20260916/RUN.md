# H100 workspace: eight classifiers and HTDemucs adaptation

Workspace: `wsp-fu3a51jw77ln` / `fullft8-h100-20260916-051256`.
Console: https://cloud.vessl.ai/qwdkw/Default/workspaces/wsp-fu3a51jw77ln
Notebook: `MERT_EAT_HTDemucs_H100x1_8Experiments.ipynb`.
Run ID: `run-20260916-051256`.

The user authorized execution, then requested the previously discussed HTDemucs
experiment and prohibited further screen interaction. All subsequent operations
use VESSL CLI or authenticated Jupyter API. The older seven-trial Job
`job-gxeb388rndlt` is terminated; it is not this run.

Hardware: H100 SXM 80GB x1, 15 vCPU, 200GB RAM, 200GB temporary disk.
Persistent root volume: `clustervol-qgze68bffy6l`, mounted at `/root`.
Input/output object volume: `objvol-a5gxdg7er70i`, mounted at `/shared`.
Existing input assets: `fullft7/h100x1-20260916/`, plus the original V6/C3 ZIPs.
New notebook/launcher: `fullft8/h100x1-20260916/`.
Results: `fullft8/h100x1-20260916/runs/run-20260916-051256/`.
Local downloads: `results/fullft8_h100x1_20260916/downloaded/`.

Existing 21 input objects matched recorded sizes (11,905,259,278 bytes total).
New deployment files were downloaded again and matched their SHA-256 hashes.
Runtime rechecks model and original audio/PCM hashes. No duplicate audio upload
was needed. GPU creation occurred after these storage checks and upload readback.

Execution: actual pretrained HTDemucs one-step adaptation and MERT/EAT backward
preflight; frozen-separator S0 trials 01-03; train-only separator adaptation;
trial08 with the tune-selected encoder freshly initialized from pretrained
weights; original S1/S2 trials04-07. The separator uses a fixed teacher, four
parent-disjoint sources per remix, 400 optimizer steps, and held S0-train parent
groups for pseudo-validation. It is inspired by RemixIT, not a reproduction of
its continual teacher-update algorithm. It has no clean-stem supervision.

Only the music branch changes. Historical voice/presence predictions stay fixed.
Pseudo-target reconstruction loss is not ground-truth separation quality, and
Music EER is not the official composite leaderboard score. Seed41 and the
original historical/held-generator evaluation limitations remain.

Training deadline: start epoch1789535576 plus15 hours. Bootstrap/runtime timeout:
15.5 hours. Pause deadline: epoch1789593176 (2026-09-17 06:12:56 KST).
Compute rate:2.98 credits/hour;16 hours=47.68 compute credits, storage separate.
These are execution/monitoring controls, not an account-enforced spending cap.
Do not allocate another GPU or extend the budget automatically.

Heartbeat automation `h100-15` checks every15 minutes. On completion/error it
preserves and downloads results, verifies file integrity, pauses this workspace,
and checks stopped/standby. If download would cross the deadline, pause after
durable remote persistence and finish downloading afterward. Never terminate or
delete volumes. The automation then pauses itself. It stays quiet on routine,
unchanged progress. Token-bearing endpoint data lives only under ignored
`_workspace/fullft8_workspace_private.json`; never print or commit it.

Local checks: standalone notebook/worker compilation and execution with training
off; cache identity rejects a changed separator/recipe or changed waveform bytes;
mirror excludes partial checkpoints and copies completed replacements; real
small BagOfModels optimization, parent-disjoint remix, held-parent selection,
checkpoint reload, and zero-gradient rejection. These are not pretrained H100
performance results. Live preflight/training status must be read from current
logs and result artifacts before claiming success.

Live verification 2026-09-16 05:26 UTC: all three real pretrained GPU backward/optimizer gates passed. HTDemucs: 41,984,456 parameters connected, parameter-update L2 0.05180749. MERT and EAT nonzero optimizer updates are recorded in encoder_gpu_preflight.json. Main notebook reached cell24 and launched prepare_all.log. No classifier trial has completed yet.

The first attempt failed only when EAT resolved a transitive local Python import. The GPU was paused, the failure reproduced in an empty local HF cache, all snapshot Python modules were primed with the native Transformers helper, and real cold-cache EAT loading passed. The same workspace resumed, preserving its inputs and original deadline. notebook.log retains the earlier exception; judge the latest attempt by run_state and the last CELL_START sequence. CLI Object Storage access works; direct Capella SSH/Jupyter ports timed out from this host.

Throughput update authorized after the user observed low H100 utilization:
- Training/evaluation batch size4; gradient accumulation2; effective8 songs per
  optimizer update. Spawn4 loader workers with pinned host memory and nonblocking
  transfer. Recreate workers each epoch so crop selection sees the current epoch.
- Sample-weight the final incomplete accumulation group and preserve the number
  of scheduler/optimizer updates. The sample sequence and effective batch remain;
  dropout random draws and floating-point reduction order are not bitwise equal.
- Trials01/02/03/08 retain completed original results. Trial04 retains its running
  original implementation. Its wall-clock epoch time includes a brief process
  pause for the benchmark and is unsuitable for timing comparisons.
- Trials05/06/07 use the new worker and runtime. Their execution_config.json and
  identities record these settings. Old code/config are batch1_worker.py and
  batch1_runtime.json; old notebook is batch1_notebook.ipynb in the run output.
- A real H100 benchmark compared72 songs per mode (first8 warmup,64 timed), using
  pretrained MERT and EAT, full backward, gradient clipping and AdamW updates.
  MERT10.92 ->34.46 songs/s (3.15x); EAT11.53 ->32.53 songs/s (2.82x).
  Peak allocated tensor memory: MERT13.88GiB, EAT10.94GiB with batch4. Loader/model
  warmup is excluded from these throughput ratios (about7.5s for batch4 workers).
  The live training process was suspended during measurement; its GPU allocation
  remained resident, but it performed no concurrent training.
- Local numerical check matches batch1 vs batch4 SGD updates to1e-12 including a
  partial group of5 songs. Actual spawn workers reproduce direct sample contents,
  preserve order, and change crops when the epoch changes. GPU benchmark checks
  both real encoder paths with9 optimizer updates each, including8 measured ones.
- No extra GPU or budget extension. Same training/pause deadlines. Do not restart
  the whole notebook into this mixed historical output; the identity guard will
  reject incompatible old checkpoints. Continue the already-running supervisor.
Batch4 deployment/readback hashes are in batch4_manifest.json; measured results
and actual deployment record are batch_benchmark.json and batching_deployment.json.

Completion check 2026-09-16 07:28 UTC: eight classifier complete.json records and separator completion verified. GPU utilization/memory were zero; training finished. A 37,601,239,040-byte tar of156 result files is preserved on persistent root, including every best.pt/last.pt. Regenerable audio/teacher/frozen-feature caches remain on root and are excluded. HTTP Range download is in progress; inspect completion_receipt.json and the range ledger, never the preallocated local file size. The heartbeat will verify the archive SHA256 and checkpoint identities before pausing the workspace. GPU is not yet stopped at this checkpoint. Final metrics are in final_metrics.json and RESULTS.md.

Final completion 2026-09-16T08:18:49.043974+00:00: local archive SHA-256 matches the remote manifest (e7690d84e2df4fa12b8405b88b4e56ec9447aef494adc2c56576976288eeb89a). All 156 member paths/sizes and eight selected checkpoint identities verified. Archive size is 37,601,239,040 bytes; large checkpoints remain inside the tar, with small metadata extracted locally. VESSL confirms workspace wsp-fu3a51jw77ln stopped/standby after pause. Persistent root and results retained. Heartbeat h100-15 is PAUSED. No user-screen interaction.
