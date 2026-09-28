# Existing9504 A100 workspace replacement — 2026-09-19

User explicitly authorized replacing the hung A100 workspace while preserving both volumes and resuming the existing experiment. Scheduled five-minute monitoring remains PAUSED.

- Original: `wsp-5i3knd3gxeox`; termination requested. Control plane still reports `terminating` / `terminated`; completed deletion/billing release is not yet independently established.
- Replacement: `wsp-y519wpuarqc0`, `dacon-existing9504-a100-resume-20260919`, cluster `cluster-betelgeuse`, A100-SXM4-80GB.
- Cluster volume `clustervol-4jnjmzdw5rik` remains mounted at `/root`; object volume `objvol-a5gxdg7er70i` at `/root-1`. No volume deletion requested.
- An initial replacement `wsp-p6ra2xz0ucd2` exposed no endpoints at the time of inspection. It was terminated without running the experiment. Explicit ports 22/8888 were rejected as reserved. The final replacement includes an authenticated additional Jupyter service on 8889; normal Jupyter/SSH endpoints subsequently became available. Operations use the normal Jupyter API.

## Preserved experiment

The 9,504-row dataset and train/val/test counts 6,652/1,426/1,426 are unchanged, as are frozen assets, architecture, and author-code archive SHA256 `d386b833c94d3bce4683f79e67ced8dbf81c2194a404e502de8b09af1575b683`. The notebook now reads the workspace ID from the deployment descriptor when building configuration. This preserves cache implementation fingerprints and prevents the old workspace ID from entering new handoff metadata. Earlier notebook/deployment snapshots are preserved locally and under `before_workspace_replacement_20260919/` on the object volume.

## Storage and cache recovery

Actual 4 MiB write, fsync, SHA256 readback and removal passed on both mounts: `/root` 0.056 s, `/root-1` 2.817 s. Before compute, GPU memory was 0 MiB and no GPU process was listed. The original job's exclusive controller lock was acquired successfully on the replacement.

All 681 indexed cache files were checked. 677 matched; four failed SHA256 and were moved, together with the previous index and discrepancy receipt, to `/root/dacon_existing9504_a100_20260919/recovery_20260919_hash_mismatch/`. Only these four entries were removed from the active index, so they will be recomputed by the normal extractor. No source audio was deleted. See `storage_probe.json` and `quarantine_receipt.json` for exact IDs and hashes. Why those four byte streams differ is not proven.

Full input validation passed for all 9,504 audio files, exact split counts, frozen assets and source modules. The existing 83-package venv was reused; FFmpeg 6.1.1 was reinstalled into the new container and real torchcodec decoding passed. The pilot fingerprint matched and the prior measured MERT batch 256 / peak 25.1 GiB receipt was reused. See `preflight_resume.log`. This run has historical test exposure and is not an independent evaluation or an official leaderboard result.

## Verified resume and stopping point

Feature extraction resumed at 14:29:33 UTC with 677 fully validated existing cache entries. At 14:31:11 UTC, progress reached 843/9504 (166 newly produced rows, including the four quarantined entries). All four recomputed cache files were read back and SHA256-verified against the new index. The latest saved progress is 1002/9504 at 2026-09-19T14:32:11+00:00. No optimizer training or final ZIP completion is claimed. The same notebook kernel is executing features → train → test_eval → parity → handoff sequentially. The API client detaches without interrupting that kernel.

The user requested that active assistance stop after healthy execution was established. No recurring monitoring was enabled; the heartbeat remains PAUSED. Original H100 remains stopped. Old A100 lifecycle status is preserved in `lifecycle_final.json`; do not equate a termination request with confirmed final control-plane deletion.
