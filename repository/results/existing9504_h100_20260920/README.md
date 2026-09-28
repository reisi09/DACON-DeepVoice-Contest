# Existing9504 H100 execution

User-authorized reuse of workspace `wsp-9683432r72m7`. The A100 replacement `wsp-y519wpuarqc0` was verified stopped before dispatch. No recurring monitor was enabled.

## Unchanged experiment

9,504 rows: 6,652 train, 1,426 validation, 1,426 test; each split globally real/fake balanced. HTDemucs base, frozen MERT330M layer12 WindowMIL, original DF-Arena fc5 and PANNs; seeds 41/42/43. Original model assets, dataset manifests, provenance and source-module hashes are unchanged. Test includes 1,001 historically exposed rows and is not independent.

## Deployment

- H100 notebook: `DACON_Existing9504_H100.ipynb` in this directory.
- Object root: `/shared/existing-only-9504-h100-20260920`.
- Runtime root: `/root/dacon_existing9504_h100_20260920` on existing H100 cluster volume `clustervol-qgze68bffy6l`.
- Shared originals remain under `existing-only-9504-20260919` on `objvol-a5gxdg7er70i`.
- Immutable startup entry: `/shared/redesign-full-20260917/redesign_full_refit_run.sh`; previous idle contents preserved in `previous_h100_startup.sh` and uploaded backup.
- `run_notebook.py` executes the notebook bootstrap and the same Pipeline stages. It does not create notebook cell outputs. GPU stages run in the isolated environment.
- Expected final artifact: `DACON_Existing9504_H100_Handoff.zip`; not yet produced.

## Verified before start

75 audio shard metadata records and all 9,504 selected upload entries checked. New notebook, author archive and deployment descriptor passed full upload readback comparison. Archive SHA-256 is recorded in `port_receipt.json` and bound in the notebook. H100 GPU guards, workspace and mount paths were ported; model and training logic unchanged. A100 feature caches are preserved and are not reused across GPU identities.

## Recovery evidence

The Jupyter session was created but its execution connection timed out before any execute request. The native startup hook then passed 4 MiB write/fsync/SHA-256 readback probes on both volumes and preflight. H100 reported NVIDIA H100 80GB HBM3, driver 580.82.07 and 9.8 TiB filesystem free (filesystem-level reading, not a private quota guarantee).

The first native attempt was later observed stopped during frozen-asset staging. There is no Python exception in its captured log; cause is unconfirmed. Its snapshots are under `interrupted_native_attempt/`. Startup was changed to retain the pipeline in the foreground and stream to platform logs, then the same stopped workspace was restarted. This is not proof the underlying interruption was fixed.

Runtime evidence and final health status will be recorded separately. No completed training, quality improvement, independent-test or L4 claim is made.

## User-requested wait

The user requested waiting while VESSL still shows Initializing. Further changes and restarts are suspended. Latest observed control state: Initializing/active; pipeline mirror: preflight complete, stage_inputs running. The foreground initialization script remains active, so a Running UI state has not been established. No GPU feature extraction or optimizer training is verified. The existing heartbeat remains PAUSED.

## Verified completion

A later live inspection found the pipeline complete at 2026-09-19 21:45 UTC (2026-09-20 06:45 KST). The user-requested continuation did not launch duplicate training. The 6,316,973,258-byte handoff was downloaded and its entire SHA-256 matched the remote completion receipt. `verify_handoff.py` then verified all 136 ZIP members by CRC and SHA-256, exact split/label/group invariants, 9,504-row feature-index coverage, six checkpoint hashes and the prediction/metric/selection-lock bindings.

- Validation file EER: 0.07994389901823283 (1,426 rows).
- Test file EER: 0.07433380084151474 (1,426 rows).
- Test contains 1,001 historically exposed rows and is not independent. These are local masked metrics, not DACON leaderboard scores.
- Recorded raw-audio re-extraction parity: bitwise on 10 selected samples; pilot-vs-full comparison covers 24 pilot rows. This does not establish final DACON packaged inference or L4 compatibility.
- Completed archive: `completed/DACON_Existing9504_H100_Handoff.zip`. This is the model/evidence handoff requested by the user, not a DACON `submit.zip`.
- SHA-256: `53a3236bbe201ce4ac2fd021800c4b891731c9eee5146fbf9733fae355460a7d`.
- Re-run local integrity/evidence checks: `.venv/bin/python results/existing9504_h100_20260920/verify_handoff.py`.
- Do not interpret the resumed stage total as total job wall time: the final feature stage reused previously extracted caches after a restart.

The H100 pause was requested after hash-verified download. Final stop evidence is recorded separately in `completed/gpu_stopped_receipt.json`. The original idle startup file was restored and readback-verified, preventing an accidental rerun on a later workspace start. Original datasets, old results and cloud handoff remain preserved; the heartbeat remains PAUSED.
