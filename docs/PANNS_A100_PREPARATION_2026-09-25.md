# PANNs A100 execution preparation

Scope: finish D0 feature extraction, run the fixed PANNs comparison, and produce development diagnostics. This preparation does not start paid compute, submit to DACON, or authorize model adoption.

Current status: code, data uploads and local integration checks are complete. **Not yet GPU-ready:** the CPU-only Linux environment build needs paid-resource approval. The existing A100 is still stopped/standby.

## Execution contract

- Existing workspace: `qwdkw / wsp-nvxqw1jf9zdv` (`lb-e00-e04-a100-20260924-v1`), A100 SXM ×1; observed stopped/standby.
- Object volume: `objvol-a5gxdg7er70i`, mounted at `/shared`.
- Inputs: `panns-cnn-v1-20260925/inputs-v3/{run_bundle.tar,launch_lock.json,panns_a100_launch.sh}`; the verified noise tar remains at `panns-cnn-v1-20260925/inputs/noise-v2.tar`.
- Existing canonical audio and pretrained assets remain under `lb-improvement-20260924/a100-preparation-v1/audio/` and `lb-improvement-20260924/weights/baseline-assets/`. They are reused, not uploaded again.
- D0 has 42,939 admitted files, including 28,315 training files and the 900 recovered noise files. Historical roles, masks and parent components remain fixed. D+ is excluded.
- R0 is frozen. R1/R2/R3/R5 each run seeds 41/42/43, effective batch 4, **21,237 fixed optimizer steps**. This is approximately three uniform-sampling passes, not a guarantee of exactly three visits per file. The budget is fixed before inspecting candidate quality.
- The six fake heads, encoders, presence logistic and FILE ensemble remain frozen. Changed presence values feed the original FILE rule.

## Pipeline

`panns_a100_launch.sh` verifies both uploaded archive hashes, rejects changed extraction contents, verifies every reference asset and admitted noise file, and creates the immutable run manifest. It restores a CPU-built, hash-verified Python 3.11 environment to `/root/panns-cnn-v1-20260925/env` before GPU work. Missing CPU environment artifacts stop execution. The A100 stage performs no dependency installation or model download. The CPU environment builder uses the existing hash-pinned dependency lock and strips workload credentials from build subprocesses. Its CPU-only instance requires the pending explicit paid-resource approval.

`panns_cloud_pipeline.py` runs these stages in separate processes:

1. Extract the 43-file CUDA preflight panel and compare against the preserved CPU gold. Maximum tag/presence absolute error is `1e-5`; CPU/CUDA R3 gradient relative L2 must be at most `1e-3`. Failure stops bulk work.
2. Extract the complete D0 block5/stem cache. Each file is committed separately; explicit resume validates existing entries. There is no repeated full-index rewrite.
3. Train the 12 fixed conditions. Only train-role rows are sampled. Save optimizer/sampler/RNG state every 100 steps; SIGTERM retains a resumable checkpoint. Finished conditions are reused only with matching identities and hashes.
4. Infer frozen fake outputs once on development/selection audio with the hash-pinned preserved package. No fake-head retraining or unproven historical CSV reuse.
5. Evaluate R0 and all 12 candidates, recomputing FILE from each candidate's presence values.
6. Produce known-label coverage, presence AUC/Brier/log loss, FILE EER, source/presence strata and paired parent bootstrap diagnostics (2,000 draws, seed 20260925), including R3−R1 structural contrasts.

Durable results are under `/shared/panns-cnn-v1-20260925/run-v1`; intermediate execution files stay under `/root/panns-cnn-v1-20260925`. A hash-verified local SSD activation cache prevents repeated object-volume reads during training. Allow roughly 12 GiB for activations plus package, checkpoints and temporary head-loading memory.

The process ceiling is 48 hours per invocation and 6 hours per training condition, with partial progress preserved. These are stopping bounds, not measured A100 duration estimates. The launcher uses the workload's `vesslctl` to pause the named workspace on exit. `PANNS_LOCAL_QA=1 PANNS_PAUSE_ON_EXIT=0` plus an explicit interpreter is allowed only on Darwin for local packaging QA; Linux production cannot bypass runtime verification or automatic pause. Every restored runtime member is hashed again on reuse, so a stale marker cannot hide corruption. A killed shell or unavailable control plane still requires the operator to confirm stopped/standby; a timeout alone is not proof billing stopped.

## Start only after preparation and compute authorization

The existing workspace retains an old init script with an expired deadline and a guard on its historical `run_started.json`. That object is retained; do not remove the guard artifact or replace the old prefix. Starting the workspace does not select this new PANNs workflow automatically. Connect to the resumed workspace and execute:

```bash
bash /shared/panns-cnn-v1-20260925/inputs-v3/panns_a100_launch.sh
```

The host can inspect, resume, connect and pause with the already authenticated CLI:

```bash
vesslctl --org qwdkw workspace show wsp-nvxqw1jf9zdv
vesslctl --org qwdkw workspace start wsp-nvxqw1jf9zdv
vesslctl --org qwdkw workspace ssh wsp-nvxqw1jf9zdv
vesslctl --org qwdkw workspace pause wsp-nvxqw1jf9zdv
```

No command above was used to start a GPU in this preparation. CLI pause/resume behavior and workload authentication are described in the [official VESSL workspace reference](https://docs.cloud.vessl.ai/cli/commands/workspace).

## Evidence and limits

See `docs/evidence/panns_cnn_execution_20260925/cloud_preparation.json` for artifact hashes, remote readback and storage receipts. Local real-audio checks cover cache reuse, strict resume, exact continuous/resumed training equality, frozen fake inference, adapted presence evaluation and bootstrap ROC arithmetic. The uploaded launcher is exercised locally through real extraction and complete input/asset verification, then must reject this Mac because it is not an A100.

Final uploaded bundle SHA-256: `9356724b1b5ad13d3581c738e8214b39e8101bb303d7a2eadfe585a617c05c3b` (167,014,400 bytes). Recovered-noise archive SHA-256: `6b871c982dbc63c9cbf159551c4d948cb908e6ff3db2ecc841bf4de6d5516381` (272,404,480 bytes). Both were fully read back from object storage and hashed. The CPU environment receipt and Linux restorer bind this exact final bundle.

CUDA throughput, CUDA numerical gates and full training have not run. Linux environment materialization/validation is pending CPU resource approval; the prior `/root` environment is not assumed to survive pause. See the [temporary-storage rule](https://docs.cloud.vessl.ai/member/workspace/create). Preparation is not marked GPU-ready until that archive is uploaded and read back. No independent holdout is available: the workflow stops at development diagnostics and does not automatically adopt/package a winner. Conditional W6 expansion and W7 independent evaluation/submission remain gated by the approved design.

## Storage retention

Kept canonical audio/weights, role/admission manifests, previous selected checkpoints/results, new uploaded inputs and durable PANNs outputs. Deleted four original redundant archives (24,905,374,958 bytes) only after matching retained local backups by size and multipart ETag. Local SHA-256 and exact backup paths remain in the private cleanup receipt. Also removed four superseded task staging objects (334,019,098 bytes) after validating their retained local copies against the earlier upload/readback receipts. All deletions were followed by absence checks.

The storage snapshot fell from 175,759,024,281 to 151,293,095,779 bytes after final input uploads, a net reduction of 24,465,928,502 bytes. A subsequent 1,505-byte cleanup receipt is outside that snapshot. The 27 canonical audio/weight objects retained their original sizes and ETags. Unverified duplicates and sole source copies remain untouched; deleted objects can be restored from the verified local backups.
