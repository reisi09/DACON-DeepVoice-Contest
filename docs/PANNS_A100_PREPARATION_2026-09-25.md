# PANNs A100 execution preparation

Scope: prepare D0 feature extraction, the fixed PANNs comparison and development diagnostics. One explicitly approved CPU-only workspace built and verified the runtime. No GPU or DACON submission was started, and model adoption remains separate.

Current status: the design-alignment revision, inputs-v5 uploads and local CPU integration checks are complete. The historical CPU-only Linux environment remains reusable because its runtime and probe inputs are byte-identical; the revised application code has not run on Linux/CUDA yet. The final 4.49 GB environment archive passed full object-storage readback SHA-256 verification and an independent local comparison of all 29,410 member hashes/links. Inputs are ready for A100 preflight. The CPU workspace and existing A100 are both stopped/standby. A100 startup, CUDA numerical gates and full training have not run.

## Execution contract

- Existing workspace: `qwdkw / wsp-nvxqw1jf9zdv` (`lb-e00-e04-a100-20260924-v1`), A100 SXM ×1; observed stopped/standby.
- Object volume: `objvol-a5gxdg7er70i`, mounted at `/shared`.
- Inputs: `panns-cnn-v1-20260925/inputs-v5/{run_bundle.tar,launch_lock.json,panns_a100_launch.sh}`; the verified noise tar remains at `panns-cnn-v1-20260925/inputs/noise-v2.tar`.
- Existing canonical audio and pretrained assets remain under `lb-improvement-20260924/a100-preparation-v1/audio/` and `lb-improvement-20260924/weights/baseline-assets/`. They are reused, not uploaded again.
- D0 has 42,939 admitted files, including 28,315 training files and the 900 recovered noise files. Historical roles and masks remain fixed. The 900 regenerated files now have derived IDs tied to their historical parents and byte hashes; parent closure is re-audited. The audit retains the 900 missing historical rows instead of calling their new bytes a restoration. D+ is excluded.
- R0 is frozen. R1/R2/R3/R5 each run seeds 41/42/43. Each shuffled pass visits every eligible training file exactly once, in batches of four with the final remainder preserved. A real throughput pilot measures 40 updates across all four arms, including the longest batch. It fixes one common whole-pass step budget from an explicit aggregate training allowance, with a 20% timing reserve, before candidate quality is evaluated. CPU measurements cannot authorize CUDA training. No A100 training step count has been measured or fixed yet.
- The six fake heads, encoders, presence logistic and FILE ensemble remain frozen. Changed presence values feed the original FILE rule.

## Pipeline

`panns_a100_launch.sh` verifies both uploaded archive hashes, rejects changed extraction contents, verifies every reference asset and admitted noise file, and creates the immutable run manifest. It restores a CPU-built, hash-verified Python 3.11 environment to `/root/panns-cnn-v1-20260925/env` before GPU work. Missing CPU environment artifacts stop execution. The A100 stage performs no dependency installation or model download. The CPU environment builder uses the existing hash-pinned dependency lock and strips workload credentials from build subprocesses. The user approved the one CPU-only instance used for this preparation.

`panns_cloud_pipeline.py` runs these stages in separate processes:

1. Extract the 43-file CUDA preflight panel and compare against the preserved CPU gold. Maximum tag/presence absolute error is `1e-5`; CPU/CUDA R3 gradient relative L2 must be at most `1e-3`. Failure stops bulk work.
2. Extract the complete D0 block5/stem cache. Each file is committed separately; explicit resume validates existing entries. There is no repeated full-index rewrite.
3. Measure suffix throughput and freeze the training budget. Train the 12 fixed conditions using complete shuffled passes over eligible train-role rows. Group gradient norms/clipping fractions and R5 attention entropy/peak weight are retained across resume. Save optimizer/sampler/RNG state every 100 steps; SIGTERM retains a resumable checkpoint. Finished conditions are reused only with matching identities and hashes.
4. Infer frozen fake outputs once on development/selection audio with the hash-pinned preserved package. No fake-head retraining or unproven historical CSV reuse.
5. Evaluate R0 and all 12 candidates, recomputing FILE from each candidate's presence values.
6. Produce known-label coverage, presence AUC/Brier/log loss, FILE EER, source/presence strata and paired parent bootstrap diagnostics (2,000 draws, seed 20260925), including same-draw three-seed means and R3−R1/R5−R1 structural contrasts. Only the literal selection role ranks candidates; development is error analysis. The deployment seed stays 41.

Durable results are under `/shared/panns-cnn-v1-20260925/run-v3`; intermediate application files stay under `/root/panns-cnn-v1-20260925/application-v5`. The verified environment remains at its original absolute prefix `/root/panns-cnn-v1-20260925/env`. A hash-verified local SSD activation cache prevents repeated object-volume reads during training. Allow roughly 12 GiB for activations plus package, checkpoints and temporary head-loading memory.

The launcher requires explicit `--hours` (session ceiling) and `--training-hours` (aggregate allowance for all 12 conditions). There are no inherited 48-hour/6-hour defaults. Both limits must be finite and at most 72 hours; they are operator limits, not measured A100 duration estimates. The session supervisor forwards termination to the active process group, allows checkpoint saving, and refuses to start a subsequent stage after cancellation. A later invocation resumes the same immutable step budget. The launcher uses the workload's `vesslctl` to pause the named workspace on exit. `PANNS_LOCAL_QA=1 PANNS_PAUSE_ON_EXIT=0` plus an explicit interpreter is allowed only on Darwin for local packaging QA; Linux production cannot bypass runtime verification or automatic pause. Every restored runtime member is hashed again on reuse, so a stale marker cannot hide corruption. A killed shell or unavailable control plane still requires the operator to confirm stopped/standby; a timeout alone is not proof billing stopped.

## Start only after preparation and compute authorization

The existing workspace retains an old init script with an expired deadline and a guard on its historical `run_started.json`. That object is retained; do not remove the guard artifact or replace the old prefix. Starting the workspace does not select this new PANNs workflow automatically. Connect to the resumed workspace and execute:

```bash
bash /shared/panns-cnn-v1-20260925/inputs-v5/panns_a100_launch.sh \
  --hours "$APPROVED_SESSION_HOURS" --training-hours "$APPROVED_TRAINING_HOURS"
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

See `docs/evidence/panns_cnn_execution_20260925/cloud_preparation.json` for artifact hashes, remote readback and storage receipts, and `cpu_environment_ready.json` in the same directory for the approved CPU build/restore and final archive verification. Local real-audio checks cover cache reuse, strict resume, exact continuous/resumed training equality, frozen fake inference, adapted presence evaluation and bootstrap ROC arithmetic. The uploaded launcher is exercised locally through real extraction and complete input/asset verification, then must reject this Mac because it is not an A100.

Current inputs-v5 bundle SHA-256: `7feb15575fd025724cc99d610898645e383e7b60939f4694b72ae54444a5c527` (167,208,960 bytes). Recovered-noise archive SHA-256: `6b871c982dbc63c9cbf159551c4d948cb908e6ff3db2ecc841bf4de6d5516381` (272,404,480 bytes). Both were fully read back from object storage and hashed. The original CPU receipt still binds the preserved inputs-v3 bundle `9356724b1b5ad13d3581c738e8214b39e8101bb303d7a2eadfe585a617c05c3b`. The restorer pins that receipt and historical archive, then verifies that current requirements, reference package and all 43 gold NPZ files match those historical inputs. Only application code and admitted metadata changed; this does not claim that revised code was executed in the historical CPU job.

The Linux runtime was built on `wsp-2yoknqpl0k4s` (`panns-env-cpu-20260925`, 3.5 CPU, 14 GB RAM, no GPU). Its 83 installed packages matched the frozen dependency lock. The 4,485,481,689-byte archive is stored at `panns-cnn-v1-20260925/environment/linux-env.tar.gz`, SHA-256 `1778246a643d4ef0381c831bc54f66950e3088731230a4f263c755c584fc5c01`. The job moved the original environment aside, restored the archive to the production path, verified all 29,410 runtime members and ran real Cnn14 suffix inference again. Maximum tag error against CPU gold was `7.450580596923828e-08`, below `1e-5`. The bounded build/restore job exited 0; the workspace is stopped/standby.

CUDA throughput, CUDA numerical gates and full training have not run. The prior `/root` environment is not assumed to survive pause; see the [temporary-storage rule](https://docs.cloud.vessl.ai/member/workspace/create). No independent holdout is available: the workflow stops at development diagnostics and does not automatically adopt/package a winner. Conditional W6 expansion and W7 independent evaluation/submission remain gated by the approved design.

## Storage retention

Kept canonical audio/weights, role/admission manifests, previous selected checkpoints/results, new uploaded inputs and durable PANNs outputs. Deleted four original redundant archives (24,905,374,958 bytes) only after matching retained local backups by size and multipart ETag. Local SHA-256 and exact backup paths remain in the private cleanup receipt. Also removed four superseded task staging objects (334,019,098 bytes) after validating their retained local copies against the earlier upload/readback receipts. All deletions were followed by absence checks.

The storage snapshot fell from 175,759,024,281 to 151,293,095,779 bytes after final input uploads, a net reduction of 24,465,928,502 bytes. A subsequent 1,505-byte cleanup receipt is outside that snapshot. The 27 canonical audio/weight objects retained their original sizes and ETags. Unverified duplicates and sole source copies remain untouched; deleted objects can be restored from the verified local backups.

After adding the reusable Linux environment and its verification records, the pre-revision snapshot was **155,783,736,090 bytes**, still **19,975,288,191 bytes below** the original inventory. The environment archive is an input needed for offline A100 startup, not a second copy of the training data.

## Design-alignment evidence

See `docs/evidence/panns_cnn_execution_20260925/design_alignment.json`. Seven real audio files exercised fresh extraction, partial cache reuse, measured CPU throughput, and R3/R5 updates. Five training files were processed as batches of four and one. Interrupted R3 and uninterrupted R3 had exactly equal final weights, sampler/RNG state and accumulated diagnostics. Two real evaluation files exercised the preserved six fake heads plus R0/R3/R5 presence and FILE recomputation; fake probabilities were unchanged. Incomplete arm/seed comparisons were rejected. These are engineering checks, not a full quality experiment.

Numerical and integration regressions cover masked-label gradients, SE updates through frozen block6, source/parent isolation, nonfinite cache rejection, stale provenance/CDF/checkpoint rejection, paired seed means, selection-only ranking, process-group cancellation, runtime corruption and archive containment. LSP was unavailable because its server is not installed; no dependency installation was performed. The initial isolated frozen-inference attempt hit the protected default Hugging Face cache; explicitly setting the task-local offline HF_HOME resolved it without expanding sandbox access.

The inputs-v5 archive, lock and launcher were fully read back from the existing object volume with matching SHA256. New snapshot: **156,118,163,804 bytes**, **19,640,860,477 bytes below** the original inventory. Existing audio, weights, environment and historical inputs-v3 were reused. The local launcher verified all input bytes and then correctly rejected the Mac as a non-A100 device. Both named cloud workspaces were last observed stopped/standby before the v5 upload; this upload used object storage only. No new compute or official submission was started.

The final review identified an MPS-incompatible float64 attention diagnostic and unsafe stale-PID cleanup in a cancellation test. Attention statistics now transfer to CPU before float64 calculations, and cancellation cleanup acts only on the owned parent process. The real-audio CPU training/evaluation and cancellation checks were rerun against these corrected bytes. No new MPS run was performed. Historical inputs-v4 and its evidence remain preserved; inputs-v5 is the current application bundle. Current private artifact directory: `_workspace/panns_cnn_v1/cloud/design_fix_v5`.
