# A100 E00/E04 preparation

Code commit: 32f11bdc1094fec7eae0dcdc614982fe6dd9108b. This is a new code bundle; the earlier H100-only prepared-v1 is unchanged.
Archive: a100-preparation.tar.gz
SHA-256: ff0f79538af12dbc518fe57d77847d222beb78a235f872bfe26ce4f4a16d93b6
Prepared SHA256SUMS.json SHA-256: a6f6380ef2c7e7bcedac09cf0bf42104b961739f90775da43c4342419852d9ac

No GPU has been started for this A100 change. One separately authorized CPU job exported and uploaded the audio; it was terminated after independent full archive readback verified the transfer. The final log and downloaded export_receipt.json confirmed cloud-mounted readback also completed at2026-09-24T11:56:12.475Z; both verification paths passed. Local checks cover CLI routing, CPU-only default, a simulated low-memory decision with real CPU optimization, role/cache/F0 boundaries and export byte integrity. Actual A100 model loading, performance and peak VRAM remain unverified.

VESSL resource resourcespec-a100x1 reports one A100 SXM80GB, CPU11, RAM128GB, 1.48credit/hour in cluster-betelgeuse. Existing A100 workspace wsp-5i3knd3gxeox is terminating/terminated. Verified audio is on clustervol-qgze68bffy6l in cluster-capella. Do not assume this root can be mounted by the A100 resource. The existing-workspace-only instruction has not been waived; no new workspace was created.

CPU audio export and transfer are complete. Job job-3votgz7mfrno verified all43516audio files (12329271166bytes), created portable-audio.tar (12369408000bytes), and copied it to the existing project object volume. While mounted readback had no completion in the last observed log, an independent full VESSL CLI download to the local CPU reproduced SHA-256 030ff2d8109cfcb39aa0b7c298d68882a829ea4e21dafd41b4343347d038cfa9 in490.026seconds (download plus hash). The job was then terminated; do not describe its status as succeeded. Final logs and the downloaded audio/export_receipt.json subsequently confirmed the mounted readback also passed. Use audio/transfer_receipt.json for combined evidence of both paths and the actual job state. cpu_export_job_request.json is the preserved original request, not a pending instruction; do not rerun it. Original audio was preserved. Both audio and the already-uploaded weights must still be staged on the authorized A100 environment before GPU experiment execution.

## Deployment prerequisites

- Use a fresh directory for this bundle, such as /root/lb-a100-20260924/prepared-v1. Verify the archive hash above BEFORE extraction; preserve its exact31regular-file members and member hashes. Do not mix it with the old code or include local __pycache__ files.
- Stage the exported audio into /root/lb-a100-20260924/input, preserving audio/... paths. Check the archive hash against transfer_receipt.json before extracting into a NEW directory with tar --keep-old-files. CPU preflight below rechecks every audio file against the frozen CSV.
- Reuse the25pinned assets from lb-improvement-20260924/weights/baseline-assets/ on the SAME project object volume; stage into /root/lb-a100-20260924/assets with the original relative paths. Do not redownload model weights from the internet.
- Provision an explicitly authorized Python environment compatible with the original data environment (Python3.11; historical recipe torch/torchaudio2.7.1cu128, transformers4.57.6, demucs4.0.1,panns-inference0.1.1,librosa0.10.2.post1). The observed source Python is3.11.13. No target environment was installed or verified. requirements-reference.in in the bundle is HISTORICAL REFERENCE ONLY, not an installer or proof of runtime equivalence.
- Use fresh cache/output directories: changing feature source invalidates older cache identities. E00 and E04 share this bundle's union cache; data,encoder,code and row checks remain strict.
- Retain at least4GiB free for cache/outputs AFTER staging; this is a reserve threshold,not a guaranteed final size. GPU launch requires a separate budget and a workspace pause deadline. Process timeout does not stop workspace billing.

## CPU preflight first

Set these paths to the verified deployment. This command does not load models or run a GPU:

```bash
LB_PY=/root/envs/data/bin/python
LB_PREPARED=/root/lb-a100-20260924/prepared-v1
LB_INPUT=/root/lb-a100-20260924/input
LB_ASSETS=/root/lb-a100-20260924/assets
LB_WORK=/root/lb-a100-20260924
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$LB_PY" "$LB_PREPARED/code/data_v4/gpu_experiments.py" preflight --gpu-name A100 \
 --feature-manifest "$LB_PREPARED/inputs/feature_union.csv" --e00-manifest "$LB_PREPARED/inputs/E00.csv" \
 --e04-manifest "$LB_PREPARED/inputs/E04.csv" --config "$LB_PREPARED/inputs/seed41.json" \
 --input "$LB_INPUT" --assets "$LB_ASSETS" --asset-hashes "$LB_PREPARED/inputs/asset_hashes.json" \
 --cache "$LB_WORK/cache" --output "$LB_WORK/runs"
```

## GPU experiment only after separate authorization

3600seconds here is an EXAMPLE bound,not an execution-time estimate or spending approval. The outer timeout covers preflight as well. Retain the original absolute workspace pause deadline across resumes.

```bash
unset CUDA_VISIBLE_DEVICES
"$LB_PY" -c 'import shutil,sys; n=shutil.disk_usage(sys.argv[1]).free; raise SystemExit(0 if n >= 4*1024**3 else "Need4GiB free after staging")' "$LB_WORK" && \
timeout --kill-after=30s 3600s "$LB_PY" "$LB_PREPARED/code/data_v4/gpu_experiments.py" all --gpu-name A100 --max-seconds 3600 \
 --feature-manifest "$LB_PREPARED/inputs/feature_union.csv" --e00-manifest "$LB_PREPARED/inputs/E00.csv" \
 --e04-manifest "$LB_PREPARED/inputs/E04.csv" --config "$LB_PREPARED/inputs/seed41.json" \
 --input "$LB_INPUT" --assets "$LB_ASSETS" --asset-hashes "$LB_PREPARED/inputs/asset_hashes.json" \
 --cache "$LB_WORK/cache" --output "$LB_WORK/runs"
```

The actions features,train-e00,train-e04,compare can resume completed stages. MERT batch size is remeasured on every invocation with new work; an H100 measurement is never trusted as an A100 measurement. The real device is recorded in device.json; hardware-query mocks are not hardware validation. Features remain FP32 with TF32 disabled; head training retains native BF16 autocast. All model weights are frozen except the unchanged music WindowMIL and voice linear head. E00 uses32197rows,E04uses33674rows including1477newtrain-only clips; development9842rows is shared. Seed41,max50epochs,patience5; F0 is fitted on E00 calibration only and reused for both development comparisons. No official score or independent model-quality improvement is established.
