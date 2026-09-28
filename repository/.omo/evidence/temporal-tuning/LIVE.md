# Temporal Conv1D tuning — COMPLETE

User explicitly authorized creating one H100 workspace and executing the prepared experiment.

- H100 workspace: temporal-conv1d-h100 / wsp-yiuufw51tke8, eu-north1, 2.98 credits/hour. See run_config.json for compute and pause deadlines.
- Old A100 jolly-routine / wsp-5i3knd3gxeox is stopped. Readiness session 38740 exited without dispatching training. Do not restart A100.
- CPU-only cache export job job-fml78fjtmo81 verified source audio SHA256, exported 1446 cached records, and completed its command. Job metadata incorrectly remained workloadState=created despite FEATURES_READY logs; billing was zero before H100 start.
- Exported feature ZIP: 82,829,419 bytes, SHA256 ddd3d481269e95172762f3ec988ccb15b541424d10ee05f9925deed463dcb13c. Local verified backup in _workspace/h100_portable_backup; object prefix temporal/h100-transfer.
- H100 input ZIP: 458,243 bytes, SHA256 beda657ce4f34a568a97350b4638b14ae298ccad8a5152bf496df6d721a6bbd4; object prefix temporal/h100-run-20260915.
- Remote working directory /root/Temporal_Tuning. Notebook Temporal_H100_Tuning.ipynb. Kernel temporal-h100. Base image torch2.9.1 CUDA13.0.1 Python3.13; lightweight dependencies in /root/temporal-venv.
- Frozen train/tune/test split: 1008 / 248 / 190 via locked_split.json. Historical dev reused, not newly untouched external validation. Baseline prediction data remains fixed. No all-data C3 checkpoint loaded.
- Eight combinations of lr(3e-4,1e-4), dropout(.2,.4), weight decay(.01,.05). Selected on tune Music EER then balanced BCE. Three seeds 41/42/43 all evaluated. Max300 epochs, patience20, min_delta1e-4. Existing source and quality masks retained.
- Latest initial execution check at 07:03 UTC: run_state running, eight candidates finished, progress seed_repeat/T7_seed43. T0 completed22 epochs with selected epoch2 (best validation BCE0.7741246577); later validation BCE4.2606757078 demonstrates overfit under that setting. Final comparison has not yet been inspected.
- Startup script automatically runs once and checks durable object run_started.json to prevent duplicate launches. Do not rerun using test metrics.
- No new cluster storage was created. H100 local /root is ephemeral and wiped on Pause. Bootstrap copies final result ZIP and download_manifest.json into existing object storage before announcing persistence.
- Download Temporal_H100_results.zip and manifest from object prefix temporal/h100-run-20260915 (or Jupyter if needed), verify bytes/SHA256/CRC/paths, extract to results/temporal_h100_20260915/unpacked, then immediately Pause H100 and independently confirm stopped and billing burn=0. Do not lose the sole local remote results by pausing before a verified download or object backup.
- Monitor every15min, quiet on unchanged state. Read _workspace/v7_jupyter.mjs; refresh its ignored _workspace/v7_workspace_private.json from the H100 workspace, never expose token. Review saved executed notebook, test_report.json, comparison.png and disable heartbeat after completion.
- Local checks verified source export/load parity, rejection of changed audio/cache, padding isolation, all temporal/attention gradients, best checkpoint restore, nondefault dropout reload, frozen selection, baseline hybrid output invariants, and real data source/PCM/ID split isolation. Synthetic checks are not real-data quality evidence.

## Final state

All five notebook code cells completed. Verified local result download and object backup; H100 stopped and billing burn=0 independently confirmed. Final balance12.224411, initial12.793524, delta0.569113 credits. Monitor disabled. Final artifacts and measured results: results/temporal_h100_20260915/README.md and Temporal_H100_Report.ipynb. Conv1D did not beat the fresh same-split linear control; do not automatically continue training or submit.
