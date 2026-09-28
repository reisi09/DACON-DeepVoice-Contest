# Live H100 export race audit

Read-only scope: inspect the frozen wrapper and export helper while the real runner remains active. No cloud or source mutation was performed.

## Concrete race

- Writer: `redesign_h100_run.sh:15-16` continuously appends `ROOT/output/gpu_telemetry.csv` every 10 seconds until shell exit.
- Export roots: the notebook export and `redesign_h100_run.py:281` both recursively package `OUTPUT`.
- Non-atomic interval: `redesign_notebook_runtime.py:242` hashes all source files, lines 243-245 later write them into the ZIP, and lines 252-255 compare archived bytes against the earlier hashes.
- Binary consequence: one telemetry append between those operations produces `ZIP hash readback failed: output/gpu_telemetry.csv`.
- Recovery gap: the verified helper replaces `results.zip` before readback. If readback fails, the shell sees a nonempty archive at line 78 and skips its emergency export.

## Operational recommendation

Terminate only the exact background `nvidia-smi ... --loop=10` sampler before export so `output/gpu_telemetry.csv` becomes immutable. The shell EXIT trap already tolerates an exited monitor. For continued sampling, write a replacement stream under `/shared/redesign-h100-20260917/gpu_telemetry.live.csv`, outside every export root, and stop it through the watcher after runner completion. Do not signal the training runner.

`output/bootstrap.log` is also an exported open file, but its known final shell writes occur only after the runner and its export return, so telemetry is the active periodic writer and the immediate race.
