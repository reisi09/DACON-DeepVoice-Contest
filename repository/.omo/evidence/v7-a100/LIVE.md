# V7 A100 data ablation — COMPLETE, DOWNLOADED, PAUSED

Do not resume automatically. Notebook completed 2026-09-15 03:16:13 UTC (12:16 KST). Final archive downloaded and verified in results/v7_a100_20260915. A100 workspace state=stopped and burnRatePerHour=0 confirmed; heartbeat PAUSED. Final credit balance 13.977432, decrease 1.415478 from starting balance. See results/v7_a100_20260915/REPORT.md and download_verification.json. Best exploratory candidate C3: file EER 0.3160256410, music EER 0.3138888889 on deduplicated dev, not a leaderboard result. No DACON submission.

## Archived execution record


Authorized: run this notebook on A100 x1, total workspace runtime <=4 hours including setup/download. Monitor at 15-minute intervals. Download and verify results, then Pause. No DACON submission or credit purchase.

Workspace: jolly-routine, wsp-5i3knd3gxeox. CLI: _workspace/bin/vesslctl. Existing storage /root contains /root/V6_L40S and its verified V6 caches. New files uploaded to object volume objvol-a5gxdg7er70i prefix v7/data-ablation-20260915/. Update SHA256 cb1811e090660c0a56a6b57b04ce8279cad9c807f2afee9b5350a5ede8db3e51.

Read run_config.json for absolute deadlines. Local Jupyter API helper: _workspace/v7_jupyter.mjs (bundled Node runtime), private endpoint configuration _workspace/v7_workspace_private.json. Do not print tokens. Jupyter API paths are relative to /root. New experiment root: V7_A100. Progress: V7_A100/run_state.json and V7_A100/results/progress.json. Log: V7_A100/notebook.log. Final archive: V7_A100/V7_A100_results.zip and V7_A100/download_manifest.json. Executed notebook: V7_A100/V7_A100_Executed.ipynb.

If setup is still in progress, inspect the current root turn before starting a duplicate runner. Once running, do not restart or run a second notebook. On completion/error/time limit, download archive plus manifest into results/v7_a100_20260915, verify SHA256, inspect findings, then CLI workspace pause wsp-5i3knd3gxeox and confirm stopped. If archive unavailable on failure/cutoff, preserve available logs and partial results, Pause promptly. Disable heartbeat after downloads and verified Pause.

Current phase: notebook launched on A100, input cache verification passed for all 1806 records. Check live run_state.json before acting. Existing V6 probability reproduction allows <=1e-5 float32 differences across CPU hosts but requires exact metric parity to 1e-12. Original start/deadlines unchanged. Results are not yet final.

Setup correction: V6 split metadata uses sorted JSON semantic hash, whereas input_manifest also verifies raw file bytes. Corrected verifier to match the original producer. Failed attempt completed; next launch is a replacement, not concurrent training.

Jupyter Contents API root is filesystem /, but the local v7_jupyter.mjs helper now automatically prepends root/ for get, put, download actions. Pass V7_A100/... paths as documented above. Bootstrap and float-replay failed attempts are preserved remotely as setup_hash_failure.zip and setup_float_failure.zip. The currently running notebook was launched directly after uploaded notebook/input-manifest correction; do not run old bootstrap while it is active.

Initial execution verified at about 02:29 UTC: run_state=running; progress=training/C3_equal_windows. C0-C2 had completed. Saved-head maximum probability drift 1.4597686859296743e-7; every available V6 evaluation metric reproduced exactly. No final comparison claims yet. Next scheduled monitoring should inspect the long four-second extraction stage, then collect results and Pause when done.
