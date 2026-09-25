# PANNs notebook execution audit, 2026-09-25

These records separate engineering verification from model quality. No result here establishes a leaderboard improvement or official L4 runtime compliance.

## Resume and persistence

- A real seven-file CPU extraction/recovery took 14.418 seconds with 2.811 GiB peak RSS. Recovering missing/truncated cache publication markers preserved payload and entry-receipt hashes and modification times, without loading the models again. A finalized hash mismatch remains an error.
- Real CPU Cnn14/AdamW training was interrupted during the first checkpoint write: a 681,080,216-byte checkpoint was truncated to 65,536 bytes. Resume is permitted only with the matching pre-run identity receipt and no committed progress or foreign files. The resumed final model, optimizer, sampler, RNG, diagnostics and frozen digest matched continuous execution exactly. This check took 27.194 seconds.
- Duplicate final checkpoint writes were removed. The training budget now measures the real checkpoint write and final hash costs alongside optimizer-step time, retaining common complete data passes for all four arms and three seeds plus the 20% timing reserve. Mac filesystem timings do not predict cloud object-volume timings.
- CPU report readers verify archived code/config/cache/budget/checkpoint identities without requiring a live CUDA device. Live training and evaluation still validate their runtime and hardware.
- Decode scratch files use local temporary storage; persistent artifacts remain in their separate experiment namespace.

## Historical feature recovery

- The explicitly authorized CPU-only job `job-ghnoc09nki2p` finished with state `succeeded`. It preserved the source archive on the Capella volume and copied it into the existing project object volume after checking every indexed NPZ payload.
- The archive contains 42,039 NPZ features: 35,589 train/validation-bank rows and 6,450 historical test-bank rows. Its size is 1,713,551,360 bytes and SHA-256 is `07479d870724db5d4c410aaa9f0e68f6d2c5e99521fc92d4bba3d36e52065eb9`.
- Source and destination readback hashes matched. Export took 219.082 seconds; a separate local full download and confined extraction took 79.591 seconds and matched the same archive hash.
- All 42,039 historical original IDs, audio hashes and five-label values/knownness match current D0. Current component IDs differ for 39,993 rows, so old cache identity and current role assignment must be authenticated separately. Historical split names must not replace the current audited roles.
- Current role counts in this recovered bank are train 27,788, selection 2,386, calibration 2,023 and development 9,842. The other 900 current D0 rows are recovered noise without these historical features.
- Known-present training support includes 22,807 voice-fake labels and 7,760 music-fake labels. Unknown labels remain masked. The earlier 5,952-row local partial cache was not the complete cloud inventory.
- Exact historical `full_bank_features.py` bytes were found in Git commit `a5cb7d47d0fd56869e63e1152c169dc2aef8e738`, with SHA-256 `2ac3a1c133176a7e7ccfa0963a9f00bf6cc3be42cd8e7eddec0f53ff89143f67`, matching both cache indices.

## Evidence still required

Full cached-feature cohort experiments, successful notebook code execution, revised Linux/CUDA execution, and the A100 experiment outcomes are recorded separately when they actually complete. Historical feature integrity alone does not prove raw-runtime numerical parity, model quality, or independence of evaluation data.
