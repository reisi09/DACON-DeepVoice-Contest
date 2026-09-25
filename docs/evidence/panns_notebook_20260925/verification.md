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

## Completed cached CPU comparison

The portable notebook executed all four Python code cells under the offline sandbox. The first attempt exposed a virtual-environment interpreter path bug: resolving its symlink selected base Python without NumPy. Preserving the configured interpreter path fixed the actual execution. The final notebook differs from the executed source only in markdown describing who owns cloud pause; code cells are identical. Jupyter transport was not exercised by this local run.

All 42,039 legacy arrays passed producer, asset, row and payload authentication before current roles were applied. Three approximately parameter-matched head designs ran seeds 41/42/43 for 20 complete epochs each (4,360 updates), using four CPU threads. The CLI took 115.767081 seconds and peaked at 2.945740 GiB RSS. No GPU or MPS was used.

| Method | Development diagnostic ADS | Music EER |
|---|---:|---:|
| Preserved heads replayed on cache | 0.892041 | 0.136377 |
| Learned FILE combination only | 0.891937 | 0.136377 |
| Independent five heads, seed mean | 0.855586 | 0.226796 |
| Shared five heads, seed mean | 0.857133 | 0.224671 |
| Joint nine states, seed mean | 0.850080 | 0.245883 |

No candidate was adopted. New heads improved exposed selection/calibration metrics but worsened development music EER. This comparison does not establish that MTL generally fails: pooling, initialization and head architecture differ from the preserved model. The real-voice/fake-music state has only three train rows versus 113 development rows. The 900 recovered noise rows have no historical cache and were excluded from this CPU experiment.

Saved predictions reproduced all 33 role/model metric reports; restored checkpoints replayed 64 real rows within 2.33e-10 maximum absolute error (declared tolerance 1e-6). Sampler coverage, actual trainable-parameter updates and FILE-only preservation of the other four outputs were verified. Evaluation masks omit unknown labels; these weighted diagnostics are not official scores, independent holdouts, or proof of equivalence to the leaderboard-best model.

## Historical stem reuse

All 42,039 historical stem-statistic entries were authenticated against the 42,939 current row/audio identities and actual reference asset bytes. Only the 900 noise rows still require Demucs. Fresh seven-file CPU integration reused four historical stems and computed three noise stems: Cnn14 prefixes, tags and segment starts remained bit-identical, and maximum presence difference was 1.8784308442043596e-08. The separate 32-row historical/gold comparison stayed within the fixed 1e-5 tolerance. Neither check proves full-bank CUDA parity; the 43-row CUDA gate remains mandatory.

## Packaging and lifecycle audit

Actual bundle preparation initially exhausted local disk during a 4.59 GB weight copy and exposed publication of a partial final file. The source remained intact. Preparation now checks space, copies into an owned temporary file, verifies SHA before atomic promotion, and refuses mismatched finalized files or unsafe destination symlinks. Interruption, silent corruption, retry and space rejection were exercised. The complete revised bundle then prepared successfully against actual audio/reference bytes in 10.575 seconds. Large local QA assets used authenticated APFS clones to avoid redundant disk allocation; this is not a Linux copy-throughput measurement.

Actual Claude Opus 5.5 xhigh reviews completed, including a final 358.38-second notebook/controller review with tools and MCP disabled. Two real controller defects were reproduced and repaired: expiry blocking the stop command itself, and malformed state inspection skipping the stop attempt. Cleanup now has one bounded 30-second allowance that can only inspect/pause the owned target; it cannot start or extend compute. A real identity mismatch still refuses action. Protocol/process tests validate these paths, but do not establish cloud API success.

The prepared Linux environment contains no VESSL CLI. Therefore the Mac-side pinned controller owns starting the existing A100, polling, and verifying actual stopped state. The remote guard bounds staging, environment restoration and all compute stages. Local controller loss cannot itself guarantee cloud pause; the remote deadline still stops owned compute. No new workspace, package installation, or official submission is performed by the notebook.

## Evidence still required

Revised Linux/CUDA execution, actual cloud lifecycle integration and the A100 experiment outcomes remain incomplete. Historical features do not contain the block5 activations required by the new PANNs experiment. Raw fixed fake predictions also remain necessary because matching-input cached/raw inference parity has not been established. The experiment preserves existing voice/music fake heads and trains only the declared PANNs presence adaptations.
