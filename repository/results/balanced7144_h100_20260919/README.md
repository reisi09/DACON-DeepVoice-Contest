# Balanced 7,144 H100 run — 2026-09-19

Started on the existing `wsp-9683432r72m7` after the user explicitly authorized immediate training. Object prefix: `balanced7144-h100-20260919/` on `objvol-a5gxdg7er70i`.

- Input: `../balanced7144_20260919/selected.csv` (7,144 rows).
- Train: 5,716 (2,858 real / 2,858 fake). Validation: 1,428 (714 / 714).
- Preserve group isolation and near-80:20 source/generator ratios; route-specific classes are not separately balanced.
- Selected original audio: 10,345,454,546 bytes. All 7,144 audio files and 27 model assets were hash-verified on the H100. Existing audio was reused through hard links; no new audio objects were uploaded. Existing shared shard containers total 15,054,431,524 bytes and include unselected files.
- Preserve frozen HTDemucs base, MERT330M layer12 features, WindowMIL music head, original DF-Arena fc5 voice head, and PANNs presence. Frozen features are regenerated for this split in a separate cache. Seeds 41/42/43; max 50 epochs per head/seed, early stopping patience 5; no wall-clock cutoff. Effective head batch 256, microbatch measured on the H100; MERT batches benchmarked from 16 through 256.
- Startup observation: new feature process PID 86, runner PID 80; feature extraction confirmed at 213 / 7,144 rows; optimizer training follows after all features complete. Measured MERT batch 256, peak allocated VRAM 26,929,689,088 bytes. This is a launch receipt, not a quality result.
- Validation includes 561 historically seen rows. Do not claim an independent test, leaderboard improvement, or L4 compliance.

## Operations

```sh
.venv/bin/python full_bank_ops.py watch --balanced7144
```

The detached watcher writes `/tmp/balanced7144_watch.log` and polls every 300 seconds. The updated `full-bank-h100-completion` heartbeat monitors this new run. Verified result download precedes automatic GPU pause. Do not launch a duplicate job or restart the superseded 16,042-row deployment. Do not submit to DACON automatically.

`launch_verification.json` and `live/storage_preflight.json` retain the measured launch evidence. Live progress files are snapshots, not completion evidence.

## Decoder recovery, attempt 2

After 2,951 completed feature rows, libsndfile rejected `bank_fma_97601` and librosa could not find an executable fallback. The watcher hash-verified the failed export and stopped the GPU; that evidence is preserved under `attempt1/` locally and in the same cloud prefix.

Commit `fd1d553` exposes the already-installed offline FFmpeg binary on the child process PATH. The actual MP3 regression fails with `NoBackendError` without the fix and decodes successfully with it. The manifest, all feature/training implementation files, originals and architecture are unchanged. Existing 2,951 feature rows were verified and reused.

The existing H100 resumed. Live cache verification found 3,001 rows and the formerly failing file now has a completed feature entry: 1,343,321 samples, 41 windows. This remains feature extraction, before optimizer training. See `recovery2/recovery_receipt.json` for executable hash, manifest/source continuity and live evidence. Historical errors remain in the append-only feature log; use fresh progress and terminal run state to judge the current attempt.

Before final packaging, derive package row counts from the selected manifest and include a matching FFmpeg fallback in offline inference checks; legacy gain-preserving entries still need their separate parity check.
