# MERT12 cached-vs-runtime parity diagnosis

## Scope
CPU-only read-only diagnosis of the four-file H100 cached-feature vs A100 packaged raw-audio music probability mismatch. No GPU operations, tolerance changes, or runtime changes.

## Observed failure
- Order and subset runs agree by ID.
- Cached feature direct scoring vs packaged raw-audio inference exceeds the required 1e-5 tolerance.

## Hypotheses
1. MERT batch construction differs: cached extraction batches up to 8 equal-length windows with `padding=True`; runtime invokes one window at a time without padding.
2. Source/audio/asset identity or window layout differs between cached feature and runtime input.
3. Library or accelerator numerical differences occur in HTDemucs/MERT despite matched source logic and are amplified by the trained head.

## Planned distinguishing evidence
- Bind the four input SHA-256 values, cached NPZ identity, starts, valid lengths, and head-scored cached probabilities.
- Compare exact source implementations and source hashes.
- Recover H100 and A100 package/runtime versions from receipts and logs.
- Quantify per-ID/order/subset deltas and determine what remains unobserved (A100 intermediate features).

## Findings
- The four smoke WAV byte hashes match each NPZ `audio_sha256`.
- Cached `eight_features.py` SHA-256 equals the current source (`b0304f...`); baseline script, MERT, and HTDemucs asset hashes also match.
- All four cached records contain the same three full windows: starts `[0, 32300, 63400]`, valid samples `[64600, 64600, 64600]`.
- Original, reversed-order, and strict-subset packaged results agree by ID at the written 10-decimal precision. This rules out cross-file order/state as the observed cause.
- The feature extractor sends each file's three windows through one MERT batch (maximum configured batch 8); runtime sends three singleton forwards. A processor-only CPU probe on equal-length float32 windows was bitwise identical, so any batch effect is in model kernel geometry, not processor padding.
- TF32 flags match after model initialization. `eight_features._Engine` calls `verify_native`, whose `AntiDeepfake` initialization sets matmul/cudnn TF32 false before HTDemucs and MERT; the attempt-6 strict receipt records both false. TF32 is not a live cause candidate.
- The environments also differ: cached H100 used Python 3.11.13 and torch 2.7.1+cu128; raw A100 used Python 3.13.9 and torch 2.9.1+cu130. FFmpeg was static 7.0.2 versus system 6.1.1. The latter run is not the official Python 3.11.15 / torch 2.7.1+cu128 target environment.
- Current artifacts do not contain A100 backing or MERT vectors, so they cannot identify which live factor caused the probability delta.

## Decision
Do not change the runtime or loosen the 1e-5 gate. Run one bounded factor-isolation probe in the official/cache-matched Python 3.11 + torch 2.7.1+cu128 environment: decode once, retain the backing, compare MERT batch-3 versus singleton on that exact backing and exact processor tensor. Persist backing, both feature arrays, cached starts, feature-level max/RMS error, and per-seed/ensemble probabilities.

## Correction
The initial journal incorrectly treated the framework default `cudnn.allow_tf32=true` as the extraction-time value. The actual initialization path changes it to false before feature extraction; `_workspace/eight_h100_20260917/attempt6_state.AGajao/parity_gpu_strict.json` confirms `precision_after_model_init.cudnn_allow_tf32=false`.

## Attempt 5 and root cause
- In matched Python 3.11 / torch 2.7.1+cu128, processor batch/single tensors were identical. MERT batch-of-three versus singleton feature max error was at most `3.0517578125e-05`, while ensemble probability changed by at most `1.58e-9`; batching is ruled out.
- On one identical persisted input tensor, Transformers 4.51.3 and 4.57.6 CPU features were bitwise identical. CPU versus A100 GPU MERT error was `2.14e-4` max / `1.39e-5` RMS, far below the H100-cache gap (`2.87` max / `0.3555` RMS).
- Explicit `torchaudio.load + demucs.audio.convert_audio` (Julius resampling), followed by CPU HTDemucs and CPU MERT, reproduced all four H100-cache ensemble probabilities within `1.35e-7`. The A100 FFmpeg path differed by as much as `1.91e-4`.
- The environment-dependent decoder branch caused the failed parity. The H100 run did not record literal ffmpeg/ffprobe PATH, so the claim is based on exact numerical reproduction of the cached behavior rather than a direct H100 process receipt.

## Resolution
The packaged runtime now pins `torchaudio.load + demucs.audio.convert_audio` before baseline stem separation and binds that decoder in the package manifest/runtime receipt. Heads, training receipts, MERT batching, tolerance, pooling, and fusion remain unchanged. The rebuilt ZIP passed local complete member-hash and CRC verification; GPU end-to-end QA remains pending.
