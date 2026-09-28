# Feature preflight and feasibility audit

Audit time: epoch `1789651948` (2026-09-17 22:32:28 KST). Immutable hard deadline: `1789663360`; export reserve begins at `1789662760`. At audit time, `10,812 s` remained before the export reserve.

## Confirmed next-preflight contract

- The pinned DF-Arena backbone calls `self.ssl_model(x.unsqueeze(0))` (`_workspace/redesign_full_20260917/assets/baseline/model/df_arena_1b/backbone.py:46-47`). Its native callers pass one 1-D `[T]` waveform.
- The prior implementation passed `[B,T]`, which would become invalid `[1,B,T]`. The authorized fix in `eight_features.py:128-135` now forwards each 1-D window separately and retains `_batches()` deadline checks and input order.
- Regression invocation: `.venv/bin/python -m pytest -q test_eight_features.py --junitxml=.omo/evidence/eight-h100-20260917/task-2-df-contract.junit.xml`.
- Observable: `9 passed in 0.92s`; the focused seam rejects non-1-D input and verifies probabilities remain ordered for windows with first samples `[-2,0,2]`.
- Artifact SHA256: `5209de2e2c0601da7948ada90973e63cdf5f7c9458558a61537ce145f8fabf35`.
- Final `eight_features.py` SHA256: `f399b991e694e5cbddd69791499829351f5e2d30c69166911f93e79a2e643a4d`.

The H100 AntiDeepfake parity failure is independently confirmed in failed attempt 04: actual logits `[-3.9105923, 4.035076]`, CPU-native reference `[-3.9040575, 4.0290451]`, maximum error about `0.006535`, versus `atol=0.0003`. Earlier working H100 code explicitly disabled CUDA matmul TF32, cuDNN TF32, and cuDNN benchmark before parity (`voice_music_h100_experiments.py:31-34`). The runner owns that precision correction; this audit does not claim it has passed.

## Exact workload from the sealed 7,101-row manifest

Using each row's declared duration, 64,600-sample windows, 32,300 hop, and one deduplicated tail:

| Work | Full dataset | Remaining after the 3-record preflight |
|---|---:|---:|
| Audio duration | 71,825.124 s (19.951 h) | 71,749.664 s |
| Aligned windows | 29,275 | 29,240 |
| Separator calls | 7,101 | 7,098 |
| DF native 1-D forwards, raw + vocals | 58,550 | 58,480 |
| Anti batches, raw + vocals, batch size 8 within each file | 15,930 | 15,918 |
| MERT batches, batch size 8 within each file | 7,965 | 7,959 |
| PANNs per-record calls | 7,101 | 7,098 |
| Spectral window transforms, raw + stem | 58,550 | 58,480 |

The preflight records are `core_000209f68a0dc6367dcb_base`, `core_000cbfa7068dfcdde08e_base`, and `e5_0154ad46e66363f42aa8699a`: 75.46 seconds, 35 aligned windows, and 210 learned-model window items.

Window distribution is 4 rows with 1 window, 444 with 2, 6,005 with 3, 216 with 5, 216 with 14, and 216 with 29. Because batching is confined to one file, the 29,275 windows create 7,965 batches per single batched view: average occupancy is only 3.68 of 8. Feature arrays alone have an approximately 915 MB uncompressed lower bound before NPZ/container overhead.

## Timing conclusion

The three-record preflight total cannot honestly predict full extraction from one elapsed number. It combines at least four unidentified terms: asset hashing/model loading/native parity, separation proportional mainly to audio duration, encoder work proportional to windows but with model-specific batching, and per-file hashing/compression/progress I/O. The full pass then repeats asset validation, model loading, and native parity because preflight frees the engine.

An honest projection requires stage rates and uses this accounting equation for the post-preflight remainder:

`T = L_reload + 71,749.664*r_separator_second + 58,480*r_DF_forward + 15,918*r_Anti_batch + 7,959*r_MERT_batch + 29,240*r_PANNs_window + 58,480*r_spectrum_window + 7,098*r_record_io`.

Without those rates, scaling the whole preflight time is only sensitivity analysis: window ratio `836.43x`, audio-duration ratio `951.83x`, and row ratio `2367x`. None is a runtime estimate because fixed load cost is included in the numerator.

At audit time, consuming the entire remaining pre-export interval on extraction alone required at least 2.704 aligned windows/s, 5.409 serial DF forwards/s, 6.636x real-time separation throughput, and 0.656 records/s simultaneously. The actual requirement is higher because E1-E8, E7 training, report generation, and archive export still need time.

A prior H100 run processed 568 rows, 5,638 audio seconds, and 1,486 non-overlap windows in 276.253 s, but it used two MERT views, one serial DF view, PANNs, and Demucs without AntiDeepfake. A rough transformer-layer-window proxy makes the new encoder load `44.32x` that run (`29,275*216 / (1,486*96)`), or about 12,244 s before accounting for the remaining experiments. This is an inference, not a measured prediction, but it exceeds the 10,812 s available before reserve and makes full completion within the original deadline unlikely.

## Precision, device, and layout path

- No feature path uses autocast, FP16, or BF16. NumPy windows are float32; Torch tensors preserve float32; logits are explicitly converted to float32 before softmax.
- AntiDeepfake loads safetensors into a meta-created Wav2Vec2 model, assigns weights strictly, moves the FP32 model to the requested device, forces eager attention, disables SpecAugment, normalizes each waveform with `layer_norm`, and pools time before the 2-class head.
- DF-Arena loads the pinned local 1B wrapper and moves it to the device in eval mode. The wrapper itself creates the batch dimension, so each caller input must remain 1-D.
- MERT loads local weights with the default FP32 dtype, moves to the device, and requests all hidden states. Input is CPU-resampled 16 kHz to 24 kHz before transfer.
- Demucs is loaded on CPU in eval mode; `apply_model` receives the CUDA device and float32 waveform. PANNs is initialized with the CUDA device string and receives CPU-resampled float32 arrays.
- `eight_features.extract` bounds Torch CPU threads but has no local TF32/cuDNN policy. The earlier successful H100 implementation set both CUDA matmul and cuDNN TF32 false and cuDNN benchmark false before model construction. The runner owns restoring that global policy before Anti verification and all later model initialization.

## Remaining implementation hazards after parity

1. Full extraction repeats the large asset/model hashes, Anti parity, and all model loads after preflight; the three cached records save only 35 of 29,275 windows.
2. DF must remain serial under the pinned wrapper, producing 58,550 1-D 1B-model forwards. Treating `feature_batch_size=8` as DF batching would be false.
3. Demucs separation, PANNs inference, CPU resampling, spectral transforms, and NPZ compression do not check the deadline internally. A deadline can be crossed inside one separator/PANNs/CPU stage before the next model batch guard.
4. `_Engine` retains separator, DF-Arena, PANNs, MERT, and AntiDeepfake together. H100 capacity is likely adequate, but no successful peak-memory artifact exists for this combined runtime.
5. Raw Anti and MERT request all hidden states and retain only selected layers. This increases activation traffic; it is required for E6 but is a throughput cost.
6. Cache identity binds audio, parents, model files, source files, layers, and window settings, but it does not bind PyTorch/Transformers/CUDA/cuDNN versions or effective TF32/determinism flags. A later precision-policy change outside a hashed source file could reuse numerically different caches.
7. 7,101 separate compressed NPZ writes and later archive compression are material CPU/I/O work. No measured throughput artifact exists yet.

No GPU, remote workspace, dataset, archive, or frozen protocol was mutated by this audit.
