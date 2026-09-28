# H100 temporal resolution × softmax

## Results

All four settings completed with seeds 41–43. Lower EER is better.

| Setting | Tune music EER | Historical music EER | Historical file EER |
|---|---:|---:|---:|
| R1 sigmoid, fresh coarse control | 0.276415 | 0.311111 | 0.326709 |
| R1 softmax | 0.303012 | 0.337963 | 0.322650 |
| R8 sigmoid | 0.276415 | 0.309259 | 0.326709 |
| R8 softmax, requested combination | 0.286832 | 0.300000 | 0.310684 |
| Prior C08, previous cache | 0.275484 | 0.295370 | 0.321368 |
| Official baseline on local panel | — | 0.552778 | 0.432051 |

The requested combination improved historical file EER but did not improve music EER versus prior C08. Softmax alone worsened music EER on fresh coarse input. These observations do not establish a statistically significant or external-test gain.

**Selection audit:** the original execution selected R8_sigmoid because two mathematically tied EER means differed by 5.55e-17 in floating-point arithmetic. Correct rounding to 12 decimals followed by the stated seed-variance tie-break selects **R1_sigmoid**, whose tune EER std is 0.011637 versus 0.021471. `tune_selection_audit.json` uses the existing tune summaries only. The original executed selection and archive remain unchanged; all 12 historical comparisons were predeclared. The selector is fixed for future runs and checked against this concrete numerical failure. R8_softmax is not promoted using its historical result.

Feature extraction took 2296.85 seconds (38m17s); the 12 temporal-head fits totaled 257.54 seconds (4m18s). Result ZIP: 5,604,128 bytes, SHA-256 `a4812bf7cb238ae8dc90565b08a05ac22b96404b4f2aa7fb7bed76cf244312ad`. Feature ZIP: 716,740,891 bytes, SHA-256 `3be777517142157d7928d4d605ce68ca8fcd0a92a019cbf30b8a1b5d34b231eb`, 1446 NPZ files; persisted in object storage and verified by SHA-256 and ZIP CRC.

Local verification recalculates all 12 tune predictions and 14 historical reports, reloads the checkpoints, checks audio/cache receipts, and confirms unchanged voice/presence predictions. The comparison plot was visually inspected. See `Temporal_H100_Resolution_Softmax_Report.ipynb` for executed output plus the explicit post-run audit.

## Fixed experiment

- Frozen MERT-v1-95M, 768 features. HTDemucs separation matches the official baseline helper.
- Each 64,600-sample window at 16 kHz produces both one full-frame mean and eight disjoint temporal-bin means from the same MERT output.
- Bin timestamps use the convolutional receptive-field centres derived from MERT's kernel/stride configuration. The final overlapping window's bins are sorted by time. Contextual MERT vectors are not independent half-second audio crops.
- Compact head: LayerNorm(768), projection 768→64, GELU, three LayerNorm→Conv1D(kernel 3)→GELU layers, residual addition, masked temporal max, Linear(64,1 or 2).
- Four predeclared settings: R1_sigmoid, R1_softmax, R8_sigmoid, R8_softmax; seeds 41, 42, 43. Sigmoid uses weighted BCEWithLogits; two-logit softmax uses per-file weighted CrossEntropy and fake probability column 1.
- Fixed prior C08 hyperparameters: learning rate 0.0001, batch 64, weight decay 0, dropout 0, crop probability 0.7, feature-mask probability 0.3, NSynth mass 0, at most 300 epochs, patience 20.
- Crops select original windows, preserving duration across resolutions. Feature masks remove 38 of 768 channels across all bins.
- Train/tune/historical test = 1008/248/190 files. Shared source-group split and existing training quality gates remain fixed. All 12 trials are predeclared historical comparisons; selection uses only three-seed tune EER mean, standard deviation, then BCE.

## Reproduction and evidence

`Temporal_H100_Resolution_Softmax.ipynb` executes `temporal_resolution.main()` then creates the comparison plot. The input ZIP contains source, locked split, baseline predictions, quality plan, and MERT/Demucs assets. Input manifest lists member hashes.

`h100_bootstrap.sh` pins Python 3.13.9, torch/torchaudio 2.9.1 CUDA 13.0, and other dependencies; verifies inputs; runs the notebook on H100 with a two-hour ceiling; persists result and feature archives to the mounted object store. Upload `existing_audio_map.json` as `/root/resolution_existing_audio_map.json` before launching the bootstrap. Its entries reuse 1296 original WAVs from the existing V6 archive and 150 from the existing C3 archive. Every extracted WAV must match the locked split's SHA-256. The locally prepared audio ZIP provides an alternative complete copy; its slow redundant upload was cancelled.

Run `verify_results.py` after downloading and safely unpacking the result ZIP to `unpacked/`. It recalculates tune and historical scores, checks source-group separation, selected epochs, checkpoint loading/hashes, unchanged voice/presence probabilities, paired-cache receipts, and tune-only selection.

## Runtime recovery

The first extraction attempt stopped before generating any feature cache because TorchAudio 2.9 required an audio decoder. The bootstrap now installs FFmpeg and TorchCodec 0.8.1. `decoder_check.py` validates the real WAV decoder and Demucs audio loader against lossless reference samples with absolute tolerance 1e-7. `recover_decoder.sh` reuses verified inputs and preserves the original deadline. The successful result archive retains the first failure log.

## Interpretation boundaries

This is a resolution × output/loss parameterization experiment. The finer sequence also reduces the temporal convolution's receptive field in seconds. Two-logit softmax and scalar sigmoid are algebraically related, but independent optimization need not produce identical results. The shared fresh extraction makes the four settings comparable; comparisons against older cached runs can additionally reflect extraction/runtime differences.

Historical evaluation has been reused. It does not establish performance on a new held-out dataset or the official leaderboard. H100 runtime does not establish the L4 submission time limit. No full-data retraining or submission package integration is part of this experiment.
