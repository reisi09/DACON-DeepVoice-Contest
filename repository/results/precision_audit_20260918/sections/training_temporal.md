# Temporal and fullft8 optimization audit

Scope: downloaded fullft8 workers, runtime receipts, gradient receipts, training histories, and cached-feature temporal code/archives. Read-only audit; no production changes, GPU execution, or heavy training. Paths below are relative to `/Users/choiwuseck/Documents/ChatGPT/데이콘`.

## Confirmed implementation facts

### Fullft8 classifier fine-tuning is connected

- Downloaded `results/fullft8_h100x1_20260916/downloaded/h100x1_fullft8_run/batch1_worker.py:351-410` loads the real pretrained encoder, freezes it only for the frozen control, disables MERT SpecAugment/layerdrop, freezes the unused masking token, and executes the full encoder under enabled gradients during training. The frozen path uses cached embeddings and refuses cached embeddings for a non-frozen encoder.
- `batch1_worker.py:443-452` places every `requires_grad` encoder/head parameter into AdamW groups, using encoder LR 1e-5, head LR 1e-4, and weight decay 0.01 only for tensors with at least two dimensions. Scalar/vector parameters and biases do not decay. `batch1_worker.py:562-566` creates a 5% warmup and cosine schedule toward 1% of the peak LR over the configured 20 epochs.
- Each actual MERT full-FT `gradient_connectivity.json` contains 402 encoder tensors and 20 head tensors; each EAT full-FT trial, including adapted trial 08, contains 293 encoder tensors and 20 head tensors. Every recorded L1 gradient is positive. The frozen trial contains 20 head tensors and no encoder tensors. These receipts establish sampled backward connectivity, not that every parameter improved prediction quality throughout training.
- Downloaded `fullft8_runtime.py:61-73` additionally performed actual pretrained optimizer updates. `h100x1_fullft8_run/preflight/encoder_gradients.json:4-5` records MERT first-parameter update L1 `0.04747401922941208`; lines 433-434 record EAT `0.010220732539892197`.
- The saved worker hashes match their runtime receipts: batch1/adapted worker `5efcd568ae23d4dbc32c03ee60d01564f813ebd2231bd50a8b2bd400150ca41a`; later batch4 worker `9c076487890c6e438a9deac80a83be0e04ce4b96d512e6219612efd0cbfbd727`.

### Batch optimization changed without changing effective batch size

- Early workers use microbatch 1 and accumulation 8 (`batch1_worker.py:584-604`; `batch1_runtime.json:12`). Later trials 05-07 have recorded microbatch 4 and accumulation 2 (`05_S1_EAT_fullft/execution_config.json:12,49672-49673`, also the corresponding 06/07 receipts).
- Both are effective batch 8. The batch4 loss weighting at `worker.py:508-511,604-618` correctly weights a final short accumulation group by its actual sample count. This is not an eightfold/fourfold update-budget mismatch.
- This does not make the microbatch implementations bitwise paired: dropout execution and floating-point reductions differ. The S1 MERT/EAT comparison spans the microbatch transition, so it is a comparison of the recorded training recipes rather than a strictly identical execution trace.

| Trial | Usable rows | Updates per epoch | Selected epoch | Selected-checkpoint updates | Completed epochs | Total updates |
|---|---:|---:|---:|---:|---:|---:|
| 01 S0 MERT frozen | 916 | 115 | 5 | 575 | 8 | 920 |
| 02 S0 MERT fullFT | 916 | 115 | 3 | 345 | 6 | 690 |
| 03 S0 EAT fullFT | 916 | 115 | 1 | 115 | 4 | 460 |
| 04 S1 MERT fullFT | 472 | 59 | 8 | 472 | 11 | 649 |
| 05 S1 EAT fullFT | 472 | 59 | 6 | 354 | 9 | 531 |
| 06 S2 MERT fullFT | 473 | 60 | 2 | 120 | 5 | 300 |
| 07 S2 EAT fullFT | 473 | 60 | 4 | 240 | 7 | 420 |
| 08 S0 adapted EAT fullFT | 916 | 115 | 6 | 690 | 9 | 1035 |

Counts are derived from each trial's `training_rows.json`, `selection.json`, and `trained.json`, with `ceil(usable_rows/8)`. They are reconstruction from the executed loop, not separately logged optimizer counters. Selection epochs also appear in downloaded `notebook.log:139-146`.

### Temporal training is head-only; masking and augmentation remain connected

- `music_temporal.py:275-293` constructs and optimizes `TemporalMusic` over cached vectors. It contains no pretrained encoder parameters. Frozen-embedding results must not be presented as end-to-end encoder adaptation.
- Inputs are masked before projection (`music_temporal.py:159-169`) and after each residual convolution (`music_temporal.py:78-83`). The earliest downloaded implementation has equivalent masking at `results/temporal_h100_20260915/unpacked/music_temporal.py:50-55,76-83`.
- Augmentation is fed to the actual forward call at `music_temporal.py:316-319`; validation uses the untouched validation records at lines 338-348. `temporal_augmentation.py:45-68` restricts random cropping to eligible whole-track labels, keeps at least one audible window, preserves temporal order, and copies features before channel masking. No masking or augmentation disconnection was found.
- Temporal weighted minibatch loss is `sum(per_file_loss * global_source_weight) * N / B` (`music_temporal.py:321-331`). This is an unbiased minibatch estimate of the global weighted objective under shuffled uniform row sampling. The normalization is not a demonstrated defect.

## Confirmed comparison limits and observed optimization behavior

### Equal epochs are not equal optimizer budgets in the temporal batch comparison

- Original T0 uses batch 16, LR 3e-4, 300 maximum epochs and patience 20. Its receipt selects epoch 2 and stops after epoch 22: `results/temporal_h100_20260915/unpacked/results/T0/training.json:5-17`.
- Batch-128 T0 preserves LR/epoch/patience settings, selects epoch 4 and stops after epoch 24, logging 192 updates: `results/temporal_h100_b128_20260915/unpacked/results/T0/training.json:5-18`.
- Both have 1008 train IDs. The original loop steps once per minibatch (`results/temporal_h100_20260915/unpacked/music_temporal.py:155-174`), so batch16 has 63 updates/epoch, 126 updates at selection and 1386 total. Batch128 has 8 updates/epoch, 32 at selection and 192 total. The selected batch16 model saw 3.9375 times as many parameter updates; the completed run used 7.21875 times as many.
- This is a valid comparison of two complete recipes under epoch-based early stopping. It cannot isolate the causal effect of batch size under equal updates or equal optimization time. Similarly C08 is an explicitly joint search over LR, decay, dropout, batch size, crop probability, feature masking and NSynth mass (`temporal_tuning.py:38-48,65-71`), not a single-factor architectural ablation.

### S0 fullFT exhibits train/validation divergence; its cause is not established

- `02_S0_MERT_fullft/history.json:3-18,41-56` shows training loss falling from `0.6589431723002263` at epoch 1 to `0.058082235776853605` at selected epoch 3, while tune BCE increases from `0.5993429717350587` to `1.4516976697368718`. EER improves from `0.3030119018702939` to `0.23046716865031175`, so the code deliberately selects the worse-calibrated but better-ranked epoch.
- Selection is lexicographic `(tune EER, tune balanced BCE)` and patience is three stale epochs (`batch1_worker.py:579-580,605-624`). It does not select the minimum BCE checkpoint. This is consistent with the stated EER objective, not an accidental checkpoint-selection bug.
- The S0 MERT/EAT fullFT runs stop after six/four of the configured twenty epochs, with recorded final LR factors approximately 0.8401/0.9403 of peak (`02.../history.json`, `03.../history.json`). Their selected checkpoints occur while LR is still near peak. Thus the experiment has not tested whether lower learning rates or a different schedule would recover calibration/generalization. The recorded curves support divergence, not a specific diagnosis such as excessive LR or encoder collapse.
- Further caution: `batch1_worker.py:294-297` fixes each row's chosen window subset using `(seed, epoch, ID)`, while the balanced sampler samples with replacement at lines 584-585. Repeated copies of one row within an epoch reuse the same audio views. This limits augmentation diversity for repeatedly sampled real examples. A quality impact remains a hypothesis.

### Separator adaptation is a separate optimization stage

- `fullft_separator.py:113-146` enables all HTDemucs parameters and optimizes pseudo-stem L1, not classification loss. The downloaded separator receipt reports 41,984,456 optimizer/model parameters, all with nonzero gradients, no disconnected parameters, positive branch gradients, and parameter-update L2 `0.9930063636869454`.
- Downstream classification loads the adapted checkpoint and then freezes the separator when producing audio caches (`batch1_worker.py:215-220`). Trial 08 is separator adaptation followed by EAT fullFT, not simultaneous separator/classifier training.
- The adaptation loop initializes `best = inf`, and the selection condition excludes step 0 (`fullft_separator.py:120,157-163`). Therefore baseline fallback is not enforced if every adapted checkpoint is worse. In the actual archived run this did not produce that failure: pseudo-validation L1 was `0.011605126783251762` at step 0 and reached `0.011324429186061025` at step 250 (`separator_adaptation/history.json`). This proxy decrease does not prove better ground-truth separation.

## Provenance caveat

The current `temporal_backbone.py:67-83` launches cosine heads, whereas `results/mert330_a100_20260916/unpacked/temporal_backbone.py:65-82` launched MLP heads. The archived `results/search_protocol.json:20-22,29-30` confirms `mlp_output=true`. Historical model descriptions must use the archived code and recipes, not the current entrypoint.

## Decisive follow-up checks

1. If the decision is batch size, compare batch16/batch128 at a fixed optimizer-update budget and a fixed evaluation cadence, then separately report sample exposure and wall time. Use the same cached input IDs and loss objective. Existing epoch-budget results alone do not answer this question.
2. If the decision is S0 fullFT regularization, preserve the split and test seal; compare a lower encoder LR or short plateau schedule using tune-only selection. Record both EER and BCE to distinguish ranking gains from confidence/calibration degradation. No rerun was started by this audit.
3. If separator adaptation must guarantee non-regression on its selection proxy, a small controlled validation-loss sequence where all adapted values exceed step-0 loss decisively checks whether baseline fallback exists. The current selection logic intentionally/implicitly omits that fallback; the actual completed run improved the proxy.
4. Lightweight CPU implementation checks already performed by the masking subaudit covered padding/batch invariance, every temporal convolution and attention gate receiving gradients, crop/cache integrity and paired RNG, partial-label crop exclusion, and the real training augmentation path with unchanged validation and checkpoint replay. These establish behavior, not predictive quality.

Exact CPU command run from the project root (completed in 1.083 s, exit 0; output was observed in the subagent tool result, with no durable test log saved):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import unittest
import torch
from test_music_temporal import TemporalMusicChecks
from test_temporal_augmentation import AugmentationChecks

torch.set_num_threads(1)
suite = unittest.TestSuite([
    TemporalMusicChecks('test_padding_other_files_and_empty_music_cannot_change_a_file_prediction'),
    TemporalMusicChecks('test_every_temporal_convolution_and_attention_gate_receives_training_gradients'),
    AugmentationChecks('test_crops_preserve_time_masks_and_cache_and_pair_across_mask_conditions'),
    AugmentationChecks('test_partial_unknown_labels_and_single_window_are_not_cropped'),
    AugmentationChecks('test_actual_training_augments_inputs_without_changing_validation'),
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```
