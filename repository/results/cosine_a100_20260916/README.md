# A100 cosine experiment: complete

Six heads completed normally on NVIDIA A100-SXM4-80GB. Both frozen MERT feature
caches were reused after checking 1,446 audio hashes and 1,446 cache hashes per
backbone. The ordered features, split, quality policy and augmentation are shared
with the previous MLP run. Configuration: cosine classifier scale 10, smoothing
0.05 during training only, dropout 0.1, AdamW lr 1e-4 / weight decay 0.01,
ReduceLROnPlateau after five stale validation epochs, factor 0.5, floor 1e-6.
Seeds: 41, 42, 43. Checkpoints minimize unsmoothed balanced validation loss;
backbone selection uses tune EER mean, standard deviation, then balanced BCE.

| Model | Tune music EER | Historical music EER | Historical file EER |
|---|---:|---:|---:|
| Previous MERT95 MLP | 27.12% | 29.72% | 32.14% |
| New MERT95 cosine | 29.97% | 36.02% | 31.60% |
| Previous MERT330 MLP | 25.56% | 22.04% | 26.67% |
| New MERT330 cosine | 28.59% | 23.52% | 27.74% |

Values are three-seed means, not ensembles; lower is better. The new MERT330
combination worsened music EER by 1.48 percentage points and file EER by 1.07
points. Its tune EER also worsened. This run changed the head, label smoothing
and scheduler together, so it does not isolate the cause of the regression.
Historical evaluation contains 190 files (150 eligible for music EER) and has
been used in prior development. Incomplete labels prevent an official total
score claim. No new submission was packaged or uploaded.

`verify_results.py` verified six checkpoints and their parameter counts,
recomputed tune scores and eight historical reports, checked split integrity,
confirmed the frozen tune selection, replayed the learning-rate schedule, and
matched all MERT330 feature receipts to the original cache. The executed notebook
has no failed or unexecuted code cells; its saved comparison chart was inspected.
Recorded LR reductions were three per run, except MERT330 seed43 with five.

The result ZIP is 5,243,029 bytes with local and object SHA-256:
`a7f11693ee47d60668b4f9cc6badaa093fce5bca7cb5720834ef03e174261d6f`.
Object location: `/root-1/temporal/cosine-a100-20260916/C08_Cosine_A100_results.zip`.
Local outputs are in `unpacked/`; see `verification.json` and
`shutdown_verification.json` for machine-readable evidence. The five-minute
monitor is paused after completion.
