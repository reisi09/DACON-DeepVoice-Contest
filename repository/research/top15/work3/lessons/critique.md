# Skeptic review: LESSONS lane (work3/lessons)

Reviewed: the lane's deliverable summary (sections A and B up to G8; the text I was given is cut off mid-G8 at
"Real content from an unseen sour", so nothing after that was reviewed), plus `val_proxy_check.py`, `.log` and `.json`.
The folder has no written guardrail document. The guardrails exist only in the summary.
Cross-read: `work3/DESIGN.md`, `antideepfake_runtime.py`, `work3/trainer/finetune_ad.py` and `score_ft.py`,
`work3/integration/full_bank_submission.py`, `component_fusion.py`, `full_bank_batches.py`,
`_workspace/members_20260927/members_all.csv`, the bank CSVs and `_workspace/backbone_cache`.

**Verdict: fix-needed.** The core numbers reproduce exactly and the splits are clean. But G1 cannot pass as written,
the step-0 anchor conflicts with the trainer's default bf16 eval, and a few smaller claims are overstated or wrong.

## CPU checks run (2)

1. `check1_stats.py` (scratchpad `lessons_skeptic/`, about 11 s, CSV and npz metadata only). It recomputes the counts, the
   zero-shot anchor, saturation, the sampler draws with the trainer's own `voice_label`/`loss_weights`/`duration_balance`,
   Spearman correlations from `val_proxy_check.json`, and the share of short clips.
2. `check2_precision.py` (82 s, RSS about 4 GB, 4 threads). It runs AD-1B on the first 64,600-sample window of each of the
   10 parity files in three settings: fp32 original, CPU bf16 autocast, and weights rounded to fp16 with fp32 compute.

## Findings (most severe first)

**F1. G1's "export -> reload of the unmodified model must pass `verify_native`" cannot pass. [high]**
- (a) `verify_native` reads `native_reference.json` and `parity_audio.npz` from the model directory. Neither file exists
  anywhere in the project: not in `_workspace/models/xls-r-1b-anti-deepfake`, and not in the listed VESSL
  `/root/models/xlsr1b` (model.safetensors + antideepfake_config.json only). The call raises FileNotFoundError.
- (b) Even with those files, DESIGN's fp16 export alone moves the logits by up to **6.9e-3** [measured]. That is far above
  verify_native's `atol=3e-4, rtol=1e-4`, so the check would fail even with a correct key mapping.
- Replace the gate with three checks:
  1. The trainer's `save_native` missing/extra key check. It already enforces the key set, including
     `mask_emb` and pos_conv `weight_g`/`weight_v`.
  2. Reload the export and compare it against the in-memory model after `p.copy_(p.half().float())`. This should agree
     to float noise and is the real key-mapping proof.
  3. Compare it against fp32 with a log-odds tolerance of about 0.05.

**F2. The step-0 anchor ".1181 must reproduce" conflicts with the trainer's default `--eval-precision bf16`. [medium]**
- `evaluate(0)` runs under bf16 autocast on CUDA. CPU bf16 autocast moved window log-odds by up to **0.46** and fake prob
  by up to **0.006** against fp32 [measured]. CUDA autocast differs in detail but is of the same order.
- So step 0 will not return exactly .11806. Also, the trainer's val set is whatever has stems (`missing_stems`), so a
  fixed number is the wrong target.
- Fix:
  - Run step 0 (and a final confirmation of the chosen checkpoint) with `--eval-precision fp32`, or accept |ΔEER| ≤ 0.003.
  - Compare against `members_all.csv` `xlsr1b_max` on the trainer's actual val IDs.
  - G2's kill and G7's selection inherit this bf16 noise, so EER differences of about 0.003 between evals do not mean much.

**F3. Tolerances in probability space are ill-posed. This covers G5's "1e-3 in prob" and the trainer's round-trip `ok: diff <= 1e-3`. [medium]**
- fp16 rounding gave a log-odds shift of 0.013 [measured]. The measured Δprob of 4.6e-4 is small only because none of the
  10 windows was near p = 0.5.
- At p ≈ 0.5 the same shift is about 3e-3 in probability. A correct checkpoint could therefore fail the round trip.
- Use a log-odds tolerance instead (≤ 0.05).

**F4. G6's "runtime would need `windows()` to return logits" is not needed. [medium, saves a source edit]**
- `AntiDeepfake.windows()` already returns the pooled fp32 embedding `emb` (the classifier input).
- The caller can compute log-odds as `classifier(emb)[:, 0] - classifier(emb)[:, 1]`.
- `full_bank_submission.fused_voice` currently uses `windows(voice)[1].max()`. Only integration code changes;
  `antideepfake_runtime.py` stays untouched.

**F5. G3's evidence is overstated. [medium]**
- Spearman correlation with held-out, across epochs 1-30 [measured from the lane's JSON]:

  | Head | val | val_ood |
  |---|---|---|
  | Linear | 0.285 | 0.384 |
  | MLP | 0.429 | −0.244 |

- For the Linear head, val_ood tracked held-out *better*. Its pick was the best held-out epoch (.13562, against .13659 for
  the val pick).
- The "val pick within .001/.010 of the best held-out epoch" result comes from flat held-out curves, not from val
  tracking held-out.
- Fair reading: there is no evidence that carving helps, but val selection is weak, and the flatness may not hold for
  roughly 315M trainable parameters. This makes G7 (the CE stop) and G8 (the held-out check) load-bearing.

**F6. The short-clip regime differs between training and eval/runtime. [low-medium, not flagged by the lane]**
- Training tiles clips shorter than 64,600 samples (`np.tile`). `windows()`, and therefore val and the runtime, scores the
  unpadded short clip instead.
- 26.2% of val voice rows are shorter than one window, and 72% of those are fake. On held-out the share is 5.0% [measured].
  So about a quarter of val EER comes from an input shape the model never trained on and the competition test does not contain.
- G3 suggests reporting EER on rows of at least 64,600 samples. The test minimum is 4 s = **64,000** samples, but only 20
  val rows fall in [64,000, 64,600), so the difference is negligible.
- Suggestion (a deviation from DESIGN): use val EER on rows of at least 64,000 samples as the G2 kill metric, and at least
  as a co-metric for selection.

**F7. The arithmetic in A.4 is wrong. [low]**
- Start from 0.3·ΔMusicEER + 0.5·ΔFileEER = 0.113, with the new Music EER at most 0.5.
- ΔFileEER ≥ 0.05 requires ΔMusicEER ≤ 0.293, so M330's Music EER must be at least **about 0.21**, not 0.12.
- If M330's Music EER was 0.12, the bound on ΔFileEER is about 0. If it was 0.15, the bound is only about 0.016.
- The mechanism story may still hold, but this bound does not support it.

**F8. G4 figures are rounded. The direction is right. [low]**
- These reproduce exactly: median 0.359, p99 5.46, max 74.0, 7.4% of draws to rows drawn 20+ times, 9.8% of rows below 0.1.
- The per-cell figures are rounded. Draws per epoch by cell:

  | Cell | Rows | Draws per epoch |
  |---|---|---|
  | constructed_music REAL | 13 | 41-74 (summary says 43) |
  | constructed_enhanced | 17 | 25-60 (summary says 43) |
  | FMA | 9 | 59.1 |

- Over 4 epochs that is up to about 300 exposures, not 170-240.
- Clipping at 4 and then renormalizing gives a max of 4.51, with 618 rows above 4. Real-class mass is .449 (the summary says .446).
- Either iterate the clip or state the cap as about 4.5.

**F9. G8's per-cell veto (real-cell FPR worsens by ≥ 0.05 absolute) has no cell sizes and no paired test. [low]**
- In small cells a 0.05 change is noise.
- Report n per cell, and veto only when a paired, group-level bootstrap shows the worsening beyond its 90% CI.

**F10. Diagnostic hygiene. [low]**
- In `val_proxy_check.json`, `mlp.test_frozen` (.2835) is a random-init MLP at epoch 0, not a frozen reference. Do not cite it.
- The diagnostic scored held-out at every epoch (62 times), and those results informed the G2 and G3 thresholds. This is
  acceptable for frozen-feature heads, but disclose it. The held-out stays single-use for the fine-tuned model itself.

**F11. DESIGN note: random gain of ±6 dB is a no-op. [low]**
- `AntiDeepfake.forward` layer-normalizes each window, which removes any gain change. Noise added at a set SNR is
  scale-invariant as well.
- It is harmless, but it should not be counted as augmentation.

**F12. Cross-lane note on size and runtime. [info]**
- The fine-tuned fp16 checkpoint is 962.5M params × 2 B ≈ **1.93 GB**, plus the training-only tensors that `save_native`
  copies. That fits under 10 GB if it *replaces* `ad`.
- `full_bank_submission.load_fusion` loads a separate AntiDeepfake for every nonzero member.
- If the val fusion fit keeps both `ad` and `ft`, the package carries two 1B models and runs AD twice per file on L4.
  S3 already needs a 48-min fallback that skips AD.
- Enforce `weights['ad'] = 0`, as DESIGN says ("replaces"), or re-time the package.

## Held up (no change needed)
- **Splits.** Bank train, val and test share 0 components and 0 groups. The trainer reads only `split == 'train'`/`'val'`
  and asserts component disjointness. The diagnostic carves val_ood by component, and `epoch_order` refuses non-train rows.
- **Counts.** 24,623 train / 3,785 val / 5,139 held-out voice rows, all present in `backbone_cache` (100% coverage).
  10,583 components, 22 sources, 17 real sources, 358 fake cells.
- **Anchor and polarity.**
  - `members_all.csv` `xlsr1b_max` on all 3,785 val voice rows gives EER .11806 and AUC .944, with index 0 = fake.
  - Held-out gives .18779. Both match linear-head epoch 0 on the cached embeddings exactly, so cache embedding +
    classifier equals the runtime member.
  - The trainer's targets use class 0 = fake.
- **Saturation.** On val, 34.9% of rows are above .9999 and 76.1% of values are unique. On held-out: 19.9% and 86.8%.
  No value is exactly 1.0.
- **G5 facts.**
  - The config has layerdrop 0.1, mask_time_prob .075, and 0.1 attention/hidden/feat_proj dropout.
  - HF's layerdrop keys on the encoder's `training` flag. The trainer keeps that module in eval, so layerdrop and the
    frozen part's dropout are off.
  - The trainable set is layers ≥ 32 + `encoder.encoder.layer_norm` + classifier. It excludes `pos_conv_embed` and
    `masked_spec_embed`, so there is no unused-parameter DDP crash.
  - `changed_outside_trainable` is fingerprinted.
- **DDP sampler.** Each rank seeds its generator with `seed*1000+rank`, so ranks draw different samples. Draws per epoch
  across ranks equal N, as the lane assumed.
- **fp16 range.** Max |w| is 5.64, so there is no overflow. 0.14% of nonzero weights fall in fp16's subnormal range,
  with a negligible effect.
