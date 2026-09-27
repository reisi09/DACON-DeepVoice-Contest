# Skeptic review: Track C trainer (research/top15/work3/trainer)

Reviewed 2026-09-27 (KST): finetune_ad.py, score_ft.py, run_ft.sh, full_bank_batches.py, smoke_test.py and smoke/*. I read them against DESIGN.md, the brief, and the project modules they mirror: antideepfake_runtime.py, full_bank_train.py (head_labels/read_items/_known), full_bank_features.load_rows, graph_scoring._roc_metrics, component_fusion.detector_members, full_bank_submission.py, data_v4/dump_stems.py and package_datav4_submission.py. I ran two CPU checks. No project, cache or lane files were edited.

**Verdict: ok to run on GPU.** I found no correctness bug in labels, splits, leakage, polarity, key layout, freezing or DDP mechanics. The main risks are design-level: sampler concentration and crop label noise on songs. There are also two cheap gates the orchestrator should enforce: missing stems, and replacing rather than adding the checkpoint in the package.

## Checks run

1. **Data, splits and weights on the real bank CSVs** (scratchpad `ft_skeptic/check_data.py`, 5 s). It uses the trainer's own `voice_label`, `loss_weights` and `duration_balance`. Silent flags come from cache_tv (train) and members_all (val/test).
   - Split values: bank_trainval has `train` 30,658 and `val` 4,931; bank_test has `test` 6,450. So `--splits val test` in score_ft matches.
   - Overlap: 0 component train/val, 0 component train+val/test, 0 group train/test, 0 ID trainval/test.
   - Train voice items: 24,623 (17,171 fake). 3,328 of them are mixed (music-route) rows. 0 mixed rows were dropped as silent, and 0 voice-route train rows have silent stems. 33.1% are shorter than one window.
   - Val voice items: 3,785. Zero-shot xlsr1b_max EER with silent set to 0 is **0.11806**. Test: 5,139 items, **0.18779**. Both match the deliverable's sanity anchor exactly.
2. **2-rank gloo DDP run of the real finetune_ad.main** (scratchpad `ft_skeptic/check_ddp.py` + `ddp_driver.py`, 4 s). It uses a tiny random Wav2Vec2 (4 layers, first trainable layer 2) written in the native fairseq key layout plus 2 training-only tensors, on the real smoke stems, with a crop logger patched in and augmentation on.
   - The inverse key map round-trips through `hf_key`, and AntiDeepfake loads the checkpoint strictly.
   - Ranks draw different item sequences and different random crops. The only 2 shared crop checksums are the deterministic repeat-padded 3.1 s file.
   - The sharded eval gathers 29 windows, the same as the 1-process smoke, so no row is lost or duplicated.
   - Patience 2 stopped both ranks at the same step (20 of 40), and the final barrier exited cleanly.
   - `changed_outside_trainable=[]`, `trainable_unchanged=[]`, `weights_exact=true`, prob diff 9.8e-5.

## Findings (most important first)

### 1. Design risk: WeightedRandomSampler turns loss weights into heavy repetition
This is DESIGN-conformant, so it is flagged here and was not changed. `loss_weights` gives every (class, source, generator) cell equal mass. `full_bank_train` applies that as a loss multiplier while visiting each row once per epoch. Here it is used as sampling with replacement, which means a 315M-parameter backbone sees the same few files many times. At the default 4 epochs (98,816 draws):
- The effective sample size is **4,211 of 24,623** items.
- The top item is expected about **297 times** (297 random crops of one file). 524 items are drawn 20 or more times, and the top 1% of items take 13.7% of all draws.
- **4,165 items (17%) are expected fewer than 0.5 times**, so each has less than a 40% chance of being seen at all.
- **MLAAD fakes take 32.5% of all draws** (about 32.1k), because MLAAD's many generator cells each get a full cell share.

This is a memorisation and over-specialisation risk that the frozen-head trainer never had. It may also explain a val-versus-held-out gap if one appears.

Cheap mitigations (the orchestrator decides):
- Cap the per-item sampling probability, for example `w = np.minimum(w, 5 * w.mean())` before the sampler.
- Or sample half the batch uniformly.

Whatever the choice, log the ESS in the setup record.

### 2. Design risk: random crops of song voice stems carry the file label into windows with no vocals
3,328 mixed train rows carry 17.5% of the sampling mass. Among them, the fakes are constructed_mix, sonics, echoes and generated_music (1,928 rows), and the reals are openmic, FMA, haim, Jamendo and real constructed mixes (1,400 rows).
- A random 4.04 s crop of a song's HTDemucs voice stem can fall in an instrumental passage, which is only separation residual, and still carries the song's voice label.
- The member is the **file max** over windows. A model that learns "residual/quiet voice stem means fake" will fire on instrumental passages of real songs.
- `voice_silent` guards whole-file silence only. `event_intervals` is empty for all 3,328 mixed rows, so this could not be quantified locally.
- Val contains the same kinds of rows, so selection partly protects against it.

Option (a DESIGN deviation, so the orchestrator decides): energy-gated cropping, e.g. best-of-4 random crops by RMS, or reject crops below `SILENCE_RMS`.

### 3. Gate gap: missing stems are silently tolerated
- `finetune_ad` counts missing stems in `setup.missing_stems` but trains on whatever exists.
- `score_ft` skips rows without a stem, so `ft_members.csv` can lack rows. Only `score.log` says "N without stems".
- `run_ft.sh` asserts neither.
- An incomplete `/root/stems` would shrink training and quietly bias the held-out gate onto a subset.

Recommended fix, orchestrator-side or a one-line assert in run_ft.sh: require `missing_stems == [0, 0]`, and require the ft_members row count to equal the val+test row count (4,931 + 6,450 = 11,381).

### 4. Packaging: the ft checkpoint must replace the frozen AD, not sit beside it
- S4's zip already ships the frozen AD as fp16: `model/assets/antideepfake/ad/model.safetensors` is 1,931,134,588 B, and the zip is 8,295,658,583 B.
- RUN/best/model.safetensors has the same layout and about the same size, so replacing it keeps the package at about 8.30 GB.
- Keeping both gives about **10.23 GB**, which is over a 10 GB (decimal) limit.
- Replacing also keeps the L4 time unchanged, since there is one AD forward per file. The runtime integration should read `AntiDeepfake.windows(voice)[1]` (which pads to 400 samples like `load_voice`) and not `predict()` (which raises below 400 samples).
- `package_datav4_submission.py` copies only the config and safetensors, so `meta.json` does not leak into the package.

### 5. bf16 selection versus fp32 members
- Selection evals run under bf16 autocast (the frozen trunk runs in bf16 too). The members in ft_members.csv and the L4 runtime are fp32 with TF32 disabled.
- The deliverable already flags this, and the step-0 eval measures it directly: compare step-0 `val_eer` with the fp32 anchor 0.11806.
- If the gap exceeds about 0.003, rerun with `--eval-precision fp32`. That costs roughly 30 s or more per eval at fp32 without TF32, about 9 min for 18 evals.
- Note that `val_best.csv` holds bf16-eval probabilities. Downstream code must use ft_members.csv.

### 6. Minor or cosmetic
- **Step and eval counts.** The default run is 772 steps (4 x 193), not 770, with 18 evals including step 0, not 17.
- **Save cost.** Each improving eval makes rank 0 write a 1.93 GB checkpoint while the other ranks wait in the next all-reduce. That is under the 10 min NCCL timeout on local disk, but it can add minutes of wall time if the run directory is on a network volume. Keep OUT on local NVMe.
- **Non-atomic swap.** `save_native` does rmtree(best) and then rename(best.tmp). A crash between the two leaves only best.tmp.
- **No finiteness guard.** An fp16 overflow (very unlikely at lr 1e-5) would save inf, and `weights_exact` would still pass. `roundtrip.ok` would go false without failing the run. An `isfinite` assert in `save_native` would close this.
- **Repeat-pad mismatch.** Short clips are repeat-padded in training but scored unpadded in eval and at runtime. This affects 33.1% of train items, as the deliverable states. It is a known, accepted DESIGN choice.
- **Val used twice.** Val selects the checkpoint (up to 18 looks) and is then reused for the fusion weight and CDF tables. That is optimistic on val, and only the held-out gate is decisive.
- **Seen versus unseen sources.** Report the held-out gain split into rows AntiDeepfake saw in pretraining and rows it did not (`research/top15/work/orchestrator/s3_unseen_check.py`). Fine-tuning gains on seen sources, such as the ASVspoof family, are the likely inflation.
- **Augmentation placement.** Augmentation is applied to the separated stem, whereas real phone-channel test audio passes through HTDemucs first. This is DESIGN-fixed.

## Verified OK (no action)
- **Labels.** `voice_label` equals `head_labels`' voice head together with the read_items silent-mixed drop. Class 0 is fake in the targets, eval uses softmax index 0, and EER treats fake as positive.
- **No test leakage.** Test is never read by finetune_ad. Selection uses val only. The component check runs at startup.
- **Key layout.** `hf_key` is a verbatim copy of the runtime mapping. All native keys are written, the 7 training-only tensors are copied, and reload is strict.
- **Freezing.**
  - The default trainable set is layers 32..47 plus the encoder LN and the classifier: 260 tensors, 314.8M parameters. That is 16 x 16 + 2 + 2 tensors.
  - With the encoder module in eval, layerdrop (0.1 in the config) is off, because HF gates it on `encoder.training`. feat_proj and hidden dropout below layer 32 are off. Frozen layers build no autograd graph.
- **Windows and pooling.** The window layout, the 400-sample pad, single unpadded clips for files shorter than one window, and max/top30/mean all match `AntiDeepfake.windows`/`predict` and `detector_members`. `voice_silent` in the stems uses the runtime's `calculate_rms < SILENCE_RMS` on fp32 audio.
- **DDP.**
  - The sampler generators and global seeds differ per rank. Each rank yields exactly `total` batches, so there is no uneven-step hang.
  - Early stopping is computed from gathered data identically on every rank.
  - `gradient_as_bucket_view` with `zero_grad(set_to_none)` is a supported combination, and `clip_grad_norm_(error_if_nonfinite)` raises on every rank together.
- **Environment.** The code is Python 3.11 compatible (local venv 3.11.15). Stems are uncompressed `np.savez`, so reads are cheap.

## Not verified (needs the GPU run)
NCCL with 4 ranks, CUDA bf16 numerics, peak memory at batch 32, loader throughput from /root/stems, rank-0 save and reload time on VESSL storage, and any real val or held-out gain.
