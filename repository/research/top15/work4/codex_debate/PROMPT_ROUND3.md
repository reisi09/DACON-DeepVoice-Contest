Round 3 (planning only). The user asks us to design GPU experiment plans together. NOTHING will be executed now:
no GPU, no downloads, no submissions. Produce plans that a human could launch later.

## Facts (VERIFIED in this project today)
- Remaining time: ~32 h to 2026-09-29 10:00 KST. Slots: 09-28 x3 (planned S5 -> S6 -> S4), 09-29 x3 (S7 exploratory + 2 open).
  Any new package must be trained, packaged (zip <= 10 GB, currently ~8.9 GB; S7 ~9.73 GB) and CPU-verified before ~09-29 08:00.
- Credit $81.53 (VESSL). Workspaces: H100x1 (wsp-9683432r72m7), H100x4 (wsp-9v8efltansxx), A100x1 (other cluster), CPU-only
  (other cluster, cannot see /root). H100x1 and H100x4 share one cephfs /root. Cost ESTIMATE: x1 ~$2.5/h, x4 ~$10/h.
- VESSL auto-stops an idle GPU workspace (~10-20 min idle) -> pull results immediately; HF anonymous downloads from VESSL hit
  rate limits (use the local Mac with auth + upload ~3.5 MB/s, or already cached weights in /root/models).
- Stored HTDemucs stems (fp16) exist for all 42,039 data-v4 rows at /root/stems -> encoder extraction without separation:
  4 encoders (SSLAM pt/ft, WavLM-L, HuBERT-L) over 42,039 rows took ~20 min on H100x4 (12 shards, code/extract_s6.py).
- Extraction with demucs on H100x1: EAT-only ~5 files/s (5 shards); voice feats (DF-Arena+AD) ~4.75 files/s (3 shards;
  6 shards OOMed at ~9 GB/process + long-clip spikes).
- Partial fine-tune of AntiDeepfake XLS-R-1B on H100x4 DDP: 772 steps in ~12 min, then rejected (in-domain overfit).
- Generation envs on VESSL: /root/envs/acestep (ACE-Step, code/generate_music.py; 500 generated_music rows already in the
  bank), /root/envs/cosyvoice2 (code/generate_speech.py backends xtts/chatterbox/cosyvoice2/melo/seedvc; no generated
  speech rows are in the bank). Competition rule: every generated file used for training must be submitted in round 2.
- Held-out weaknesses (RT, rt_s6enc_report.json/unseen_report.json): unseen music generators lyria-3 pro and
  fx_encoder_style_transfer (music EER ~0.14 even with SSLAM), singing voice inside music (music-context voice EER ~0.13),
  held-out -> LB gap ~0.09 (cause unknown). Train has only 5 rows of real-vocal + AI-accompaniment (held-out 107).
- Adopted/validated levers so far were all frozen-encoder representation changes (EAT +0.012, SSLAM+WavLM +0.010);
  data additions (SingMOS, Music Arena, AI-Hub calls) were noise-level; fine-tuning hurt unseen generators.

## Claude's draft GPU plans (critique every one: keep/kill + why; then add up to 5 of your own)
G1 CUDA runtime smoke + timing calibration of S7 on H100x1 with torch.cuda memory capped to 22.4 GiB: run the real
   package (hardlinked assets) on 300 data-v4 test files, also S3 on the same files; predicted L4 time = 24 min x t(S7)/t(S3).
G2 Second-wave frozen-encoder screen on stored stems (H100x4, ~20-30 min): BEATs iter3+, Dasheng-base, CLAP-music audio
   tower, Whisper-medium encoder layers, EnCodec/DAC encoder latents (codec-artifact detectors for music), MERT-95M.
   Same locked protocol (layer screen -> val gate -> held-out once, RT).
G3 Mixture-vs-stem features: run EAT/SSLAM/MERT on the raw mixture (not the backing stem) as an extra music member
   (H100x1 ~1-2 h with decoding) - separation may erase generator artifacts.
G4 Separation quality: re-separate with htdemucs_ft (4-model bag) for the voice stem, re-extract DF/AD/WavLM voice feats,
   test music-context voice EER (H100x4 ~1-2 h); runtime cost x~4 for demucs may break the 60-min limit.
G5 Source-disjoint fine-tuning: LoRA or last-N-block adaptation of EAT/SSLAM on the backing stem with early stopping on a
   leave-generator-family-out validation carved from train (not the held-out), H100x4 DDP ~2-3 h.
G6 Synthetic AI covers: real vocals (VocalSet/CSD/SingMOS gt, never AI-Hub abroad) + AI accompaniment (ACE-Step
   instrumentals generated on GPU) + the 3 matched control combinations with identical processing; features on x4; DB gate.
G7 New fake-speech families with the installed TTS/VC envs (CosyVoice2, Chatterbox, Seed-VC), choosing families NOT in the
   held-out so the unseen-generator slice stays unseen; features; DB gate on unseen-generator voice EER.
G8 New AI-music families with ACE-Step variants/prompts and other open generators to push unseen-music robustness
   (Lyria/fx_encoder are closed or unavailable) - generate, stem, extract, DB gate.
G9 Proposal C (cross-stem window-token transformer on cached per-window features): GPU x1 for fast seed/hyper sweeps.
G10 Test-time robustness: codec/resampling/loudness augmentations of held-out audio (GPU inference of the frozen stack)
   to measure sensitivity and check whether the held-out->LB gap is a channel effect (diagnostic only).

## What to return (Korean, <= ~1,300 words)
1. Keep/kill verdict per G1-G10 with the decisive reason (cite facts above or files).
2. Up to 5 new plans (G11+), same level of detail.
3. A ranked schedule for the next ~30 h using x1/x4 in parallel with go/no-go gates, dependencies on the submission
   slots, ESTIMATED wall-clock and $ per step, and the single fact that would change the ranking.
4. For the top 3: exact protocol (data, split/leakage guard, primary metric = RT held-out ADS + unseen-generator slice,
   acceptance/rejection), integration path into a package within L4/60-min/10-GB limits, and rights/submission obligations.
Label claims VERIFIED / INFERRED / ESTIMATE / UNKNOWN.
