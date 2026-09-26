# Audit evidence log (2026-09-27, KST)

Observation-time evidence gathered while the runs were active. Read-only unless stated.

## Cloud GPU run: VESSL wsp-9683432r72m7 (capella-emea, H100 SXM 80GB), zero-shot member scoring (inference, no training)
- Launch: /shared/members-20260927/run.sh (setsid nohup). Env /root/envs/data (uv Python 3.11.13), torch 2.7.1+cu128,
  torchaudio 2.7.1+cu128, torchvision 0.22.1+cu128, transformers 4.57.6, demucs 4.0.1, numpy 2.4.6 (installed 2026-09-23),
  timm 1.0.15 + pandas 3.0.6 (installed 2026-09-26 17:38Z by this session).
- Env incident (this session): `pip install timm pandas` pulled torch 2.14.0 / torchvision 0.29.0 and broke torchaudio;
  restored torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 (cu128) at 17:40Z, before any scoring process started
  (first start 17:42Z). `pip check` clean; triton 3.3.1. Leftover nvidia-*-cu13 wheels remain (unused by torch cu128).
- Processes observed 18:13Z: PIDs 2551-2554 (shards 1-4) started 17:53:25Z, cmd score_members.py ... --shards 5,
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3; loaded code uploaded 17:52:59Z (antideepfake_runtime.py sha 80732031, score_members.py
  sha 2c78f81c, component_fusion.py sha ab1885df, decision_heads.py sha f807e7a9). Manifest members_manifest.csv sha f172902c
  (4,931 data v4 val + 6,450 held-out test rows; 0 missing audio paths under /root/data_v4/out).
- Launch history: run 1 (17:42Z) unthrottled torch threads (128 visible cores, 15 vCPU quota) -> ~20 s/file; killed. Run 2
  overlapped run 1 for ~30 s (duplicate-row risk); killed. Run 3 (17:53Z) = batched AntiDeepfake windows (batch 16; batch-1 vs
  batch-16 max prob diff 3.3e-9 on CPU). Shard 0 died 18:1xZ with CUDA OOM (5 procs x ~15 GB on 80 GB); resumed 18:14Z as PID
  3013 (window batch 4, expandable_segments). /root/finish_split.sh splits the shard-0 remainder after shards 1-4 exit.
- Partial pull at 03:22 KST: 5,653 rows, 0 duplicate IDs, 0 empty/non-finite fields.
- Weights: xlsr1b = nii-yamagishilab/xls-r-1b-anti-deepfake rev 60c96543 (model.safetensors sha 57da6a47..., identical to local
  copy); xlsr2b rev 0c12cfd4 (unpinned download, recorded); SONICS awsaf49/sonics-spectttra-alpha-120s rev 094b32a5
  (pytorch_model.bin sha 082df7e3..., identical to local copy). SONICS loads with strict=False (HF wrapper).
- Precision: fp32; AntiDeepfake init sets allow_tf32=False (matmul, cudnn) process-wide, like the L4 runtime.
- Preprocessing parity (static): score_members = runtime Engine.predict path (librosa load 16 kHz mono, peak cap 0.6,
  HTDemucs via baseline separate_voice_and_music, voice stem padded/cropped to len(raw)); SONICS on the capped mix.

## Local MPS run: S2 feature chain (feature extraction, no training)
- PID 52326, started 2026-09-27 01:39:52 KST, `.venv/bin/python local_features.py --manifest _workspace/s2_extra/bank_A_train.csv
  ... --device mps`, env PYTORCH_ENABLE_MPS_FALLBACK=1, HF_HOME=_workspace/diag_aihub/empty_hf. torch 2.7.1, fp32.
- Cache identity (index.json) implementation hashes == current files: decision_heads f807e7a9, full_bank_features 2ac3a1c1,
  local_features a1476015, mert12_submission 30beefd1, v6_runtime fb2149c0; device mps; manifest b1e34717.
- 3,182 / 5,341 items at 03:13 KST. Paused with SIGSTOP at 03:17 KST (memory pressure: 24 GB RAM, ~60 MB free, blocking
  the offline package verification). Resumable (per-item index).

## Checks run
- A1 bf16 vs fp32 head inference (P1 heads, 4,931 val rows): max |d| voice 0.0014, music 0.005; Spearman 1.0; val EER
  identical (voice 0.05017, music 0.11074). Criterion |dEER|<0.002 -> pass.
- A2 device parity vs H100 CUDA caches (Existing9504 handoff parity/full_cache, 10 files: speech/music/song, wav/flac/mp3):
  MPS: windows identical, min cosine 1.000000 (voice, MERT), voice max abs 1.9e-4, presence 4e-6, P1-head |dFv|<=1.5e-5,
  |dFm|<=5.4e-5. CPU: cosine 1.0, voice max abs 1.2e-4, |dFv|<=1.4e-5, |dFm|<=3.9e-5. Pre-declared criteria met. 10 files only.
- A3 S2 extra splits: A train/dev 0 overlap in ID/orig_id/group/component/audio bytes; B: 55 music_arena battle groups have
  one side (non-Sonauto) in train and the Sonauto side in dev (shared prompt, different generator); extra vs primary bank
  0 overlap (ID, audio bytes, component, group). AI-Hub 571: probe train (800, S2a) and S2 A rows 0 group overlap with the
  400 held-out calls. Channel profiles: code uses train_calls[:120] for train, held-out calls for dev (static only).
- A6 one real optimizer step (full_bank_train.step, CPU, 64 real train items per route): finite loss, nonzero grads on all
  trainable tensors, every tensor changed by ~lr, val-in-optimizer guard raises. Buffers mean/scale untouched.
- A7 traceability: P1 and S2a zips' heads == local checkpoints == candidate_lock == model/manifest.json; S2a run_identity =
  primary bank_trainval (aeeaca1a) + extra bank_test_as_val (50e993a1) + aihub_probe/train.csv (fe13205c).
- A8 metric: independent DACON-formula recomputation of P1 validation metrics == recorded values (|d| < 1e-9, all five).
- A17 primary split (42,039 rows, full population): 0 overlap across train/val/test for ID, audio bytes, source bytes,
  component, group, orig_id; 0 val/test constructed rows with a train parent item; 0 shared parent items.
- Presence heads: fitted on train split only (data_v4/presence.py train()).
- Dry-run fused package (P1 heads + dummy fusion, 8.37 GB): offline guard active, 10 parity files, independence 0.0,
  vendored timm/SONICS and fp16 AntiDeepfake load; presence columns unchanged vs P1, fusion columns changed.
- AntiDeepfake fp16 storage vs fp32 (CPU, 6 files): max prob diff 0.0032.

## Leakage / compliance notes
- AntiDeepfake post-training data (arXiv 2506.21090): ASVspoof challenges, MLAAD, FLEURS(+FLEURS-R), LibriTTS(+R), VoxCeleb,
  WaveFake/LJSpeech, CVoiceFake, SpoofCeleb, DSD, HABLA, VoiceMOS synth, AISHELL3, CNCeleb2, MLS, codec speech. Restored
  (enhanced) speech is in the artifact (fake) class; DACON counts enhancement-only real audio as REAL.
- Val/test rows touching member training data (source or constructed parent contains asvspoof/mlaad/fleurs/librispeech/
  sonics): val 2,563/4,931, test 2,818/6,450 -> fusion_fit --unseen.
- DACON rulings: AI-Hub allowed (417844, no answer on raw-data third-party provision for 2nd-round data submission);
  fixed training-data reference tables allowed (417885); LB probing discouraged (417702); per-file independence rule 2-4.
