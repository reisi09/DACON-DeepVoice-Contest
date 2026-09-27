# R3 — external methods (primary sources) — notes, 2026-09-27 KST

All URLs accessed 2026-09-27. "fetched" = page body read with WebFetch; "snippet" = search-result text only.
All CPU checks are read-only, under 1 min, and at most 1.6 GB RSS, with one exception: the first `music_statpool_check.py` run peaked
at 5.05 GB for about 8 s, above the 4 GB cap. It was rerun at 1.6 GB, and the output file holds the rerun.

## 1. Sources

| # | Source | URL | Access | What it supports |
|---|---|---|---|---|
| S1 | Ge et al., Post-training for Deepfake Speech Detection (AntiDeepfake), ASRU 2025 | https://arxiv.org/html/2506.21090 | fetched | 56k h real + 18k h "speech with artifacts", >100 languages. The artifact data includes restored speech (FLEURS-R, LibriTTS-R) and vocoded speech. Architecture: GAP of last layer, then FC. Zero-shot 1B EER: ITW 12.14/10.91, DF-Eval-2024 26.76/26.17. Fine-tuning (6k steps, lr 1e-6, RawBoost) on DF-Eval-2024: 1B 11.86 (4 s) / 9.27 (13 s) / 8.28 (50 s), vs 19.96 / 16.31 / 11.20 without post-training. Train crops 10–13 s. No frozen or linear-probe experiment. |
| S2 | AntiDeepfake repo | https://github.com/nii-yamagishilab/AntiDeepfake | fetched | Code BSD-3; checkpoints CC BY-NC-SA 4.0; fine-tuning lr 1e-6 with RawBoost |
| S3 | Zhang et al., XLS-R + SLS classifier, ACM MM 2024 | https://doi.org/10.1145/3664647.3681345 | snippet | Learned weighting of the 24 XLS-R layers; EER DF21 1.92, ITW 7.46 |
| S4 | Serrano et al., Layer selection and fusion of SSL CMs (arXiv 2509.12003) | https://arxiv.org/html/2509.12003 | fetched | Encoders are frozen. Intermediate layers are best (XLS-R L6, WavLM-L L8, MMS L9) and error rises toward the output layer. Score-level fusion of 4 encoders is better out of domain (ASV5-eval 4.7 vs best single 5.3–9.1). |
| S5 | Pascu et al., Interspeech 2024 (arXiv 2309.05384) | https://arxiv.org/html/2309.05384 | fetched | Frozen last-layer mean features with logistic regression (<2k params) average 8.8% EER over 8 sets. On 4 sets, LR gets 6.5% mean EER, better than a 3-layer MLP (8.2) and self-attention (11.3). |
| S6 | Liu et al., Nes2Net (arXiv 2504.05657), repo Apache-2.0 | https://arxiv.org/html/2504.05657 ; https://github.com/Liu-Tianchi/Nes2Net | fetched | Back-end takes frame-level B×D×T features. CtrSVDD 2.53 vs AASIST 3.30; DF21 1.65 vs 2.85. Needs frame sequences. |
| S7 | SVDD 2024 challenge summary (arXiv 2408.16132) | https://arxiv.org/html/2408.16132v2 | fetched | CtrSVDD top 1.65% EER. Top systems use SSL front ends (XLS-R), ensembles, RawBoost and AASIST/ResNet. ACESinger is about 50% EER. WildSVDD: separated vocals beat mixtures (8.48 vs 10.50; 14.91 vs 16.85). |
| S8 | Guragain et al., CtrSVDD ensembles (arXiv 2409.02302) | https://arxiv.org/html/2409.02302v1 | fetched | Fine-tuned WavLM/XLS-R + AASIST with layer aggregation. Best single system 2.70, ensemble 1.79. |
| S9 | SingFake (arXiv 2309.07525) | https://arxiv.org/abs/2309.07525 | fetched (abstract) | Speech-trained CMs degrade on singing; retraining on singing helps. Open challenges: unseen singers, codecs, languages, musical context. |
| S10 | SingGraph, Interspeech 2024; Phukan et al. (arXiv 2409.14131) | https://github.com/xjchenGit/SingGraph ; https://arxiv.org/abs/2409.14131 | snippet | SingGraph fuses MERT (pitch/rhythm) with wav2vec2 (lyrics). Phukan et al.: speaker-recognition SFMs are the best single models for SVDD, and fusion helps. |
| S11 | SONICS, ICLR 2025 (arXiv 2408.14080), CC BY-NC 4.0 | https://arxiv.org/html/2408.14080 | fetched | Long context helps: 120 s vs 5 s gives +6 to +17% F1. Generator-OOD drops (SeedMusic F1 0.80 SpecTTTra-γ vs 0.64 ViT). |
| S12 | Afchar et al., A Fourier explanation of AI-music artifacts, ISMIR 2025 | https://arxiv.org/html/2506.19108 | fetched | Deconvolution creates periodic spectral peaks. Pipeline: LR on peak residual, 1–8 kHz band at 16 kHz. Suno/Udio-130 100%, but unseen Udio-32 only 39.8%. Manipulation robustness was not tested. Code CC BY-NC-SA. |
| S13 | Afchar et al., ICASSP 2025 (arXiv 2405.04181) | https://arxiv.org/abs/2405.04181 | snippet | 99.8% in-domain, but brittle to unseen models, pitch shift and compression |
| S14 | Dugelay et al., ISMIR 2026 (arXiv 2607.27454) | https://arxiv.org/abs/2607.27454 | fetched (abstract) | Detectors collapse under speed and pitch changes; proposes a log-frequency invariant fix |
| S15 | Morosanu et al., RobustifAI@IJCAI 2026 (arXiv 2608.14916) | https://arxiv.org/abs/2608.14916 | fetched (abstract) | Edited human music is a hard negative (clip F1: edited 0.720 vs AI 0.836) |
| S16 | Han et al., Sofia (arXiv 2606.16612) | https://arxiv.org/html/2606.16612 | fetched | Frozen MuQ + MERT + Fx-Encoder++ (only RawNet2 fine-tuned) with MoE fusion. MUSIC8K-O F1 97.2 vs CLAM 78.7 (+18.5). MoE beats concatenation by 8.7 acc. Train: SONICS Suno-3.5/Udio. |
| S17 | Park et al., ICML 2026 ML4Audio (arXiv 2606.08663) | https://arxiv.org/abs/2606.08663 | fetched (abstract) | Under a fake-source restriction, token space matters: X-Codec is best when training on Udio, MERT when training on Suno-3.5 |
| S18 | Xie et al., All-type ADD / WPT, AAAI (arXiv 2504.06753) | https://arxiv.org/html/2504.06753 | fetched | A speech-trained model gets ≈49.8 EER on music. Speech-trained frozen XLSR-AASIST averages 32.58, full FT 27.68. MERT-AASIST is best when trained on music only; co-trained WPT-XLSR-AASIST averages 3.58. |
| S19 | AT-ADD Grand Challenge summary, ACM MM 2026 (arXiv 2608.14249) | https://arxiv.org/html/2608.14249v1 | fetched | T1 #1: w2v-BERT2 frozen-LoRA→joint FT, with noise/reverb/codec/replay augmentation. T1 #3: XLSR-1B with partial FT of the last 6 layers, 4.04 s windows median-pooled, DANN + GroupDRO against generator/condition shortcuts. T2 #1: BEATs router; XLSR-AASIST for speech, EAT-large-AASIST for non-speech; class-specific thresholds; conservative multi-crop pooling. T2 #2: EAT-large + XLSR-300M with learnable softmax layer weights. T2 #3: XLSR (3,11,24) + BEATs (3,6,9). |
| S20 | AT-ADD benchmark (arXiv 2608.23437) | https://arxiv.org/html/2608.23437 | fetched | Frozen-encoder WPT 73.35 vs full FT 76.73 (T1). FT-XLSR-AASIST per type: speech 79.5, sound 66.8, singing 96.3, music 75.3 F1. |
| S21 | Fan et al., dual-domain SSL fusion, AT-ADD T2 #2 (arXiv 2608.29021) | https://arxiv.org/abs/2608.29021 | fetched (abstract) | EAT-large + XLS-R-300M with layer-weighted + token-level fusion; 95.58 Macro-F1 |
| S22 | MuQ model card; EAT repo | https://huggingface.co/OpenMuQ/MuQ-large-msd-iter ; https://github.com/cwx-worst-one/EAT | fetched | MuQ: weights CC-BY-NC 4.0, ~300M params, 24 kHz, hidden states exposed. EAT: MIT, base 88M / large 309M, HF AutoModel. |
| S23 | DACON public code | /tmp/dacon_codeshare.json ; https://github.com/kimdong20020619-ctrl/Deep-voice ; https://github.com/sideotod/deepvoice-detection ; https://github.com/hyunI0326/dacon_voice | local + fetched | Codeshare lists only the official baseline (cs_id 14153). The two public repos are baseline derivatives (PANNs + HTDemucs + DF-Arena; fusion max/soft-or) with no score. The third returns 404. |

Local facts used:
- DF-Arena front end already has layer weighting: `_workspace/assets_e9504/baseline/model/df_arena_1b/backbone.py` lines 9-60. It does per-layer attentive pooling, then a sigmoid fc0 gate per layer, then a weighted sum over layers, then a 4-block conformer. Our voice heads read the conformer fc5 input.
- Cache contents (`np.load`):
  - `backbone_cache`: `ad_embeddings` (n_win, 1280) fp16 and `mert_layers` (n_win, 8, 1024) for layers 3..24.
  - `cache_v4`: `voice_embeddings` (n_seg, 1280) fp32 and `mert_windows` (n, 1024), layer 12.
- Only pooled vectors are cached. SLS, AASIST and Nes2Net need frame sequences, so none of them can run on the caches.
- An earlier in-repo EAT full fine-tune (`_workspace/fullft8_live/h100x1_fullft8_run/*/evaluation.json`, 09-16, n=40/subset) did better than frozen MERT on seen music (EER 0.10 vs 0.217) but worse on OOD music (0.30 vs 0.20).

## 2. CPU checks (question → result). Scripts and outputs are in this folder.

1. `ad_restored_check.py`. Does zero-shot AD-1B over-flag enhanced, phone-coded or singing REAL voice?
   - Enhanced reals vs their zeroth/fleurs parents: AUC 0.748; fraction above 0.5 is 0.233 vs 0.038.
   - tel_opus8 speech reals: 0.408 above 0.5 (other codecs 0–0.05).
   - Singing/music-context reals: 0.677 above 0.5; fakes: 0.886.
2. `codec_heads_check.py`. Do the TRAINED DB-protocol heads keep these weak cells? Held-out, at each member's EER threshold:
   - Real FPR: tel_opus8 0.300 (DF) / 0.412 (AD), vs about 0.05–0.18 for other codecs.
   - Phone-codec fake misses: 0.15–0.23, vs about 0.07–0.16 for wideband codecs.
   - musan_music reals: 0.326 / 0.426. constructed_enhanced: 0.242 / 0.158.
3. `fourier_peak_check.py` / `fourier_logo_check.py`. Tests the Afchar fingerprint on local audio at 16 kHz.
   - Within-generator CV: Lyria vs FMA AUC 0.992; vs SonicMaster 1.000.
   - Leave-one-generator-out vs SonicMaster REAL: 0.01–0.66 (Lyria 0.579).
   - Leave-one-generator-out vs held-out jamendo/FMA reals: 0.71–0.93, with audioldm at 0.16.
   - Conclusion: the detector does not transfer to an unseen real source or to AI-mastered real music.
4. `routed_fusion_check.py`. Type-routed AD weight, with speech/music context chosen on val. Val picks (1, 4). Held-out voice EER 0.1171 vs 0.1106 for global 0.5; delta −0.0069, CI [−0.016, +0.002]. Rejected.
5. `window_pool_check.py`. Voice window pooling, using the DurBal DF heads.
   - max: val 0.0510, test 0.1107 (len-rho +0.24).
   - Val-selected top2: test 0.1060; gain +0.0043, CI [−0.0009, 0.0104].
   - median: test 0.1403 (len-rho 0.01; files ≥15 s 0.283).
   - Conclusion: length-normalising the pooling removes the length correlation but costs accuracy.
6. `music_statpool_check.py`. LR on file-mean MERT features (val-selected [3,6,9,12], C 0.01) gives test music EER 0.1884. WindowMIL 6+12 on the same rows gives 0.1847. Every layer subset lands at 0.184–0.20, so the music representation is the limit, not the head.
7. `cross_component_check.py` / `cross_component_music_check.py` / `cross_component_ads.py`. Song-aware VOICE: add the file's own music-head CDF to VOICE when music is present.
   - Val-selected setting: ours 1, ad 1, music 0.5·Pm (runtime presence, soft gate).
   - Held-out official ADS, FILE decoupled: 0.87267 → 0.87731. Voice EER 0.1106 → 0.0875; FILE and music unchanged.
   - Coupled variant (FILE also sees the new Fv): 0.87605 (FILE 0.0996 → 0.1021). Decoupling was checked after seeing the coupled test result; val also prefers it (0.93044 vs 0.93003).
   - Voice gain under the oracle-gated rule: +0.0245, CI [0.013, 0.039].
   - Cost on discordant rows: real-voice+fake-music FPR 0.105 → 0.337 (n 86); fake-voice+real-music miss 0.084 → 0.238 (n 214).
   - The reverse direction (voice evidence into MUSIC) hurts: −0.0056, CI [−0.013, +0.003].

## 3. Caveats
- Every held-out number above is DB protocol (heads trained on the train split only), on the 74% backbone-cache subset. The shipped S3 heads were trained on held-out rows, so they have no clean estimate.
- Past held-out → LB ratios were 0.32–0.63 (lane2).
- The share of discordant voice/music files in the official test is UNVERIFIED. It decides the sign of experiment 1 on the LB.
- The AD restored-speech label convention is taken from S1's text ("speech with artifacts" includes restored speech) and was not quoted from a label table, so it is marked SUPPORTED WITH LIMITATIONS.
