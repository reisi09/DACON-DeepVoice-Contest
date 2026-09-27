# Lane 4 notes: model, mathematics, execution (S3 = DataV4_S3_Backbones, sha 062fc236…)
Observed 2026-09-27 KST. Read-only. CPU checks only (each < 25 s, < 0.7 GB RSS). Scripts and raw outputs are in this folder:
pkg_check.py -> pkg_check.out.json | cache_check.py -> cache_check.out.json | semantics_check.py -> semantics_check.out.json |
weights_check.py -> weights_check.out.json. Also ran the existing _workspace/audit_20260927/s3_runtime_parity.py (unchanged), which printed WORST voice 1.2e-4, music 5.3e-4 PASS.

## Implemented objective (full_bank_train.py step/forward, lines 77-117; full_bank_batches.py; train_backbone_heads.train, lines 70-106)
- Voice (DF-Arena "ours" heads and AntiDeepfake heads): e_{i,t} in R^1280 per window t. DF-Arena: fc5 input = conformer CLS token,
  one 64,600-sample segment per step, no overlap, exact tail; short audio is tiled (script.py get_segment_starts/extract_segment).
  AD: mean over frames of the last hidden state; 64,600 window, hop 32,300, exact tail, short audio kept as one shorter clip
  (antideepfake_runtime.windows). Head l = W e + b, W in R^{2x1280}. a_{i,t} = l_{t,f} - l_{t,1-f}, f = 0 (fake index) for both.
  z_i = max_t a_{i,t}. p_i = sigmoid(z_i) = max_t softmax(l_t)_f. Only the argmax window gets gradient.
- Music: x_{i,t} in R^D, D = 2048 = [mean_tau h6 ; mean_tau h12] (S3) / 1024 = h12 (P1), MERT windows 64,600 @16 kHz resampled to 24 kHz,
  hop 32,300, exact tail, right zero-pad. x~ = (x - mu)/s (mu, s = mean/std over train files of per-file window means; buffers).
  s_{i,t} = v^T GELU(U x~ + c) + d, U in R^{128xD}. z_i = logsumexp_t s_{i,t} - log T_i (log-mean-exp, T-invariant to duplication).
- Loss per 256-row batch B (one AdamW step): L_B = (1/|B|) sum_i w_i BCE(z_i, y_i) + 0.25 (1/max(1,|K_B|)) sum_{(i,t) in K_B} BCE(s_{i,t}, q_{i,t})
  (second term music only; q = known fake-occupancy fraction of the window; K_B = windows with known truth).
- Weights: w_i = N / (2 C_{y_i} n_cell(i)), cell = (label, source, generator), C_y = cells with label y; sum w = N. Duration balance:
  10 pooled quantile bins of log(samples); per class m_i = clip(P_pool(b)/P_class(b), 0.1, 10) (both weighted by w), renormalised to keep
  class total; w_i <- w_i m_i.
- AdamW lr 1e-3 music / 1e-4 voice, wd 0.01 (all params), grad clip global norm 1.0, epoch = full permutation (seed+100003*epoch),
  max 50 epochs, patience 5 on lexicographic (val EER, val BCE), silent voice val predictions = 0, best state kept, seeds 41/42/43,
  seed-mean of probabilities. Voice init = DF-Arena fc5 / AD proj_fc; music init = default. New heads trained on CPU fp32.
- Runtime (full_bank_submission.Engine.predict lines 88-138): Fm = mean_s sigmoid(z^(s)); Fv_ours = mean_s max_t softmax_0;
  silent (RMS(voice stem) < 1e-5) -> Fv = 0, no fusion; else Fad = mean_s max_t softmax_0(AD head); c = (CDF_ours(Fv_ours) + 0.5 CDF_ad(Fad))/1.5,
  Fv = CDF_ours^{-1}(c) (1001-point val quantile tables, rows with ours > 0); after 48 min: Fv = Fv_ours. FILE = sum_k w_k CDF_k(g_k)/sum w_k,
  w = {max 0, noisy_or 9, max_nopresence 1}.

## Architecture (config.json in the zip)
- XLS-R 1B (df_arena_1b/facebook/wav2vec2-xls-r-1b/config.json and antideepfake/ad/antideepfake_config.json): hidden 1280, 48 layers,
  16 heads (head dim 80), FFN 5120, conv encoder 7x512, strides 5,2,2,2,2,2,2 (320x -> 50 frames/s), stable layer norm.
- DF-Arena extra (conformer.py FinalConformer, backbone.py): attention-weighted sum of all 49 hidden states, BN+SELU, 4 conformer blocks,
  emb 1280, 4 heads (dim_head 320), kernel 31, CLS -> fc5 Linear(1280,2). label2id spoof 0.
- MERT-v1-330M (mert/config.json): hidden 1024, 24 layers, 16 heads (head dim 64), FFN 4096, same conv strides at 24 kHz (75 frames/s);
  preprocessor do_normalize False.
- Parameter counts (mmap / safetensors header): DF-Arena 1,147,865,740 (4.276 GiB fp32); AD 965,517,314 stored fp16, 962.5 M resident
  (3.586 GiB fp32); MERT 315,428,992 (1.175 GiB). Trainable per seed: voice 2,562; AD 2,562; music S3 262,401; P1 music 131,329.

## Key measured numbers
- Package: 61/61 manifest hashes match; trained members == _workspace/final_backbone_20260927 == candidate_lock; voice_* == p1_refit_cpu;
  encoder assets == cache_v4 identity; AD asset = fp16 copy 352649fd… (local fp32 original 57da6a47…).
- Parity (400 random rows): samples/windows/voice_silent 400/400; L12 slot vs cache_v4 max abs 0.06236 == max fp16 rounding 0.06236,
  rel L2 <= 2.6e-4; L6 slot vs cache_v4 min max-abs 17.3.
- Val reproduction (3,680 rows): file 0.054381, voice 0.036837, music 0.111939, AUCs 0.998896/0.996272 == validation_metrics.json.
  Variants: no-AD fallback file 0.063892 voice 0.048242 (ADS -0.0070); 30% partial fallback ADS -0.0025; equal FILE weights file 0.058523;
  fp16 dequant noise: metrics unchanged, max |dFm| 7.8e-4, |dFv| 2.3e-4, |dFILE| 5.7e-4. 0 silent-voice val rows.
- AD index 0 zero-shot: AUC 0.947, EER 0.124 (5,067 val+test voice rows).
- Coverage 74.3/74.6/74.8% (train/val/test); all route/label and duration bins 72-77%; no source (>=100 rows) outside 60-90%.
- Window truth: 0 length mismatches; 449/10,416 music rows have any known window label.
- Weights (final S3 training rows): voice N 22,101, 384 cells (20 real), max w 105.9, Kish ESS 4,732; music N 9,130, 52 cells (11 real),
  max w 31.0, ESS 1,777.
- Duration: Spearman(score, samples | real voice, val) ours 0.204, AD 0.131, fused 0.206; fake voice ~0; real music LME 0.115.
  val duration p50 7.5 s, p90 16.2 s, 3.6% > 30 s.
- Head drift from init: rel ||dW||/||W0|| 1.39-2.16, logit-direction cosine with init 0.20-0.35, logit norm x2.1-3.2.
- Epochs: P1 voice 43 improved at epoch 50 (cap); AD heads ran 46/29/37 epochs (best 41/24/32); music 16/25/22 (best 11/20/17).

## L4 estimate (not measured)
Resident fp32 weights ~9.4 GiB (+PANNs ~0.3, HTDemucs ~0.08 by file size) + AD conv0 activations (16x512x12,919x4 B = 0.39 GiB per tensor,
~3-4 live) + context -> ~11.5-12.5 GiB of 22.4 GiB. AD compute ~104 GFLOP per audio-second x2 overlap; L4 FP32 30.3 TFLOPS
(TF32 dense 60, https://www.nvidia.com/en-us/data-center/l4/, accessed 2026-09-27) with TF32 disabled -> ~6-8 min if mean duration 30 s,
~13-17 min if all 60 s. With the brief's ~17.5 min previous runtime (UNVERIFIED provenance) -> ~24-35 min, under the 48-min switch.
