# Track C design (orchestrator-fixed, 2026-09-27 ~16:10 KST) — SSL fine-tuning bet for top-15

Goal: a step change on the VOICE route (and FILE) by fine-tuning AntiDeepfake XLS-R-1B on HTDemucs voice stems.
Prior evidence against: FullFT8 (MERT/EAT full fine-tuning, pre-data-v4) scored LB 0.638, -0.10 vs frozen; AST FT inverted
dev gains on LB. So: partial fine-tuning, strong regularization, DB-protocol held-out gate before any packaging.

Data: data v4 bank rows (results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/bank_trainval.csv, bank_test.csv),
voice-labelled rows per full_bank_train.head_labels semantics. Stems: /root/stems/<ID>.npz on VESSL with 'voice' and
'backing' float16 16 kHz arrays (runtime-identical preprocessing: librosa load 16k mono, peak cap 0.6, HTDemucs via
baseline separate_voice_and_music, pad/crop to len(raw)). No AI-Hub data anywhere (VESSL is abroad).
Protocol: train split only -> select on val -> held-out test reported once (DB protocol). No refit on held-out for the bet.

Model: antideepfake_runtime.AntiDeepfake (fairseq->HF Wav2Vec2 conversion, classifier Linear(1280,2), fake index 0),
weights _workspace/models/xls-r-1b-anti-deepfake (fp32; VESSL copy /root/models/xlsr1b). Freeze feature extractor +
feature projection + encoder layers 0..31; train layers 32..47, encoder layer norm and classifier. bf16 autocast,
AdamW (backbone lr 1e-5, classifier 1e-4, wd 0.01), cosine with 5% warmup, grad clip 1.0, DDP over 4 x H100 (torchrun).
Batch: 32 windows per GPU of 64600 samples (4.04 s) randomly cropped from each file's voice stem (files shorter than a
window: whole clip, zero-padded to the batch max is NOT allowed because AD pooling uses all frames -> pad by repeating
the clip like baseline extract_segment). Sampling: WeightedRandomSampler with full_bank_batches.loss_weights x
duration_balance weights; plain CE.
Augmentation (train only, on the fly, torch on GPU): random gain +-6 dB; additive Gaussian/pink noise SNR 10-40 dB (p .5);
random FIR low-pass (cutoff 3.4-8 kHz, p .3); telephone band 300-3400 Hz with 8 kHz resample (p .15); random
polarity. No codec simulation (CPU cost).
Eval: every 1/4 epoch on val voice rows: AntiDeepfake windows (hop 32300) -> window fake prob -> file max (the runtime
member definition); val voice-route EER; keep best; max 4 epochs; early stop patience 3 evals.
Deliverables: best checkpoint (fp16 safetensors in the AntiDeepfake key layout so antideepfake_runtime loads it
unchanged), per-row member CSV for val and held-out test (ID, ft_max, ft_top30, ft_mean), training log.
Integration: member 'ft' replaces the frozen AntiDeepfake embedding heads in the voice CDF fusion (package size
unchanged); fusion weight chosen on val; DB-protocol held-out official ADS gate: >= +0.01 over S4-equivalent with group
bootstrap P(>0) >= 0.95, else do not ship.
