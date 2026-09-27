Round 4 (planning only, short). Two disagreements with your round-3 plan; give a final verdict on each (<= 250 words total, Korean).

D1. Component entanglement in G3/G11. The pipeline separates stems precisely so that MUSIC_FAKE is judged on the
accompaniment and VOICE_FAKE on the vocal (labels are per component; mixtures exist: held-out fakeV+realM 286 rows,
realV+fakeM 107 rows). Feeding the RAW mixture to the music head (G3) lets fake vocals raise MUSIC scores on fakeV+realM
files, and feeding it to the WavLM voice member (G11) lets AI accompaniment raise VOICE scores on realV+fakeM (AI-cover)
files. Claude proposes: keep G3/G11 only with mandatory mixture-slice guards (MUSIC EER on fakeV+realM and VOICE EER on
realV+fakeM must not worsen, per seed) in addition to your gates, and ranks G11 below G3 because the voice member is
fused into VOICE for every file. Agree / disagree?

D2. Reduced G2' = ONE codec encoder only (transformers-native EncodecModel 24 kHz ~15M params or DacModel 44 kHz ~75M;
fp16/fp32 < 0.3 GB, fits the 0.27-0.5 GB zip headroom only if S7 is not packaged, or replaces SSLAM if it wins), encoder
latents per runtime window on the backing stem, same locked screen. Hypothesis: several music generators decode through
neural codecs, so codec-latent statistics may expose the unseen lyria-3 pro / fx_encoder families that SSL encoders miss.
Extractor code ~30 min (AutoModel path like WavLM), x4 ~20 min from stored stems. Keep or kill?
