# Early speech research

Observer: /root/speech_early. 27 distinct queries; 12 primary papers and 5 pinned author repositories. Dates/methods/results read. No models downloaded or performance reproduced.

Source inventory IDs A01–A12 retain the primary sources. Most useful additions are matched-content contrastive training, separate corruption-family controls, frequency-band diagnostics, and equal-budget acquisition of difficult real sources. These are hypotheses for this pipeline, not proven local gains.

Counterevidence: vocoder-paired contrastive improves some corpora but regresses on ASV2021 LA; LA-optimized RawBoost underperforms DF-specific policy on DF; SAMO headline uses target-only/enrollment protocol; VIB benefits are not uniform; large vocoded SSL regresses on some codec-heavy sets. Do not import vocoder-positive labels without reconciling the competition's component-generation definition.

Pinned implementation observations:

- https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/data_utils.py#L39 — fixed64,600 crop/repeat; caller must handle sample rate.
- https://github.com/TakHemlata/RawBoost-antispoofing/blob/4f161a8b4d0d9f4a8431509ddd86c645869ef6c4/RawBoost.py#L54 — separate distortion families.
- https://github.com/TakHemlata/SSL_Anti-spoofing/blob/4acaa61dcef5f7610f43aa4d0b29c4559b970cd2/data_utils_SSL.py#L111 — actual algorithm switch.
- https://github.com/sivannavis/samo/blob/aef2bff72a9bd8da67d71fbe8cba361faebb66ae/samo/main.py#L82 — default enrollment/target filters.
- https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/b529705b65588d62db17b56f98636ee734ff725e/project/09-asvspoof-vocoded-trn/model-ID-7/model.py#L447 — actual contrastive objective; same file also includes costly Monte Carlo uncertainty inference defaults, not required for plain detection.

## EXPAND

- W2V-ASDG split audit: reconcile source partitions, do not allege leakage without evidence. Assigned J and wave2 A.
- VIB and Elastic-Margin author code: targeted wave2 A.
- Layer-conditioned SSL/supervised fusion2023: wave2 A/B overlap check.
- CLAD2024: duplicate C scope; date clarified.
- OC-Softmax2021-04-28: closed as outside requested window, background only.
- Generic/multimodal model leads: closed as lower-specificity or outside audio-only scope; no superiority claim retained.
