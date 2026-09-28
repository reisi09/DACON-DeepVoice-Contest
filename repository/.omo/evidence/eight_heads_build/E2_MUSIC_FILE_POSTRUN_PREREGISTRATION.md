# E2 Music/File post-run analysis preregistration

Status: specified before reading any E2 Music/File GPU result.

The analysis uses only the downloaded manifest, cached NPZ features, and the twelve main E1 checkpoints. It performs no training, GPU work, arm selection, or source mutation.

- Report both `parent` and `balanced` E1 arms. Average seeds 41, 42, and 43 within each arm; do not select an arm or seed.
- Reconstruct the music `mert_l24` and voice `anti_l24` normalizers from labeled `train/base` rows only, using NumPy population standard deviation and replacing scales below `1e-6` with `1`. Save the ordered training IDs and normalizer artifact hash.
- Require checkpoint metadata: 500 updates, batch size 32, the expected seed and sampler mode, MLP architecture, zero consistency weight, 8,000 REAL and 8,000 FAKE draws, and a 64-character hexadecimal draw hash.
- Score Music and Voice with the corresponding E1 ensemble. Use maximum valid-window PANNs presence. Compute `File = 1 - (1 - voice_presence * Voice) * (1 - music_presence * Music)`.
- Primary continuous estimand: SONICS-fake backing score minus the mean score of the two FMA-real backings for the same vocal, split, and gain level. Report Music, gated music evidence, Voice, gated voice evidence, and File deltas. The FMA-a versus FMA-b variation remains the backing-identity control.
- Fit high-is-fake Music and File thresholds only from calibration-split FMA-real swap controls at the predeclared target FPR. Atomically save the lock before reading dev/test feature files.
- Music truth contrasts require `MUSIC_FAKE` to be known. File truth-changing contrasts require FMA `FILE_FAKE=0` and SONICS `FILE_FAKE=1`. Pairs already labeled File-fake before the backing change are reported separately as saturation diagnostics.
- Noise rows retain unknown Music truth and never enter truth contrasts or threshold fitting.
- Use global transitive provenance components. A population CI is omitted when fewer than five components are available. Any CI is conditional on the already fitted models.

Claim boundary: this is a constructed, source-confounded response diagnostic. Separator behavior and presence-gate changes remain part of the measured pipeline. It is not an official score or evidence of population-wide generalization.
