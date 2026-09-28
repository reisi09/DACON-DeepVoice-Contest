# Exact feature-key dependency audit

Audit date: 2026-09-17. This traces current consumers; no extraction or GPU run was performed.

- Universal record metadata used by consumers: `identity_json`; pooled learned features also require `valid_samples` (`eight_fixed_io.py:87-95`, `eight_e2_music_file_analysis.py:37-49`).
- Core music base: train/dev/test require `mert_l6`, `mert_l12`, `mert_l24`, and `spectrum_stem`; E4 consumes only `spectrum_stem[:, :384]`. Calibration requires `mert_l24` for E8. No current experiment consumes `spectrum_raw`.
- Core music processed views: E3 requires `mert_l24` for base/telephone/mp3 train and base/telephone/mp3/opus dev/test.
- Core speech base: E1/E3/E6 require `anti_l12`, `anti_l24`, `anti_l48` according to branch/layer; the conservative E2 scope additionally requires `df_raw`, `df_vocals`, `anti_raw`, `anti_vocals` for all core speech views.
- Swap rows: original E2 requires `df_raw`, `df_vocals`, `anti_raw`, `anti_vocals`. The preregistered postrun Music/File analysis additionally requires `mert_l24`, `anti_l24`, `presence`, and `valid_samples` on calibration/dev/test swaps.
- Partial rows: E2/E5 union requires `df_raw`, `df_vocals`, `anti_raw`, `anti_vocals`.
- `starts`, `padding_samples`, `raw_mask`, `stem_mask`, and `spectrum_raw` have no current downstream experiment consumer. They may remain cheap structural data, but must not force unused model inference.

Current compatibility blockers are `eight_feature_schema.py:11-46` (global all-key requirement), eager loading and unconditional inference in `eight_features.py:67-169`, cache validation at `eight_features.py:218-223`, and preflight at `eight_features.py:248-273`. Sparse records require a row-role required-key plan bound into `identity_json`, row-specific validation, lazy model loading, and preflight representatives covering each role. Do not use dummy arrays. `spectrum_stem` must be validated as `(N,448)`; current schema checks only `spectrum_raw`, while E4 would otherwise silently slice malformed stem features. MERT for E4 remains accompaniment/stem input. The absence of a `spectrum_raw` consumer means omitting it preserves the implemented primary E4 analysis but cannot support an unimplemented future raw-spectrum secondary plot.
