# Feature role pruning audit (read-only, 2026-09-17)

Scope: sealed `manifest.csv` (`fc21a54a...`), 7,101 rows. No source or protocol files were changed.

## Blocking consumer mismatch

The manifest contains 4,617 `kind=mixed` rows and 2,484 `kind=speech` rows, with zero `kind=music` rows. `eight_ablation.py` and the E2 post-run normalizer select music rows with `kind == "music"`; their music arms are therefore empty before extraction. The intended core-music predicate is the 4,457 `probe_family=core`, `kind=mixed`, `MUSIC_PRESENT=1` rows, but correcting that predicate and the corresponding `MUSIC_FAKE` label selection is required before pruning.

Also, the current E2 selector includes every one of the 7,101 rows: all partial/swap rows, all core speech rows, and all core mixed rows through `kind in {speech, voice, mixed}`. Therefore a pruning plan that gives ordinary core rows no DF/Anti probabilities changes the current E2 population. This must be an explicit E2 scope correction, not described as cache-only optimization.

## Intended row-role matrix requested by the protocol owner

All records retain `starts`, `valid_samples`, `padding_samples`, `raw_mask`, and exact identity. `stem_mask` exists only when separation/accompaniment is actually produced. No placeholder arrays are permitted.

| Role | Rows | Windows | Required learned/forensic arrays |
|---|---:|---:|---|
| core music, base | 1,322 | 3,966 | `mert_l6`, `mert_l12`, `mert_l24`, `spectrum_stem`, `stem_mask` |
| core music, processed | 3,135 | 9,405 | `mert_l24`, `stem_mask` |
| core speech, base | 512 | 1,424 | `anti_l12`, `anti_l24`, `anti_l48` on raw |
| core speech, processed | 1,324 | 3,672 | `anti_l24` on raw |
| E2 vocal parent reference (overlap with core-speech base) | 20 | 55 | additionally `df_raw`, `df_vocals`, `anti_raw`, `anti_vocals`; recovered-vocal view |
| E2 swap | 160 | 440 | `df_raw`, `df_vocals`, `anti_raw`, `anti_vocals`, `mert_l24`, `anti_l24`, `presence`, `stem_mask`; recovered-vocal and accompaniment views |
| partial E2/E5 | 648 | 10,368 | `df_raw`, `df_vocals`, `anti_raw`, `anti_vocals`; recovered-vocal view |

`spectrum_raw` has no consumer. Partial rows have no MERT, hidden Anti layer, spectrum, or PANNs consumer. PANNs `presence` is consumed only for swaps by `eight_e2_music_file_analysis.py`. E5 itself reads raw probabilities only, but vocals probabilities remain required by E2's paired view.

## Exact compute accounting for that intended matrix

Counts use the production 64,600/32,300 deduplicated-tail schedule and batch size 8.

| Work | Current all-row extraction | Role-mask extraction | Reduction |
|---|---:|---:|---:|
| separator records | 7,101 / 71,825.124 audio s | 5,285 / 58,980.239 audio s | 25.57% calls / 17.88% audio s |
| DF serial forwards (two views) | 58,550 | 21,726 | 62.89% |
| Anti window-items | 58,550 | 26,767 | 54.28% |
| Anti batches | 15,930 | 5,200 | 67.36% |
| MERT window-items | 29,275 | 13,811 | 52.82% |
| MERT batches | 7,965 | 4,617 | 42.03% |
| PANNs window-items | 29,275 | 440 | 98.50% |
| PANNs record calls | 7,101 | 160 | 97.75% |
| spectral window transforms | 58,550 (raw+stem) | 3,966 (base music stem only) | 93.23% |

Float payload lower bound falls from about 872.63 MiB to 132.82 MiB (84.78%) before NPZ compression and metadata. This is not a wall-time prediction: separator audio decreases only 17.88%, while model rates were not separately measured on the admitted H100.

## Smallest coherent implementation surface

1. `eight_ablation.py`, `eight_fixed_io.py`, and `eight_e2_music_file_analysis.py`: repair the absent `kind=music` predicate and select `MUSIC_FAKE` for core music.
2. `eight_probes.py`: explicitly settle E2 scope. Keeping its current selector requires detector arrays on all 7,101 rows and prevents the proposed detector pruning. The intended matrix requires E2 to use swaps, partials, and the 20 referenced core-vocal parents.
3. `eight_feature_schema.py`: derive an exact required-key set from a global manifest role map (including the transitive set of `swap.parent_ids[0]`), validate required shapes/domains, and reject both missing keys and undeclared placeholder keys.
4. `eight_feature_identity.py`: bind the sorted required-key/profile set and exact views/layers to each cache identity. A cache produced under one role must be stale under another role.
5. `eight_features.py`: compute the global role map once, conditionally separate only roles requiring recovered vocals/accompaniment, call only required encoders/views, and cache-validate against that row's exact key set. Preflight's selected speech record is one of the 20 swap parents, so its detector/reference profile remains exercised.

The consumers already request named arrays and do not require uniform NPZ schemas; `eight_probes._load` copies available keys. The uniform-schema assumptions are confined to feature extraction/cache validation and preflight.
