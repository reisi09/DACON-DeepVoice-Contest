# Actual manifest CPU admission audit

Verdict: **PASS for the source and metadata admission gate; native H100 execution remains pending.** This audit used the sealed local manifest and current source selectors. It did not read feature NPZ payloads, run feature extraction, start a remote workspace, or claim model performance.

## Inputs and exact invocations

- Manifest: `_workspace/eight_h100_20260917/dataset/manifest.csv`, SHA-256 `fc21a54a7d9aeec69575ad1b6a9c0d0c35910f13ba317fde434587686309b537`.
- Sources: `_workspace/eight_h100_20260917/dataset/sources.csv`, SHA-256 `ab7de8e9d077d64008381b7cd78831db5b5ace5b7acf219636f1269ee965fdb9`.
- Receipt: `_workspace/eight_h100_20260917/dataset/data_receipt.json`, SHA-256 `bd49bc583abac831b6b751c42e8d890e9e922ae16a346612414c609e155c6743`.
- Counts and checks were run with `.venv/bin/python` using `eight_contract.read_manifest`, the current `eight_ablation.music_model_rows`, `eight_heads.balanced_draws`, and `eight_feature_schema.feature_profiles`; the complete machine-readable output is `actual_manifest_admission_counts.json`.
- Real dataset contract check: `.venv/bin/python` calling `eight_contract.validate_manifest(Path("_workspace/eight_h100_20260917/dataset"))` returned 7,101 rows, 7,101 audio hashes, 876 provenance atoms, and core/swap/partial counts 6,293/160/648. Transcript: `actual_manifest_validate.log`.
- Source syntax/import admission: `.venv/bin/python -m py_compile eight_ablation.py eight_feature_schema.py eight_features.py eight_probes.py eight_e2_music_file_analysis.py` (PASS).
- Focused regression admission after the selector/profile fixes: `.venv/bin/python -m pytest -q test_eight_features.py test_eight_fixed.py test_eight_lora.py test_eight_integration.py` (21 passed; fixture integration only, no performance evidence).
- Source fingerprints at this audit: `eight_ablation.py` `5d6c8fc7b214bcae3701a0ead2798fc97dfc1e1fea34023d37a2a41d10d12246`; `eight_feature_schema.py` `cdc62ffb8da7773deee4d030c68ca7fda1cd2cb99ce2294a866e88ba79d5e9ff`; `eight_features.py` `b0304f4f9aa2d13524d56622f494c4b1fab160b5009e09a51fb77987ed23acac`; `eight_probes.py` `1661764b7214f5b7cdc2724656aaf8fe865bda11ffec920ec86067fbceb10f52`.

## Population and selector checks

The manifest contains 7,101 unique IDs and 7,101 unique audio hashes. Its `kind` values are `mixed=4,617` and `speech=2,484`; literal `kind=music` is absent. The current music selector therefore uses `probe_family=core`, `MUSIC_PRESENT=1`, and known `MUSIC_FAKE`, rather than the absent `kind=music` value. It selects 1,322 base rows and has both labels in every train/calibration/dev/test split: train 139/692, calibration 31/169, dev 25/139, test 23/104 (REAL/FAKE). The voice selector remains `kind=speech`, `probe_family=core`, known `VOICE_FAKE`; it selects 512 base rows with train 106/106, calibration 102/102, dev 24/24, and test 24/24.

The E1 source-generator-parent sampler was exercised with seed 41, batch size 2, and two steps for both populations. Both produced one row of each class in every checked batch. Music has 21 source strata and 12 non-empty generated/REAL generator strata; voice has 21 source strata and 14 non-empty generator/REAL strata. Generator-held-out sensitivity is feasible for all 12/12 music generators and 4/13 voice generators. The other voice folds are recorded as data-insufficient because their held-out dev slice contains only REAL; this is an explicit sensitivity limitation, not a skipped main arm.

E3 retains matched views for both classes. Music has 1,322 complete base/telephone/MP3 parent groups and 4,457 selected rows after adding evaluation-only Opus; voice has 512 complete groups and 1,836 selected rows. The dev Opus slice has both labels for music (25/139) and voice (24/24). Training excludes Opus as declared. E4 and E6 use the same non-empty base populations above. E8 has 139 REAL training rows, 31 REAL calibration rows, and both labels in dev (25/139) and test (23/104); its REAL-only fit and calibration threshold therefore have an admissible metadata population.

E7 selects 1,236 core speech rows: train base/telephone/MP3 = 636 rows (318/318), calibration base/Opus = 408 (204/204), dev = 96 (48/48), and test = 96 (48/48). The speaker allocation is 3/3/2/2 across train/calibration/dev/test. Its `balanced_schedule` has both trusted classes available. No GPU topology or native checkpoint claim is made here.

## E2 and E5 joins and controls

E2 selects 2,644 rows: core speech 1,836, swap 160, and partial 648. The constructed swap/partial subset has 808 rows; every declared parent ID exists and is in the same split. The 1,324 processed core-speech rows retain source parent IDs that are not manifest record IDs by design, so they are not treated as missing constructed-control joins. E2 has 12 swap rows per background in calibration (FMA-a, FMA-b, SONICS-fake, and noise), with FMA-a/FMA-b REAL controls available for both `MUSIC_FAKE` and `FILE_FAKE` threshold calibration. The independent byte-level E2 gain audit is recorded in `independent_data_recompute_final2.log`: raw-parent subtraction gives 80/80 groups with ratio 0.5011869669–0.5011870861 against the expected 0.501187.

E5 selects 648 partial rows, balanced `FILE_FAKE` 324/324. There are 324 `pair_id` cells, each exactly one REAL and one FAKE, spanning 12/30/60 seconds, 0.25/0.5/1/2 seconds, and start/middle/end positions. Calibration, dev, and test are all non-empty and contain both labels; the constructed-composite limitation remains in force.

## Provenance and feature role admission

An independent union-find over every `group_id` atom and explicit `parent_ids` produces 256 global connected components across 876 group atoms. No group atom or connected component crosses train, calibration, dev, or test.

Current role-specific feature profiles map all 7,101 rows exactly once: `core_music_base=1,322`, `core_music_processed=3,135`, `core_speech_base=512`, `core_speech_processed=1,324`, `swap=160`, and `partial=648`. The declared consumer keys are present in the profile contract: MERT L6/L12/L24 for music layer selection, Anti L12/L24/L48 for voice layer selection, MERT L24 plus the first 384 spectrum bins for E4, MERT L24 for E8, Anti probabilities for E2/E5, and swap MERT L24/presence for the post-run E2 music/file analysis. Exact per-profile key sets and widths are in the JSON. `actual_npz_inspected=false` is intentional because feature extraction is a later GPU-stage operation.

## Findings and remaining gate

The previously empty music arm is resolved in the current source by the manifest-driven selector and is populated with two labels in every required split. No other actual-manifest population, sampler, source holdout, paired-control, provenance, or feature-role blocker was found. Voice generator-held-out sensitivity remains partially infeasible for 9/13 named generators and must be reported as such if emitted.

One report-contract caveat was found for the owner to resolve separately: `eight_ablation.run()` still computes its generic report `label_denominators` through `eight_fixed_io.model_rows(rows)`. Because the sealed manifest has no literal `kind=music`, `label_for()` treats mixed rows as `FILE_FAKE`; those generic denominators combine mixed FILE labels with speech VOICE labels and do not describe the active experiment branch. The branch-specific fit and metric populations above are correct, but this report-level aggregate should be scoped to the experiment population before relying on those fields.

This is a source/data admission result. Native H100 feature extraction, checkpoint loading, runtime duration, and final experiment execution remain pending and require their separate GPU evidence.
