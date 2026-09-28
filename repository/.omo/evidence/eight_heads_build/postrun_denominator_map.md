# Post-run denominator and component reconstruction map

Date: 2026-09-17. Scope: report-only correction after the frozen H100 run. Do not change or rerun model fitting or scoring.

Verification scenario: apply the exact branch predicates below to the complete 7,101-row manifest, build global provenance components with `connected_components`, and tabulate each role. Invocation: `.venv/bin/python` using `read_manifest`, `music_model_rows`, `model_rows`, and `connected_components`. Binary observable: every listed evaluated slice has both labels; the row, label, codec, and global-component counts equal the table below. This file is the captured artifact.

## Why the top-level fields need correction

`eight_ablation.run` builds `known = model_rows(rows)` over the whole manifest and writes that aggregate into every report's top-level `label_denominators` and `component_count` (`eight_ablation.py:330-336`). That population mixes speech `VOICE_FAKE` with mixed-row `FILE_FAKE` and is not any E1/E3/E4/E6/E8 branch population. It does not affect fitting, selection, predictions, nested metrics, or bootstrap calculations.

For every reconstruction, resolve provenance components once over the complete manifest with `eight_fixed_io.connected_components(all_manifest_rows)`, then subset by selected IDs. Do not count raw `group_id` strings and do not recompute connected components inside a subset. Prediction CSVs already contain the resulting global `component_id` (`eight_fixed_io.py:137-147`).

Labels are branch-specific:

- Music: `probe_family == 'core'`, `MUSIC_PRESENT == '1'`, `MUSIC_FAKE in {'0','1'}`; label is `int(MUSIC_FAKE)`.
- Voice: `kind == 'speech'` and `VOICE_FAKE in {'0','1'}`; label is `int(VOICE_FAKE)`.

The source manifest audited here is `_workspace/eight_h100_20260917/dataset/manifest.csv`, SHA-256 `fc21a54a7d9aeec69575ad1b6a9c0d0c35910f13ba317fde434587686309b537`.

## Saved prediction mapping

| Experiment / branch | Saved prediction rows | Correct reconstruction |
|---|---|---|
| E1 music | `output/predictions_E1_music.csv` | Test-only, core music, `variant/base`; count CSV `label`, and unique `component_id`. Dev denominators/components come from the same manifest filter with `split/dev`. |
| E1 voice | `output/predictions_E1_voice.csv` | Test-only, speech, `variant/base`; count CSV `label` and unique `component_id`. Dev uses the corresponding manifest filter. |
| E1 generator holdouts | No per-file holdout CSV | Use the branch population above. Training membership is `split/train` and `generator != held_generator`; evaluation membership is `split/dev` and (`generator == held_generator` or label is REAL). Nested `control`/`balanced` metrics already expose `negative_n`, `positive_n`, and `components`. Scores cannot be reconstructed from saved predictions. |
| E3 music | `output/predictions_E3_music.csv` | CSV contains every test codec. Primary unseen-chain denominator is `codec == 'opus_telephone'`; per-codec denominators group by `codec`. Dev primary uses the manifest filter with `split/dev` and `variant/opus_telephone`. Training includes base, telephone, mp3 only. Calibration is unused. |
| E3 voice | `output/predictions_E3_voice.csv` | Same as E3 music with speech/`VOICE_FAKE`. |
| E4 | `output/predictions_E4.csv` | Test-only core music `variant/base`; the same rows apply to MERT, spectrum, and fusion columns. Dev denominators/components use the matching manifest filter. |
| E6 MERT | `output/predictions_E6_MERT.csv` | Test-only core music `variant/base`; the same denominator applies to all three MERT layers. Dev layer selection uses the matching manifest filter. |
| E6 Anti | `output/predictions_E6_Anti.csv` | Test-only speech `variant/base`; the same denominator applies to all three Anti layers. Dev layer selection uses the matching manifest filter. |
| E8 | `output/predictions_E8.csv` | CSV contains both dev and test core-music base rows; group separately by `split`, then count `label` and unique `component_id`. Fit denominator is train REAL only. Threshold denominator is calibration REAL only. |

## Expected manifest-derived counts

These counts are an admission receipt for interpreting the downloaded archive; recheck the downloaded manifest hash before applying them.

| Population | Train REAL/FAKE, components | Calibration REAL/FAKE, components | Dev REAL/FAKE, components | Test REAL/FAKE, components |
|---|---:|---:|---:|---:|
| Core music base (E1/E4/E6/E8) | 139/692, 175 | 31/169, 30 | 25/139, 27 | 23/104, 24 |
| Speech base (E1/E6) | 106/106, 1 | 102/102, 1 | 24/24, 1 | 24/24, 1 |
| Core music E3 all scheduled codecs | 417/2076, 175 | unused by E3; 124/676, 30 | 100/556, 27 | 92/416, 24 |
| Speech E3 all scheduled codecs | 318/318, 1 | unused by E3; 408/408, 1 | 96/96, 1 | 96/96, 1 |
| E3 music primary Opus chain | not trained | unused | 25/139, 27 | 23/104, 24 |
| E3 voice primary Opus chain | not trained | unused | 24/24, 1 | 24/24, 1 |

For E8, report role-specific counts instead of one experiment-wide pair: fit `139 REAL / 127 components`, threshold `31 REAL / 19 components`, dev `25 REAL / 139 FAKE / 27 components`, and test `23 REAL / 104 FAKE / 24 components`. The fit and threshold component counts come from their REAL-only ID subsets, not the full split populations.

## Other report-only interpretation ambiguities

1. The top-level `population` and `comparison_scope` strings are also generic. Final reporting should name branch, split, variant set, and truth column explicitly.
2. Nested `evaluate` outputs already contain correct `negative_n`, `positive_n`, `rows`, and `components` (`eight_fixed_io.py:123-134`). Prefer those for a scored slice; use CSV/manifest reconstruction as an integrity check.
3. E3's `ce_dev_eer` and `consistent_dev_eer` seed summaries are computed only on the unseen Opus+telephone dev slice (`eight_ablation.py:193-210`), despite the shorter field names. Test CSVs contain all codecs, while `test_unseen_chain` is Opus-only.
4. E4's report key named `dev` contains each seed's dev results plus a later nested `test` result (`eight_ablation.py:238-262`). Read the nested level rather than treating the whole object as dev-only.
5. E6 stores dev layer summaries and selected layer, but only test per-seed metrics. Dev denominators therefore come from the manifest filter, not a prediction CSV.
6. E1 holdout sensitivity has no per-file prediction artifact. Its nested denominators/components are auditable from the manifest membership rule, but its scores cannot be independently replayed without checkpoints/features.
7. Speech has one connected component per split in this manifest. Any speech slice with fewer than five components has no population CI under the declared rule; three seeds are optimization repeats and must not be treated as independent provenance units.
8. Prediction CSVs omit `parent_id`; source/generator/codec are present. Reconstruct parent-level strata only by joining CSV `ID` back to the manifest.
9. Calibration rows are not part of E1/E3/E4/E6 fitting or scoring. E8 alone uses calibration, and only REAL calibration rows set its fixed FPR threshold.
