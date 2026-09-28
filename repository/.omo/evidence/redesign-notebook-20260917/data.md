# CPU data preparation evidence

## Delivered API

- `redesign_data.prepare_data(manifest: Path, output: Path, *, seed=41, diagnostic=False, make_mixtures=True) -> dict`
- `redesign_data.read_rows(path: Path) -> list[dict[str, str]]`
- `redesign_data.write_source_template(path: Path) -> None`
- `redesign_data.SOURCE_FIELDS`
- CLI header template: `.venv/bin/python redesign_data.py template /path/to/sources.csv`
- CLI preparation: `.venv/bin/python redesign_data.py build --sources /path/to/sources.csv --output /path/to/prepared`

The template is header-only. It contains exactly the binding source columns and creates no example row, license, provenance, or historical-data claim.

## Verification invocation

```text
.venv/bin/python -m pytest -vv test_redesign_data.py --basetemp .omo/evidence/redesign-notebook-20260917/data-fixtures --junitxml .omo/evidence/redesign-notebook-20260917/data-pytest.xml
```

Observed result: `4 passed in 0.24s`. Machine-readable result: `data-pytest.xml`.

The real CLI surface was also invoked against the preserved WAV manifest:

```text
.venv/bin/python redesign_data.py build --sources .omo/evidence/redesign-notebook-20260917/data-fixtures/test_prepare_preserves_unknown0/sources.csv --output .omo/evidence/redesign-notebook-20260917/cli-prepared-final
.venv/bin/python redesign_data.py template .omo/evidence/redesign-notebook-20260917/source-template.csv
```

Observed CLI audit: 53 input rows, 51 accepted sources, 24 generated rows, 12 verified pairs, zero lineage overlap, two quarantined rows, `sealed_panel_identity=ad2a4be053526346a889cfcf97fcdf225b3622e11d75585c2b5ed31be9e5070c`, and `coverage_gate: unmet`. Captured products: `cli-prepared-final/{train,validation,sealed}.csv`, `cli-prepared-final/{protocol,audit}.json`, `cli-prepared-final/audio/*.wav`, and `source-template.csv`.

## Success criteria and artifacts

| Scenario | Binary observable | Captured artifact |
| --- | --- | --- |
| Parent, artist, speaker, exact-file, and decoded-PCM lineage is split before mixing; generated parent links cannot cross a split | linked sources share `group_id`; a byte-identical WAV under an unrelated ID and a differently encoded AIFF with identical decoded PCM share their source group; `lineage_overlap` is `{}`; injected cross-split parent raises `DataContractError` | `data-fixtures/test_prepare_preserves_unknown0/prepared-a/audit.json`; JUnit testcase `test_existing_output_and_cross_split_parent_fail_closed` |
| Real WAV validation rejects corruption and false component claims | `corrupt` is quarantined with decode failure; `false-claim` is quarantined because `VOICE_FAKE` was declared for known-absent voice | `data-fixtures/test_prepare_preserves_unknown0/prepared-a/audit.json`; source fixture `data-fixtures/test_prepare_preserves_unknown0/corrupt.wav` |
| Singing authenticity unknowns remain unknown, including isolated singing with music absent | `singing-unknown` retains empty `VOICE_FAKE` and `MUSIC_FAKE`, `VOICE_PRESENT=1`, `MUSIC_PRESENT=0` | `data-fixtures/test_prepare_preserves_unknown0/prepared-a/train.csv` |
| Pure/mixed alternatives have identical pair, split, group, duration, accompaniment parent/gain/event truth and replace rather than add exposure | 12 verified pairs; waveform check reconstructs `mixed - pure` from the 0.5-gain voice parent; each pair has one `pure` and one `mixed` row | `data-fixtures/test_prepare_preserves_unknown0/prepared-a/train.csv`; paired WAVs under `prepared-a/audio/` |
| All four real/fake component states and interval modes are represented when eligible disjoint parents exist | mixed states are `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)`; modes are simultaneous, sequential, sparse; sparse fake interval is 0.5 s | `data-fixtures/test_prepare_preserves_unknown0/prepared-a/protocol.json`; `prepared-a/train.csv` |
| Same seed and sources are deterministic; existing outputs fail closed | IDs, splits, pair IDs, audio SHA-256, and event intervals match across `prepared-a` and `prepared-b`; altered output WAV is rejected instead of overwritten | JUnit testcases `test_prepare_preserves_unknowns_and_builds_deterministic_paired_truth` and `test_existing_output_and_cross_split_parent_fail_closed` |
| Generated final paths load after atomic staging rename | every recorded path exists; generated WAV decodes as 4,800 samples at 8 kHz in the fixture | paired WAVs under `data-fixtures/test_prepare_preserves_unknown0/prepared-a/audio/` |
| Header-only source workflow does not invent provenance | template parses as zero rows and its header equals `SOURCE_FIELDS` | JUnit testcase `test_template_is_header_only`; `data-fixtures/test_template_is_header_only0/sources.csv` |
| Sealed registry identity ignores relocation and row serialization | `audit.json.sealed_panel_identity` is unchanged after copying the same source audio to a new root and reversing manifest row order; the digest contains only sorted sealed `ID`, `audio_sha256`, and atomic groups | JUnit testcase `test_sealed_panel_identity_survives_source_relocation_and_manifest_reordering`; prepared audits under its fixture directory |

Additional static checks:

```text
.venv/bin/python -m py_compile redesign_data.py redesign_data_validation.py redesign_mix_audio.py test_redesign_data.py
.venv/bin/python /Users/choiwuseck/.codex/plugins/cache/sisyphuslabs/omo/4.19.2/skills/programming/scripts/python/check-no-excuse-rules.py redesign_data.py redesign_data_validation.py redesign_mix_audio.py test_redesign_data.py
git diff --check -- redesign_data.py redesign_data_validation.py redesign_mix_audio.py test_redesign_data.py
```

Expected observables: successful compile, `no violations in 4 file(s)`, and empty `git diff --check` output.

## Limits

These fixtures are generated locally for integration verification and use test-only `example.invalid` provenance. They are not usable training sources. No external data was collected, no source license was inferred, no historical data was declared independent, and no GPU/network/UI path ran. The fixture audit reports `coverage_gate: unmet`; it proves data-contract behavior, not sample sufficiency or model performance. Exact bytes and exact decoded PCM are grouped, but this does not claim near-duplicate detection. Sample-rate-mismatched candidate pairs are skipped and counted in `paired_state_slots.skipped_sample_rate_mismatch`; they are never silently resampled.
