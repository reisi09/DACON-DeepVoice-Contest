# V4 data final review

Date: 2026-09-13

## Verdict

- `codeQualityStatus`: `WATCH`
- `recommendation`: `APPROVE`
- `blockers`: none for the completed local corpus and data-readiness scope.

The prior speaker-identity blocker is resolved in the reviewed corpus. This
does not claim that Drive upload/IDs or any Colab/GPU execution is complete.

## Evidence verified

### Speaker identity

`v4_data_acquisition.py:62-83` reconstructs every prior fake-speech row from
the pinned MLAAD `meta.csv` `original_file` field, derives the speaker from
the source path at `:55-59`, and fails if any prior row is missing or
ambiguous. `prepare_v4_data.py:410-412` passes that set into new speech
selection. `speech_candidates` excludes an exact derived speaker identifier
at `v4_data_acquisition.py:183-198`; a live cached-metadata check observed
302 candidates before exclusion and removed all 24 candidates for
`mlaad_de_DE_speaker:erzaehlungen_poe` when it was supplied as excluded.

The finished `v4_validation/data_protocol_audit.json` is hash-pinned and
records 220 prior rows mapped to four English speakers; it records one German
dev speaker (`erzaehlungen_poe`) and one German sealed speaker (`karlsson`),
with `speech_speaker_overlap: []`. Provenance independently agrees: train
speech uses the four recorded English speaker IDs; dev and sealed use the two
separate German IDs. The actual speaker sets have no dev/sealed, new/prior, or
new/train intersections.

### Corpus, provenance, and archive evidence

- `v4_validation/data_integrity.json` reports 1,936 rows: train 1,516, dev
  210, sealed 210. The train set holds 10 named providers, each with at least
  31 parent groups; neither held-out provider (`stableaudio`, `mubert`) occurs
  in train.
- Train provenance contains both `ATA` and `TTA`; dev/sealed music contains
  the intended `TTA` evaluation variants. The acquisition loop includes all
  available ATA/TTA variants for a selected training track whose provider is
  in the fixed included-provider set (`prepare_v4_data.py:184-198`).
- `.omo/evidence/v4-data/dataset_lineage_audit.json` records decoding all
  1,936 manifest waveforms, matching decoded duration and PCM hash, finite
  samples, manifest/provenance ID closure, zero derivative-parent closure
  failures, and no missing license rows.
- `.omo/evidence/v4-data/archive_integrity.json` records matching streamed
  SHA-256 and successful ZIP-member tests for all four archives: 1,517 train,
  261 dev, 261 sealed, and 14 metadata members. All upload parts are at or
  below 95 MiB.
- All 11 source/artifact hashes in
  `.omo/evidence/v4-data/source_hashes.json` and all six code/test hashes in
  `.omo/evidence/v4-data/code_hashes.json` match the files reviewed here.
  `final_drive_inputs.json` is a matching immutable local receipt for the
  four archives. It has no Drive IDs, which is intentionally outside this
  review.

## Validation run

`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python
-m pytest -q test_v4_data.py` passed: **5 passed in 0.44s**. It exercises
full-source decode beyond the legacy ten-second crop, artist assignment,
manifest lineage and labels, injected parent/provider leakage rejection, and
evaluation duration/splice semantics. `compileall` on the reviewed Python
files also passed.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

1. **The direct old-speaker exclusion behavior lacks a dedicated regression
   test.** `test_v4_data.py` checks the produced corpus but never calls
   `prior_speech_speakers` or `speech_candidates` with an excluded speaker.
   Because the present historical rows are English and the new panel is
   German, deleting the `speaker not in excluded_speakers` predicate at
   `v4_data_acquisition.py:194` could leave the current artifact audit and all
   five tests green. Add one small cached-metadata test that supplies a known
   candidate speaker in `excluded_speakers` and proves every matching candidate
   is removed. This should be a behavior test, not a pinned prose/constant
   test.

2. **`prepare_v4_data.py` is 401 pure LOC without a documented size opt-out.**
   It combines acquisition orchestration, materialization, panel construction,
   validation, report writing, and archive writing. That exceeds the loaded
   programming perspective's 250-LOC threshold and makes the next change to a
   data invariant harder to isolate. This did not invalidate the reviewed
   corpus, so it is a maintenance watch item rather than a release blocker.

### LOW

None.

## Required skill-perspective check

This review loaded and applied `omo:programming`, `omo:remove-ai-slops`, and
Ponytail. The slop pass found no deletion-only/requested-removal tests,
tautological expected values, prompt/prose assertions, implementation-mirroring
tests, untyped escape hatches, or needless data extraction/parsing in the
speaker fix. The only programming-perspective violation is the oversized
orchestration module noted above. The suite's existing tests are high-signal
for the completed corpus, but they do not lock the newly introduced speaker
filter; that is the MEDIUM finding above.

## Reviewed hashes

| Artifact | SHA-256 |
| --- | --- |
| `prepare_v4_data.py` | `e2e5047b9f9593b57a770a43909b6bb17d1ad9c275ccdce2f9012d9b05a9ef08` |
| `v4_data_acquisition.py` | `c37ce5514c9ed5defe2b32a66b39d8627369d30889a5459ef8244b792d607140` |
| `v4_validation/provenance.json` | `2d250402253d292517d823ed1c8de8b789bc684eddb7d7e44689554c4157d00a` |
| `v4_validation/data_protocol_audit.json` | `9aa8699cb65544b0a0026ccdd0ed0dbf1900b41450c3c196b7bdd1af82444b5e` |
| `v4_validation/train.csv` | `df5c083d7deee6f5efa195c4fce2c8b4298f2e8ca743f2c7bce7f7129877c07f` |
| `v4_validation/dev.csv` | `f78185faee0578a311eea420a1776256ccea0346d2152e77b2288db477c605d9` |
| `v4_validation/sealed.csv` | `36892df96e1286600f7c9d9b674d935b5ad2fcb334577b843d51bc82f982fb58` |
| `_workspace/v4_data/final_drive_inputs.json` | `1462522e7ab374f4dfb77c84c32ce4faade6a3298d197b0be5dff1440ed7d511` |
