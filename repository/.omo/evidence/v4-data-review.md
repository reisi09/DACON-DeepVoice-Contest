# V4 data code review

Date: 2026-09-13

Scope reviewed: `prepare_v4_data.py`, `v4_data_acquisition.py`,
`v4_data_augmentation.py`, `v4_data.py`, `v2_data.py`, and `test_v4_data.py`,
against `.omo/plans/trainable-recovery-v4.md`.

## Verdict

- `codeQualityStatus`: `BLOCK`
- `recommendation`: `REQUEST_CHANGES`
- `blockers`:
  1. The data path cannot establish or enforce the required exclusion of new
     dev/sealed speech identities from old/training speech identities.

## Findings

### CRITICAL

None.

### HIGH

1. **Speech-identity leakage is permitted despite the explicit role-disjoint
   requirement.**

   The plan requires the 20 dev and 20 sealed speech pairs to be disjoint from
   old and training *speech identities* (`trainable-recovery-v4.md:97`).
   `prepare_v4_data.py:391` supplies `prior_groups` plus training group IDs to
   `speech_candidates`, but `v4_data_acquisition.py:162-178` filters only the
   utterance-derived `group` (`mlaad:<filename stem>`). It does derive a true
   speaker field for new German MLAAD candidates, but the old V2 and V3
   provenance records have no speaker identity: `prepare_v4_data.py:173`
   substitutes the utterance group as `speaker_id` for historical MLAAD rows.
   The post-build validator checks dev-versus-sealed groups and the explicit
   artist lists (`prepare_v4_data.py:292-298`), never new-panel speakers versus
   train/prior speakers.

   This allows different utterances from the same speaker to occur in training
   and a fresh evaluation role. The historical records inspected contain 400
   V2 speech rows and 40 V3 speech rows, all without a `speaker_id`, so the
   intended check cannot be reconstructed from the current provenance. Derive
   the historical MLAAD speaker identity from the pinned metadata, persist it,
   exclude it during candidate selection, and validate/report train/dev/sealed
   speaker-set intersections before treating either panel as independent.

### MEDIUM

None.

### LOW

None.

## Confirmed source-level behavior

- `v2_data.decode_audio(..., limit_seconds=None)` preserves the V2 default and
  provides explicit full decoding (`v2_data.py:66-78`). V4 materialization
  stores full decoded WAVs and records raw-file, source-PCM, stored-PCM, and
  original-duration values (`prepare_v4_data.py:123-139`). Evaluation excerpts
  are a separate manifest path (`prepare_v4_data.py:142-151`), so the code does
  not relabel a short evaluation clip as a recovered full original.
- Training includes every available ATA and TTA row for each selected track
  whose provider meets the fixed quota: variants are loaded at
  `v4_data_acquisition.py:38-44` and all eligible included-provider variants
  are appended at `prepare_v4_data.py:186-196`.
- Generated-song voice truth remains unknown unless fake speech is a retained
  parent: mixture construction writes `VOICE_FAKE=""` for real speech plus fake
  music (`prepare_v4_data.py:258-269`); splices keep it unknown
  (`prepare_v4_data.py:270-284`).
- Derived rows store the union of their immediate parent group IDs, and the
  intended manifest test recomputes that union from `parent_ids`
  (`prepare_v4_data.py:267-269`, `test_v4_data.py:53-60`).

## Artifact and archive state

`v4_validation/` exists but is empty at review time. Therefore
`protocol.json`, `provenance.json`, `integrity_report.json`, manifests,
duration/label audits, and archive receipts/hashes are **pending**, not
verified. `.omo/evidence/v4-data-build.log` is zero bytes. This is not treated
as a completed-data claim or as a separate code defect while acquisition is
still running. The staged `_workspace/v4_data/final_drive_inputs.json` handoff
is intentionally separate from the later Drive-ID receipt.

## Validation performed

- `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python
  -m pytest -q test_v4_data.py::test_full_decode_preserves_content_after_ten_seconds
  test_v4_data.py::test_exclusive_first_artist_assignment_matches_audited_pool`
  passed: 2 passed in 0.39s. These tests exercise full-source recovery beyond
  the V2 ten-second crop and deterministic exclusive-first panel assignment.
- Full `test_v4_data.py` result: 2 passed, 3 failed only because the pending
  `v4_validation/provenance.json` does not exist. No artifact-backed lineage,
  labels, durations, or provider-hold result can be inferred from this failure.
- `compileall` over the reviewed Python files passed.

## Required skill-perspective check

This review loaded `omo:programming`, `omo:remove-ai-slops`, and Ponytail.
The slop pass found no deletion-only tests, requested-removal tests,
tautological expected values, prompt/prose assertions, implementation-mirroring
tests, unnecessary parsing/normalization, or speculative production
abstractions in this scope. The programming perspective finds the semantic
identity boundary above incomplete; otherwise the reviewed diff has no
material violation relevant to this data goal. The tests that do run protect
real recovery and assignment behavior; the artifact-dependent tests are
appropriately aimed at parent union, unknown labels, provider holds, and
duration semantics, but cannot validate them until the real manifest exists.
