# V2 data preflight code review

## Result

- `codeQualityStatus`: WATCH
- `recommendation`: APPROVE
- `blockers`: none

No CRITICAL or HIGH finding remains.  The one WATCH item is a code-organization
concern and is not a data-integrity or GPU-preflight blocker.

## Independently verified acceptance criteria

- `_workspace/v2_sources/source_inventory.json` contains exactly 740 new rows:
  200 `mlaad_real`, 200 `mlaad_fake` (200 matched pairs across 10 accepted
  generators), 180 `echoes_tta`, 60 `fma_real`, and 100 `nsynth`.  The NSynth
  rows span 25 `nsynth_instrument_*` groups with four notes each.  Every
  retained original and prepared WAV exists; all source-file SHA-256 and
  prepared, decoded PCM SHA-256 values recomputed successfully.
- The MLAAD original and fake license records are present under
  `_workspace/v2_sources/licenses/`.  The inventory contains pinned revisions,
  URLs/member identifiers, source checksum, PCM checksum, license, attribution,
  and group for every new row.  No inventory or manifest source field refers to
  DACON/official competition audio; the only legacy input is `validation/`,
  which was explicitly reclassified as development-only.
- The two persisted manifests contain 1,029 unique IDs and exactly the expected
  role counts: `dev_train=633`, `dev_select=122`, `test_generator=176`, and
  `test_source=98`.  All 225 IDs from `validation/manifest.csv` are present and
  all are in a development role.  `split_audit.json` records 64 new mixtures,
  197/40/58/30 connected components, and the expected locked-label coverage.
- A full recomputation over all 1,029 actual stored WAVs matched every
  `pcm_sha256.csv` entry and found no duplicate decoded PCM.  All persisted WAVs
  are at most ten seconds, so the shared decoder's ten-second cap does not omit
  any persisted validation PCM in this delivered bank.
- Atomic source groups never cross roles.  Test-generator providers do not
  occur outside `test_generator`.  The 64 new mixtures are 16 per role.  The
  generated-music rule is implemented at `prepare_v2_validation.py:170-195`:
  real foreground plus fake background yields an unknown `VOICE_FAKE`, while a
  fake foreground remains `VOICE_FAKE=1`; all mixture `FILE_FAKE` values follow
  the foreground/background fake OR.
- Meaningful existing checks passed: `uv run --with numpy==2.4.6 --with
  imageio-ffmpeg==0.6.0 pytest -q test_prepare_v2_sources.py
  test_prepare_v2_validation.py` -> `11 passed`.  These tests cover CRC/range
  extraction, duplicate/legacy lineage rejection, artist distinctness,
  persisted-WAV hashing, unknown-label preservation, group isolation, valid
  mixture parents, and the generated-music authenticity mask.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

1. `prepare_v2_sources.py` (265 LOC) and `prepare_v2_validation.py` (289 LOC)
   exceed the programming skill's 250 pure-LOC module guideline.  They are
   standalone, deterministic preparation CLIs and the review found no resulting
   correctness or preflight failure.  Split only when either script needs another
   behavior change; do not delay this A100 run for a mechanical refactor.

## Skill-perspective check

Ran `omo:programming` and `omo:remove-ai-slops` before judging maintainability
and test relevance.  The remove-ai-slops pass found no deletion-only,
tautological, implementation-constant, or needless-normalization test, and no
unnecessary production parsing/extraction for this goal.  The programming pass
found no untyped escape hatch, brittle prompt test, or implementation-mirroring
test; it does flag the two over-250-LOC scripts noted above.  This is a LOW
maintainability observation, not a data criterion blocker.
