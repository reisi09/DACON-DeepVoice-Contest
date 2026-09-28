# Real input bundle evidence — 2026-09-17

## Decision

The available verified local assets support a **historical diagnostic run
only**. They do not provide a fresh holdout. The portable manifest contains
1,696 rows and 3,916,530,384 logical audio bytes. Every component label that
was not established by the existing provenance remains empty/unknown.

The V2 inventory, V4 provenance, Echoes/FMA metadata, and existing H100 input
manifests contain no verified fake instrumental/accompaniment row. FMA metadata
has 18 tracks marked instrumental, but this supports real accompaniment only;
it does not prove that an Echoes-generated derivative has no voice. Therefore
the preparer correctly produces zero paired mixtures. C duplicates A's
exposure and D duplicates B's exposure until a verified fake accompaniment is
available, so only meaningful A/B diagnostics should be run from this bundle.

The recovery check loaded 740 V2 inventory rows and all V4 rows. Both returned
zero rows with known fake music, known voice absence, and usable music
presence truth. Its hashes for six prior H100 input manifests and exact counts
are captured in `source-recovery.log`.

## Deliverables

- Portable source manifest: `_workspace/redesign_20260917/input-preparation/sources.csv`
- Portable audio directory: `_workspace/redesign_20260917/input-preparation/audio/`
- Source and license audit: `_workspace/redesign_20260917/input-preparation/input_audit.json`
- Pinned local receipts: `_workspace/redesign_20260917/input-preparation/receipts/`
- Reproducible converter: `_workspace/redesign_20260917/input-preparation/build_inputs.py`
- CPU-prepared manifests/audit: `_workspace/redesign_20260917/input-preparation/prepared/`
- Full preparation output: `_workspace/redesign_20260917/input-preparation/prepare.log`
- Independent verification output: `_workspace/redesign_20260917/input-preparation/verification.log`
- Existing-source recovery check: `_workspace/redesign_20260917/input-preparation/source-recovery.log`

`sources.csv` SHA-256 is
`47ad5a01c1d88365ee1e8c72cd3b125858578d50041e93280516d4372004ae89`.
The source provenance SHA-256 is
`2d250402253d292517d823ed1c8de8b789bc684eddb7d7e44689554c4157d00a`.

## Provenance and label coverage

The rows reuse the existing V4 panel or its existing TTA/ATA augmentations:

- 1,116 whole-song music rows (`kind=mixed`): 1,026 File/Music fake and 90
  File/Music real. Their voice presence and voice authenticity remain unknown.
- 480 speech rows: 240 real voice and 240 fake voice. Music is known absent.
- 100 NSynth rows: File/Music real, voice known absent. NSynth remains a
  stress source and is not evidence of natural-real accompaniment quality.
- Unknown values retained: `VOICE_FAKE=1,216`, `VOICE_PRESENT=1,116`, and
  `MUSIC_FAKE=480`.
- License counts and exact receipt hashes are recorded in `input_audit.json`.
  License metadata is preserved; competition eligibility was not independently
  adjudicated in this task.

Whole-song rows are not called verified singing strata. `kind=mixed` reflects
the whole-song file category while known-mask fields retain the component truth
boundary. They cannot seed component mixture generation.

## CPU preparation scenario

Requirement protected: a portable manifest must resolve its real audio, reject
hash drift/corruption, preserve unknown labels, prevent lineage overlap, and
show actual achievable design coverage before any upload or GPU creation.

Invocation:

```bash
.venv/bin/python redesign_data.py build \
  --sources _workspace/redesign_20260917/input-preparation/sources.csv \
  --output _workspace/redesign_20260917/input-preparation/prepared \
  --seed 41 --diagnostic \
  | tee _workspace/redesign_20260917/input-preparation/prepare.log
```

Binary observables from `prepared/audit.json`:

- accepted sources: 1,696; quarantined: 0
- split rows: train 1,469; validation 129; sealed 98
- independent connected groups: train 97; validation 12; sealed 12
- lineage overlap: `{}`
- generated rows / verified pairs: 0 / 0
- paired slots: train 0; validation 0; sealed 0
- coverage gate: `diagnostic_only`
- sealed panel identity:
  `f47612f1d424f80a25bf4431fab7b417c186d0488d0ad49c63fcc5dbe4074b90`

The connected lineage split puts all 480 speech rows in training. Validation
and sealed contain only mixed whole-song and NSynth rows, so they cannot gate a
Voice candidate. This is an observed limitation, not a passing Voice result.

## Portable integrity scenario

Requirement protected: upload paths must be relative and every payload file
must retain the manifest SHA-256.

Invocation: the Python assertion block captured in `verification.log` resolves
all 1,696 relative paths, recomputes every file SHA-256, checks the logical byte
total, and reads the actual prepared split CSVs. It then runs:

```bash
du -sh _workspace/redesign_20260917/input-preparation/audio
wc -l _workspace/redesign_20260917/input-preparation/sources.csv
shasum -a 256 \
  _workspace/redesign_20260917/input-preparation/sources.csv \
  _workspace/redesign_20260917/input-preparation/prepared/audit.json
```

Binary observables:

- `relative_paths: true`
- `all_audio_sha256_verified: true`
- logical audio bytes: 3,916,530,384 (`du`: 3.7G)
- manifest lines: 1,697 including the header
- prepared audit SHA-256:
  `8d6dafb03e5f69fe87479fa623b6d040ce9a92d8285909e027776c81a965aeb7`

## Cloud handoff

Upload the contents of `input-preparation/` to
`/root/redesign_h100_20260917/input/`, preserving `sources.csv` beside
`audio/`. Then rerun the same preparation command with `--diagnostic` before
starting a GPU experiment. No cloud resource was created and no upload was
performed by this task.
