# Frozen feature/cache and asset preparation evidence

Date: 2026-09-17 (Asia/Seoul)

## Delivered callable surface

- `redesign_assets.prepare_assets(archive: Path, assets: Path, *, expected_sha256: str) -> dict[str, str]`
- `redesign_features.extract_features(rows, assets: Path, cache: Path, *, device='cuda', deadline=None, views=('stem','raw')) -> dict`
- `redesign_features.load_records(rows, cache: Path) -> dict`

The extractor uses an absolute `time.time()` deadline, emits `progress.json` with elapsed time and ETA, and returns device, total window count, elapsed seconds, peak RSS, and peak CUDA allocation. A caller can obtain a 100-file pilot by passing `rows[:100]`; no separate benchmark API was added.

## Source contract inspection

Scenario: preserve the submitted baseline preprocessing and bag semantics while adding aligned frozen features.

Invocation:

```bash
rg -n "load_track\(|apply_model\(|register_forward_pre_hook|requires_grad_\(False\)|make_panns_segments|predict_presence\(|predict_fake\(|last_hidden_state.mean" redesign_features_runtime.py baseline_original/script.py
```

Binary observable:

- `baseline_original/script.py:265,279` loads native-channel audio through Demucs `load_track` and calls `apply_model` before mono component reduction.
- `redesign_features_runtime.py:49,51,59` freezes HTDemucs, DF-Arena, and MERT; it never enables global gradients.
- `redesign_features_runtime.py:99` captures the exact 1280-dimensional input to `backbone.conformer.fc5` with a forward pre-hook while using native logits.
- `redesign_features_runtime.py:78` uses the shared offline MERT model with mean-pooled last hidden state for raw and stem windows.
- `redesign_features_runtime.py:131,155-156` uses baseline PANNs support and runs bounded first-record parity against the original full-file PANNs and DF-Arena bag functions. Every loaded cache record also checks that evidence is the exact window maximum.

Captured artifact: this file and `redesign_features_runtime.py`.

## Actual M330 archive anatomy

Scenario: inspect the user-provided archive without extracting weights or starting a GPU.

Invocation: Python `zipfile` opened `submissions/M330_MLP_full1206/M330_MLP_full1206_submit.zip`, read only its member table plus `model/manifest.json`, and compared required member names with the adjacent receipt.

Binary observable:

- receipt SHA256: `1f6cfd6c9feacead01d4ebcfd217d10c1796969b1f863ba4b0283d46e7432c7e`
- archive bytes: `6267568297`
- members: `47`; selected baseline/MERT members: `27`
- baseline script: `14552` bytes, declared SHA256 `c17d776b7c17c2d70f1e55c2186ea54f0af8695dcdf07a106523cd80213aed9e`
- DF-Arena weight: `4591794734` bytes, declared SHA256 `780bc14fd4c15e65d58efdef728427cf03cd29cd60be528e97badf8c89087988`
- PANNs weight: `327428481` bytes, declared SHA256 `0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31`
- MERT weight: `1261846489` bytes, declared SHA256 `0c03774d821a6ad972cb48220b128a10ebea2c790116bd0fd35e5a013f8017f6`

`prepare_assets` requires the full external archive pin, rejects absolute/backslash/traversal paths, selected symlinks, duplicate or colliding members, CRC errors, internal digest mismatches, and an existing destination. It stages only baseline script/model families plus MERT, then writes the required flat `asset_hashes.json`.

Captured artifact: this file and `redesign_assets.py`.

## CPU integrity and corruption QA

Scenario: protect practical package/cache failure modes without pretending that a tiny fixture validates encoder quality.

Invocation:

```bash
.venv/bin/python -m pytest -q test_redesign_features.py
```

Binary observable: `5 passed in 0.05s`.

The five tests establish:

1. selected ZIP traversal is rejected; safely prepared files validate; post-extraction asset tampering is rejected;
2. a changed current audio file is rejected and a checksum-adjusted but semantically corrupted evidence vector is still rejected;
3. modification of a content-addressed immutable extraction receipt is rejected;
4. `load_records` accepts a verified train/validation subset of a larger manifest;
5. extraction into the same receipt preserves other partition entries and reports cached windows/resource telemetry.

Captured artifact: this file and `test_redesign_features.py`.

## Static and source-integrity checks

Invocations and binary observables:

```text
uvx ruff check redesign_assets.py redesign_features.py redesign_features_cache.py redesign_features_runtime.py test_redesign_features.py
All checks passed!

.venv/bin/python .../check-no-excuse-rules.py redesign_assets.py redesign_features.py redesign_features_cache.py redesign_features_runtime.py test_redesign_features.py
no violations in 5 file(s)

git diff --check -- redesign_assets.py redesign_features.py redesign_features_cache.py redesign_features_runtime.py test_redesign_features.py
exit 0
```

Current source SHA256 values:

```text
3039d85626bf28c5e5374316c97aa3b49dec37956fd550116e1eb8288575d933  redesign_assets.py
e5453b39c0e1acec0966bc19642538442ba16d3d9a7f90d41c6f44fd14559eef  redesign_features.py
5594b86ca3effffeb40893317cad04debf510da11db84158bdc0590508fb299a  redesign_features_cache.py
33799ddc1efc017a6b04f2ca52a8f22c50b57d32ed05b18cc1eb477447ffec99  redesign_features_runtime.py
bd26ba9bf3613ebfabeda7543fe449806a430134ad3d523aea00dac8737da618  test_redesign_features.py
```

`uvx basedpyright` completed with `0 errors`; it reported warnings at dynamic boundaries in the imported untyped baseline, Transformers, Demucs, and NumPy archive APIs. No ignore directives or casts were added.

Captured artifact: this file.

## Explicit limitation

No real encoder, separator, PANNs, DF-Arena, MERT, CUDA, H100, or L4 inference ran in this local QA. The first actual extraction performs native baseline bag-parity assertions and records runtime telemetry; H100/L4 runtime, memory fit, throughput, and prediction parity remain unverified until that run. The cache always contains both raw and stem views because the contracted NPZ schema requires both; A/C may consume stem only, while B/D reuse both.
