# V5 submission runtime verification

Invocation:

`.venv/bin/python -m pytest -q test_v5_submission.py test_v5_experts.py test_v5_features.py test_v5_fusion.py test_v5_final_refit.py`

Binary observable: exit code 0; `24 passed`.

Captured artifact: `.omo/evidence/v5-final-submission-runtime/pytest.log`

Validated scenarios:

- B1/B2 raw inference: the strongest tail-only window changes the component maximum; FILE_FAKE equals `max(VOICE_PRESENT*VOICE_FAKE, MUSIC_PRESENT*MUSIC_FAKE)`.
- B3/B4 feature parity: per-kind feature order is raw embedding, stem embedding, raw logits, stem logits, raw-minus-stem logits; timestamps use the tail-anchored 16 kHz grid; encoding a companion file does not change the target file.
- Decoder parity: packaged stereo-FLAC decode is sample-identical to `v2_data.decode_audio`, including ffmpeg downmix and 16 kHz resampling.
- Input failures: wrong submission columns, test/sample ID mismatch, and undecodable audio all raise `InferenceError` before output is accepted.
- Packaging: four archives are produced with only `script.py`, `requirements.txt`, and `model/` at the root; B3/B4 alone contain Demucs and fusion state; original DF, SONICS, and PANNs tensors are absent.
- Provenance packaging: archives retain baseline model info/checksums, DF source license/readme, SONICS license/provenance, final-data audit, source provenance, and available source-license files.
- Completion gate: incomplete status, incomplete epoch/exposure coverage, identity mismatch, missing/empty final state, and SHA-256 mismatch are rejected by the shared gate. The focused test exercises incomplete status; the remaining branches are direct predicates in the same gate and are covered by final-refit artifact tests.
- Existing V5 invariants: strict expert adapters, temporal feature coverage, nine-state marginalization, and final full-data refit completion contracts remain passing.

Static invocation: `.venv/bin/python -m py_compile v5_submission.py package_v5_submission.py test_v5_submission.py`

Binary observable: exit code 0.

Captured artifact: `.omo/evidence/v5-final-submission-runtime/py_compile.log`

Source integrity invocation: `shasum -a 256 v5_submission.py package_v5_submission.py test_v5_submission.py`

Binary observable: three SHA-256 entries.

Captured artifact: `.omo/evidence/v5-final-submission-runtime/source_sha256.txt`

Real full-size checkpoints and CUDA/Demucs execution are not present locally. `verify_offline_archive()` is the callable fresh-unpack parity gate for the root Colab run; it requires exact ID order and maximum absolute probability delta <= `1e-4`.

Pinned constructor invocation: downloaded `configuration_antispoofing.py`, `modeling_antispoofing.py`, `backbone.py`, and `config.json` from Hugging Face model revision `fb6ce85de12c2c5a509d89114adaf827dd75f49f`; compared the three Python files byte-for-byte with `baseline_original/baseline_submit.zip`; extracted the stripped source tree and instantiated it under a meta device.

Binary observable: source comparisons exit 0; constructor reports exactly 1,147,845,253 parameters, Wav2Vec2 hidden size 1280, and 48 layers without the original pretrained tensor.

Captured artifacts: `.omo/evidence/v5-final-submission-runtime/pinned_configuration_antispoofing.py`, `.omo/evidence/v5-final-submission-runtime/pinned_modeling_antispoofing.py`, `.omo/evidence/v5-final-submission-runtime/pinned_backbone.py`, `.omo/evidence/v5-final-submission-runtime/df_meta_constructor.log`.
