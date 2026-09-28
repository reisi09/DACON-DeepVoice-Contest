# SONICS runtime evidence

## Strict local assets

- Scenario: a checkpoint whose bytes differ from the pinned SHA-256 must fail before deserialization.
- Invocation: `.venv/bin/python -m pytest -q test_recovery_sonics.py test_recovery_fusion.py`
- Observable: `test_strict_asset_hash_rejects_tampered_weights_before_model_load` passes; its tampered `pytorch_model.bin` raises `SonicsContractError` containing `SHA-256 mismatch`.
- Capture: `final_meaningful_behavior_tests.log`.

## Full-file tail coverage and aggregation

- Scenario: a 168,000-sample file with a synthetic high late tail reaches an anchored final full window.
- Invocation: `.venv/bin/python -m pytest -q test_recovery_sonics.py test_recovery_fusion.py`
- Observable: `test_predict_covers_late_tail_and_aggregates_file_locally` passes with starts `[0, 80000, 88000]`, a gap-free interval union ending at `168000`, and the expected tail-dependent mean.
- Capture: `final_meaningful_behavior_tests.log`.

## Fixed, file-local fusion

- Scenario: the five candidates retain G0 voice/presence values while changing music and its dependent file probability; an unrelated file cannot affect the result.
- Invocation: `.venv/bin/python -m pytest -q test_recovery_sonics.py test_recovery_fusion.py`
- Observable: `test_candidates_change_only_music_and_recompute_file_head` and `test_candidates_are_file_local_and_reject_invalid_probabilities` pass.
- Capture: `final_meaningful_behavior_tests.log`.

## Real CPU load and prediction

- Scenario: strict local SONICS loading and serial inference on `validation/audio/music_43fa43208594a31f.wav`.
- Invocation:
  ```bash
  .venv/bin/python - <<'PY' | tee .omo/evidence/recovery_sonics_runtime/real_cpu_inference.json
  import json
  from pathlib import Path
  import torchaudio
  from recovery_sonics import SAMPLE_RATE, SonicsPredictor
  path = Path('validation/audio/music_43fa43208594a31f.wav')
  audio, rate = torchaudio.load(path)
  if rate != SAMPLE_RATE:
      audio = torchaudio.functional.resample(audio, rate, SAMPLE_RATE)
  waveform = audio.mean(dim=0).numpy()
  predictor = SonicsPredictor(Path('_workspace/recovery_v3/sonics_assets'), 'cpu')
  result = predictor.predict(waveform)
  print(json.dumps({'input': str(path), 'input_samples': int(waveform.size),
                    'input_sample_rate': SAMPLE_RATE,
                    'model_parameters': sum(parameter.numel() for parameter in predictor.model.parameters()),
                    'device': str(predictor.device), 'result': result}, sort_keys=True))
  PY
  ```
- Observable: CPU model loaded with `17172493` parameters; 160,000 samples produced two window probabilities, mean `0.04136451659724116`, `audio_seconds: 10.0`, and a measured `elapsed_seconds`.
- Capture: `real_cpu_inference.json` and `real_cpu_inference_and_fusion.json`.

## Portable offline layout parity

- Scenario: an independent process can load a copied `WORK/sonics_assets` plus sibling `WORK/sonics_vendor/sonics` without a project-relative vendor path or Hub access.
- Invocation: `.venv/bin/python -m pytest -q test_recovery_sonics.py test_recovery_fusion.py`
- Observable: `test_offline_portable_directory_matches_source_bundle` copies and byte-compares the pinned source tree, then compares source and copied-directory CPU predictions from independent `HF_HUB_OFFLINE=1` processes.
- Capture: `final_meaningful_behavior_tests.log`.
