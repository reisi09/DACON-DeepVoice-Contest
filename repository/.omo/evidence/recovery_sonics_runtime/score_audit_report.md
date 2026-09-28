# Locked SONICS score audit

## Checkpoint identity

- Scenario: determine whether the inspected runtime differs from the locked A100 run.
- Invocation: `sha256sum recovery_sonics.py _workspace/recovery_v3/sonics_assets/config.json _workspace/recovery_v3/sonics_assets/pytorch_model.bin` and `rg -n 'recovery_sonics.py|sonics_assets/config.json|sonics_assets/pytorch_model.bin' results/recovery_v3/colab_20260913_044621/candidate_lock.json`.
- Observable: runtime hash `4882c036f504695df46162b81ca794a897acde7a671c564231eea2066b82e0ba`, config hash `754bc15b3f914afaf4e002e50bfaa5e81055c9fb0e2b76408e52fd88f09f7def`, and weights hash `11c384cea651c56028d6f02e04aa467c2099cf1102687353fe4dd9504c4db868` exactly match `candidate_lock.json`.
- Capture: `score_audit.json` and this report.

## Logit and probability boundary

- Scenario: distinguish the implemented single sigmoid from a double sigmoid on a real 16 kHz waveform.
- Invocation: the `.venv/bin/python` probe recorded in `score_audit.json`, loading `validation/audio/music_43fa43208594a31f.wav` with `librosa.load(sr=16000, mono=True, dtype=float32)`.
- Observable: raw logit `-4.803152084350586`; `sigmoid(logit)` and predictor probability both `0.008137091062963009` with delta `0.0`; counterfactual double sigmoid `0.5020342469215393` differs by `0.4938971558585763`.
- Capture: `score_audit.json`.

## Pinned label direction

- Scenario: determine whether the model's positive class is fake.
- Invocation: `curl -L --fail --silent --show-error https://raw.githubusercontent.com/awsaf49/sonics/9156ffad151f797c71556923c4a02fa01fa8fc91/data_split.py | rg -n 'real_df\["target"\]|fake_df\["target"\]|concat'` plus the corresponding pinned `train.py` grep for `BCEWithLogitsLoss|torch.sigmoid(preds)`.
- Observable: source sets real to `0`, fake to `1`; training uses `BCEWithLogitsLoss` and applies sigmoid only when converting logits to prediction.
- Capture: this report; source revision is pinned in `_workspace/recovery_v3/sonics_assets/provenance.json`.

## Preprocessing parity

- Scenario: rebuild the original feature path from the loaded checkpoint configuration and compare its logit with `HFAudioClassifier.forward`.
- Invocation: the `.venv/bin/python` probe recorded in `score_audit.json`.
- Observable: configured 16 kHz, 80,000-sample, 2,048/512/2,048, 128-mel, 20-8,000 Hz, power-2, top-db-80, mean/std path yields manual-rebuilt logit delta `4.76837158203125e-07`. The original training loader's waveform-standardization step changes this real input's logit by `0.0`.
- Capture: `score_audit.json`.

## Conclusion

No implementation error is proven. The locked runtime, assets, single-sigmoid conversion, fake-positive direction, and preprocessing path all agree with the pinned SONICS contract. The poor development result is evidence that this specialized detector does not generalize to the evaluated corpus; it is not evidence for inversion, retuning, or a locked-code change.
