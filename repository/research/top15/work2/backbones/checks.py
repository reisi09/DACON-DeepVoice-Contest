"""Equivalence / input checks for the vendored encoders (CPU, fp32 HF weights).
  python checks.py eat   -> vendored EAT (no timm) == original remote code (timm) on the same fbank; manual layer loop == extract_features
  python checks.py muq   -> MuQ input scaling: normalised mel of a runtime window vs the checkpoint's N(0,1) training statistics
Answers: is the vendored code numerically the published model, and is the waveform fed at the scale/rate MuQ was trained on?"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'vendor'))
import extract_new_backbones as X  # noqa: E402

torch.set_num_threads(4)
AUDIO = HERE.parents[3] / '_workspace/audit_20260927/parity_audio/fake_16900_udio_1__81062a7b5a.mp3'
wav = librosa.load(AUDIO, sr=16000, mono=True)[0].astype(np.float32)
chunks = np.stack([wav[s:s + 64600] for s in (0, 32300, 64600, 96900)])
report = {}
if sys.argv[1] == 'eat':
    import shutil
    import tempfile
    from eat_core import EAT, EATConfig
    d = HERE / 'ckpt/EAT-large_epoch20_finetune_AS2M'
    cfg = json.loads((d / 'config.json').read_text())
    cfg = EATConfig(**{k: v for k, v in cfg.items() if k not in ('architectures', 'auto_map', 'model_type', 'torch_dtype', 'transformers_version')})
    state = {k.removeprefix('model.'): v for k, v in load_file(str(d / 'model.safetensors')).items()}
    tmp = Path(tempfile.mkdtemp()) / 'eat_orig'      # the unmodified HF files as a package (the local timm 1.0.15 is used)
    tmp.mkdir()
    for f in ('eat_model.py', 'model_core.py', 'configuration_eat.py'):
        shutil.copy(d / f, tmp / f)
    (tmp / '__init__.py').write_text('')
    sys.path.insert(0, str(tmp.parent))
    from eat_orig.eat_model import EAT as OriginalEAT
    original, vendored = OriginalEAT(cfg), EAT(cfg)
    for m in (original, vendored):
        m.load_state_dict(state, strict=True)
        m.eval()
    import torchaudio
    mel = torch.stack([(m[:len(m) // 16 * 16] + 4.268) / (4.569 * 2) for m in (torchaudio.compliance.kaldi.fbank(
        torch.from_numpy(c - c.mean())[None], htk_compat=True, sample_frequency=16000, use_energy=False, window_type='hanning',
        num_mel_bins=128, dither=0.0, frame_shift=10) for c in chunks)])[:, None]
    with torch.inference_mode():
        a, b = original.extract_features(mel), vendored.extract_features(mel)
        manual = X.eat_features(vendored, chunks, 'cpu').astype(np.float32)[:, -1]   # block 24, fp16-rounded like the cache
    report['eat_vendored_vs_original_max_abs'] = float((a - b).abs().max())
    report['eat_manual_loop_vs_extract_features_rel'] = float(np.linalg.norm(manual - b[:, 1:].mean(1).numpy()) / np.linalg.norm(manual))
    report['eat_tokens'] = list(a.shape)
else:
    from muq_core import MuQ, MuQConfig
    d = HERE / 'ckpt/MuQ-large-msd-iter'
    model = MuQ(MuQConfig(**json.loads((d / 'config.json').read_text())))
    model.load_state_dict(load_file(str(d / 'model.safetensors')), strict=True)
    with torch.inference_mode():
        x = torch.from_numpy(np.stack([librosa.resample(c, orig_sr=16000, target_sr=24000, res_type='soxr_hq') for c in chunks]))
        m = model.model.normalize(model.model.preprocessing(x, features=['melspec_2048']))['melspec_2048']
    report['muq_normalised_mel_mean_std'] = [round(float(m.mean()), 3), round(float(m.std()), 3)]
    low = torch.from_numpy(librosa.mel_frequencies(n_mels=130, fmin=0, fmax=12000, htk=True)[1:-1] < 7500)   # 16 kHz source band
    report['muq_bins_below_7p5khz'] = int(low.sum())
    report['muq_normalised_mel_mean_std_below_7p5khz'] = [round(float(m[:, low].mean()), 3), round(float(m[:, low].std()), 3)]
    report['muq_normalised_mel_mean_above_8p5khz'] = round(float(m[:, torch.from_numpy(librosa.mel_frequencies(n_mels=130, fmin=0, fmax=12000, htk=True)[1:-1] > 8500)].mean()), 3)
    report['muq_mel_frames_per_window'] = int(m.shape[-1])
    report['muq_tokens_per_window'] = int(model(x, output_hidden_states=True).hidden_states[0].shape[1])
    with torch.inference_mode():   # a mis-scaled waveform (int16 range) moves the statistics far from N(0,1)
        bad = model.model.normalize(model.model.preprocessing(x * 32768, features=['melspec_2048']))['melspec_2048']
    report['muq_normalised_mel_mean_std_if_int16_scaled'] = [round(float(bad.mean()), 3), round(float(bad.std()), 3)]
print(json.dumps(report))
(HERE / f'logs/checks_{sys.argv[1]}.json').write_text(json.dumps(report, indent=1))
