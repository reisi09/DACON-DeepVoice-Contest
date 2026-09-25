from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio
from demucs.audio import convert_audio

import panns_adaptation_cache as cache_module
from mert12_submission import load_track_torchaudio


def test_normalized_wav_retains_decoder_numerics_without_torchcodec(tmp_path: Path, monkeypatch) -> None:
    audio = np.random.default_rng(41).uniform(-.6, .6, 16003).astype(np.float32)
    path = tmp_path/'normalized.wav'
    sf.write(path, audio, 16000, subtype='FLOAT')
    expected = convert_audio(torch.from_numpy(audio[None]), 16000, 44100, 2)
    def unavailable(*args, **kwargs):
        raise RuntimeError('libtorchcodec unavailable')
    monkeypatch.setattr(torchaudio, 'load', unavailable)
    with pytest.raises(RuntimeError, match='libtorchcodec'):
        load_track_torchaudio(path, 2, 44100)
    actual = cache_module.load_normalized_wav(path, 2, 44100)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for rate, channels, subtype, format in ((8000, 1, 'FLOAT', 'WAV'),
            (16000, 2, 'FLOAT', 'WAV'), (16000, 1, 'PCM_16', 'WAV'),
            (16000, 1, 'PCM_16', 'FLAC')):
        samples = np.tile(audio[:, None], (1, channels))
        sf.write(path, samples, rate, subtype=subtype, format=format)
        with pytest.raises(ValueError, match='mono 16 kHz FLOAT WAV'):
            cache_module.load_normalized_wav(path, 2, 44100)
