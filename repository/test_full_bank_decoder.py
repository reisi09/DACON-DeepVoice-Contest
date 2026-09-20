"""The real FMA MP3 must remain decodable when libsndfile rejects it."""
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import audioread
from audioread.ffdec import FFmpegAudioFile
import librosa
import numpy as np
import soundfile as sf

from full_bank_run import decoder_path, digest


def test_fma_libsndfile_failure_uses_existing_offline_ffmpeg():
    path = Path(__file__).parent/'_workspace/data_expansion_20260918/fma/originals/fma_97601.mp3'
    assert digest(path) == '4f6ecece8770e8e1fbee421e3002c652ff6d88fb703d2de55c68db8231a04e1b'
    try:
        sf.read(path, dtype='float32')
    except sf.LibsndfileError:
        pass
    else:
        raise AssertionError('This regression requires the observed libsndfile decoder failure')
    with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, PATH=''):
        os.environ['PATH'] = decoder_path(Path(temporary))
        # Linux has no macOS CoreAudio backend to mask a missing FFmpeg executable.
        with audioread.audio_open(str(path), backends=[FFmpegAudioFile]) as decoder:
            audio, rate = librosa.load(decoder, sr=16000)
        assert np.isfinite(audio).all()
        assert abs(len(audio)/rate - 83.9575625) < .05, 'Decoded audio must retain the full source duration'
    assert digest(path) == '4f6ecece8770e8e1fbee421e3002c652ff6d88fb703d2de55c68db8231a04e1b'
