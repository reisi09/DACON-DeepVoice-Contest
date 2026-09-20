"""Controlled V4 mixture and splice construction."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from v2_data import decode_audio, write_wav


def load_audio(path: Path) -> np.ndarray:
    """Load a complete mono 16 kHz waveform."""
    return decode_audio(path, limit_seconds=None)


def crop_or_tile(audio: np.ndarray, samples: int, key: str) -> np.ndarray:
    if len(audio) >= samples:
        maximum = len(audio) - samples
        offset = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (maximum + 1)
        return audio[offset:offset + samples].copy()
    return np.pad(audio, (0, samples - len(audio)))


def mix_at_snr(music: np.ndarray, speech: np.ndarray, snr_db: float) -> np.ndarray:
    music_rms = max(float(np.sqrt(np.mean(np.square(music), dtype=np.float64))), 1e-8)
    speech_rms = max(float(np.sqrt(np.mean(np.square(speech), dtype=np.float64))), 1e-8)
    speech_gain = music_rms / (speech_rms * 10 ** (snr_db / 20))
    mixed = music + speech * speech_gain
    peak = max(float(np.max(np.abs(mixed))), 1.0)
    return (mixed / peak * 0.98).astype(np.float32)


def splice(first: np.ndarray, second: np.ndarray, samples: int, crossfade_samples: int = 800) -> tuple[np.ndarray, int]:
    first = crop_or_tile(first, samples, "splice-first")
    second = crop_or_tile(second, samples, "splice-second")
    boundary = samples // 2
    half = crossfade_samples // 2
    start, end = boundary - half, boundary + half
    output = np.concatenate((first[:start], np.zeros(crossfade_samples, dtype=np.float32), second[end:]))
    fade = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
    output[start:end] = first[start:end] * (1.0 - fade) + second[start:end] * fade
    return output, boundary


def write_derived(path: Path, audio: np.ndarray) -> None:
    write_wav(path, audio)
