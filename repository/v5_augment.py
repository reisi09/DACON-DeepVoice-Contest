"""Deterministic label-independent V5 channel augmentation."""
from __future__ import annotations

import hashlib
import random
import subprocess

import imageio_ffmpeg
import numpy as np


def _seed(identifier: str, epoch: int, seed: int, purpose: str) -> int:
    return int(hashlib.sha256(f"{seed}|{identifier}|{epoch}|{purpose}".encode()).hexdigest()[:16], 16)


def channel_augment(audio: np.ndarray, identifier: str, epoch: int, seed: int) -> np.ndarray:
    """Apply gain, low-pass/8 kHz telephony, noise, and a 64 kbps codec replayably."""
    rng = np.random.default_rng(_seed(identifier, epoch, seed, "channel"))
    gain = 10.0 ** (random.Random(_seed(identifier, epoch, seed, "gain")).uniform(-6.0, 6.0) / 20.0)
    filtered = np.convolve(audio * gain, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    down = filtered[::2]
    telephony = np.interp(np.arange(len(audio)) / 2.0, np.arange(len(down)), down).astype(np.float32)
    rms = max(float(np.sqrt(np.mean(np.square(telephony), dtype=np.float64))), 1e-8)
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    noise *= rms / (max(float(np.sqrt(np.mean(np.square(noise), dtype=np.float64))), 1e-8) * 31.62)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    encoded = subprocess.run(
        [ffmpeg, "-v", "error", "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "-",
         "-f", "mp3", "-c:a", "libmp3lame", "-b:a", "64k", "-"],
        input=(telephony + noise).astype("<f4").tobytes(), check=True, capture_output=True,
    ).stdout
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-i", "-", "-f", "f32le", "-ar", "16000", "-ac", "1", "-"],
        input=encoded, check=True, capture_output=True,
    ).stdout
    result = np.frombuffer(decoded, dtype="<f4").copy()
    return np.pad(result[:len(audio)], (0, max(0, len(audio) - len(result)))).astype(np.float32)
