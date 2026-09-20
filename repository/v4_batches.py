"""Deterministic full-source crop scheduling for V4 training."""

from __future__ import annotations

import hashlib
import math
import random
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import imageio_ffmpeg

SEED: Final = 20_260_913
STRATA: Final = 8


def _seed(*parts: str | int) -> int:
    text = "|".join(map(str, (SEED, *parts)))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def epoch_crop(
    audio: np.ndarray,
    identifier: str,
    epoch: int,
    window_samples: int,
) -> np.ndarray:
    """Draw one replayable crop while cycling all eight temporal strata."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("training audio must be nonempty mono samples")
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if waveform.size <= window_samples:
        return np.pad(waveform, (0, window_samples - waveform.size))
    start = epoch_crop_start(waveform.size, identifier, epoch, window_samples)
    return waveform[start : start + window_samples].copy()


def epoch_crop_start(samples: int, identifier: str, epoch: int, window_samples: int) -> int:
    if samples <= window_samples:
        return 0
    positions = samples - window_samples + 1
    strata = min(STRATA, positions)
    stratum = (_seed(identifier) + epoch) % strata
    lower = positions * stratum // strata
    upper = positions * (stratum + 1) // strata
    start = random.Random(_seed(identifier, epoch, "crop")).randrange(lower, upper)
    return start


def apply_gain(audio: np.ndarray, identifier: str, epoch: int) -> np.ndarray:
    gain_db = random.Random(_seed(identifier, epoch, "gain")).uniform(-6.0, 6.0)
    return (audio * 10 ** (gain_db / 20.0)).astype(np.float32)


def epoch_order(rows: Sequence[Mapping[str, str]], epoch: int) -> list[Mapping[str, str]]:
    ordered = list(rows)
    random.Random(_seed(epoch, "order")).shuffle(ordered)
    return ordered


def music_supervision_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """Keep only rows with justified music-present authenticity supervision."""
    return [row for row in rows if row.get("MUSIC_PRESENT") == "1" and row.get("MUSIC_FAKE") in {"0", "1"}]


def p4_starts(samples: int, window_samples: int, identifier: str, epoch: int) -> tuple[int, ...]:
    """Spread up to four distinct native windows across the complete source."""
    if samples <= window_samples:
        return (0,)
    positions = samples - window_samples + 1
    if positions <= 4:
        return tuple(range(positions))
    starts: list[int] = []
    for stratum in range(4):
        lower = positions * stratum // 4
        upper = positions * (stratum + 1) // 4
        starts.append(random.Random(_seed(identifier, epoch, stratum, "bag")).randrange(lower, upper))
    return tuple(starts)


def p4_splice(
    real: np.ndarray,
    donor: np.ndarray,
    starts: Sequence[int],
    window_samples: int,
    identifier: str,
    epoch: int,
    *,
    fake_donor: bool,
) -> tuple[np.ndarray, tuple[float, ...], float]:
    """Make one aligned real/fake or matched real/real partial edit."""
    samples = min(len(real), len(donor))
    if samples < 5 * 16_000:
        raise ValueError("P4 splice parents must both contain at least five seconds")
    duration = min(samples, random.Random(_seed(identifier, epoch, "splice-duration")).randint(5, 20) * 16_000)
    offset = random.Random(_seed(identifier, epoch, "splice-offset")).randrange(samples - duration + 1)
    output = real.copy()
    output[offset : offset + duration] = donor[offset : offset + duration]
    crossfade = min(800, duration // 2)
    for boundary, entering in ((offset, True), (offset + duration, False)):
        lower, upper = max(0, boundary - crossfade), min(samples, boundary + crossfade)
        if upper > lower:
            donor_weight = np.linspace(0.0, 1.0, upper - lower, dtype=np.float32)
            if not entering:
                donor_weight = 1.0 - donor_weight
            output[lower:upper] = (real[lower:upper] * (1.0 - donor_weight)
                                   + donor[lower:upper] * donor_weight)
    labels: list[float] = []
    for start in starts:
        end = start + window_samples
        wholly_replaced = start >= offset + crossfade and end <= offset + duration - crossfade
        wholly_real = end <= offset - crossfade or start >= offset + duration + crossfade
        labels.append(float(fake_donor) if wholly_replaced else (0.0 if wholly_real else math.nan))
    retained_fake = any(max(start, offset) < min(start + window_samples, offset + duration)
                        for start in starts)
    bag_target = float(fake_donor and retained_fake)
    return output, tuple(labels), bag_target


def _aligned(audio: np.ndarray, samples: int) -> np.ndarray:
    return np.pad(audio[:samples], (0, max(0, samples - len(audio))))


def p3_augment(
    audio: np.ndarray,
    identifier: str,
    epoch: int,
    cache_root: str,
) -> np.ndarray:
    """Apply the fixed, label-independent P3 choice and cache codec work on CPU."""
    choice = random.Random(_seed(identifier, epoch, "p3-choice")).random()
    if choice < 0.5:
        return audio.copy()
    if choice < 0.8:
        codec_index = _seed(identifier, epoch, "codec") % 6
        codec = "libmp3lame" if codec_index < 3 else "aac"
        bitrate = (64, 128, 192)[codec_index % 3]
        key = hashlib.sha256(audio.astype("<f4", copy=False).tobytes()).hexdigest()
        cache = Path(cache_root) / f"{key}-{codec}-{bitrate}.npy"
        if cache.is_file():
            return np.load(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        encoded = subprocess.run(
            [ffmpeg, "-v", "error", "-f", "f32le", "-ar", "16000", "-ac", "1",
             "-i", "-", "-f", "mp3" if codec == "libmp3lame" else "adts",
             "-c:a", codec, "-b:a", f"{bitrate}k", "-"],
            input=audio.astype("<f4", copy=False).tobytes(), check=True, capture_output=True,
        ).stdout
        decoded = subprocess.run(
            [ffmpeg, "-v", "error", "-i", "-", "-f", "f32le", "-ar", "16000",
             "-ac", "1", "-"], input=encoded, check=True, capture_output=True,
        ).stdout
        result = _aligned(np.frombuffer(decoded, dtype="<f4").copy(), len(audio))
        temporary = cache.with_suffix(".tmp.npy")
        np.save(temporary, result)
        temporary.replace(cache)
        return result
    if choice < 0.9:
        down = np.interp(
            np.linspace(0, len(audio) - 1, round(len(audio) * 0.75)),
            np.arange(len(audio)), audio,
        )
        return np.interp(
            np.linspace(0, len(down) - 1, len(audio)), np.arange(len(down)), down,
        ).astype(np.float32)
    rng = np.random.default_rng(_seed(identifier, epoch, "noise"))
    snr_db = random.Random(_seed(identifier, epoch, "snr")).uniform(20.0, 35.0)
    rms = max(float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))), 1e-8)
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    noise *= rms / (max(float(np.sqrt(np.mean(np.square(noise), dtype=np.float64))), 1e-8)
                    * 10 ** (snr_db / 20.0))
    return audio + noise


def p3_variant(identifier: str, epoch: int) -> str:
    choice = random.Random(_seed(identifier, epoch, "p3-choice")).random()
    if choice < 0.5:
        return "identity"
    if choice < 0.8:
        codec_index = _seed(identifier, epoch, "codec") % 6
        return f"{'mp3' if codec_index < 3 else 'aac'}-{(64, 128, 192)[codec_index % 3]}k"
    return "resample-16k-12k-16k" if choice < 0.9 else "noise-20-35db"
