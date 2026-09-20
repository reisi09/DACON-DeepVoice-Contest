"""Fixed CPU spectral representation for E4 forensic residuals."""

from __future__ import annotations

from typing import Final

import numpy as np

SAMPLE_RATE: Final = 16_000
N_FFT: Final = 2_048
RESIDUAL_BINS: Final = 384
BAND_BINS: Final = 64
FLOOR: Final = 1e-12


class SpectrumError(RuntimeError):
    """A waveform cannot produce the fixed spectral representation."""


def _pool_axis(values: np.ndarray, bins: int) -> np.ndarray:
    edges = np.linspace(0, values.shape[1], bins + 1, dtype=np.int64)
    return np.stack(
        [values[:, edges[index] : edges[index + 1]].mean(axis=1) for index in range(bins)],
        axis=1,
    )


def spectrum_features(windows: np.ndarray) -> np.ndarray:
    """Return 384 local-trend residual bins plus 64 coarse log-power bands."""
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < N_FFT:
        raise SpectrumError("Spectral input must be a batch of windows at least 2048 samples long")
    frame_count = 1 + (values.shape[1] - N_FFT) // (N_FFT // 2)
    frames = np.stack(
        [values[:, index * (N_FFT // 2) : index * (N_FFT // 2) + N_FFT] for index in range(frame_count)],
        axis=1,
    )
    window = np.hanning(N_FFT).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    log_power = np.log(np.maximum(power[:, :, (frequencies >= 1_000) & (frequencies <= 7_000)].mean(axis=1), FLOOR))
    padded = np.pad(log_power, ((0, 0), (8, 8)), mode="edge")
    trend = np.stack([padded[:, index : index + 17].mean(axis=1) for index in range(log_power.shape[1])], axis=1)
    residual = _pool_axis(log_power - trend, RESIDUAL_BINS)
    bands = _pool_axis(log_power, BAND_BINS)
    result = np.concatenate((residual, bands), axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise SpectrumError("Spectral representation contains non-finite values")
    return result
