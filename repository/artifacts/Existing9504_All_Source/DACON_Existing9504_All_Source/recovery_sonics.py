"""Pinned, file-local SONICS music-authenticity inference."""

from __future__ import annotations

import hashlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import torch

SAMPLE_RATE: Final = 16_000
WINDOW_SAMPLES: Final = SAMPLE_RATE * 5
CONFIG_SHA256: Final = "754bc15b3f914afaf4e002e50bfaa5e81055c9fb0e2b76408e52fd88f09f7def"
WEIGHTS_SHA256: Final = "11c384cea651c56028d6f02e04aa467c2099cf1102687353fe4dd9504c4db868"


class SonicsPrediction(TypedDict):
    mean_probability: float
    window_probabilities: list[float]
    starts: list[int]
    covered_samples: int
    seconds: float
    audio_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SonicsContractError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _sha256(path: Path) -> str:
    """Return one asset's full SHA-256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verified_assets(assets: Path) -> None:
    """Reject a missing or altered local asset before checkpoint deserialization."""
    for name, expected in (("config.json", CONFIG_SHA256), ("pytorch_model.bin", WEIGHTS_SHA256)):
        path = assets / name
        if not path.is_file():
            raise SonicsContractError(f"SONICS local asset is missing: {path}")
        if _sha256(path) != expected:
            raise SonicsContractError(f"SONICS {name} SHA-256 mismatch")


class SonicsPredictor:
    """Run the pinned classifier serially over one 16 kHz mono waveform."""

    model: torch.nn.Module
    device: torch.device

    def __init__(self, assets: Path, device: str) -> None:
        _verified_assets(assets)
        vendor_root = assets.parent / "sonics_vendor"
        if not (vendor_root / "sonics").is_dir():
            raise SonicsContractError(f"Pinned SONICS source is missing beside assets: {vendor_root}")
        sys.path.insert(0, str(vendor_root))
        from sonics.models.hf_model import HFAudioClassifier

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        self.model = HFAudioClassifier.from_pretrained(str(assets), map_location="cpu", strict=True).to(self.device).eval()

    def predict(self, audio: np.ndarray) -> SonicsPrediction:
        """Return a mean of serial 5-second window probabilities for one file."""
        waveform = np.asarray(audio)
        if waveform.ndim != 1 or waveform.size == 0:
            raise SonicsContractError("SONICS audio must be nonempty and one-dimensional")
        try:
            waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise SonicsContractError("SONICS audio must be convertible to float32") from error
        if not np.isfinite(waveform).all():
            raise SonicsContractError("SONICS audio must contain only finite values")
        starts = self._window_starts(waveform.size)
        started = time.perf_counter()
        probabilities = [self._window_probability(waveform[start : start + WINDOW_SAMPLES]) for start in starts]
        elapsed = time.perf_counter() - started
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
            raise SonicsContractError("SONICS model returned a non-finite probability")
        covered_samples = self._covered_samples(starts, waveform.size)
        if covered_samples != waveform.size:
            raise SonicsContractError("SONICS windows must cover every original sample")
        audio_seconds = waveform.size / SAMPLE_RATE
        return {
            "mean_probability": float(np.mean(probabilities)),
            "window_probabilities": probabilities,
            "starts": starts,
            "covered_samples": covered_samples,
            "seconds": audio_seconds,
            "audio_seconds": audio_seconds,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _window_starts(samples: int) -> list[int]:
        """Cover the tail with a final full window; only short files are padded."""
        if samples <= WINDOW_SAMPLES:
            return [0]
        final_start = samples - WINDOW_SAMPLES
        starts = list(range(0, final_start + 1, WINDOW_SAMPLES))
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    @staticmethod
    def _covered_samples(starts: list[int], samples: int) -> int:
        """Measure the covered union, rejecting a gap rather than reporting input length."""
        covered_end = 0
        for start in starts:
            if start > covered_end:
                return covered_end
            covered_end = max(covered_end, min(samples, start + WINDOW_SAMPLES))
        return covered_end

    def _window_probability(self, samples: np.ndarray) -> float:
        """Infer exactly one window, preserving per-example feature normalization."""
        padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        padded[: samples.size] = samples
        with torch.no_grad():
            logits = self.model(torch.from_numpy(padded).unsqueeze(0).to(self.device))
        if logits.numel() != 1:
            raise SonicsContractError("SONICS model must return exactly one logit per window")
        logit = float(logits.item())
        if not math.isfinite(logit):
            raise SonicsContractError("SONICS model returned a non-finite logit")
        return float(torch.sigmoid(torch.tensor(logit)).item())
