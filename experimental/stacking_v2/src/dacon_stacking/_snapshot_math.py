"""Verbatim numerical definitions extracted from the input snapshot, not legacy.
See configs/snapshot_symbols.json for source hashes and original line ranges.
Only imports/constants are adapted to make this frozen subset CPU-importable.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping
import numpy as np
import torch
from torch import nn
from ._snapshot_scoring import PREDICTION_COLUMNS
WINDOW = SEGMENT_SAMPLES = 64600
HOP = 32300
SEEDS = (41,42,43)
SILENCE_RMS = 1e-5


class FrozenHead(nn.Module):
    """Small matched MLP used by every frozen-feature comparison."""

    def __init__(self, width: int, architecture: str = "mlp") -> None:
        super().__init__()
        hidden = min(128, max(16, width // 4))
        self.network = (nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1))
                        if architecture == "mlp" else nn.Sequential(nn.Identity(), nn.Identity(), nn.Linear(width, 1)))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network[:2](features)
        return self.network[2](hidden).squeeze(-1), hidden


class LocalizationContractError(ValueError):
    """Features, truth, or masks violate the localization comparison contract."""


@dataclass(frozen=True, slots=True)
class BatchOutput:
    """Binary logits; every supplied boolean mask uses True for valid positions."""

    file_logits: torch.Tensor
    window_logits: torch.Tensor | None = None
    window_mask: torch.Tensor | None = None
    patch_logits: torch.Tensor | None = None
    patch_mask: torch.Tensor | None = None


def _check_mask(values: torch.Tensor, mask: torch.Tensor) -> None:
    if values.ndim != 2 or mask.shape != values.shape or mask.dtype != torch.bool:
        raise LocalizationContractError("Expected aligned [batch, positions] values and boolean valid mask")


def masked_logmeanexp(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Fixed-temperature (1) MIL; duplicating a whole bag cannot change its score."""
    _check_mask(logits, valid_mask)
    count = valid_mask.sum(-1)
    if torch.any(count == 0):
        raise LocalizationContractError("Every bag must contain at least one valid position")
    return logits.float().masked_fill(~valid_mask, -torch.inf).logsumexp(-1) - count.float().log()


class FrozenMeanMLP(nn.Module):
    """Raw-window mean -> train-file-mean standardization -> existing MERT12 MLP."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        if (mean.ndim != 1 or not mean.numel() or scale.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
                or torch.any(scale <= 0)):
            raise LocalizationContractError("Supply finite fitted mean and positive scale vectors")
        self.register_buffer("mean", mean.detach().float().clone())
        self.register_buffer("scale", scale.detach().float().clone())
        self.head = FrozenHead(mean.numel(), "mlp")

    def _features(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.mean.numel():
            raise LocalizationContractError("Expected [files, windows, fitted feature width] features")
        _check_mask(features[..., 0], valid_mask)
        if torch.any(~valid_mask.any(-1)):
            raise LocalizationContractError("Every file must contain a valid window")
        features = features.masked_fill(~valid_mask[..., None], 0)
        if not torch.isfinite(features).all():
            raise LocalizationContractError("Valid MERT windows must contain finite features")
        return features

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> BatchOutput:
        features = self._features(features, valid_mask)
        pooled = features.sum(1) / valid_mask.sum(1, keepdim=True)
        file_logits, _ = self.head((pooled - self.mean) / self.scale)
        window_logits, _ = self.head((features - self.mean) / self.scale)
        return BatchOutput(file_logits, window_logits.masked_fill(~valid_mask, 0), valid_mask)


class WindowMIL(FrozenMeanMLP):
    """Same scaler/MLP on windows, aggregated by log-mean-exp with no tuning knob."""

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> BatchOutput:
        features = self._features(features, valid_mask)
        logits, _ = self.head((features - self.mean) / self.scale)
        return BatchOutput(masked_logmeanexp(logits, valid_mask),
                           logits.masked_fill(~valid_mask, 0), valid_mask)


class SubmissionError(ValueError):
    """The packaged runtime, checkpoint, or prediction violates its fixed contract."""


def window_layout(samples: int) -> np.ndarray:
    """Return half-hop starts plus one deduplicated exact tail."""
    if samples <= 0:
        raise SubmissionError("Decoded audio is empty")
    last = max(0, samples - WINDOW)
    starts = list(range(0, last + 1, HOP))
    if starts[-1] != last:
        starts.append(last)
    return np.asarray(starts, dtype=np.int64)


def windows(audio: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Right-zero-pad each exact 16 kHz window."""
    result = np.zeros((len(starts), WINDOW), dtype=np.float32)
    for index, start in enumerate(starts):
        piece = audio[int(start): int(start) + WINDOW]
        result[index, :len(piece)] = piece
    return result


def get_segment_starts(audio_length):
    if audio_length <= SEGMENT_SAMPLES:
        return [0]

    last_start = audio_length - SEGMENT_SAMPLES
    starts = list(range(0, last_start + 1, SEGMENT_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(audio, start):
    if audio.size < SEGMENT_SAMPLES:
        repeat_count = SEGMENT_SAMPLES // audio.size + 1
        audio = np.tile(audio, repeat_count)
        return audio[:SEGMENT_SAMPLES].astype(np.float32)

    end = start + SEGMENT_SAMPLES
    return audio[start:end].astype(np.float32, copy=False)


def calculate_rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def load_heads(root: Path, device):
    music, voice = [], []
    fake_index = None
    for seed in SEEDS:
        for route, collection in (('music', music), ('voice', voice)):
            record = torch.load(root/f'{route}_{seed}.pt', map_location='cpu', weights_only=True)
            if record['route'] != route or record['seed'] != seed:
                raise SubmissionError('Checkpoint route or seed mismatch')
            state = record['model']
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise SubmissionError('Nonfinite checkpoint')
            model = WindowMIL(state['mean'], state['scale']) if route == 'music' else nn.Linear(1280, 2)
            model.load_state_dict(state, strict=True)
            collection.append(model.to(device).eval().requires_grad_(False))
            if fake_index is not None and record['fake_label_index'] != fake_index:
                raise SubmissionError('Inconsistent fake-label index')
            fake_index = record['fake_label_index']
    return music, voice, fake_index


@torch.no_grad()
def score_arrays(arrays: Mapping[str, np.ndarray], heads, device: str) -> dict[str, Any]:
    """Five scores (+ per-seed extras) for one file's cached/recomputed feature arrays."""
    music_heads, voice_heads, fake_index, _ = heads
    features = torch.from_numpy(np.asarray(arrays["mert_windows"])).to(device)
    embeddings = torch.from_numpy(np.asarray(arrays["voice_embeddings"])).to(device)
    presence = np.asarray(arrays["presence"]).astype(np.float64)
    silent = bool(arrays["voice_silent"])
    music_scores, voice_scores = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for head in music_heads:
            logits, _ = head.head((features - head.mean) / head.scale)
            music_scores.append(float((logits.float().logsumexp(0) - np.log(len(features))).sigmoid()))
        for head in voice_heads:
            voice_scores.append(0. if silent else float(head(embeddings).float().softmax(-1)[:, fake_index].max()))
    music, voice = float(np.mean(music_scores)), float(np.mean(voice_scores))
    values = [max(voice * presence[0], music * presence[1]), voice, music, float(presence[0]), float(presence[1])]
    record: dict[str, Any] = dict(zip(PREDICTION_COLUMNS, values, strict=True))
    record.update({f"MUSIC_FAKE_PROB_seed{seed}": value for seed, value in zip(SEEDS, music_scores)})
    record.update({f"VOICE_FAKE_PROB_seed{seed}": value for seed, value in zip(SEEDS, voice_scores)})
    record["VOICE_FAKE_PROB_df_arena_original_fc5"] = float(arrays["voice_original"])
    record["voice_stem_silent"] = silent
    return record
