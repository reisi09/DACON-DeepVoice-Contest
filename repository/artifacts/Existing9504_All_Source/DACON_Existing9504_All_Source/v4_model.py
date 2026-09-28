"""Full-fine-tuning adapters for the pinned V4 music encoders."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final, assert_never

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional

from recovery_sonics import SonicsPredictor
from v4_model_components import (
    DisabledAugment,
    FilePredictions,
    LoadedModel,
    ModelKind,
    ParameterGroups,
    PerExampleAmplitudeToDB,
    PreparedWindows,
    V4ModelError,
    load_ast,
)

SONICS_SAMPLES: Final = 80_000
AST_SAMPLES: Final = 160_000
AST_KALDI_MIN_SAMPLES: Final = 400


def load_model(
    kind: ModelKind,
    assets: Path,
    device: str | torch.device,
    *,
    state_path: Path | None = None,
) -> LoadedModel:
    """Load a verified original, then optionally overlay one strict trained state."""
    match kind:
        case "sonics":
            model = SonicsPredictor(assets, str(device)).model
            feature_extractor = model.ft_extractor
            feature_extractor.amplitude_to_db = PerExampleAmplitudeToDB(
                feature_extractor.amplitude_to_db,
            )
            model.augment = DisabledAugment()
            processor = None
            window_samples = SONICS_SAMPLES
        case "ast":
            model, processor = load_ast(assets, device)
            window_samples = AST_SAMPLES
        case unreachable:
            assert_never(unreachable)
    if state_path is not None:
        model.load_state_dict(load_file(state_path, device="cpu"), strict=True)
    model.requires_grad_(True)
    return LoadedModel(kind, model, processor, 16_000, window_samples)


def parameter_groups(loaded: LoadedModel) -> ParameterGroups:
    """Separate the existing/new classifier from every encoder parameter."""
    head_prefix = "classifier." if loaded.kind == "sonics" else "head."
    named = tuple(loaded.model.named_parameters())
    return ParameterGroups(
        encoder=tuple(item for item in named if not item[0].startswith(head_prefix)),
        head=tuple(item for item in named if item[0].startswith(head_prefix)),
    )


def _validated_audio(audio: np.ndarray) -> np.ndarray:
    waveform = np.asarray(audio)
    if waveform.ndim != 1 or waveform.size == 0:
        raise V4ModelError("Audio must be nonempty and one-dimensional")
    try:
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise V4ModelError("Audio must be convertible to float32") from error
    if not np.isfinite(waveform).all():
        raise V4ModelError("Audio must contain only finite values")
    return waveform


def _native_starts(samples: int, kind: ModelKind) -> tuple[int, ...]:
    match kind:
        case "sonics":
            return tuple(SonicsPredictor._window_starts(samples))
        case "ast":
            return tuple(range(0, samples, AST_SAMPLES))
        case unreachable:
            assert_never(unreachable)


def prepare_windows(
    loaded: LoadedModel,
    audios: Sequence[np.ndarray],
    *,
    starts: Sequence[Sequence[int]] | None = None,
    window_targets: Sequence[Sequence[float]] | None = None,
) -> PreparedWindows:
    """Prepare each file on its native deterministic grid without cross-file state."""
    if not audios:
        raise V4ModelError("At least one audio file is required")
    waveforms = tuple(_validated_audio(audio) for audio in audios)
    if starts is None:
        resolved_starts = tuple(
            _native_starts(audio.size, loaded.kind) for audio in waveforms
        )
    else:
        if len(starts) != len(waveforms):
            raise V4ModelError("Explicit starts must contain one grid per file")
        resolved_starts = tuple(tuple(file_starts) for file_starts in starts)
        for audio, file_starts in zip(waveforms, resolved_starts, strict=True):
            last_full_start = max(0, audio.size - loaded.window_samples)
            if (
                not 1 <= len(file_starts) <= 4
                or len(set(file_starts)) != len(file_starts)
                or any(start < 0 or start > last_full_start for start in file_starts)
            ):
                raise V4ModelError(
                    "Explicit starts must be 1-4 distinct full-window starts"
                )
    if window_targets is not None and len(window_targets) != len(waveforms):
        raise V4ModelError("Window labels must contain one grid per file")
    windows: list[np.ndarray] = []
    indices: list[int] = []
    labels: list[float] = []
    for file_index, (audio, file_starts) in enumerate(
        zip(waveforms, resolved_starts, strict=True)
    ):
        supplied = None if window_targets is None else tuple(window_targets[file_index])
        if supplied is not None and len(supplied) != len(file_starts):
            raise V4ModelError("Window labels must match the native window grid")
        for window_index, start in enumerate(file_starts):
            window = audio[start : start + loaded.window_samples]
            windows.append(window)
            indices.append(file_index)
            labels.append(math.nan if supplied is None else supplied[window_index])
    if loaded.kind == "sonics":
        values = np.zeros((len(windows), loaded.window_samples), dtype=np.float32)
        for index, window in enumerate(windows):
            values[index, : window.size] = window
        tensor = torch.from_numpy(values)
    else:
        if loaded.processor is None:
            raise V4ModelError("AST processor is unavailable")
        processor_windows = [
            np.pad(window, (0, max(0, AST_KALDI_MIN_SAMPLES - window.size)))
            for window in windows
        ]
        tensor = loaded.processor(
            processor_windows,
            sampling_rate=loaded.sample_rate,
            return_tensors="pt",
        )["input_values"].float()
    return PreparedWindows(
        tensor,
        torch.tensor(indices, dtype=torch.long),
        resolved_starts,
        torch.tensor(labels, dtype=torch.float32),
    )


def window_logits(loaded: LoadedModel, prepared: PreparedWindows) -> torch.Tensor:
    """Return one fake-music logit for every prepared native window."""
    device = next(loaded.model.parameters()).device
    logits = loaded.model(prepared.values.to(device))
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 1 or logits.shape[0] != prepared.file_indices.shape[0]:
        raise V4ModelError("Model must return one logit per native window")
    return logits


def pool_bag_logits(
    logits: torch.Tensor,
    file_indices: torch.Tensor,
    file_count: int,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Apply fixed normalized log-sum-exp independently within each file."""
    if (
        logits.ndim != 1
        or file_indices.shape != logits.shape
        or file_count < 1
        or temperature <= 0
    ):
        raise V4ModelError("Bag pooling input contract mismatch")
    pooled: list[torch.Tensor] = []
    for file_index in range(file_count):
        selected = logits[file_indices.to(logits.device) == file_index]
        if selected.numel() == 0:
            raise V4ModelError("Every file must contain at least one window")
        pooled.append(
            torch.logsumexp(temperature * selected, dim=0) / temperature
            - math.log(selected.numel()) / temperature
        )
    return torch.stack(pooled)


def _known_binary(values: torch.Tensor, name: str) -> torch.Tensor:
    known = torch.isfinite(values)
    if torch.any(known & (values != 0) & (values != 1)):
        raise V4ModelError(f"{name} must contain binary values or NaN")
    return known


def music_loss(
    logits: torch.Tensor,
    prepared: PreparedWindows,
    targets: torch.Tensor,
    *,
    p4: bool,
) -> torch.Tensor:
    """Compute ordinary window BCE or P4 bag BCE plus known-segment BCE."""
    if targets.ndim != 1 or len(targets) != len(prepared.starts):
        raise V4ModelError("Music targets must contain one value per file")
    targets = targets.to(logits.device)
    known_bags = _known_binary(targets, "Music targets")
    if not p4:
        window_targets = targets[prepared.file_indices.to(logits.device)]
        known_windows = torch.isfinite(window_targets)
        if not torch.any(known_windows):
            raise V4ModelError("Batch has no known music labels")
        return functional.binary_cross_entropy_with_logits(
            logits[known_windows], window_targets[known_windows]
        )
    losses: list[torch.Tensor] = []
    pooled = pool_bag_logits(logits, prepared.file_indices, len(targets))
    if torch.any(known_bags):
        losses.append(
            functional.binary_cross_entropy_with_logits(
                pooled[known_bags], targets[known_bags]
            )
        )
    segment_targets = prepared.window_targets.to(logits.device)
    known_segments = _known_binary(segment_targets, "Window targets")
    if torch.any(known_segments):
        losses.append(
            0.5
            * functional.binary_cross_entropy_with_logits(
                logits[known_segments],
                segment_targets[known_segments],
            )
        )
    if not losses:
        raise V4ModelError("Batch has no known bag or segment labels")
    return torch.stack(losses).sum()


def predict_files(
    loaded: LoadedModel, audios: Sequence[np.ndarray], *, p4: bool
) -> FilePredictions:
    """Predict full files with their native grid and fixed requested pooling."""
    prepared = prepare_windows(loaded, audios)
    with torch.inference_mode():
        windows = window_logits(loaded, prepared)
        if p4:
            logits = pool_bag_logits(windows, prepared.file_indices, len(audios))
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.stack(
                [
                    torch.sigmoid(
                        windows[prepared.file_indices.to(windows.device) == index]
                    ).mean()
                    for index in range(len(audios))
                ]
            )
            logits = torch.logit(probabilities)
    return FilePredictions(logits.cpu(), probabilities.cpu(), prepared.starts)
