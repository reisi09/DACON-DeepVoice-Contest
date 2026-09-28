"""Matched MERT heads and a last-block AST head for known-interval localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import ASTModel

from ast_multitask import load_processor
from eight_heads import FrozenHead


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


def frame_targets_to_patches(
    frame_targets: torch.Tensor, frame_mask: torch.Tensor, *, patch_size: int = 16,
    time_stride: int = 10,
) -> torch.Tensor:
    """Mean fake occupancy per receptive patch; any unknown real frame makes it NaN.

    Truth is [0, 1], with NaN/-1 for unknown. Right padding does not enter the
    denominator. Frame labels must come from known intervals, never file labels.
    """
    _check_mask(frame_targets, frame_mask)
    if patch_size < 1 or time_stride < 1 or frame_targets.shape[1] < patch_size:
        raise LocalizationContractError("Patch geometry cannot cover the provided frames")
    known = torch.isfinite(frame_targets) & (frame_targets >= 0) & (frame_targets <= 1)
    invalid = ~known & ~torch.isnan(frame_targets) & (frame_targets != -1)
    if (invalid & frame_mask).any():
        raise LocalizationContractError("Frame truth must be [0,1], NaN, or -1")
    valid = frame_mask.unfold(1, patch_size, time_stride)
    observed = known.unfold(1, patch_size, time_stride)
    values = frame_targets.masked_fill(~(known & frame_mask), 0).unfold(1, patch_size, time_stride)
    count = valid.sum(-1)
    eligible = (count > 0) & (observed | ~valid).all(-1)
    fraction = values.sum(-1) / count.clamp_min(1)
    return fraction.masked_fill(~eligible, torch.nan)


def masked_localization_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor,
) -> torch.Tensor:
    """BCE on known valid truth only; an unlabeled batch contributes zero gradient."""
    _check_mask(logits, valid_mask)
    if targets.shape != logits.shape:
        raise LocalizationContractError("Local truth must align with local logits")
    known = torch.isfinite(targets) & (targets >= 0) & (targets <= 1)
    if (valid_mask & ~known & ~torch.isnan(targets) & (targets != -1)).any():
        raise LocalizationContractError("Local truth must be [0,1], NaN, or -1")
    selected = known & valid_mask
    if not selected.any():
        return logits.masked_fill(~valid_mask, 0).sum() * 0
    return F.binary_cross_entropy_with_logits(logits[selected].float(), targets[selected].float())


class ASTLocalizer(nn.Module):
    """Frequency-pooled AST patch head; only last block, final norm, and head train.

    Native AST attention still sees the processor's padded spectrogram. Masks
    remove padding-only patches from pooling/loss, not from self-attention.
    """

    def __init__(self, backbone: ASTModel) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, 1)
        self.backbone.requires_grad_(False)
        self.backbone.encoder.layer[-1].requires_grad_(True)
        self.backbone.layernorm.requires_grad_(True)
        self.train()

    @classmethod
    def from_pretrained(cls, local_dir: Path) -> ASTLocalizer:
        """Reuse the pinned processor and AudioSet backbone, without network access."""
        load_processor(local_dir)
        return cls(ASTModel.from_pretrained(local_dir, local_files_only=True,
                                            attn_implementation="sdpa"))

    def train(self, mode: bool = True) -> ASTLocalizer:
        super().train(mode)
        self.backbone.eval()
        self.backbone.encoder.layer[-1].train(mode)
        return self

    @torch.no_grad()
    def encode_prefix(self, input_values: torch.Tensor) -> torch.Tensor:
        """Cache frozen embeddings and all blocks before the last in current dtype.

        The runner owns persistence/dtype provenance. Casting this result to fp16
        is lossy; only replay with unchanged dtype has exact parity.
        """
        config = self.backbone.config
        if input_values.ndim != 3 or input_values.shape[1:] != (config.max_length, config.num_mel_bins):
            raise LocalizationContractError("Input spectrogram must match the AST checkpoint dimensions")
        hidden = self.backbone.embeddings(input_values)
        for layer in self.backbone.encoder.layer[:-1]:
            hidden = layer(hidden)
        return hidden.detach()

    def _patch_grid(self, sequence: torch.Tensor) -> tuple[int, int]:
        """Reject stale native/cached tokens before applying the AST suffix."""
        config = self.backbone.config
        frequency = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
        time = (config.max_length - config.patch_size) // config.time_stride + 1
        if sequence.ndim != 3 or sequence.shape[1:] != (2 + frequency * time, config.hidden_size):
            raise LocalizationContractError("AST token sequence does not match native patch geometry")
        return frequency, time

    def temporal_embeddings(self, sequence: torch.Tensor) -> torch.Tensor:
        """Remove CLS/distillation tokens; Conv2d flattening is frequency-major."""
        frequency, time = self._patch_grid(sequence)
        return sequence[:, 2:].reshape(sequence.shape[0], frequency, time, sequence.shape[-1]).mean(1)

    def forward_encoded(self, hidden: torch.Tensor, frame_mask: torch.Tensor) -> BatchOutput:
        """Run the trainable suffix on a detached frozen-prefix cache."""
        config = self.backbone.config
        self._patch_grid(hidden)
        if (frame_mask.shape != (hidden.shape[0], config.max_length)
                or frame_mask.dtype != torch.bool or not frame_mask.any(1).all()):
            raise LocalizationContractError("Supply nonempty boolean valid-frame masks per AST window")
        hidden = hidden.detach().to(dtype=self.backbone.layernorm.weight.dtype)
        if hidden.device.type == "mps" and torch.is_grad_enabled():
            hidden.requires_grad_(True)
        sequence = self.backbone.encoder.layer[-1](hidden)
        sequence = self.backbone.layernorm(sequence)
        logits = self.head(self.temporal_embeddings(sequence)).squeeze(-1)
        patch_mask = frame_mask.unfold(1, config.patch_size, config.time_stride).any(-1)
        file_logits = masked_logmeanexp(logits, patch_mask)
        return BatchOutput(file_logits, file_logits[:, None],
                           torch.ones_like(file_logits[:, None], dtype=torch.bool),
                           logits.masked_fill(~patch_mask, 0), patch_mask)

    def forward(self, input_values: torch.Tensor, frame_mask: torch.Tensor) -> BatchOutput:
        return self.forward_encoded(self.encode_prefix(input_values), frame_mask)
