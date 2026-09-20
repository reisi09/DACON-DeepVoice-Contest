"""Learned per-file temporal fusion with joint partial-label likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn
from torch.nn import functional

OUTPUT_NAMES: Final = (
    "FILE_FAKE",
    "VOICE_FAKE",
    "MUSIC_FAKE",
    "VOICE_PRESENT",
    "MUSIC_PRESENT",
)
STATE_COUNT: Final = 9


class FusionContractError(ValueError):
    """A temporal-fusion tensor violates the public shape or label contract."""


@dataclass(frozen=True, slots=True)
class FusionOutput:
    """Five ordered probabilities and the logits needed for a stable loss."""

    probabilities: torch.Tensor
    logits: torch.Tensor


def joint_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Map voice x music absent/real/fake logits to five output probabilities."""
    if logits.ndim != 2 or logits.shape[1] != STATE_COUNT:
        raise FusionContractError("Joint logits must have shape [batch, 9]")
    states = logits.softmax(dim=-1).reshape(-1, 3, 3)
    voice_present = states[:, 1:, :].sum(dim=(1, 2))
    music_present = states[:, :, 1:].sum(dim=(1, 2))
    voice_fake_mass = states[:, 2, :].sum(dim=1)
    music_fake_mass = states[:, :, 2].sum(dim=1)
    file_fake = voice_fake_mass + states[:, :2, 2].sum(dim=1)
    tiny = torch.finfo(states.dtype).tiny
    return torch.stack(
        (
            file_fake,
            voice_fake_mass / voice_present.clamp_min(tiny),
            music_fake_mass / music_present.clamp_min(tiny),
            voice_present,
            music_present,
        ),
        dim=1,
    )


def _known_binary(labels: torch.Tensor) -> torch.Tensor:
    known = ~torch.isnan(labels)
    if torch.any(known & (labels != 0) & (labels != 1)):
        raise FusionContractError("Labels must contain only 0, 1, or NaN")
    return known


def joint_partial_label_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return negative log likelihood after marginalizing every unknown label."""
    if logits.ndim != 2 or logits.shape[1] != STATE_COUNT:
        raise FusionContractError("Joint logits must have shape [batch, 9]")
    if labels.shape != (logits.shape[0], len(OUTPUT_NAMES)):
        raise FusionContractError("Labels must have shape [batch, 5]")
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    known = _known_binary(labels)
    voice = torch.arange(3, device=logits.device).repeat_interleave(3)
    music = torch.arange(3, device=logits.device).repeat(3)
    voice_fake = voice == 2
    music_fake = music == 2
    compatible = torch.ones_like(logits, dtype=torch.bool)

    file_target = labels[:, 0, None].bool()
    compatible &= ~known[:, 0, None] | ((voice_fake | music_fake) == file_target)
    compatible &= ~known[:, 3, None] | ((voice > 0) == labels[:, 3, None].bool())
    compatible &= ~known[:, 4, None] | ((music > 0) == labels[:, 4, None].bool())
    compatible &= ~known[:, 1, None] | (voice == 0) | (
        voice_fake == labels[:, 1, None].bool()
    )
    compatible &= ~known[:, 2, None] | (music == 0) | (
        music_fake == labels[:, 2, None].bool()
    )
    if torch.any(~compatible.any(dim=1)):
        raise FusionContractError("Known labels have no compatible joint state")
    return (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(logits.masked_fill(~compatible, -torch.inf), dim=1)
    ).mean()


def balanced_known_label_bce(
    probabilities: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Compute an optional class-balanced BCE over known, applicable labels."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(OUTPUT_NAMES):
        raise FusionContractError("Probabilities must have shape [batch, 5]")
    if labels.shape != probabilities.shape:
        raise FusionContractError("Labels must match probability shape")
    labels = labels.to(device=probabilities.device, dtype=probabilities.dtype)
    known = _known_binary(labels)
    known[:, 1] &= ~(known[:, 3] & (labels[:, 3] == 0))
    known[:, 2] &= ~(known[:, 4] & (labels[:, 4] == 0))
    losses: list[torch.Tensor] = []
    for column in range(len(OUTPUT_NAMES)):
        selected = known[:, column]
        if not torch.any(selected):
            continue
        target = labels[selected, column]
        item_loss = functional.binary_cross_entropy(
            probabilities[selected, column].clamp(1e-7, 1 - 1e-7),
            target,
            reduction="none",
        )
        positive = target == 1
        negative = ~positive
        if torch.any(positive) and torch.any(negative):
            losses.append((item_loss[positive].mean() + item_loss[negative].mean()) / 2)
        else:
            losses.append(item_loss.mean())
    return torch.stack(losses).mean() if losses else probabilities.sum() * 0


class TemporalFusion(nn.Module):
    """Fuse ordered windows independently within each file."""

    frequencies: torch.Tensor

    def __init__(self, input_dim: int, joint: bool) -> None:
        super().__init__()
        if input_dim < 1:
            raise FusionContractError("input_dim must be positive")
        self.input_dim = input_dim
        self.joint = joint
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, 256)
        self.frequencies = torch.logspace(0, -3, 128)
        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.pool_score = nn.Linear(256, 1)
        self.output = nn.Linear(256, STATE_COUNT if joint else 4)

    def _encoding(self, values: torch.Tensor) -> torch.Tensor:
        phase = values.unsqueeze(-1) * self.frequencies.to(values)
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
        times: torch.Tensor,
    ) -> FusionOutput:
        """Return FILE/VOICE/MUSIC fake then VOICE/MUSIC presence probabilities."""
        if features.ndim != 3 or features.shape[2] != self.input_dim:
            raise FusionContractError("Features must have shape [batch, windows, input_dim]")
        if padding_mask.shape != features.shape[:2] or padding_mask.dtype != torch.bool:
            raise FusionContractError("Padding mask must be bool [batch, windows]")
        if times.shape != features.shape[:2]:
            raise FusionContractError("Times must have shape [batch, windows]")
        if torch.any(padding_mask.all(dim=1)):
            raise FusionContractError("Every file must contain at least one real window")
        times = times.to(device=features.device, dtype=features.dtype)
        positions = torch.arange(features.shape[1], device=features.device).to(features)
        hidden = self.input_projection(self.input_norm(features))
        hidden = hidden + self._encoding(times) + self._encoding(positions)[None, :, :]
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        scores = self.pool_score(hidden).squeeze(-1).masked_fill(padding_mask, -torch.inf)
        pooled = (scores.softmax(dim=1).unsqueeze(-1) * hidden).sum(dim=1)
        logits = self.output(pooled)
        if self.joint:
            probabilities = joint_probabilities(logits)
        else:
            component = logits.sigmoid()
            voice_fake, music_fake, voice_present, music_present = component.unbind(dim=1)
            file_fake = torch.maximum(
                voice_present * voice_fake, music_present * music_fake
            )
            probabilities = torch.stack(
                (file_fake, voice_fake, music_fake, voice_present, music_present), dim=1
            )
        return FusionOutput(probabilities=probabilities, logits=logits)
