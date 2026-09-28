"""Small residual heads around the unchanged compact TemporalMusic model."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from music_temporal import TemporalMusic


@dataclass(frozen=True, slots=True)
class MusicOutput:
    bag_logits: torch.Tensor
    window_logits: torch.Tensor


class StemMusicWrapper(nn.Module):
    """Expose hidden windows using only TemporalMusic's public layers."""

    def __init__(self, control: TemporalMusic) -> None:
        super().__init__()
        if not (control.simplified and control.compact and control.input_dim == 1024
                and control.softmax_output and isinstance(control.output, nn.Sequential)):
            raise ValueError("Expected compact 1024-d two-class MLP TemporalMusic control")
        self.control = control

    def encode_hidden(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != 1024:
            raise ValueError("Expected music features [batch, windows, 1024]")
        if valid.shape != features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Expected boolean valid mask [batch, windows]")
        padding = ~valid
        mask = padding[:, None, :]
        hidden = self.control.projection(
            self.control.input_norm(features.masked_fill(padding[..., None], 0)).transpose(1, 2)
        )
        hidden = F.gelu(hidden).masked_fill(mask, 0)
        for block in self.control.blocks:
            hidden = block(hidden, mask)
        return hidden.transpose(1, 2)

    def classify_hidden(self, hidden: torch.Tensor, valid: torch.Tensor) -> MusicOutput:
        window_logits = self.control.output(hidden)
        empty = ~valid.any(dim=1)
        pooled = hidden.masked_fill(~valid[..., None], -torch.inf).amax(dim=1)
        pooled = pooled.masked_fill(empty[:, None], 0)
        bag_logits = self.control.output(pooled)
        bag_logits = torch.where(empty[:, None], bag_logits.new_tensor([0.0, -100.0]), bag_logits)
        return MusicOutput(bag_logits, window_logits)

    def forward(self, stem: torch.Tensor, valid: torch.Tensor) -> MusicOutput:
        return self.classify_hidden(self.encode_hidden(stem, valid), valid)


class MusicResidual(nn.Module):
    """Aligned raw/stem residual whose zero initialization is the stem control."""

    def __init__(self, control: TemporalMusic) -> None:
        super().__init__()
        self.encoder = StemMusicWrapper(control)
        self.delta_in = nn.Linear(198, 32)
        self.delta_out = nn.Linear(32, 64)
        self.gate = nn.Linear(70, 1)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

    def forward(
        self,
        stem: torch.Tensor,
        raw: torch.Tensor,
        aux: torch.Tensor,
        valid: torch.Tensor,
    ) -> MusicOutput:
        if raw.shape != stem.shape or aux.shape != (*stem.shape[:2], 6):
            raise ValueError("Raw/stem windows and six auxiliary values must align")
        stem_hidden = self.encoder.encode_hidden(stem, valid)
        raw_hidden = self.encoder.encode_hidden(raw, valid)
        joined = torch.cat((stem_hidden, raw_hidden, raw_hidden - stem_hidden, aux), dim=-1)
        delta = self.delta_out(F.gelu(self.delta_in(joined)))
        gate = torch.sigmoid(self.gate(torch.cat((stem_hidden, aux), dim=-1)))
        return self.encoder.classify_hidden(stem_hidden + gate * delta, valid)


class VoiceResidual(nn.Module):
    """Zero-initialized correction to frozen per-window voice logits."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(1280, 32)
        self.output = nn.Linear(32, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, embeddings: torch.Tensor, baseline_logits: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.shape[:2] != baseline_logits.shape or valid.shape != baseline_logits.shape:
            raise ValueError("Voice embeddings, logits and masks must share windows")
        corrected = baseline_logits + self.output(F.gelu(self.input(embeddings))).squeeze(-1)
        corrected = corrected.masked_fill(~valid, -torch.inf)
        empty = ~valid.any(dim=1)
        bag = corrected.amax(dim=1).masked_fill(empty, -100.0)
        return bag, corrected.masked_fill(~valid, -100.0)


def make_control(seed: int) -> TemporalMusic:
    """Construct the reviewed compact MERT330 control with matched initialization."""
    torch.manual_seed(seed)
    return TemporalMusic(
        0.1,
        simplified=True,
        compact=True,
        input_dim=1024,
        softmax_output=True,
        mlp_output=True,
    )
