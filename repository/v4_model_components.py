"""Small model-specific components shared by the V4 adapter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

import torch
from torch import nn
from transformers import ASTFeatureExtractor, ASTModel

from ast_multitask import load_processor

AST_HASHES: Final = {
    "config.json": "a93d525511d77e8ecc933d09674b85099815bbbb417c228a4edd655e252fb9ff",
    "preprocessor_config.json": "8d04ba5a9c6fca5d39d0de2b1fd05ecf79deb589fbba279728bbebac39934231",
    "model.safetensors": "ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d",
}
ModelKind = Literal["sonics", "ast"]


class V4ModelError(ValueError):
    """A V4 model asset, waveform, or label violates the fixed contract."""


@dataclass(frozen=True, slots=True)
class LoadedModel:
    kind: ModelKind
    model: nn.Module
    processor: ASTFeatureExtractor | None
    sample_rate: int
    window_samples: int


@dataclass(frozen=True, slots=True)
class PreparedWindows:
    values: torch.Tensor
    file_indices: torch.Tensor
    starts: tuple[tuple[int, ...], ...]
    window_targets: torch.Tensor

    def to(self, device: str | torch.device) -> PreparedWindows:
        return replace(
            self,
            values=self.values.to(device),
            file_indices=self.file_indices.to(device),
            window_targets=self.window_targets.to(device),
        )


@dataclass(frozen=True, slots=True)
class ParameterGroups:
    encoder: tuple[tuple[str, nn.Parameter], ...]
    head: tuple[tuple[str, nn.Parameter], ...]


@dataclass(frozen=True, slots=True)
class FilePredictions:
    logits: torch.Tensor
    probabilities: torch.Tensor
    starts: tuple[tuple[int, ...], ...]


class ASTMusicModel(nn.Module):
    def __init__(self, backbone: ASTModel) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.head(
            self.backbone(input_values=input_values).pooler_output
        ).squeeze(-1)


class PerExampleAmplitudeToDB(nn.Module):
    """Give torchaudio a channel axis so top_db is computed per example."""

    def __init__(self, transform: nn.Module) -> None:
        super().__init__()
        self.transform = transform

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.transform(spectrogram.unsqueeze(1)).squeeze(1)


class DisabledAugment(nn.Module):
    def forward(
        self,
        spectrogram: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return spectrogram, targets


def load_ast(
    assets: Path, device: str | torch.device
) -> tuple[nn.Module, ASTFeatureExtractor]:
    """Verify and load the pinned AST backbone plus one new music head."""
    for name, expected in AST_HASHES.items():
        path = assets / name
        if not path.is_file():
            raise V4ModelError(f"Pinned AST asset is missing: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise V4ModelError(f"Pinned AST asset hash mismatch: {name}")
    processor = load_processor(assets)
    backbone = ASTModel.from_pretrained(
        assets,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    return ASTMusicModel(backbone).to(device), processor
