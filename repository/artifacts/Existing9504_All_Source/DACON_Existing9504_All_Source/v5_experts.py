"""Trainable V5 adapters around the pinned DACON baseline experts."""

from __future__ import annotations

import csv
import importlib
import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, assert_never

import torch
from torch import nn
from torch.nn import functional
from torchaudio.functional import resample

ExpertKind = Literal["voice", "music", "presence"]
VOICE_SAMPLES: Final = 64_600
MUSIC_SAMPLES: Final = 80_000
PRESENCE_SAMPLES: Final = 64_600


class V5ExpertError(ValueError):
    """A V5 expert asset or input violates the adapter contract."""


@dataclass(frozen=True, slots=True)
class ExpertParameterGroups:
    """Optimizer-owned encoder/head parameters and intentional exclusions."""

    encoder: tuple[tuple[str, nn.Parameter], ...]
    head: tuple[tuple[str, nn.Parameter], ...]
    excluded: tuple[str, ...]


def _validate_waveforms(waveforms: torch.Tensor, samples: int) -> None:
    if waveforms.ndim != 2 or waveforms.shape[1] != samples:
        raise V5ExpertError(f"Expected float waveforms shaped [B, {samples}]")
    if not waveforms.is_floating_point() or not torch.isfinite(waveforms).all():
        raise V5ExpertError("Waveforms must contain finite floating-point samples")


class VoiceExpert(nn.Module):
    """Batch-correct DF-Arena wrapper exposing its 1280-dimensional seam."""

    embedding_dim: Final = 1280

    def __init__(self, wrapper: nn.Module, window_samples: int = VOICE_SAMPLES) -> None:
        super().__init__()
        self.wrapper = wrapper.requires_grad_(True)
        self.window_samples = window_samples
        backbone = self.wrapper.get_submodule("backbone")
        ssl_model = backbone.get_submodule("ssl_model")
        checkpointing = getattr(ssl_model, "gradient_checkpointing_enable", None)
        if checkpointing is not None:
            checkpointing()
        children = dict(ssl_model.named_children())
        if "pooler" in children:
            children["pooler"].requires_grad_(False)
        masked_spec_embed = getattr(ssl_model, "masked_spec_embed", None)
        if isinstance(masked_spec_embed, nn.Parameter):
            masked_spec_embed.requires_grad_(False)
        positional = backbone.get_parameter("conformer.positional_emb")
        positional.requires_grad_(False)

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_waveforms(waveforms, self.window_samples)
        backbone = self.wrapper.get_submodule("backbone")
        ssl_model = backbone.get_submodule("ssl_model")
        ssl_output = ssl_model(waveforms)
        pooled, features = backbone.get_attenF1D(ssl_output.hidden_states)
        weights = backbone.sig(backbone.fc0(pooled)).view(
            waveforms.shape[0], pooled.shape[1], 1, 1,
        )
        sequence = backbone.selu(
            backbone.first_bn((features * weights).sum(1).unsqueeze(1)),
        ).squeeze(1)
        conformer = backbone.get_submodule("conformer")
        sequence = sequence + conformer.positional_emb[:, : sequence.shape[1]]
        token = conformer.class_token.unsqueeze(0).expand(len(sequence), -1, -1)
        sequence = torch.cat((token, sequence), dim=1)
        for block in conformer.encoder_blocks:
            sequence, _ = block(sequence)
        embeddings = sequence[:, 0]
        native_logits = conformer.fc5(embeddings)
        return (native_logits[:, :1] - native_logits[:, 1:2]), embeddings


class MusicExpert(nn.Module):
    """SONICS binary classifier with its exact pre-classifier embedding."""

    def __init__(self, model: nn.Module, window_samples: int = MUSIC_SAMPLES) -> None:
        super().__init__()
        self.model = model.requires_grad_(True)
        self.embedding_dim = int(model.embed_dim)
        self.window_samples = window_samples

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_waveforms(waveforms, self.window_samples)
        spectrogram = self.model.ft_extractor(waveforms).unsqueeze(1)
        spectrogram = functional.interpolate(
            spectrogram,
            size=tuple(self.model.input_shape),
            mode="bilinear",
        )
        features = self.model.encoder(spectrogram)
        embeddings = features.mean(dim=1)
        return self.model.classifier(embeddings), embeddings


class PresenceExpert(nn.Module):
    """PANNs Cnn14 with ordered voice/music logits from one shared embedding."""

    embedding_dim: Final = 2048

    def __init__(
        self,
        cnn: nn.Module,
        head: nn.Linear,
        window_samples: int = PRESENCE_SAMPLES,
    ) -> None:
        super().__init__()
        self.cnn = cnn.requires_grad_(True)
        for module_name in ("spectrogram_extractor", "logmel_extractor", "fc_audioset"):
            cnn.get_submodule(module_name).requires_grad_(False)
        self.head = head
        self.window_samples = window_samples

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_waveforms(waveforms, self.window_samples)
        outputs = self.cnn(resample(waveforms, 16_000, 32_000))
        embeddings = outputs["embedding"]
        return self.head(embeddings), embeddings


ExpertAdapter = VoiceExpert | MusicExpert | PresenceExpert


def parameter_groups(expert: ExpertAdapter) -> ExpertParameterGroups:
    """Return every optimizer-owned parameter, with non-forward weights excluded."""
    match expert:
        case VoiceExpert():
            head_prefix = "wrapper.backbone.conformer.fc5."
        case MusicExpert():
            head_prefix = "model.classifier."
        case PresenceExpert():
            head_prefix = "head."
        case unsupported:
            assert_never(unsupported)
    named = tuple(expert.named_parameters())
    return ExpertParameterGroups(
        encoder=tuple(
            item for item in named
            if item[1].requires_grad and not item[0].startswith(head_prefix)
        ),
        head=tuple(
            item for item in named
            if item[1].requires_grad and item[0].startswith(head_prefix)
        ),
        excluded=tuple(name for name, parameter in named if not parameter.requires_grad),
    )


@contextmanager
def deterministic_gradient_audit(expert: ExpertAdapter) -> Iterator[None]:
    """Keep every DF SSL layer active while auditing full-depth connectivity."""
    match expert:
        case VoiceExpert():
            ssl_model = expert.wrapper.get_submodule("backbone.ssl_model")
            previous = ssl_model.config.layerdrop
            ssl_model.config.layerdrop = 0.0
            try:
                yield
            finally:
                ssl_model.config.layerdrop = previous
        case MusicExpert() | PresenceExpert():
            yield
        case unsupported:
            assert_never(unsupported)


def _model_root(baseline: Path) -> Path:
    for candidate in (baseline / "model", baseline):
        if (candidate / "df_arena_1b").is_dir() and (candidate / "panns").is_dir():
            return candidate
    raise V5ExpertError(
        "Baseline model directory is unavailable; extract model/ from baseline_submit.zip",
    )


@contextmanager
def _df_import_context(model_root: Path, df_root: Path) -> Iterator[None]:
    previous = Path.cwd()
    sys.path.insert(0, str(model_root))
    os.chdir(df_root)
    try:
        yield
    finally:
        os.chdir(previous)
        sys.path.remove(str(model_root))


def _load_voice(model_root: Path, device: str) -> VoiceExpert:
    df_root = model_root / "df_arena_1b"
    with _df_import_context(model_root, df_root):
        module = importlib.import_module("df_arena_1b.modeling_antispoofing")
        model_class = module.DF_Arena_1B_Antispoofing
        wrapper, loading = model_class.from_pretrained(
            str(df_root),
            local_files_only=True,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
    problems = tuple(
        item
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        for item in loading.get(key, ())
    )
    if problems:
        raise V5ExpertError(f"DF-Arena wrapper state was not restored strictly: {problems[:3]}")
    return VoiceExpert(wrapper.to(device))


def _read_panns_labels(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        return tuple(row["display_name"] for row in csv.DictReader(stream))


def _load_presence(model_root: Path, device: str) -> PresenceExpert:
    panns_root = model_root / "panns"
    labels_path = panns_root / "class_labels_indices.csv"
    package_labels = Path.home() / "panns_data" / labels_path.name
    if not package_labels.is_file():
        package_labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(labels_path, package_labels)
    from panns_inference.models import Cnn14

    labels = _read_panns_labels(labels_path)
    cnn = Cnn14(
        sample_rate=32_000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14_000,
        classes_num=len(labels),
    )
    checkpoint = torch.load(
        panns_root / "Cnn14_mAP=0.431.pth",
        map_location="cpu",
        weights_only=False,
    )
    cnn.load_state_dict(checkpoint["model"], strict=True)
    groups = json.loads((panns_root / "component_labels.json").read_text(encoding="utf-8"))
    indices = {label: index for index, label in enumerate(labels)}
    head = nn.Linear(2048, 2)
    with torch.no_grad():
        for row, name in enumerate(("voice", "music")):
            selected = torch.tensor([indices[label] for label in groups[name]])
            head.weight[row].copy_(cnn.fc_audioset.weight[selected].mean(0))
            head.bias[row].copy_(cnn.fc_audioset.bias[selected].mean())
    return PresenceExpert(cnn.to(device), head.to(device))


def load_expert(
    kind: ExpertKind,
    baseline: Path,
    sonics: Path,
    device: str,
) -> nn.Module:
    """Load one fully trainable V5 expert from pinned local assets."""
    match kind:
        case "voice":
            return _load_voice(_model_root(baseline), device)
        case "music":
            from v4_model import load_model

            return MusicExpert(load_model("sonics", sonics, device).model)
        case "presence":
            return _load_presence(_model_root(baseline), device)
        case unreachable:
            assert_never(unreachable)
