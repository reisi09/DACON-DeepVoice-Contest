from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import soundfile
import torch
from torch import nn

from v5_experts import MusicExpert, PresenceExpert, VoiceExpert, parameter_groups

ROOT = Path(__file__).resolve().parent
SONICS = ROOT / "_workspace" / "recovery_v3" / "sonics_assets"


class TinySsl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 4), nn.Linear(4, 4), nn.Linear(4, 4)])
        self.pooler = nn.Linear(4, 4)
        self.checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        self.checkpointing = True

    def forward(self, waveforms: torch.Tensor) -> SimpleNamespace:
        hidden_states: list[torch.Tensor] = []
        value = waveforms.unsqueeze(-1)
        for layer in self.layers:
            value = torch.tanh(layer(value))
            hidden_states.append(value)
        return SimpleNamespace(hidden_states=tuple(hidden_states))


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, None]:
        mixed = values + values.mean(1, keepdim=True)
        return values + torch.tanh(self.projection(mixed)), None


class TinyConformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positional_emb = nn.Parameter(torch.zeros(1, 64, 4), requires_grad=False)
        self.class_token = nn.Parameter(torch.ones(1, 4))
        self.encoder_blocks = nn.ModuleList([TinyBlock(), TinyBlock(), TinyBlock()])
        self.fc5 = nn.Linear(4, 2)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ssl_model = TinySsl()
        self.fc0 = nn.Linear(4, 1)
        self.sig = nn.Sigmoid()
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU()
        self.conformer = TinyConformer()

    def get_attenF1D(self, layers: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = [layer.mean(1, keepdim=True) for layer in layers]
        return torch.cat(pooled, 1), torch.stack(layers, 1)


class TinyWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = TinyBackbone()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        pooled, features = backbone.get_attenF1D(
            backbone.ssl_model(waveform.unsqueeze(0)).hidden_states,
        )
        weights = backbone.sig(backbone.fc0(pooled)).view(1, len(pooled[0]), 1, 1)
        sequence = backbone.selu(backbone.first_bn((features * weights).sum(1).unsqueeze(1))).squeeze(1)
        conformer = backbone.conformer
        sequence = sequence + conformer.positional_emb[:, : sequence.shape[1]]
        sequence = torch.cat((conformer.class_token.unsqueeze(0), sequence), dim=1)
        for block in conformer.encoder_blocks:
            sequence, _ = block(sequence)
        return conformer.fc5(sequence[:, 0])


def test_voice_batches_original_path_and_updates_relevant_depths() -> None:
    # Given: a deterministic miniature of the original SSL -> layer attention -> Conformer path.
    torch.manual_seed(236749)
    wrapper = TinyWrapper().eval()
    expert = VoiceExpert(wrapper, window_samples=16).eval()
    waveforms = torch.randn(2, 16)

    # When: the corrected batched path and original one-dimensional path process the same inputs.
    logits, embeddings = expert(waveforms)
    native = torch.cat([wrapper(waveform) for waveform in waveforms])

    # Then: B>1 preserves B=1 numerics and the binary logit is spoof minus bonafide.
    torch.testing.assert_close(logits[:, 0], native[:, 0] - native[:, 1])
    assert embeddings.shape == (2, 4)
    assert wrapper.backbone.ssl_model.checkpointing
    assert all(parameter.requires_grad for parameter in wrapper.parameters() if parameter is not wrapper.backbone.ssl_model.pooler.weight and parameter is not wrapper.backbone.ssl_model.pooler.bias and parameter is not wrapper.backbone.conformer.positional_emb)
    assert not any(parameter.requires_grad for parameter in wrapper.backbone.ssl_model.pooler.parameters())

    expert.train()
    groups = parameter_groups(expert)
    trainable = groups.encoder + groups.head
    assert set(groups.excluded) == {
        "wrapper.backbone.ssl_model.pooler.weight",
        "wrapper.backbone.ssl_model.pooler.bias",
        "wrapper.backbone.conformer.positional_emb",
    }
    assert {name for name, _ in groups.head} == {
        "wrapper.backbone.conformer.fc5.weight",
        "wrapper.backbone.conformer.fc5.bias",
    }
    representatives = (
        wrapper.backbone.ssl_model.layers[0].weight,
        wrapper.backbone.ssl_model.layers[1].weight,
        wrapper.backbone.ssl_model.layers[2].weight,
        wrapper.backbone.conformer.encoder_blocks[0].projection.weight,
        wrapper.backbone.conformer.encoder_blocks[1].projection.weight,
        wrapper.backbone.conformer.encoder_blocks[2].projection.weight,
        wrapper.backbone.conformer.fc5.weight,
    )
    before = [parameter.detach().clone() for parameter in representatives]
    optimizer = torch.optim.SGD([parameter for _, parameter in trainable], lr=0.01)
    loss = expert(waveforms)[0].square().mean()
    loss.backward()
    for parameter in representatives:
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad)
    optimizer.step()
    for parameter, original in zip(representatives, before, strict=True):
        assert not torch.equal(parameter, original)


class TinyMusic(nn.Module):
    model_name = "SpecTTTra"
    input_shape = (4, 4)
    embed_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.ft_extractor = TinyFeatures()
        self.encoder = TinyMusicEncoder()
        self.classifier = nn.Linear(3, 1)


class TinyFeatures(nn.Module):
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return waveforms.repeat_interleave(2, dim=1).reshape(len(waveforms), 4, 4)


class TinyMusicEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.projection(spectrogram.mean(2))


def test_music_uses_the_native_preclassifier_embedding() -> None:
    # Given: a model whose native classifier consumes the mean encoder tokens.
    torch.manual_seed(7)
    model = TinyMusic().eval()
    expert = MusicExpert(model, window_samples=8).eval()
    waveforms = torch.randn(2, 8)

    # When: the adapter runs the same feature, resize, encoder and classifier seam.
    logits, embeddings = expert(waveforms)

    # Then: the returned embedding is exactly the native classifier input.
    torch.testing.assert_close(logits, model.classifier(embeddings))
    assert embeddings.shape == (2, 3)


class TinyPanns(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spectrogram_extractor = nn.Linear(1, 1)
        self.logmel_extractor = nn.Linear(1, 1)
        self.encoder = nn.Linear(1, 4)
        self.fc_audioset = nn.Linear(4, 527)
        self.seen_samples = 0

    def forward(self, waveforms: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen_samples = waveforms.shape[1]
        embedding = self.encoder(waveforms.mean(1, keepdim=True))
        return {"embedding": embedding}


def test_presence_resamples_16khz_and_returns_voice_music_logits() -> None:
    # Given: a differentiable PANNs stand-in and an ordered two-row head.
    cnn = TinyPanns()
    head = nn.Linear(4, 2)
    expert = PresenceExpert(cnn, head, window_samples=16)

    # When: a 16 kHz batch crosses the presence adapter.
    logits, embeddings = expert(torch.randn(2, 16, requires_grad=True))
    logits.sum().backward()

    # Then: the CNN receives 32 kHz audio and both CNN/head stay connected to the loss.
    assert cnn.seen_samples == 32
    assert logits.shape == (2, 2)
    torch.testing.assert_close(logits, head(embeddings))
    assert cnn.encoder.weight.grad is not None and torch.count_nonzero(cnn.encoder.weight.grad)
    groups = parameter_groups(expert)
    assert {name for name, _ in groups.head} == {"head.weight", "head.bias"}
    assert all(name.startswith(("cnn.spectrogram_extractor.", "cnn.logmel_extractor.", "cnn.fc_audioset.")) for name in groups.excluded)


@pytest.mark.skipif(not SONICS.is_dir(), reason="pinned local SONICS assets unavailable")
def test_real_sonics_eval_parity_and_encoder_gradient() -> None:
    # Given: the pinned real V4 SONICS weights and the first real baseline waveform.
    from v5_experts import load_expert

    torch.set_num_threads(2)
    expert = load_expert("music", ROOT / "baseline_original", SONICS, "cpu").eval()
    audio, sample_rate = soundfile.read(ROOT / "baseline_original" / "data" / "test" / "TEST_0000.wav", dtype="float32")
    assert sample_rate == 16_000
    waveform = torch.zeros(1, expert.window_samples)
    waveform[0, : len(audio)] = torch.from_numpy(audio)

    # When: original and adapter eval paths score the same real sample, then train once.
    with torch.inference_mode():
        original = expert.model(waveform)
        logits, embeddings = expert(waveform)
    expert.train()
    groups = parameter_groups(expert)
    trainable = groups.encoder + groups.head
    assert sum(parameter.numel() for _, parameter in trainable) == 17_172_493
    assert not groups.excluded
    representatives = {
        prefix: next(
            parameter
            for name, parameter in trainable
            if name.startswith(prefix) and parameter.ndim > 1
        )
        for prefix in (
            "model.encoder.transformer.blocks.0.",
            "model.encoder.transformer.blocks.6.",
            "model.encoder.transformer.blocks.11.",
            "model.classifier.",
        )
    }
    before = {prefix: parameter.detach().clone() for prefix, parameter in representatives.items()}
    train_logits, _ = expert(waveform)
    train_logits.square().mean().backward()
    optimizer = torch.optim.SGD([parameter for _, parameter in trainable], lr=0.01)
    optimizer.step()

    # Then: adapter numerics match and early/middle/late encoder parameters receive gradients.
    torch.testing.assert_close(logits, original, atol=1e-6, rtol=0.0)
    assert embeddings.shape == (1, expert.embedding_dim)
    for prefix, parameter in representatives.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad)
        assert not torch.equal(parameter, before[prefix])
