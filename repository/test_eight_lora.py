"""Numerical connectivity checks for E7's adapter-only checkpoint path."""

from pathlib import Path

import torch
import numpy as np
from torch import nn

from eight_lora import (
    adapter_state_dict,
    inject_lora,
    load_adapter_state_dict,
)
from eight_lora_train import balanced_schedule, make_windows


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.q_proj(inputs) + self.v_proj(inputs)


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = TinyAttention()


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.encoder = nn.Module()
        self.encoder.encoder.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.encoder.encoder.layers:
            hidden = layer.attention(hidden)
        return self.classifier(hidden.mean(1))


def test_adapter_gradient_preserves_base_and_round_trips(tmp_path: Path) -> None:
    """Protect gradient flow, frozen weights, and adapter-only reload fidelity."""
    torch.manual_seed(7)
    model = TinyModel()
    inject_lora(model, rank=2, alpha=4, expected_layers=2)
    inputs = torch.randn(3, 5, 4)
    pristine = adapter_state_dict(model)
    frozen_reference = model(inputs).detach()
    with torch.no_grad():
        model.classifier.weight.add_(1)
    assert not torch.equal(model(inputs), frozen_reference)
    load_adapter_state_dict(model, pristine)
    torch.testing.assert_close(model(inputs), frozen_reference, rtol=0, atol=0)
    frozen_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    target = torch.tensor([0, 1, 0])
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    loss = nn.functional.cross_entropy(model(inputs), target)
    loss.backward()
    b_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.endswith("lora_b.weight")
    ]
    assert b_gradients and all(
        gradient is not None
        and torch.isfinite(gradient).all()
        and float(gradient.abs().sum()) > 0
        for gradient in b_gradients
    )
    optimizer.step()
    assert all(
        torch.equal(frozen_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in frozen_before
    )

    expected = model(inputs).detach()
    checkpoint = tmp_path / "adapter.pt"
    torch.save(adapter_state_dict(model), checkpoint)
    torch.manual_seed(7)
    reloaded = TinyModel()
    inject_lora(reloaded, rank=2, alpha=4, expected_layers=2)
    load_adapter_state_dict(reloaded, torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(reloaded(inputs), expected, rtol=0, atol=0)


def test_shared_windows_and_schedule_preserve_equal_exposure() -> None:
    """Protect zero-padding/tail coverage and identical balanced step budgets."""
    audio = np.arange(70_000, dtype=np.float32)
    windows = make_windows(audio, window=64_600, hop=32_300)
    assert [start for start, _ in windows] == [0, 5_400]
    assert np.array_equal(windows[-1][1], audio[-64_600:])
    short = make_windows(audio[:400], window=64_600, hop=32_300)
    assert np.array_equal(short[0][1][:400], audio[:400])
    assert np.count_nonzero(short[0][1][400:]) == 0

    labels = [0, 0, 1, 1, 1]
    first = balanced_schedule(labels, steps=7, batch_size=2, seed=41)
    second = balanced_schedule(labels, steps=7, batch_size=2, seed=41)
    assert first == second
    assert all(sorted(labels[index] for index in batch) == [0, 1] for batch in first)


def test_lora_inherits_base_projection_device_and_dtype() -> None:
    """Protect injection into an already-placed non-default model such as H100 BF16."""
    model = TinyModel().to(dtype=torch.float64)
    inject_lora(model, rank=2, alpha=4, expected_layers=2)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            assert parameter.device == model.classifier.weight.device
            assert parameter.dtype == torch.float64
    assert model(torch.randn(2, 3, 4, dtype=torch.float64)).dtype == torch.float64
