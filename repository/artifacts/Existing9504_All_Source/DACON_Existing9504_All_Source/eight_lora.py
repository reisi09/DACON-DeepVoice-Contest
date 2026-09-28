"""Small, dependency-free LoRA support for AntiDeepfake E7."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Final

import torch
from torch import nn


LORA_SUFFIXES: Final = ("lora_a.weight", "lora_b.weight")


class LoRAError(RuntimeError):
    """Raised when the requested adapter topology or state is invalid."""


class LoRALinear(nn.Module):
    """Frozen linear projection plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if rank <= 0:
            raise LoRAError("LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False,
                                device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False,
                                device=base.weight.device, dtype=base.weight.dtype)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(inputs)) * self.scaling


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    expected_layers: int = 48,
) -> tuple[str, ...]:
    """Freeze the model, replace every q/v projection, and unfreeze the head."""
    model.requires_grad_(False)
    layers = model.get_submodule("encoder.encoder.layers")
    if not isinstance(layers, nn.ModuleList) or len(layers) != expected_layers:
        raise LoRAError(
            f"Expected {expected_layers} AntiDeepfake layers, found {len(layers)}",
        )
    names: list[str] = []
    for index, layer in enumerate(layers):
        attention = layer.get_submodule("attention")
        for projection in ("q_proj", "v_proj"):
            base = attention.get_submodule(projection)
            if not isinstance(base, nn.Linear):
                raise LoRAError(f"Layer {index} {projection} is not Linear")
            setattr(attention, projection, LoRALinear(base, rank, alpha))
            names.append(f"encoder.encoder.layers.{index}.attention.{projection}")
    classifier = model.get_submodule("classifier")
    if not isinstance(classifier, nn.Linear):
        raise LoRAError("AntiDeepfake classifier is not Linear")
    classifier.requires_grad_(True)
    return tuple(names)


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return the portable adapter and classifier tensors only."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.endswith(LORA_SUFFIXES) or name.startswith("classifier.")
    }


def load_adapter_state_dict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Load exactly the adapter/head tensors expected by an injected model."""
    expected = set(adapter_state_dict(model))
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise LoRAError(f"Adapter keys differ: missing={missing}, unexpected={unexpected}")
    model.load_state_dict(dict(state), strict=False)


def tensor_checksum(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata, and values in stable order."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def frozen_parameter_probe(model: nn.Module) -> str:
    """Hash bounded samples from every frozen tensor without copying the backbone."""
    probe: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        flat = parameter.detach().reshape(-1)
        indices = torch.linspace(
            0,
            max(0, flat.numel() - 1),
            min(17, flat.numel()),
            device=flat.device,
        ).long()
        probe[name] = flat[indices]
    return tensor_checksum(probe)


def frozen_parameter_checksum(model: nn.Module) -> str:
    """Stream a byte-exact checksum of every frozen parameter."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
        del tensor
    return digest.hexdigest()


def reinitialize_adapters(model: nn.Module, seed: int) -> dict[str, torch.Tensor]:
    """Create a paired-arm seed initialization while retaining the pretrained head."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LoRALinear):
                continue
            temporary = torch.empty_like(module.lora_a.weight, device="cpu")
            nn.init.kaiming_uniform_(temporary, a=5**0.5, generator=generator)
            module.lora_a.weight.copy_(temporary.to(module.lora_a.weight.device))
            module.lora_b.weight.zero_()
    return adapter_state_dict(model)
