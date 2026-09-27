"""Voice prediction heads shared by training (data_v4/train_backbone_heads.py) and the submission runtime.
  linear : nn.Linear(1280, 2)                                      (DF-Arena fc5 / AntiDeepfake classifier shape)
  mlp    : standardize -> Linear(1280, 256) -> GELU -> Linear(256, 2)
  resmlp : pretrained Linear + zero-initialised standardized 2-layer branch (equals the Linear head at step 0)
Checkpoint records carry 'arch'; build() recreates the module so load_state_dict(strict=True) restores buffers too.
"""
from __future__ import annotations

import torch
from torch import nn


class Standardize(nn.Module):
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer('mean', mean.detach().float().clone())
        self.register_buffer('scale', scale.detach().float().clone())

    def forward(self, x):
        return (x - self.mean) / self.scale


class ResMLP(nn.Module):
    def __init__(self, init: dict | None, mean: torch.Tensor, scale: torch.Tensor, hidden: int = 256, width: int = 1280):
        super().__init__()
        self.lin = nn.Linear(width, 2)
        if init is not None:
            self.lin.load_state_dict(init)
        self.branch = nn.Sequential(Standardize(mean, scale), nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, x):
        return self.lin(x) + self.branch(x)


def mlp(mean: torch.Tensor, scale: torch.Tensor, hidden: int = 256, width: int = 1280) -> nn.Module:
    return nn.Sequential(Standardize(mean, scale), nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 2))


def build(arch: str, init: dict | None = None, mean: torch.Tensor | None = None, scale: torch.Tensor | None = None,
          width: int = 1280) -> nn.Module:
    """Fresh module of the given arch; for loading, mean/scale default to placeholders overwritten by the state dict."""
    mean = torch.zeros(width) if mean is None else mean
    scale = torch.ones(width) if scale is None else scale
    if arch == 'linear':
        m = nn.Linear(width, 2)
        if init is not None:
            m.load_state_dict(init)
        return m
    if arch == 'mlp':
        return mlp(mean, scale, width=width)
    if arch == 'resmlp':
        return ResMLP(init, mean, scale, width=width)
    raise ValueError(f'unknown voice head arch {arch}')


def load(record: dict) -> nn.Module:
    m = build(record.get('arch', 'linear'))
    m.load_state_dict(record['model'], strict=True)
    return m


if __name__ == '__main__':
    x = torch.randn(5, 1280)
    for arch in ('linear', 'mlp', 'resmlp'):
        m = build(arch, None, x.mean(0), x.std(0))
        again = load({'arch': arch, 'model': m.state_dict()})
        assert torch.allclose(m(x), again(x)), arch
    init = {'weight': torch.randn(2, 1280), 'bias': torch.randn(2)}
    r = build('resmlp', init, x.mean(0), x.std(0))
    assert torch.allclose(r(x), x @ init['weight'].T + init['bias'], atol=1e-5), 'resmlp equals Linear at init'
    print('voice_heads self-check ok')
