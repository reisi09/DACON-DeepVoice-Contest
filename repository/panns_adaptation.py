"""Cnn14 suffix adaptation with the frozen DataV4 presence decision layer."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from decision_heads import SINGING, VOICE_CORE


class ChannelSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(1024, 64)
        self.up = nn.Linear(64, 1024)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        gate = 2 * torch.sigmoid(self.up(F.relu(self.down(x.mean((2, 3))))))
        return x * gate[:, :, None, None]


class TemporalPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2048))

    def forward(self, z):
        attention = torch.softmax((z * self.weight[None, :, None]).sum(1), dim=1)
        return z.max(2).values + (z * attention[:, None, :]).sum(2)


class AdaptedCnn14(nn.Module):
    def __init__(self, base, channel_se=False, temporal_pool=False, train_blocks=(6,)):
        super().__init__()
        if tuple(train_blocks) not in ((), (6,)):
            raise ValueError('v1 block5 cache permits only frozen prefix and optional block6')
        self.base = base.requires_grad_(False)
        self.se = ChannelSE() if channel_se else nn.Identity()
        self.pool = TemporalPool() if temporal_pool else None
        for number in train_blocks:
            block = getattr(base, f'conv_block{number}')
            block.conv1.requires_grad_(True)
            block.conv2.requires_grad_(True)
        self.train_blocks = tuple(train_blocks)
        self.train(False)

    def train(self, mode=True):
        # Eval disables stochastic layers and BN updates without disabling autograd.
        super().train(False)
        return self

    @torch.no_grad()
    def prefix(self, waveforms_32k):
        x = self.base.logmel_extractor(self.base.spectrogram_extractor(waveforms_32k))
        x = self.base.bn0(x.transpose(1, 3)).transpose(1, 3)
        for number in range(1, 6):
            x = getattr(self.base, f'conv_block{number}')(x, pool_size=(2, 2), pool_type='avg')
        return x

    def tags_from_block5(self, features):
        x = self.base.conv_block6(self.se(features), pool_size=(1, 1), pool_type='avg').mean(3)
        x = self.pool(x) if self.pool is not None else x.max(2).values + x.mean(2)
        return torch.sigmoid(self.base.fc_audioset(F.relu(self.base.fc1(x))))

    def forward(self, waveforms_32k):
        return self.tags_from_block5(self.prefix(waveforms_32k))

    def optimizer_groups(self):
        groups = []
        for number in self.train_blocks:
            block = getattr(self.base, f'conv_block{number}')
            groups.append(dict(params=[*block.conv1.parameters(), *block.conv2.parameters()], lr=1e-5))
        if isinstance(self.se, ChannelSE):
            groups.append(dict(params=list(self.se.parameters()), lr=1e-3))
        if self.pool is not None:
            groups.append(dict(params=list(self.pool.parameters()), lr=1e-4))
        return groups


def tag_features(tags, labels, groups):
    """Keep input float32 statistics; copy to CPU with autograd before float64 logits."""
    tags = tags.to('cpu')
    if tags.ndim != 2 or not tags.size(0) or tags.size(1) != len(labels):
        raise ValueError('Expected all segments of one file in label order')
    index = {name: i for i, name in enumerate(labels)}
    result = {}
    for name, names in (('voice', groups['voice']), ('music', groups['music']),
                        ('speech', VOICE_CORE), ('singing', SINGING), ('music_only', ['Music'])):
        indices = [index[n] for n in names if n in index]
        if not indices:
            raise ValueError(f'Empty PANNs class group: {name}')
        values = tags[:, indices].max(1).values
        for suffix, value in (('max', values.max()), ('mean', values.mean()),
                              ('q90', torch.quantile(values, .9))):
            result[f'tag_{name}_{suffix}'] = value
    return result


def presence_logits(tags, stem_features, heads, labels, groups):
    features = {**stem_features, **tag_features(tags, labels, groups)}
    logits = []
    for target in ('VOICE_PRESENT', 'MUSIC_PRESENT'):
        head = heads[target]
        x = torch.stack([torch.as_tensor(features[n], dtype=torch.float64, device='cpu')
                         for n in head['features']])
        mean, scale, coef = [torch.as_tensor(head[n], dtype=torch.float64) for n in ('mean', 'scale', 'coef')]
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid frozen logistic scale')
        logits.append(((x - mean) / scale).dot(coef) + head['intercept'])
    return torch.stack(logits)


def masked_file_loss(logits, targets, known, known_fraction):
    if logits.ndim != 2 or logits.shape != targets.shape or known.shape != logits.shape or logits.shape[1] != 2:
        raise ValueError('Expected file-batch by two targets')
    known_fraction = torch.as_tensor(known_fraction, dtype=logits.dtype, device=logits.device)
    if known_fraction.shape != (2,) or not torch.isfinite(known_fraction).all() or ((known_fraction <= 0) | (known_fraction > 1)).any():
        raise ValueError('Target has no known training labels or invalid fraction')
    safe_targets = torch.where(known.bool(), targets, torch.zeros_like(targets))
    if not torch.isfinite(safe_targets).all() or ((safe_targets < 0) | (safe_targets > 1)).any():
        raise ValueError('Known targets must be binary probabilities')
    per_target = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction='none')
    return (per_target * known / known_fraction).sum() / (2 * logits.size(0))
