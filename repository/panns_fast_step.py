from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

import torch

from panns_adaptation import AdaptedCnn14, masked_file_loss, presence_logits
from panns_adaptation_data import load_cache


class CacheEntry(TypedDict):
    ID: str
    targets: list[float | None]
    stems: dict[str, float]
    path: str
    identity: str
    sha256: str
    member: NotRequired[str]


class PresenceHead(TypedDict):
    features: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float


class AttentionRecord(TypedDict):
    segments: int
    entropy_sum: float
    peak_sum: float


class StepRecord(TypedDict):
    loss: float
    probabilities: list[list[float]]
    IDs: list[str]
    gradient_norms: dict[str, float]
    attention: AttentionRecord | None


def optimization_step(
    model: AdaptedCnn14,
    optimizer: torch.optim.Optimizer,
    entries: Sequence[CacheEntry],
    heads: Mapping[str, PresenceHead],
    labels: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    fraction: Sequence[float],
    device: str,
    microbatch: int,
    capture_gradients: bool = False,
) -> tuple[StepRecord, dict[str, torch.Tensor]]:
    """Preserve update arithmetic; clipping also rejects nonfinite gradients."""
    optimizer.zero_grad(set_to_none=True)
    loss_sum, probabilities = 0., []
    attention: AttentionRecord = dict(segments=0, entropy_sum=0., peak_sum=0.)
    for entry in entries:
        features = torch.from_numpy(load_cache(entry)['block5'])
        parts = []
        for part in features.split(microbatch):
            parts.append(model.tags_from_block5(part.to(device)))
            if model.pool is not None:
                weights = model.pool.last_attention.to(device='cpu', dtype=torch.float64)
                attention['segments'] += weights.size(0)
                attention['entropy_sum'] += float(-(weights * weights.clamp_min(1e-300).log()).sum())
                attention['peak_sum'] += float(weights.max(1).values.sum())
        logits = presence_logits(torch.cat(parts), entry['stems'], heads, labels, groups)
        known = torch.tensor([[target is not None for target in entry['targets']]])
        targets = torch.tensor([[float('nan') if target is None else target
                                 for target in entry['targets']]], dtype=torch.float64)
        loss = masked_file_loss(logits[None], targets, known, fraction) / len(entries)
        if not torch.isfinite(loss):
            message = 'Nonfinite file loss'
            raise ValueError(message)
        loss.backward()
        loss_sum += float(loss.detach())
        probabilities.append(logits.detach().sigmoid().tolist())
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                message = f'Missing gradient: {name}'
                raise ValueError(message)
            if capture_gradients:
                gradients[name] = parameter.grad.detach().cpu().clone()
    norms = {group['name']: float(torch.nn.utils.clip_grad_norm_(
        group['params'], 1., error_if_nonfinite=True)) for group in optimizer.param_groups}
    optimizer.step()
    return StepRecord(loss=loss_sum, probabilities=probabilities,
                      IDs=[entry['ID'] for entry in entries], gradient_norms=norms,
                      attention=attention if attention['segments'] else None), gradients
