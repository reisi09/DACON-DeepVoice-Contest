"""Full-coverage batches and source/class loss weights for frozen-feature training."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True, slots=True)
class Item:
    row: Mapping[str, str]
    features: np.ndarray
    truth: np.ndarray
    label: float
    silent: bool = False



def epoch_order(items: Sequence[Item], seed: int, epoch: int) -> np.ndarray:
    """No replacement: balance loss weights, never drop majority-source rows."""
    if any(item.row['split'] != 'train' for item in items):
        raise ValueError('Validation rows cannot enter optimizer batches')
    return np.random.default_rng(seed + epoch * 100003).permutation(len(items))


def loss_weights(items: Sequence[Item]) -> dict[str, float]:
    """Equal class, then source/generator contribution while visiting every row."""
    cells = [(item.label, item.row.get('source', ''), item.row.get('generator', '')) for item in items]
    counts = Counter(cells)
    per_class = Counter(cell[0] for cell in counts)
    if set(per_class) != {0., 1.}:
        raise ValueError('Both classes must occur in the training route')
    return {item.row['ID']: len(items) / (2 * per_class[cell[0]] * counts[cell])
            for item, cell in zip(items, cells, strict=True)}



def groups(items: Sequence[Item], microbatch: int, window_limit: int) -> list[list[Item]]:
    """Bound total windows; a longest-file benchmark establishes single-bag safety."""
    result, current, count = [], [], 0
    for item in items:
        if len(item.features) > window_limit:
            raise ValueError('A full file exceeds measured window capacity; benchmark must include the longest file')
        if current and (len(current) >= microbatch or count + len(item.features) > window_limit):
            result.append(current)
            current, count = [], 0
        current.append(item)
        count += len(item.features)
    if current:
        result.append(current)
    return result


