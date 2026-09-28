from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal


class SelectionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Selection:
    arm: Literal['R1', 'R3']
    seed: int
    checkpoint_sha256: str
    best_epoch: int
    stop_reason: Literal['early_stopping', 'max_epochs', 'time_budget']
    run_identity: str
    config_identity: str
    weights_sha256: str
    reference_sha256: str

    @classmethod
    def read(cls, path: Path) -> Selection:
        value = cls(**json.loads(path.read_text()))
        if (value.arm not in ('R1', 'R3') or value.seed != 41
                or type(value.best_epoch) is not int or value.best_epoch < 0
                or value.stop_reason not in ('early_stopping', 'max_epochs', 'time_budget')):
            raise SelectionError('Invalid selected PANNs experiment')
        for digest in (value.checkpoint_sha256, value.run_identity, value.config_identity,
                       value.weights_sha256, value.reference_sha256):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise SelectionError('Invalid selected PANNs identity')
        return value
