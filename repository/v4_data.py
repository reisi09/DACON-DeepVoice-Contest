"""Runtime loaders for the fixed V4 manifests and full-source audio."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from v2_data import decode_audio


def load_manifest(data_root: Path, split: str) -> list[dict[str, str]]:
    path = data_root / f"{split}.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_audio(path: Path) -> np.ndarray:
    return decode_audio(path, limit_seconds=None)
