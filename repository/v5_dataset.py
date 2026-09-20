"""Leakage-aware V5 view over the immutable V4 train/dev data."""
from __future__ import annotations

import csv
import hashlib
import math
import random
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Final, TypedDict

import numpy as np

from v4_data import load_audio
from v5_augment import channel_augment

LABELS: Final = ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")
HOLDOUT_FRACTION: Final = 0.20
HOLDOUT_SALT: Final = "v5-nsynth-vp-holdout-v1"
SOURCE_PANELS: Final = frozenset({"speech", "music_real", "music_seen", "music_ood", "mixture", "vp_holdout"})


class HoldoutAudit(TypedDict):
    protocol: str
    fraction: float
    available_groups: int
    groups: list[str]
    rows: int


class HashAudit(TypedDict):
    train_manifest_sha256: str
    dev_manifest_sha256: str
    train_audio_receipt_sha256: str
    dev_audio_receipt_sha256: str
    train_unique_audio_sha256: int
    dev_unique_audio_sha256: int


class SplitAudit(TypedDict):
    protocol: str
    counts: dict[str, int]
    nsynth_holdout: HoldoutAudit
    overlap: dict[str, int]
    hashes: HashAudit


class CompositionMetadata(TypedDict):
    kind: str
    parents: tuple[str, str]
    parent_groups: tuple[str, ...]
    epoch: int
    seed: int


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    path: Path
    groups: tuple[str, ...]
    labels: tuple[float, ...]
    panel: str


@dataclass(frozen=True, slots=True)
class Composition:
    waveform: np.ndarray
    labels: tuple[float, ...]
    metadata: CompositionMetadata


class DataLeakageError(RuntimeError):
    """V4-derived V5 splits or requested parents cross a leakage boundary."""


class DataContractError(RuntimeError):
    """A requested crop or composition has unsupported inputs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_label(value: str) -> float:
    return float(value) if value in {"0", "1"} else math.nan


def _read_rows(data: Path, split: str) -> list[Example]:
    with (data / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [Example(row["ID"], (data / row["path"]).resolve(), tuple(row["group_id"].split("|")),
                    tuple(_parse_label(row[name]) for name in LABELS), row["panel"]) for row in rows]


def _receipt(examples: list[Example], data: Path) -> tuple[set[str], str]:
    pairs = sorted((str(example.path.relative_to(data)), _sha256(example.path)) for example in examples)
    encoded = "\n".join(f"{path}\t{digest}" for path, digest in pairs).encode()
    return {digest for _, digest in pairs}, hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=2)
def _load(data: Path) -> tuple[tuple[Example, ...], tuple[Example, ...], SplitAudit]:
    train, original_dev = _read_rows(data, "train"), _read_rows(data, "dev")
    nsynth_groups = sorted({example.groups[0] for example in train
                            if example.groups[0].startswith("nsynth_instrument_")})
    holdout_count = round(len(nsynth_groups) * HOLDOUT_FRACTION)
    held = set(sorted(nsynth_groups,
                      key=lambda group: hashlib.sha256(f"{HOLDOUT_SALT}|{group}".encode()).hexdigest())[:holdout_count])
    moved = [replace(example, panel="vp_holdout") for example in train if held.intersection(example.groups)]
    if any(example.labels[3] != 0.0 for example in moved):
        raise DataContractError("NSynth VP holdout contains a row without known VOICE_PRESENT=0")
    kept = [example for example in train if not held.intersection(example.groups)]
    dev = [*original_dev, *moved]
    train_groups = {group for example in kept for group in example.groups}
    dev_groups = {group for example in dev for group in example.groups}
    train_paths, dev_paths = {example.path for example in kept}, {example.path for example in dev}
    train_hashes, train_receipt = _receipt(kept, data)
    dev_hashes, dev_receipt = _receipt(dev, data)
    overlap = {"groups": len(train_groups & dev_groups), "paths": len(train_paths & dev_paths),
               "sha256": len(train_hashes & dev_hashes)}
    if any(overlap.values()):
        raise DataLeakageError(f"train/dev overlap: {overlap}")
    audit: SplitAudit = {
        "protocol": "V4 train/dev only; deterministic pre-label NSynth instrument-group holdout; sealed and V2 excluded",
        "counts": {"v4_train": len(train), "v4_dev": len(original_dev), "train": len(kept), "dev": len(dev)},
        "nsynth_holdout": {"protocol": HOLDOUT_SALT, "fraction": HOLDOUT_FRACTION,
                            "available_groups": len(nsynth_groups), "groups": sorted(held), "rows": len(moved)},
        "overlap": overlap,
        "hashes": {"train_manifest_sha256": _sha256(data / "train.csv"),
                   "dev_manifest_sha256": _sha256(data / "dev.csv"),
                   "train_audio_receipt_sha256": train_receipt,
                   "dev_audio_receipt_sha256": dev_receipt,
                   "train_unique_audio_sha256": len(train_hashes),
                   "dev_unique_audio_sha256": len(dev_hashes)},
    }
    return tuple(kept), tuple(dev), audit


def load_splits(root: Path) -> tuple[list[Example], list[Example], SplitAudit]:
    """Load V4 train/dev and derive the fixed group-disjoint V5 VP holdout."""
    data = root / "v4_validation" if (root / "v4_validation/train.csv").is_file() else root
    train, dev, audit = _load(data.resolve())
    return list(train), list(dev), audit


def read_audio(path: Path) -> np.ndarray:
    """Decode one complete mono source at 16 kHz."""
    return load_audio(path).astype(np.float32, copy=False)


def _seed(example: Example, epoch: int, seed: int, purpose: str) -> int:
    text = f"{seed}|{example.id}|{epoch}|{purpose}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def _crop_start(samples: int, example: Example, epoch: int, window_samples: int, seed: int) -> int:
    if window_samples <= 0:
        raise DataContractError("window_samples must be positive")
    positions = samples - window_samples + 1
    if positions <= 1:
        return 0
    strata = min(8, positions)
    stratum = (_seed(example, 0, seed, "stratum") + epoch) % strata
    lower, upper = positions * stratum // strata, positions * (stratum + 1) // strata
    return random.Random(_seed(example, epoch, seed, "crop")).randrange(lower, upper)


def crop_start(example: Example, epoch: int, window_samples: int, seed: int) -> int:
    """Cycle through up to eight full-source strata with replayable jitter."""
    return _crop_start(len(read_audio(example.path)), example, epoch, window_samples, seed)


def _crop(example: Example, epoch: int, window_samples: int, seed: int) -> np.ndarray:
    audio = read_audio(example.path)
    start = _crop_start(len(audio), example, epoch, window_samples, seed)
    return np.pad(audio[start:start + window_samples], (0, max(0, window_samples - len(audio[start:])))).astype(np.float32)


def training_crop(example: Example, epoch: int, window_samples: int, augmented: bool,
                  seed: int) -> tuple[np.ndarray, tuple[float, ...]]:
    """Return one deterministic crop and only clip-level labels valid for that crop."""
    waveform = _crop(example, epoch, window_samples, seed)
    labels = example.labels if example.panel in SOURCE_PANELS else (math.nan, math.nan, math.nan, *example.labels[3:])
    return (channel_augment(waveform, example.id, epoch, seed) if augmented else waveform), labels


def _split(example: Example) -> str:
    if example.panel == "vp_holdout":
        return "dev"
    roles = {part for part in example.path.parts if part in {"train", "dev"}}
    if len(roles) != 1:
        raise DataLeakageError(f"cannot identify split for {example.id}")
    return roles.pop()


def _ensure_same_split(first: Example, second: Example) -> None:
    if _split(first) != _split(second):
        raise DataLeakageError(f"cross-split parents: {first.id}, {second.id}")


def compose_mixture(speech: Example, music: Example, epoch: int, window_samples: int,
                    seed: int) -> Composition:
    """Mix same-split speech/music crops while preserving only supported truth."""
    _ensure_same_split(speech, music)
    if speech.labels[3] != 1.0 or music.labels[4] != 1.0:
        raise DataContractError("mixture requires known present speech and music components")
    voice_fake = 1.0 if speech.labels[1] == 1.0 else (speech.labels[1] if music.labels[2] != 1.0 or music.labels[3] == 0.0 else math.nan)
    file_fake = max(speech.labels[0], music.labels[0])
    labels = (file_fake, voice_fake, music.labels[2], 1.0, 1.0)
    speech_audio, music_audio = _crop(speech, epoch, window_samples, seed), _crop(music, epoch, window_samples, seed)
    level = 10.0 ** (random.Random(_seed(speech, epoch, seed, music.id)).uniform(-8.0, -2.0) / 20.0)
    waveform = (speech_audio + level * music_audio) / max(1.0, float(np.max(np.abs(speech_audio + level * music_audio))))
    groups = tuple(dict.fromkeys((*speech.groups, *music.groups)))
    metadata: CompositionMetadata = {"kind": "speech_music", "parents": (speech.id, music.id),
                                     "parent_groups": groups, "epoch": epoch, "seed": seed}
    return Composition(waveform.astype(np.float32), labels, metadata)


def real_real_edit(base: Example, donor: Example, epoch: int, window_samples: int,
                   seed: int) -> Composition:
    """Create a deterministic same-split real/real replacement control."""
    _ensure_same_split(base, donor)
    if base.labels[0] != 0.0 or donor.labels[0] != 0.0:
        raise DataContractError("real_real_edit requires two known-real parents")
    first, second = _crop(base, epoch, window_samples, seed), _crop(donor, epoch, window_samples, seed)
    start = random.Random(_seed(base, epoch, seed, donor.id)).randrange(max(1, window_samples // 2))
    stop = min(window_samples, start + max(1, window_samples // 2))
    output = first.copy()
    output[start:stop] = second[start:stop]
    voice_present = (1.0 if 1.0 in (base.labels[3], donor.labels[3]) else
                     (0.0 if base.labels[3] == donor.labels[3] == 0.0 else math.nan))
    music_present = (1.0 if 1.0 in (base.labels[4], donor.labels[4]) else
                     (0.0 if base.labels[4] == donor.labels[4] == 0.0 else math.nan))
    labels = (0.0, 0.0 if voice_present == 1.0 else math.nan,
              0.0 if music_present == 1.0 else math.nan, voice_present, music_present)
    groups = tuple(dict.fromkeys((*base.groups, *donor.groups)))
    metadata: CompositionMetadata = {"kind": "real_real_edit", "parents": (base.id, donor.id),
                                     "parent_groups": groups, "epoch": epoch, "seed": seed}
    return Composition(output, labels, metadata)
