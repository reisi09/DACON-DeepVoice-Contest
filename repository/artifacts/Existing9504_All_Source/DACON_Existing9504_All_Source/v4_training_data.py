"""Lazy CPU batches for full-source V4 epochs."""

from __future__ import annotations

import math
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from v4_batches import (apply_gain, epoch_crop, epoch_crop_start, epoch_order, p3_augment,
                        p3_variant, p4_splice, p4_starts)
from v4_data import load_audio
from v4_model import LoadedModel, prepare_windows
from v4_training import Batch

Row = Mapping[str, str]


@lru_cache(maxsize=None)
def _audio(path: Path) -> np.ndarray:
    """Cache only deterministic decoded PCM, never learned representations."""
    return load_audio(path)


def _weights(rows: Sequence[Row]) -> dict[str, float]:
    def provider(row: Row) -> str:
        parts = row["source"].split("/")
        return parts[1] if len(parts) >= 3 and parts[0].lower() == "echoes" else row["source"]

    real = [row for row in rows if row["MUSIC_FAKE"] == "0"]
    regular = [row for row in real if "nsynth" not in row["source"].lower()]
    nsynth = [row for row in real if "nsynth" in row["source"].lower()]
    fake = [row for row in rows if row["MUSIC_FAKE"] == "1"]
    fake_sources = Counter(provider(row) for row in fake)
    if not regular or not fake or not fake_sources:
        raise ValueError("weighted training requires real and fake music from named providers")
    weights = {row["ID"]: (0.45 if nsynth else 0.5) / len(regular) for row in regular}
    if nsynth:
        weights.update({row["ID"]: 0.05 / len(nsynth) for row in nsynth})
    provider_mass = 0.5 / len(fake_sources)
    weights.update({row["ID"]: provider_mass / fake_sources[provider(row)] for row in fake})
    return weights


class EpochBatches(Sequence[Batch]):
    """Generate one CPU microbatch on demand so an epoch is never retained in RAM/GPU tensors."""

    def __init__(
        self,
        loaded: LoadedModel,
        rows: Sequence[Row],
        root: Path,
        epoch: int,
        *,
        microbatch: int,
        p3: bool,
        p4: bool,
        p4_control: bool = False,
        codec_cache: Path,
        deadline: float,
    ) -> None:
        self.loaded = loaded
        self.rows = epoch_order(rows, epoch)
        self.root = root
        self.epoch = epoch
        self.microbatch = microbatch
        self.p3 = p3
        self.p4 = p4
        self.p4_control = p4_control
        self.codec_cache = codec_cache
        self.deadline = deadline
        self.weights = _weights(rows)
        self.original_batches = math.ceil(len(rows) / microbatch)
        self.splice_pairs = len(rows) // 2 if p4 else 0
        self.real = [row for row in self.rows if row["MUSIC_FAKE"] == "0"]
        self.fake = [row for row in self.rows if row["MUSIC_FAKE"] == "1"]

    def __len__(self) -> int:
        return self.original_batches + self.splice_pairs

    def __getitem__(self, index: int | slice) -> Batch | list[Batch]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if time.time() >= self.deadline:
            raise TimeoutError("deadline reached during CPU batch preparation")
        return self._original(index) if index < self.original_batches else self._splice(index - self.original_batches)

    def _original(self, index: int) -> Batch:
        selected = self.rows[index * self.microbatch : (index + 1) * self.microbatch]
        audios = [_audio(self.root / row["path"]) for row in selected]
        if self.p4:
            starts = [p4_starts(len(audio), self.loaded.window_samples, row["ID"], self.epoch)
                      for audio, row in zip(audios, selected, strict=True)]
            prepared = prepare_windows(self.loaded, audios, starts=starts)
        else:
            crops = [apply_gain(epoch_crop(audio, row["ID"], self.epoch, self.loaded.window_samples),
                                row["ID"], self.epoch)
                     for audio, row in zip(audios, selected, strict=True)]
            if self.p3:
                crops = [p3_augment(audio, row["ID"], self.epoch, str(self.codec_cache))
                         for audio, row in zip(crops, selected, strict=True)]
            prepared = prepare_windows(self.loaded, crops)
        scale = 0.5 if self.p4 else 1.0
        records = tuple(json.dumps({"ID": row["ID"], "epoch": self.epoch,
            "source_samples": len(audio), "window_samples": self.loaded.window_samples,
            "starts": list(starts[index]) if self.p4 else [epoch_crop_start(
                len(audio), row["ID"], self.epoch, self.loaded.window_samples)],
            "augmentation": "p4_bag" if self.p4 else (p3_variant(row["ID"], self.epoch)
                if self.p3 else "gain")}, sort_keys=True)
            for index, (row, audio) in enumerate(zip(selected, audios, strict=True)))
        return self._batch(tuple(row["ID"] for row in selected), prepared,
                           [float(row["MUSIC_FAKE"]) for row in selected],
                           [scale * self.weights[row["ID"]] for row in selected], records)

    def _splice(self, index: int) -> Batch:
        if not self.real or not self.fake:
            raise ValueError("P4 needs real and fake music parents")
        base = self._long(self.real, index)
        fake = self._long(self.fake, index)
        control = self._long(self.real, index + 1)
        audio = _audio(self.root / base["path"])
        starts = p4_starts(len(audio), self.loaded.window_samples, base["ID"], self.epoch)
        edited, labels, targets, identifiers = [], [], [], []
        donors = ((fake, True), (control, False)) if not self.p4_control else ((control, False), (base, False))
        for slot, (donor, is_fake) in enumerate(donors):
            value, known, target = p4_splice(audio, _audio(self.root / donor["path"]), starts,
                self.loaded.window_samples, f"{base['ID']}|{donor['ID']}", self.epoch, fake_donor=is_fake)
            edited.append(value)
            labels.append(known)
            targets.append(target)
            identifiers.append(f"p4:{self.epoch}:{index}:{slot}:{base['ID']}|{donor['ID']}")
        prepared = prepare_windows(self.loaded, edited, starts=(starts, starts), window_targets=labels)
        weight = 0.25 / self.splice_pairs
        records = tuple(json.dumps({"ID": identifier, "epoch": self.epoch,
            "parents": identifier.split(":", 4)[-1].split("|"), "starts": list(starts),
            "window_samples": self.loaded.window_samples, "source_samples": len(audio),
            "augmentation": "partial_fake_splice" if target else "matched_real_splice"},
            sort_keys=True) for identifier, target in zip(identifiers, targets, strict=True))
        return self._batch(tuple(identifiers), prepared, targets, (weight, weight), records)

    def _long(self, rows: Sequence[Row], start: int) -> Row:
        for offset in range(len(rows)):
            row = rows[(start + offset) % len(rows)]
            if len(_audio(self.root / row["path"])) >= 5 * 16_000:
                return row
        raise ValueError("P4 has no five-second parent in the requested class")

    @staticmethod
    def _batch(ids: tuple[str, ...], prepared, targets: Sequence[float], weights: Sequence[float],
               records: tuple[str, ...]) -> Batch:
        return Batch(ids, prepared.values, torch.tensor(targets), torch.tensor(weights),
                     prepared.file_indices, prepared.window_targets, prepared.starts, records)
