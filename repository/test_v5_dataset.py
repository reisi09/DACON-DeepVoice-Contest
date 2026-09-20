"""Behavioral checks for the leakage-aware V5 data boundary."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from v5_dataset import (
    DataLeakageError,
    Example,
    compose_mixture,
    load_splits,
    real_real_edit,
    training_crop,
)

ROOT = Path(__file__).resolve().parent


def test_load_splits_holds_out_whole_nsynth_groups_without_leakage() -> None:
    # Given: the immutable V4 train/dev manifests.
    data = ROOT / "v4_validation"

    # When: V5 derives its training and identifiable VP evaluation splits.
    train, dev, audit = load_splits(data)

    # Then: five complete four-row NSynth groups move, and every leakage check is clean.
    held = set(audit["nsynth_holdout"]["groups"])
    assert (len(train), len(dev)) == (1496, 230)
    assert len(held) == 5
    assert sum(example.groups[0] in held for example in dev) == 20
    assert not any(held.intersection(example.groups) for example in train)
    assert all(example.labels[3] == 0.0 for example in dev if example.groups[0] in held)
    assert audit["overlap"] == {"groups": 0, "paths": 0, "sha256": 0}


def test_training_crop_replays_and_masks_unproven_partial_fake_truth() -> None:
    # Given: one real full-source row and one partial-edit row sharing its readable audio.
    train, _, _ = load_splits(ROOT / "v4_validation")
    source = train[0]
    partial = Example("partial", source.path, source.groups, (1.0, math.nan, 1.0, 1.0, 1.0), "splice")

    # When: the same augmented crop is requested twice.
    first, labels = training_crop(partial, 3, 16_000, True, 41)
    second, replay_labels = training_crop(partial, 3, 16_000, True, 41)

    # Then: the waveform replays exactly and clip authenticity is not invented.
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32 and len(first) == 16_000
    assert math.isnan(labels[0]) and math.isnan(labels[1]) and math.isnan(labels[2])
    assert labels == replay_labels


def test_composition_preserves_exact_voice_truth_and_real_control() -> None:
    # Given: actual same-split real/fake speech and non-vocal/possibly-vocal music parents.
    train, _, _ = load_splits(ROOT / "v4_validation")
    fake_speech = next(example for example in train if example.panel == "speech" and example.labels[1] == 1.0)
    real_speech = next(example for example in train if example.panel == "speech" and example.labels[1] == 0.0)
    nsynth = next(example for example in train if example.labels[3] == 0.0 and example.labels[4] == 1.0)
    generated = next(example for example in train if example.panel == "music_seen")
    real_music = next(example for example in train if example.panel == "music_real" and example.labels[3] != 0.0)

    # When: mixtures and a matched real/real edit are composed deterministically.
    fake_voice = compose_mixture(fake_speech, generated, 0, 16_000, 9)
    known_nonvocal = compose_mixture(real_speech, nsynth, 0, 16_000, 9)
    possible_generated_voice = compose_mixture(real_speech, generated, 0, 16_000, 9)
    control = real_real_edit(real_music, nsynth, 0, 16_000, 9)

    # Then: fake voice follows only supported parent truth, and the control stays real.
    assert fake_voice.labels == (1.0, 1.0, 1.0, 1.0, 1.0)
    assert known_nonvocal.labels == (0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isnan(possible_generated_voice.labels[1])
    assert control.labels[0] == control.labels[2] == 0.0
    assert np.array_equal(fake_voice.waveform, compose_mixture(fake_speech, generated, 0, 16_000, 9).waveform)
    assert fake_voice.metadata["parents"] == (fake_speech.id, generated.id)

    # Then: a parent from the derived dev split cannot enter a train composition.
    _, dev, _ = load_splits(ROOT / "v4_validation")
    with pytest.raises(DataLeakageError, match="cross-split"):
        compose_mixture(fake_speech, next(example for example in dev if example.panel == "speech"), 0, 16_000, 9)
