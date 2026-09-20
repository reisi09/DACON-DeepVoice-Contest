"""Protect whole-file coverage and cross-expert timeline integrity."""

from pathlib import Path

import numpy as np
import pytest

from v5_dataset import Example
from v5_features import assemble_features, window_starts


def test_every_sample_and_tail_are_covered() -> None:
    for length in (1, 64_599, 64_600, 64_601, 960_000):
        coverage = np.zeros(length, dtype=bool)
        for start in window_starts(length, 64_600):
            coverage[start:start + 64_600] = True
        assert coverage.all(), "A partial tail must not disappear from full-file inference"


def test_different_expert_timelines_are_rejected(tmp_path: Path) -> None:
    left, right = tmp_path / "voice", tmp_path / "music"
    left.mkdir()
    right.mkdir()
    np.savez(left / "sample.npz", times=np.array([0., 4.]), features=np.ones((2, 3)))
    np.savez(right / "sample.npz", times=np.array([0., 5.]), features=np.ones((2, 4)))
    example = Example("sample", tmp_path / "unused.wav", ("source",), (0., 0., 0., 1., 1.), "mixture")
    with pytest.raises(ValueError, match="alignment"):
        assemble_features([example], [left, right])
