"""Practical cache-integrity check for temporal recovery."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from v5_dataset import Example
from v5_resume_temporal import inspect_cached_views


def test_cached_trio_is_reused_only_when_context_rate_and_frames_align(
    tmp_path: Path,
) -> None:
    # Given: a long source and its deterministic 20-second raw/voice/music cache trio.
    source = tmp_path / "source.wav"
    cache = tmp_path / "views"
    cache.mkdir()
    waveform = np.linspace(-0.5, 0.5, 320_123, dtype=np.float32)
    sf.write(source, waveform, 16_000, subtype="FLOAT")
    row = Example("sample", source, ("g",), (0.0, 0.0, np.nan, 1.0, 0.0), "speech")
    offset = int(hashlib.sha256(row.id.encode()).hexdigest()[:16], 16) % 124
    context = waveform[offset : offset + 320_000]
    for name in ("raw", "voice", "music"):
        sf.write(cache / f"sample_{name}.wav", context, 16_000, subtype="FLOAT")

    # When: the complete trio is audited, then one stem is made one frame short.
    views, missing, audit = inspect_cached_views([row], cache, training=True)
    sf.write(cache / "sample_music.wav", context[:-1], 16_000, subtype="FLOAT")
    _, rejected, damaged = inspect_cached_views([row], cache, training=True)

    # Then: valid work is skipped with its exact offset; misalignment is regenerated.
    assert missing == []
    assert audit == {"valid": 1, "absent": 0, "partial": 0, "invalid": 0}
    assert views[row.id].offset_seconds == offset / 16_000
    assert rejected == [row]
    assert damaged == {"valid": 0, "absent": 0, "partial": 0, "invalid": 1}
