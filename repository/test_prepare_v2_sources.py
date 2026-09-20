import io
import struct
import wave
import zlib
from pathlib import Path

import pytest

import prepare_v2_sources as sources


def _local_member(payload: bytes, *, crc: int | None = None, descriptor: bool = False) -> tuple[bytes, dict[str, int | str]]:
    name = b"audio.mp3"
    compressed = zlib.compress(payload)[2:-4]
    expected_crc = zlib.crc32(payload) if crc is None else crc
    local_crc, local_compressed, local_size = (0, 0, 0) if descriptor else (expected_crc, len(compressed), len(payload))
    header = struct.pack("<4s5H3L2H", b"PK\x03\x04", 20, 8 if descriptor else 0, 8, 0, 0, local_crc, local_compressed, local_size, len(name), 0)
    return header + name + compressed, {
        "filename": "audio.mp3", "compress_size": len(compressed), "file_size": len(payload),
        "header_offset": 0, "compress_type": 8, "CRC": expected_crc,
    }


def test_extract_member_rejects_corrupt_crc(tmp_path: Path) -> None:
    # Given a valid local ZIP member whose central-directory CRC is wrong.
    blob, entry = _local_member(b"real audio bytes", crc=7)
    # When the retained member is decoded, then corruption is rejected before use.
    with pytest.raises(sources.SourceIntegrityError, match="CRC"):
        sources.extract_local_member(blob, entry, tmp_path / "audio.mp3")


def test_extract_member_uses_central_sizes_with_data_descriptor(tmp_path: Path) -> None:
    # Given a real ZIP member whose bit-3 local sizes are zero and central sizes are authoritative.
    blob, entry = _local_member(b"descriptor audio bytes", descriptor=True)
    # When extracted by range, then its payload is complete and CRC-verified.
    output = sources.extract_local_member(blob, entry, tmp_path / "audio.mp3")
    assert output.read_bytes() == b"descriptor audio bytes"


def test_inventory_rejects_old_group_and_duplicate_pcm(tmp_path: Path) -> None:
    # Given two licensed sources that repeat PCM and one crosses the old bank.
    rows = [
        sources.SourceRow("a", "p/a.wav", "o/a", "mlaad_real", "old:g", "real", "a", "u", "r", "x", "same", "CC", "a", sources.real_voice_labels()),
        sources.SourceRow("b", "p/b.wav", "o/b", "mlaad_fake", "new:g", "fake", "b", "u", "r", "y", "same", "CC", "b", sources.fake_voice_labels()),
    ]
    # When inventory validation runs, then both source and PCM leakage are fatal.
    with pytest.raises(sources.SourceIntegrityError):
        sources.validate_inventory(rows, {"old:g"}, {"old-pcm"})


def test_first_valid_refills_after_more_than_four_rejections() -> None:
    # Given SHA-ordered candidates with five invalid decodes before enough valid rows.
    candidates = list(range(30))
    # When selection runs in bounded parallel batches, then it refills through later candidates.
    chosen = sources.first_valid(candidates, lambda value: None if value < 5 else value, 20)
    assert chosen == list(range(5, 25))


def test_distinct_artists_does_not_spend_slots_on_second_tracks() -> None:
    # Given two valid tracks by one artist followed by another artist.
    items = [{"artist_id": "a", "track_id": "1"}, {"artist_id": "a", "track_id": "2"}, {"artist_id": "b", "track_id": "3"}]
    # When two references are reserved, then they represent two independent artists.
    assert [item["track_id"] for item in sources.distinct_artists(items, set(), 2)] == ["1", "3"]


def test_inventory_pcm_hash_names_the_persisted_prepared_wave(tmp_path: Path) -> None:
    # Given source PCM loud enough for the prepared writer's peak normalization.
    original = tmp_path / "original.wav"
    samples = struct.pack("<h", 32767) * (4 * 16000)
    with wave.open(str(original), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        handle.writeframes(samples)
    # When retained, then inventory hashes the bytes decoded from the written prepared file.
    row = sources._retain(tmp_path, original, "mlaad_real", "g", "p", "c", "u", "r", "CC", "a", sources.real_voice_labels())
    assert row.pcm_sha256 == sources.pcm_sha256(sources.decode_audio(tmp_path / row.prepared_path))
