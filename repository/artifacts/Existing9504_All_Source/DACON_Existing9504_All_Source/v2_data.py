"""Small shared data/audio primitives for the V2 bank."""
from __future__ import annotations

import hashlib
import bz2
import subprocess
import struct
import wave
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypeVar, assert_never

import imageio_ffmpeg
import numpy as np

LABEL_NAMES: Final = ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")
Candidate = TypeVar("Candidate")
Loaded = TypeVar("Loaded")


class DataIntegrityError(RuntimeError):
    """A retained data item failed an integrity boundary."""


@dataclass(frozen=True, slots=True)
class Labels:
    FILE_FAKE: str
    VOICE_FAKE: str
    MUSIC_FAKE: str
    VOICE_PRESENT: str
    MUSIC_PRESENT: str


@dataclass(frozen=True, slots=True)
class SourceRow:
    id: str
    prepared_path: str
    original_path: str
    source_kind: str
    group_id: str
    provider: str
    content_id: str
    source_url: str
    revision: str
    source_sha256: str
    pcm_sha256: str
    license: str
    attribution: str
    labels: Labels

    def json_record(self) -> dict[str, str | dict[str, str]]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_audio(path: Path, limit_seconds: int | None = 10) -> np.ndarray:
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        check=True, capture_output=True,
    )
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not np.isfinite(audio).all():
        raise ArithmeticError(f"non-finite decoded audio: {path}")
    if limit_seconds is None:
        return audio
    limit = limit_seconds * 16000
    start = max(0, (len(audio) - limit) // 2)
    return audio[start:start + limit]


def pcm_sha256(audio: np.ndarray) -> str:
    normalized = np.asarray(audio, dtype="<f4")
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    pcm = np.round(audio / max(1.0, peak / 0.98) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        handle.writeframes(pcm.tobytes())


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def first_valid(candidates: Sequence[Candidate], loader: Callable[[Candidate], Loaded | None], count: int) -> list[Loaded]:
    accepted: list[Loaded] = []
    for start in range(0, len(candidates), 8):
        with ThreadPoolExecutor(max_workers=8) as executor:
            accepted.extend(item for item in executor.map(loader, candidates[start:start + 8]) if item is not None)
        if len(accepted) >= count:
            return accepted[:count]
    return accepted


def extract_local_member(blob: bytes, entry: dict[str, int | str], output: Path) -> Path:
    if len(blob) < 30:
        raise DataIntegrityError("ZIP member local header is truncated")
    signature, _, _, method, _, _, _, _, _, name_len, extra_len = struct.unpack("<4s5H3L2H", blob[:30])
    if signature != b"PK\x03\x04" or method != int(entry["compress_type"]):
        raise DataIntegrityError("ZIP member header or compression method mismatch")
    start = 30 + name_len + extra_len
    compressed = blob[start:start + int(entry["compress_size"])]
    if method not in (0, 8, 12):
        raise DataIntegrityError(f"unsupported ZIP compression method {method}")
    match method:
        case 0:
            payload = compressed
        case 8:
            payload = zlib.decompress(compressed, -15)
        case 12:
            payload = bz2.decompress(compressed)
        case unreachable:
            assert_never(unreachable)
    if len(payload) != int(entry["file_size"]):
        raise DataIntegrityError("ZIP member size mismatch")
    if zlib.crc32(payload) != int(entry["CRC"]):
        raise DataIntegrityError("ZIP member CRC mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return output
