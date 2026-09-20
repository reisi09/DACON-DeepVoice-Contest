"""Content-bound receipts and cache records for redesign features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
from numpy.typing import NDArray

RECORD_KEYS: Final = (
    "stem", "raw", "aux", "starts", "valid", "stem_audible",
    "voice_embedding", "voice_logits", "voice_probability", "presence", "evidence",
)
CONFIG: Final = {
    "audio_sample_rate": 16_000,
    "mert_sample_rate": 24_000,
    "window_samples": 64_600,
    "views": ["stem", "raw"],
    "mert_pooling": "last_hidden_state_mean",
    "voice_embedding": "df_arena_conformer_fc5_input",
    "presence": "panns_per_window_class_max",
}


class FeatureRow(TypedDict):
    ID: str
    path: str
    audio_sha256: str


class FeatureRecord(TypedDict):
    stem: NDArray[np.float32]
    raw: NDArray[np.float32]
    aux: NDArray[np.float32]
    starts: NDArray[np.int64]
    valid: NDArray[np.bool_]
    stem_audible: NDArray[np.bool_]
    voice_embedding: NDArray[np.float32]
    voice_logits: NDArray[np.float32]
    voice_probability: NDArray[np.float32]
    presence: NDArray[np.float32]
    evidence: NDArray[np.float32]


class CacheError(ValueError):
    """A cache record is stale, incomplete, or internally inconsistent."""


def canonical_sha(value: Mapping[str, str] | dict[str, int | str | list[str]]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def extractor_sha256() -> str:
    """Bind records to all source files that define extraction and cache semantics."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in ("redesign_features.py", "redesign_features_cache.py", "redesign_features_runtime.py"):
        path = root / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def config_sha256() -> str:
    return canonical_sha(CONFIG)


def receipt_payload(assets: Path, asset_hashes: Mapping[str, str]) -> dict[str, str]:
    return {
        "extractor_sha256": extractor_sha256(),
        "model_sha256": canonical_sha(dict(asset_hashes)),
        "source_sha256": hashlib.sha256((assets / "baseline/script.py").read_bytes()).hexdigest(),
        "config_sha256": config_sha256(),
    }


def install_receipt(cache: Path, payload: Mapping[str, str]) -> str:
    encoded = (json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n").encode()
    identity = hashlib.sha256(encoded).hexdigest()
    directory = cache / "receipts"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{identity}.json"
    if target.exists() and target.read_bytes() != encoded:
        raise CacheError("Immutable extraction receipt collision")
    if not target.exists():
        target.write_bytes(encoded)
    return identity


def load_receipt(cache: Path, identity: str) -> dict[str, str]:
    path = cache / "receipts" / f"{identity}.json"
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != identity:
        raise CacheError("Extraction receipt is missing or modified")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("extractor_sha256") != extractor_sha256() or raw.get("config_sha256") != config_sha256():
        raise CacheError("Cache extractor or configuration is stale")
    return {key: str(raw[key]) for key in ("extractor_sha256", "model_sha256", "source_sha256", "config_sha256")}


def record_path(cache: Path, identifier: str) -> Path:
    return cache / "records" / f"{hashlib.sha256(identifier.encode()).hexdigest()}.npz"


def validate_record(values: FeatureRecord, identifier: str) -> None:
    lengths = {len(values[key]) for key in RECORD_KEYS if key != "evidence"}
    if len(lengths) != 1 or not lengths or values["evidence"].shape != (3,):
        raise CacheError(f"Misaligned cached features: {identifier}")
    count = lengths.pop()
    shapes = {
        "stem": (count, 1024), "raw": (count, 1024), "aux": (count, 6),
        "starts": (count,), "valid": (count,), "stem_audible": (count,),
        "voice_embedding": (count, 1280), "voice_logits": (count,),
        "voice_probability": (count,), "presence": (count, 2),
    }
    if count == 0 or any(values[key].shape != shape for key, shape in shapes.items()):
        raise CacheError(f"Invalid cached feature dimensions: {identifier}")
    float_keys = ("stem", "raw", "aux", "voice_embedding", "voice_logits", "voice_probability", "presence", "evidence")
    if any(values[key].dtype != np.float32 for key in float_keys):
        raise CacheError(f"Cached floating-point dtype is not float32: {identifier}")
    if values["starts"].dtype != np.int64 or values["valid"].dtype != np.bool_ or values["stem_audible"].dtype != np.bool_:
        raise CacheError(f"Cached timeline or masks use the wrong dtype: {identifier}")
    if any(not np.isfinite(values[key]).all() for key in RECORD_KEYS):
        raise CacheError(f"Non-finite cached features: {identifier}")
    if np.any(np.diff(values["starts"]) <= 0) or values["starts"][0] != 0:
        raise CacheError(f"Invalid window timeline: {identifier}")
    if not np.array_equal(values["valid"], values["aux"][:, 5] > 0):
        raise CacheError(f"Valid-time mask disagrees with duration ratio: {identifier}")
    if not np.array_equal(values["stem_audible"], values["aux"][:, 4].astype(bool)):
        raise CacheError(f"Stem audibility fields disagree: {identifier}")
    if np.any((values["voice_probability"] < 0) | (values["voice_probability"] > 1)):
        raise CacheError(f"Voice probability outside [0,1]: {identifier}")
    if np.any((values["presence"] < 0) | (values["presence"] > 1)):
        raise CacheError(f"Presence probability outside [0,1]: {identifier}")
    reconstructed = 1 / (1 + np.exp(-values["voice_logits"].astype(np.float64)))
    if not np.allclose(values["voice_probability"], reconstructed, atol=1e-6, rtol=0):
        raise CacheError(f"Voice logits and probabilities disagree: {identifier}")
    expected = np.asarray([
        values["voice_probability"].max(), values["presence"][:, 0].max(),
        values["presence"][:, 1].max(),
    ], dtype=np.float32)
    if not np.allclose(values["evidence"], expected, atol=1e-6, rtol=0):
        raise CacheError(f"Bag evidence is not the max of aligned windows: {identifier}")


def save_record(cache: Path, row: FeatureRow, receipt: str, values: FeatureRecord) -> tuple[Path, str]:
    validate_record(values, row["ID"])
    target = record_path(cache, row["ID"])
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"ID": row["ID"], "audio_sha256": row["audio_sha256"], "receipt": receipt}, sort_keys=True)
    temporary = target.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=np.asarray(metadata),
            stem=values["stem"], raw=values["raw"], aux=values["aux"], starts=values["starts"],
            valid=values["valid"], stem_audible=values["stem_audible"],
            voice_embedding=values["voice_embedding"], voice_logits=values["voice_logits"],
            voice_probability=values["voice_probability"], presence=values["presence"],
            evidence=values["evidence"],
        )
    temporary.replace(target)
    return target, hashlib.sha256(target.read_bytes()).hexdigest()


def read_record(cache: Path, row: FeatureRow, receipt: str, expected_sha256: str | None = None) -> FeatureRecord:
    target = record_path(cache, row["ID"])
    if expected_sha256 is not None and (not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected_sha256):
        raise CacheError(f"Cache file checksum mismatch: {row['ID']}")
    try:
        with np.load(target, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].item())
            values = {key: archive[key] for key in RECORD_KEYS}
    except (FileNotFoundError, KeyError, ValueError, OSError, json.JSONDecodeError) as error:
        raise CacheError(f"Unreadable cache record: {row['ID']}") from error
    expected = {"ID": row["ID"], "audio_sha256": row["audio_sha256"], "receipt": receipt}
    if metadata != expected:
        raise CacheError(f"Stale cache identity: {row['ID']}")
    record = FeatureRecord(**values)
    validate_record(record, row["ID"])
    return record
