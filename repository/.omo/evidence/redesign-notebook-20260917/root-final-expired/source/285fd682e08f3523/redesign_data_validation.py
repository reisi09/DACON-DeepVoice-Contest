# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile"]
# ///
# How to run: imported by `redesign_data.py`; it has no standalone command.
"""Source-manifest boundary validation for redesign data."""
from __future__ import annotations

import hashlib
import json
import csv
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf

SOURCE_FIELDS: Final = (
    "ID", "path", "audio_sha256", "source", "license", "license_url", "source_url", "parent_ids",
    "artist_id", "speaker_id", "generator", "kind", "domain", "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE",
    "VOICE_PRESENT", "MUSIC_PRESENT",
)
LABELS: Final = ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")
KINDS: Final = {"speech", "instrumental", "singing", "mixed", "neither"}


class DataContractError(RuntimeError):
    """A source or prepared row violates the redesign data contract."""


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV strings verbatim, including empty unknown-label cells."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_source_template(path: Path) -> None:
    """Write a header-only manifest template; no provenance values are invented."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=SOURCE_FIELDS).writeheader()


def sealed_panel_identity_sha256(rows: list[dict[str, str]]) -> str:
    """Hash only path- and label-independent sealed panel content identity."""
    records = sorted((
        {"ID": row["ID"], "audio_sha256": row["audio_sha256"], "groups": sorted(set(row["group_id"].split("|")))}
        for row in rows if row["split"] == "sealed"
    ), key=lambda record: record["ID"])
    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _parse_parent_ids(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DataContractError(f"parent_ids is not JSON: {error.msg}") from error
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise DataContractError("parent_ids must be a non-empty JSON string array")
    return value


def _validate_labels(row: dict[str, str]) -> None:
    if row["kind"] not in KINDS:
        raise DataContractError(f"unknown kind {row['kind']!r}")
    if any(row[label] not in {"", "0", "1"} for label in LABELS):
        raise DataContractError("labels must be 0, 1, or empty")
    allowed_presence = {
        "speech": ({"", "1"}, {"", "0"}),
        "instrumental": ({"", "0"}, {"", "1"}),
        "singing": ({"", "1"}, {"", "0", "1"}),
        "mixed": ({"", "1"}, {"", "1"}),
        "neither": ({"", "0"}, {"", "0"}),
    }[row["kind"]]
    if row["VOICE_PRESENT"] not in allowed_presence[0] or row["MUSIC_PRESENT"] not in allowed_presence[1]:
        raise DataContractError("kind contradicts component presence")
    for fake, present in (("VOICE_FAKE", "VOICE_PRESENT"), ("MUSIC_FAKE", "MUSIC_PRESENT")):
        if row[present] == "0" and row[fake] != "":
            raise DataContractError(f"{fake} claims authenticity for an absent component")
    known = [row[fake] for fake, present in (("VOICE_FAKE", "VOICE_PRESENT"), ("MUSIC_FAKE", "MUSIC_PRESENT")) if row[present] == "1"]
    if row["FILE_FAKE"] == "0" and "1" in known:
        raise DataContractError("FILE_FAKE=0 contradicts a fake component")
    if row["FILE_FAKE"] == "1" and known and all(value == "0" for value in known):
        raise DataContractError("FILE_FAKE=1 contradicts all known-real components")


def validate_row(raw: dict[str, str], base: Path) -> tuple[dict[str, str], list[str]]:
    """Parse one untrusted source row and fully decode its declared audio."""
    missing = [field for field in SOURCE_FIELDS if field not in raw]
    if missing:
        raise DataContractError(f"missing columns: {missing}")
    row = {field: raw[field].strip() for field in SOURCE_FIELDS}
    if not row["ID"]:
        raise DataContractError("empty ID")
    for field in ("source", "license", "license_url", "source_url", "domain"):
        if not row[field]:
            raise DataContractError(f"missing provenance field {field}")
    parents = _parse_parent_ids(row["parent_ids"])
    _validate_labels(row)
    audio_path = Path(row["path"])
    audio_path = audio_path if audio_path.is_absolute() else base / audio_path
    if not audio_path.is_file():
        raise DataContractError(f"audio missing: {audio_path}")
    if _digest(audio_path) != row["audio_sha256"]:
        raise DataContractError(f"audio checksum mismatch: {audio_path}")
    try:
        info = sf.info(audio_path)
        audio, _ = sf.read(audio_path, dtype="float32", always_2d=True)
    except (RuntimeError, ValueError) as error:
        raise DataContractError(f"audio decode failed: {audio_path}: {error}") from error
    if info.frames <= 0 or info.samplerate <= 0 or not np.isfinite(audio).all():
        raise DataContractError(f"invalid audio samples: {audio_path}")
    row["path"] = str(audio_path.resolve())
    pcm_identity = info.samplerate.to_bytes(8, "little") + info.channels.to_bytes(4, "little") + audio.astype("<f4", copy=False).tobytes()
    row["_pcm_sha256"] = hashlib.sha256(pcm_identity).hexdigest()
    row.update(pair_id="", view="observed", event_intervals="[]", duration_seconds=f"{info.duration:.9f}")
    return row, parents
