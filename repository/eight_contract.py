"""Manifest contract and semantic integrity audit for the eight-study dataset."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Final, TypedDict

import soundfile as sf

MANIFEST_FIELDS: Final = (
    "ID", "path", "parent_id", "parent_ids", "group_id", "split", "source",
    "generator", "kind", "variant", "probe_family", "pair_id", "FILE_FAKE",
    "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT",
    "audio_sha256", "duration_seconds", "event_intervals",
)
LABELS: Final = ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")
SPLITS: Final = frozenset({"train", "calibration", "dev", "test"})


class ManifestRow(TypedDict):
    ID: str
    path: str
    parent_id: str
    parent_ids: str
    group_id: str
    split: str
    source: str
    generator: str
    kind: str
    variant: str
    probe_family: str
    pair_id: str
    FILE_FAKE: str
    VOICE_FAKE: str
    MUSIC_FAKE: str
    VOICE_PRESENT: str
    MUSIC_PRESENT: str
    audio_sha256: str
    duration_seconds: str
    event_intervals: str


class ContractError(RuntimeError):
    """A dataset artifact violates the manifest or provenance contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_manifest(root: str | Path) -> Path:
    base = Path(root)
    remote = base / "source" / "dataset" / "manifest.csv"
    return remote if remote.exists() else base / "_workspace" / "eight_h100_20260917" / "dataset" / "manifest.csv"


def read_manifest(path: str | Path) -> list[ManifestRow]:
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ContractError("manifest fields or order changed")
        return [ManifestRow(**row) for row in reader]


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_manifest(dataset: str | Path) -> dict[str, int | dict[str, int]]:
    base = Path(dataset)
    rows = read_manifest(base / "manifest.csv")
    row_map = {row["ID"]: row for row in rows}
    if len(row_map) != len(rows):
        raise ContractError("duplicate manifest ID")
    source_map: dict[str, dict[str, str]] = {}
    source_path = base / "sources.csv"
    if source_path.exists():
        with source_path.open(newline="") as stream:
            for source in csv.DictReader(stream):
                if source["ID"] in source_map:
                    raise ContractError(f"duplicate source ID: {source['ID']}")
                source_map[source["ID"]] = source
    ids: set[str] = set()
    hashes: set[str] = set()
    atom_split: dict[str, str] = {}
    pair_shape: dict[str, tuple[str, int]] = {}
    for row in rows:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ContractError(f"unsafe relative path: {path}")
        audio_path = base / path
        if row["ID"] in ids or row["audio_sha256"] in hashes:
            raise ContractError(f"duplicate identity or audio: {row['ID']}")
        ids.add(row["ID"])
        hashes.add(row["audio_sha256"])
        if row["split"] not in SPLITS or any(row[name] not in {"", "0", "1"} for name in LABELS):
            raise ContractError(f"invalid split or label: {row['ID']}")
        if sha256_file(audio_path) != row["audio_sha256"]:
            raise ContractError(f"audio hash mismatch: {row['ID']}")
        info = sf.info(audio_path)
        frames = int(info.frames)
        if info.samplerate != 16000 or info.channels != 1 or abs(frames / 16000 - float(row["duration_seconds"])) > 1e-7:
            raise ContractError(f"audio format or duration mismatch: {row['ID']}")
        parents = json.loads(row["parent_ids"])
        events = json.loads(row["event_intervals"])
        if not isinstance(parents, list) or not parents or not isinstance(events, list):
            raise ContractError(f"invalid lineage JSON: {row['ID']}")
        atoms = row["group_id"].split("|")
        if atoms != sorted(set(atoms)) or any(not atom for atom in atoms):
            raise ContractError(f"non-canonical provenance atoms: {row['ID']}")
        for atom in atoms:
            previous = atom_split.setdefault(atom, row["split"])
            if previous != row["split"]:
                raise ContractError(f"cross-split contributor: {atom}")
        for parent in parents:
            parent_row = row_map.get(parent)
            parent_source = source_map.get(parent)
            parent_group = parent_row["group_id"] if parent_row else parent_source["group_id"] if parent_source else ""
            if not parent_group or not set(parent_group.split("|")).issubset(atoms):
                raise ContractError(f"unresolved or truncated parent provenance: {row['ID']} <- {parent}")
        for event in events:
            if not isinstance(event, dict) or not {"component", "start", "end", "label"} <= event.keys():
                raise ContractError(f"invalid event: {row['ID']}")
            if not (0 <= float(event["start"]) < float(event["end"]) <= frames / 16000 + 1e-7):
                raise ContractError(f"event outside audio: {row['ID']}")
        if row["pair_id"]:
            shape = (row["split"], frames)
            old = pair_shape.setdefault(row["pair_id"], shape)
            if old != shape:
                raise ContractError(f"paired timing/split mismatch: {row['pair_id']}")
    counts = Counter(row["split"] for row in rows)
    families = Counter(row["probe_family"] for row in rows)
    return {"rows": len(rows), "audio_hashes": len(hashes), "provenance_atoms": len(atom_split),
            "splits": dict(counts), "probe_families": dict(families)}


def adversarial_check(dataset: str | Path) -> dict[str, bool]:
    base = Path(dataset)
    rows = list(csv.DictReader((base / "manifest.csv").open()))[:2]
    outcomes: list[bool] = []
    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        for row in rows:
            target = fixture / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / row["path"], target)
        cases = ([dict(row) for row in rows], [dict(row) for row in rows])
        cases[0][0]["audio_sha256"] = "0" * 64
        cases[1][1]["group_id"] = cases[1][0]["group_id"]
        cases[1][1]["split"] = next(value for value in ("train", "dev") if value != cases[1][0]["split"])
        for case in cases:
            write_manifest(fixture / "manifest.csv", [ManifestRow(**row) for row in case])
            try:
                validate_manifest(fixture)
            except ContractError:
                outcomes.append(True)
            else:
                outcomes.append(False)
    if outcomes != [True, True]:
        raise ContractError("adversarial fixture was accepted")
    return {"actual_dataset_valid": True, "corrupted_audio_rejected": True, "cross_split_contributor_rejected": True}
