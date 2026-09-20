#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_inventory.py
"""Freeze project-history exclusions and unscored music confirmation parents."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from v2_data import DataIntegrityError, sha256_file, stable_key

ROOT: Final = Path(__file__).resolve().parent
DATA: Final = ROOT / "_workspace/local_sampler_20260918/new_data"
OUT: Final = ROOT / "results/local_sampler_20260918"


@dataclass(frozen=True, slots=True)
class History:
    """Mutable accumulator of recorded historical identities and checksums."""
    artists: set[str] = field(default_factory=set)
    sonics_ids: set[str] = field(default_factory=set)
    identities: set[str] = field(default_factory=set)
    hashes: set[str] = field(default_factory=set)
    speakers: set[str] = field(default_factory=set)
    inputs: dict[str, str] = field(default_factory=dict)


def historical_inventory() -> History:
    """Read all old manifest/source CSVs plus original provenance inventories."""
    history = History()
    files = []
    for path in ROOT.rglob("*.csv"):
        if any(part in {".venv", ".git", "node_modules", "assets", "features", "local_sampler_20260918"}
               for part in path.parts):
            continue
        if re.search(r"manifest|sources|probes|provenance|train|validation|sealed|test|dev", path.name):
            files.append(path)
    for relative in ("_workspace/v2_sources/source_inventory.json", "recovery_validation/provenance.json",
                     "v4_validation/provenance.json", "_workspace/validation_sources/music_provenance.json",
                     "_workspace/validation_sources/speech_provenance.json"):
        files.append(ROOT / relative)
    for path in sorted(files):
        history.inputs[str(path.relative_to(ROOT))] = sha256_file(path)
        if path.suffix == ".csv":
            with path.open() as stream:
                records = list(csv.DictReader(stream))
        else:
            value = json.loads(path.read_text())
            records = value.get("sources", []) if path.name not in {"music_provenance.json", "speech_provenance.json"} else value
        for row in records:
            if row.get("artist_id"):
                history.artists.add(str(row["artist_id"]))
            if row.get("speaker_id"):
                history.speakers.add(str(row["speaker_id"]))
            group = str(row.get("group_id", ""))
            history.artists.update(re.findall(r"(?:artist:fma:|fma_artist_)(\d+)", group))
            history.sonics_ids.update(re.findall(r"(?:reference:sonics:|sonics_id_)(\d+)", group))
            history.speakers.update(atom for atom in group.split("|") if "speaker" in atom)
            if row.get("source") in {"suno", "udio"} and row.get("id"):
                history.sonics_ids.add(str(row["id"]))
            for key, value in row.items():
                if value is None:
                    continue
                text = str(value)
                if re.search(r"sha256|hash$", key) and re.fullmatch("[0-9a-f]{64}", text):
                    history.hashes.add(text)
                if key in {"ID", "id", "parent_id", "parent_ids", "content_id", "reference_id", "original_file", "track_id"}:
                    history.identities.add(text)
                    history.sonics_ids.update(re.findall(r"fresh_sonics_(\d+)", text))
    return history


def main() -> None:
    """Select before acquisition; all confirmation parents remain evaluation-only."""
    history = historical_inventory()
    fma_path = ROOT / "_workspace/v2_discovery/archive_probes/fma_raw_tracks.csv"
    sonics_path = DATA / "metadata/sonics_test.csv"
    fma = []
    with fma_path.open() as stream:
        for row in csv.DictReader(stream):
            license_url = urlparse(row["license_url"])
            duration = row["track_duration"].split(":")
            duration_valid = len(duration) == 2 and all(part.isdigit() for part in duration)
            allowed = license_url.hostname in {"creativecommons.org", "www.creativecommons.org"}
            allowed &= license_url.path.startswith(("/licenses/by/", "/licenses/by-sa/"))
            if not (allowed and duration_valid and row["artist_id"] and row["track_file"]):
                continue
            seconds = int(duration[0]) * 60 + int(duration[1])
            if row["artist_id"] not in history.artists and row["track_id"] not in history.identities and 65 <= seconds <= 360:
                fma.append(row)
    selected = []
    used_artists = set(history.artists)
    for row in sorted(fma, key=lambda r: stable_key("confirm-20260918|" + r["artist_id"] + "|" + r["track_id"])):
        if row["artist_id"] in used_artists:
            continue
        used_artists.add(row["artist_id"])
        selected.append(row | {"dataset": "FMA"})
        if len(selected) == 32:
            break
    with sonics_path.open() as stream:
        sonics = [row for row in csv.DictReader(stream) if row["label"] == "full fake"
                  and row["target"] == "1" and float(row["duration"]) >= 65 and row["id"] not in history.sonics_ids]
    used_sonics = set(history.sonics_ids)
    counts = {}
    for provider in ("suno", "udio"):
        count = 0
        for row in sorted((r for r in sonics if r["source"] == provider),
                          key=lambda r: stable_key("confirm-20260918|" + r["id"] + "|" + r["filename"])):
            if row["id"] in used_sonics:
                continue
            used_sonics.add(row["id"])
            selected.append(row | {"dataset": "SONICS"})
            count += 1
            if count == 16:
                break
        counts[provider] = count
    if len(selected) != 64 or counts != {"suno": 16, "udio": 16}:
        raise DataIntegrityError(f"Cannot meet frozen confirmation quota: {len(selected)}, {counts}")
    receipt = {
        "scope": "new recorded parents; same FMA and SONICS corpora and known generators",
        "historical_inputs": history.inputs, "historical_manifest_count": len(history.inputs),
        "excluded_artists": sorted(history.artists), "excluded_sonics_ids": sorted(history.sonics_ids),
        "excluded_speakers": sorted(history.speakers), "recorded_hashes": sorted(history.hashes),
        "recorded_identities": sorted(history.identities),
        "selection_metadata_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in (fma_path, sonics_path)},
        "fma_artists": 32, "sonics_ids": 32, "provider_counts": counts,
        "evaluated": False, "selection_uses_predictions": False,
        "limitations": ["No independent listening annotation", "Artist aliases, partial-content duplicates and pretraining overlap unknown",
                        "Historical PCM hashes use differing crops/quantization; hash exclusion is not acoustic fingerprinting"],
    }
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path = DATA / "selection_music.json"
    serialized = json.dumps(selected, indent=2, ensure_ascii=False) + "\n"
    if selection_path.exists() and selection_path.read_text() != serialized:
        raise DataIntegrityError("Previously frozen selection differs; do not replace it silently")
    selection_path.write_text(serialized)
    (OUT / "data_preparation_history.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"selected": len(selected), "historical_inputs": len(history.inputs), "old_artists": len(history.artists), "old_sonics_ids": len(history.sonics_ids)}))


if __name__ == "__main__":
    main()
