#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_sunocaps.py
"""Acquire one generated track per SunoCaps prompt group from public Kaggle v7."""
from __future__ import annotations

import csv
import json
import re
import subprocess
import zipfile
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final
from urllib.parse import quote

from eight_audio import core, manifest_row, write
from eight_contract import ManifestRow, write_manifest
from new_panel_data_inventory import DATA, OUT, ROOT
from v2_data import DataIntegrityError, decode_audio, pcm_sha256, sha256_file

BASE: Final = "https://www.kaggle.com/api/v1/datasets/download/miguelcivit/sunocaps/"
VERSION: Final = 7
LICENSE: Final = "https://creativecommons.org/licenses/by-sa/4.0/"
TARGET: Final = DATA / "sunocaps"


@dataclass(frozen=True, slots=True)
class Candidate:
    identifier: str
    prompt_group: str
    member: str


def select_candidates(metadata: Path, excluded: set[str]) -> list[Candidate]:
    """Keep all variants of a MusicCaps source together; select only variant 1."""
    with metadata.open(newline="", encoding="utf-8-sig") as stream:
        identifiers = [row["ytid_V"] for row in csv.DictReader(stream)]
    if len(set(identifiers)) != len(identifiers):
        raise DataIntegrityError("Duplicate SunoCaps metadata identifier")
    selected = []
    for identifier in sorted(identifiers):
        match = re.fullmatch(r"([A-Za-z0-9_-]{11})_([1-4])", identifier)
        if match is None:
            raise DataIntegrityError(f"Invalid SunoCaps identifier: {identifier}")
        group, variant = match.groups()
        if variant == "1" and not ({group, identifier, f"musiccaps:{group}"} & excluded):
            selected.append(Candidate(identifier, group, f"audio/audio/{identifier}.mp3"))
    if len(selected) != 64 or len({row.prompt_group for row in selected}) != 64:
        raise DataIntegrityError("Expected 64 unique, unused MusicCaps prompt groups")
    return selected


def download(url: str, target: Path) -> None:
    """Bound individual transfers; preserve TLS verification with DNS-over-HTTPS."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    subprocess.run(["curl", "--doh-url", "https://1.1.1.1/dns-query", "-fLsS",
                    "--retry", "2", "--max-time", "90", "--max-filesize", "5000000",
                    url, "-o", str(temporary)], check=True)
    if temporary.stat().st_size > 5_000_000:
        raise DataIntegrityError("SunoCaps per-file transfer bound exceeded")
    temporary.replace(target)


def main() -> None:
    """Admit source-documented labels only after provenance and duplicate checks."""
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = TARGET / "kaggle_metadata.json"
    download("https://www.kaggle.com/api/v1/datasets/view/miguelcivit/sunocaps", metadata)
    card = json.loads(metadata.read_text())
    if card["licenseName"] != "CC BY-SA 4.0" or card["creatorName"] != "Miguel Civit":
        raise DataIntegrityError("Kaggle author/license differs from the inspected original release")
    source_csv = TARGET / "SunoCaps_dataset.csv"
    download(BASE + "SunoCaps_dataset.csv?datasetVersionNumber=7", source_csv)
    history_path = OUT / "data_preparation_history.json"
    history = json.loads(history_path.read_text())
    seen = set(history["recorded_hashes"])
    excluded = set(history["recorded_identities"])
    with (OUT / "data_preparation_music_sources.csv").open(newline="") as stream:
        confirmation = list(csv.DictReader(stream))
    seen.update(row[key] for row in confirmation for key in ("source_sha256", "pcm_sha256"))
    excluded.update(row[key] for row in confirmation for key in ("ID", "parent_id", "reference_id"))
    selected = select_candidates(source_csv, excluded)
    sources = []
    manifest: list[ManifestRow] = []
    receipt = {"status": "running", "origin": "https://www.kaggle.com/datasets/miguelcivit/sunocaps",
               "dataset_version": VERSION, "license_url": LICENSE, "selected_prompt_groups": 64,
               "kaggle_card_sha256": sha256_file(metadata), "metadata_sha256": sha256_file(source_csv),
               "history_sha256": sha256_file(history_path), "script_sha256": sha256_file(Path(__file__)),
               "selection": "variant 1 per unique original MusicCaps prompt; no predictions used",
               "source_annotation_only": True, "independent_listening": False,
               "voice_labels_known": False, "new_generator": False, "evaluated": False,
               "limitations": ["SunoCaps supplies MusicCaps prompt IDs, not per-file Suno platform UUIDs; cross-corpus UUID identity cannot be proved.",
                               "Exact original-byte and decoded-PCM checks do not detect every lossy re-encoding or near-duplicate."]}
    try:
        urls = [BASE + quote(row.member, safe="") + "?datasetVersionNumber=7" for row in selected]
        archives = [TARGET / "downloads" / (row.identifier + ".zip") for row in selected]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, urls, archives))
        for row in selected:
            identifier = "new_sunocaps_" + row.identifier
            raw = TARGET / "original" / (row.identifier + ".mp3")
            url = BASE + quote(row.member, safe="") + "?datasetVersionNumber=7"
            if not raw.exists():
                archive = TARGET / "downloads" / (row.identifier + ".zip")
                download(url, archive)
                raw.parent.mkdir(parents=True, exist_ok=True)
                if zipfile.is_zipfile(archive):
                    with zipfile.ZipFile(archive) as bundle:
                        members = bundle.infolist()
                        if len(members) != 1 or members[0].filename != raw.name or not 0 < members[0].file_size <= 5_000_000:
                            raise DataIntegrityError("Kaggle single-file ZIP member or size mismatch")
                        raw.write_bytes(bundle.read(members[0]))
                else:
                    payload = archive.read_bytes()
                    if not payload.startswith(b"ID3"):
                        raise DataIntegrityError("Kaggle response is neither the expected ZIP nor MP3")
                    raw.write_bytes(payload)
            audio = decode_audio(raw, limit_seconds=None)
            digest, pcm = sha256_file(raw), pcm_sha256(audio)
            if digest in seen or pcm in seen or len(audio) < 4 * 16000:
                raise DataIntegrityError(f"Duplicate source or insufficient source duration: {identifier}")
            seen.update((digest, pcm))
            group = "reference:musiccaps:" + row.prompt_group
            wav = TARGET / "audio" / (identifier + ".wav")
            window = core(raw, 8)
            write(wav, window)
            manifest.append(manifest_row(identifier, wav, [identifier], {group}, "confirmation_external", "SunoCaps", "suno",
                                         "music", "base", "core", identifier, ("1", "", "1", "", "1"),
                                         [{"component": "music", "start": 0, "end": len(window) / 16000, "label": 1}], ROOT))
            sources.append({"ID": identifier, "parent_id": identifier, "path": str(raw.relative_to(ROOT)),
                            "group_id": group, "reference_id": "musiccaps:" + row.prompt_group,
                            "artist_id": "", "speaker_id": "", "source": "SunoCaps", "generator": "suno",
                            "split": "confirmation_external", "source_url": url, "source_sha256": digest, "pcm_sha256": pcm,
                            "duration_seconds": len(audio) / 16000, "license_url": LICENSE,
                            "FILE_FAKE": "1", "MUSIC_FAKE": "1", "MUSIC_PRESENT": "1", "VOICE_FAKE": "", "VOICE_PRESENT": "",
                            "label_status": "source_documented_music_authenticity; voice_unverified; no_independent_listening"})
            print(f"SunoCaps {len(sources)}/64 {row.identifier} {len(audio) / 16000:.2f}s", flush=True)
        if sum(p.stat().st_size for p in (TARGET / "downloads").glob("*.zip")) > 500_000_000:
            raise DataIntegrityError("SunoCaps download budget exceeded")
        write_manifest(OUT / "sunocaps_manifest.csv", manifest)
        with (OUT / "sunocaps_sources.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(sources[0]))
            writer.writeheader()
            writer.writerows(sources)
        receipt.update(status="complete", rows=len(manifest), unique_prompt_groups=len({r["group_id"] for r in sources}),
                       source_bytes=sum((ROOT / r["path"]).stat().st_size for r in sources),
                       history_or_confirmation_exact_hash_overlap=0, within_cohort_exact_hash_duplicates=0,
                       manifest_sha256=sha256_file(OUT / "sunocaps_manifest.csv"),
                       sources_sha256=sha256_file(OUT / "sunocaps_sources.csv"))
    finally:
        receipt["retained_rows"] = len(sources)
        (OUT / "sunocaps_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
