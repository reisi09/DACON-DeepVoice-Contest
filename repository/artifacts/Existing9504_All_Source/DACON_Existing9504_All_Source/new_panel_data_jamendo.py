#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_jamendo.py
"""Acquire artist-excluded, individually licensed Jamendo music for training."""
from __future__ import annotations

import csv
import json
import subprocess
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Final
from urllib.parse import urlparse

from eight_audio import core, manifest_row, write
from new_panel_data_inventory import DATA, OUT, ROOT
from v2_data import DataIntegrityError, decode_audio, pcm_sha256, sha256_file, stable_key

REVISION: Final = "13090d785d071dcd617636deaa4a9775936ac995"
DEST: Final = DATA / "additional_training/jamendo"
LIMIT: Final = 4_000_000_000


def normalized(value: str) -> str:
    """Conservatively compare catalogue display names without asserting alias identity."""
    return "".join(char for char in unicodedata.normalize("NFKD", value).casefold() if char.isalnum())


def main() -> None:
    """Freeze 320 track parents, then preserve full originals and matched 8s views."""
    history = json.loads((OUT / "data_preparation_history.json").read_text())
    old_hashes = set(json.loads((OUT / "data_preparation_speech_prior_hashes.json").read_text()))
    with (OUT / "data_preparation_music_sources.csv").open() as stream:
        confirmation = list(csv.DictReader(stream))
    old_artists = set(history["excluded_artists"]) | {row["artist_id"] for row in confirmation if row["artist_id"]}
    old_hashes.update(value for row in confirmation for key, value in row.items() if "sha256" in key)
    excluded_names = set()
    resolved_artists = set()
    with (ROOT / "_workspace/v2_discovery/archive_probes/fma_raw_tracks.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["artist_id"] in old_artists:
                excluded_names.add(normalized(row["artist_name"]))
                resolved_artists.add(row["artist_id"])
    metadata_paths = sorted((DATA / "additional_training/metadata").glob("20*.jsonl"))
    grouped = defaultdict(dict)
    excluded_tracks = []
    for path in metadata_paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            license_url = urlparse(row.get("license_ccurl", ""))
            url = urlparse(row.get("audiodownload", ""))
            eligible = (row.get("audiodownload_allowed") is True and 30 <= row.get("duration", 0) <= 480
                        and license_url.hostname in {"creativecommons.org", "www.creativecommons.org"}
                        and license_url.path.startswith(("/licenses/by/", "/licenses/by-sa/"))
                        and url.scheme == "https" and url.hostname == "prod-1.storage.jamendo.com")
            if not eligible or not row.get("artist_id") or not row.get("artist_name"):
                continue
            if normalized(row["artist_name"]) in excluded_names:
                excluded_tracks.append(str(row["id"]))
                continue
            grouped[str(row["artist_id"])][str(row["id"])] = row | {"metadata_file": str(path.relative_to(ROOT))}
    ordered_artists = sorted(grouped, key=lambda key: stable_key("jamendo-train-20260918|" + key))
    by_artist = {artist: sorted(grouped[artist].values(), key=lambda row: stable_key(str(row["id"]))) for artist in ordered_artists}
    selected = []
    for pass_index in range(3):
        selected.extend(by_artist[artist][pass_index] for artist in ordered_artists if len(by_artist[artist]) > pass_index)
    selected = selected[:320]
    if len(selected) != 320:
        raise DataIntegrityError(f"Only {len(selected)} allowed tracks under the three-per-artist cap")
    DEST.mkdir(parents=True, exist_ok=True)
    selection_path = DEST / "selection.json"
    serialized = json.dumps(selected, indent=2, ensure_ascii=False) + "\n"
    if selection_path.exists() and selection_path.read_text() != serialized:
        raise DataIntegrityError("Frozen Jamendo acquisition selection changed")
    selection_path.write_text(serialized)
    receipt = {"status": "running", "repo": "amaai-lab/JamendoMaxCaps", "revision": REVISION,
               "metadata_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in metadata_paths},
               "selection_sha256": sha256_file(selection_path), "script_sha256": sha256_file(Path(__file__)),
               "selected_tracks": len(selected), "selected_artists": len({row["artist_id"] for row in selected}),
               "excluded_normalized_names": sorted(excluded_names), "excluded_fma_artists": len(old_artists),
               "resolved_fma_artists": len(resolved_artists), "unresolved_fma_artists": sorted(old_artists - resolved_artists),
               "metadata_name_overlap_tracks_excluded": sorted(set(excluded_tracks)),
               "download_limit_bytes": LIMIT, "selection_used_model_outputs": False,
               "independent_listening": False, "trained": False,
               "limitations": ["Distinct Jamendo artist IDs are not cross-catalogue identity proof",
                               "Only exact normalized artist names were excluded; aliases and collaborations remain unknown",
                               "Music voice labels remain unknown regardless of instrumental tags",
                               "Additional training does not remove real/fake source-domain confounding"]}
    paths = [DEST / "originals" / ("jamendo_" + str(row["id"]) + ".mp3") for row in selected]
    existing = {path for path in paths if path.exists()}
    accounted = sum(path.stat().st_size for path in existing)
    budget_lock = Lock()

    def download(job: tuple[str, Path]) -> Path:
        nonlocal accounted
        url, path = job
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".mp3.part")
        with budget_lock:
            if accounted + 25_000_000 > LIMIT:
                raise DataIntegrityError("Jamendo download budget would be exceeded")
            accounted += 25_000_000
        try:
            subprocess.run(["curl", "--doh-url", "https://1.1.1.1/dns-query", "-fLsS",
                            "--connect-timeout", "15", "--max-time", "180",
                            "--max-filesize", "25000000", url, "-o", str(part)], check=True)
        finally:
            with budget_lock:
                accounted -= 25_000_000 - (part.stat().st_size if part.exists() else 0)
        part.replace(path)
        return path

    sources, views, seen = [], [], set(old_hashes)
    try:
        # Reserve each complete response ceiling before starting parallel downloads.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for row, path in zip(selected, pool.map(download, zip((row["audiodownload"] for row in selected), paths)), strict=True):
                audio = decode_audio(path, limit_seconds=None)
                digest, pcm = sha256_file(path), pcm_sha256(audio)
                if digest in seen or pcm in seen or len(audio) < 8 * 16000:
                    raise DataIntegrityError(f"Prior/confirmation duplicate or short source: {path}")
                seen.update((digest, pcm))
                identifier = "new_jamendo_" + str(row["id"])
                group = "artist:jamendo:" + str(row["artist_id"])
                source = {"ID": identifier, "path": str(path.relative_to(ROOT)), "parent_id": identifier,
                          "group_id": group, "reference_id": "jamendo:" + str(row["id"]),
                          "artist_id": str(row["artist_id"]), "artist_name": row["artist_name"], "speaker_id": "",
                          "source": "JamendoMaxCaps", "generator": "original", "split": "train",
                          "source_url": row["audiodownload"], "source_sha256": digest, "pcm_sha256": pcm,
                          "duration_seconds": len(audio) / 16000, "license_url": row["license_ccurl"],
                          "metadata_duration_seconds": row["duration"],
                          "metadata_duration_delta_seconds": len(audio) / 16000 - float(row["duration"]),
                          "FILE_FAKE": "0", "MUSIC_FAKE": "0", "MUSIC_PRESENT": "1", "VOICE_FAKE": "", "VOICE_PRESENT": "",
                          "label_status": "source_documented_2008_human_music; voice_unverified; no_independent_listening",
                          "metadata_file": row["metadata_file"], "release_date": row["releasedate"]}
                sources.append(source)
                target = DEST / "audio" / (identifier + "_base.wav")
                write(target, core(path, 8))
                views.append(manifest_row(identifier + "_base", target, [identifier], {group}, "train",
                                          "JamendoMaxCaps", "original", "music", "base", "core", "source:" + identifier,
                                          ("0", "", "0", "", "1"),
                                          [{"component": "music", "start": 0, "end": 8, "label": 0}], ROOT))
                if len(sources) % 16 == 0:
                    print(f"Retained {len(sources)}/320 Jamendo originals", flush=True)
        for name, records in (("sources", sources), ("manifest", views)):
            path = OUT / f"data_preparation_jamendo_{name}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        receipt.update(status="complete", retained_rows=len(sources),
                       artist_count=len({row["artist_id"] for row in sources}),
                       artist_track_counts=dict(Counter(row["artist_id"] for row in sources)),
                       metadata_duration_delta_max_seconds=max(abs(row["metadata_duration_delta_seconds"]) for row in sources),
                       prior_or_confirmation_exact_hash_overlap=0, within_cohort_exact_duplicates=0,
                       audio_sha256=sha256_file(OUT / "data_preparation_jamendo_manifest.csv"),
                       source_manifest_sha256=sha256_file(OUT / "data_preparation_jamendo_sources.csv"))
    finally:
        receipt["retained_rows"] = len(sources)
        receipt["retained_original_bytes"] = sum(path.stat().st_size for path in paths if path.exists())
        receipt["downloaded_original_bytes_this_run"] = sum(path.stat().st_size for path in paths if path.exists() and path not in existing)
        (OUT / "data_preparation_jamendo.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
