#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_jamendo_bank.py metadata|select|download
"""Preserve licensed, pre-2016 Jamendo originals in an unassigned data bank."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Final
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

from new_panel_data_jamendo import REVISION, normalized
from new_panel_data_inventory import DATA, ROOT
from v2_data import DataIntegrityError, decode_audio, pcm_sha256, sha256_file, stable_key

BANK: Final = ROOT / "_workspace/data_expansion_20260918/jamendo"
OUT: Final = ROOT / "results/data_expansion_20260918"
LIMIT: Final = 5_000_000_000
METADATA_LIMIT: Final = 600_000_000
RESERVE: Final = 25_000_000
FREE: Final = 70 * 1024**3
WORKERS: Final = int(os.environ.get("JAMENDO_WORKERS", "16"))
TREE: Final = Path("/tmp/dacon_jamendo_tree_20260918.json")


def retrieve(url: str, path: Path) -> None:
    """Resume public downloads and publish only complete HTTP responses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    command = ["curl", "--doh-url", "https://1.1.1.1/dns-query", "-fLsS", "-C", "-",
                    "--connect-timeout", "20", "--max-time", "240", "--retry", "2",
                    "--retry-delay", "2", "--max-filesize", str(RESERVE), url, "-o", str(partial)]
    result = subprocess.run(command, capture_output=True)
    if result.returncode == 33 or (result.returncode == 22 and b"416" in result.stderr):
        # A completed-but-unpublished partial may require a fresh HTTP response.
        result = subprocess.run(command[:4] + command[6:], capture_output=True)
    result.check_returncode()
    partial.replace(path)


def metadata() -> None:
    """Sample one pinned date per month, retaining original metadata bytes."""
    entries = json.loads(TREE.read_text())
    selected = {}
    for row in entries:
        name = Path(row["path"]).name
        if name.endswith(".jsonl") and "2008" <= name[:4] <= "2015":
            selected.setdefault(name[:7], row)
    existing = {path.name: path.stat().st_size for path in (BANK / "metadata").glob("*.jsonl")}
    projected = dict(existing)
    projected.update({Path(row["path"]).name: row["size"] for row in selected.values()})
    if sum(projected.values()) > METADATA_LIMIT:
        raise DataIntegrityError("Jamendo metadata allocation would exceed 600 MB")
    for path in (DATA / "additional_training/metadata").glob("20*.jsonl"):
        target = BANK / "metadata" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(path, target)
    def load(row):
        path = BANK / "metadata" / Path(row["path"]).name
        if not path.exists():
            retrieve(f"https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps/resolve/{REVISION}/{row['path']}", path)
        payload = path.read_bytes()
        git_hash = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if len(payload) != row["size"] or git_hash != row["oid"]:
            raise DataIntegrityError(f"Pinned metadata mismatch: {path}")
        return {"path": str(path.relative_to(ROOT)), "bytes": len(payload), "git_blob_sha1": git_hash,
                "sha256": sha256_file(path), "revision": REVISION}
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(load, selected.values()))
    (OUT / "jamendo_metadata.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"metadata_files": len(records), "bytes": sum(row["bytes"] for row in records)}), flush=True)


def select() -> None:
    """Freeze artist-round-robin candidates without assigning experiment splits."""
    locked = json.loads((OUT / "locked_groups.json").read_text())
    fma_ids = set(locked["all_prior_seen"]["fma_artist_ids"])
    with (ROOT / "_workspace/v2_discovery/archive_probes/fma_raw_tracks.csv").open() as stream:
        fma_names = {normalized(row["artist_name"]) for row in csv.DictReader(stream) if row["artist_id"] in fma_ids}
    grouped = defaultdict(dict)
    excluded = Counter()
    for path in sorted((BANK / "metadata").glob("*.jsonl")):
        metadata_digest = sha256_file(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            lic, url = urlparse(row.get("license_ccurl", "")), urlparse(row.get("audiodownload", ""))
            try:
                released = date.fromisoformat(row.get("releasedate", ""))
            except ValueError:
                excluded["invalid_release_date"] += 1
                continue
            allowed = (row.get("audiodownload_allowed") is True and 30 <= row.get("duration", 0) <= 480
                       and date(2008, 1, 1) <= released < date(2016, 1, 1)
                       and lic.hostname in {"creativecommons.org", "www.creativecommons.org"}
                       and lic.path.startswith(("/licenses/by/", "/licenses/by-sa/", "/licenses/by-nc/", "/licenses/by-nc-sa/"))
                       and url.scheme == "https" and url.hostname == "prod-1.storage.jamendo.com"
                       and url.path == f"/download/track/{row.get('id')}/mp32/"
                       and str(row.get("id", "")).isdigit() and str(row.get("artist_id", "")).isdigit()
                       and row.get("artist_name") and row.get("name"))
            if not allowed:
                excluded["rights_date_duration_or_identity"] += 1
                continue
            if normalized(row.get("artist_name", "")) in fma_names:
                excluded["exact_normalized_fma_artist_name"] += 1
                continue
            row.pop("waveform", None)
            row["metadata_file"] = str(path.relative_to(ROOT))
            row["metadata_sha256"] = metadata_digest
            grouped[str(row["artist_id"])][str(row["id"])] = row
    artists = sorted(grouped, key=lambda key: stable_key("jamendo-bank-20260918|" + key))
    tracks = {artist: sorted(grouped[artist].values(), key=lambda row: stable_key(str(row["id"]))) for artist in artists}
    selected = [tracks[artist][index] for index in range(3) for artist in artists if len(tracks[artist]) > index]
    pilot = json.loads((DATA / "additional_training/jamendo/selection.json").read_text())
    selected_ids = {str(row["id"]) for row in selected}
    selected.extend(row for row in pilot if str(row["id"]) not in selected_ids)
    originals = list((BANK / "originals").glob("*.mp3"))
    acquired_ids = {path.stem.removeprefix("jamendo_") for path in originals}
    retained = [row for row in selected if str(row["id"]) in acquired_ids]
    retained_artists = {str(row["artist_id"]) for row in retained}
    current_bytes = sum(path.stat().st_size for path in originals)
    mean_bytes = current_bytes / len(originals) if originals else 6_000_000
    target = min(900, len(retained) + max(0, int((LIMIT * 0.98 - current_bytes) / mean_bytes)))
    reserved = set(locked["current_locked_eval"]["jamendo_artist_ids"])
    for row in selected:
        artist = str(row["artist_id"])
        if len(retained) < target and artist not in retained_artists and artist not in reserved:
            retained.append(row)
            retained_artists.add(artist)
    selected = retained
    selection = BANK / "selection.json"
    content = json.dumps(selected, ensure_ascii=False, indent=2) + "\n"
    if selection.exists() and selection.read_text() != content:
        raise DataIntegrityError("Frozen bank selection changed; archive explicitly before reselection")
    selection.write_text(content)
    report = {"status": "selected", "revision": REVISION, "tracks": len(selected), "artists": len(retained_artists),
              "candidate_artists": len(artists), "new_acquisition_artist_cap": 1,
              "existing_original_bytes": current_bytes, "observed_mean_original_bytes": mean_bytes,
              "candidate_tracks": sum(map(len, tracks.values())), "artist_cap": 3,
              "pilot_reuse_rows": len(pilot), "pilot_preservation_can_exceed_artist_cap": True,
              "estimated_mp3_bytes_at_192kbps": sum(row["duration"] * 24000 for row in selected),
              "excluded_metadata_rows": dict(excluded), "selection_sha256": sha256_file(selection),
              "reserved_eval_artist_ids": locked["current_locked_eval"]["jamendo_artist_ids"],
              "excluded_fma_normalized_names": sorted(fma_names), "raw_byte_cap": LIMIT, "minimum_free_bytes": FREE,
              "label_limitations": ["Release dates are public metadata, not independent authorship verification",
                                    "Voice labels remain unknown; tags are metadata only", "No independent listening",
                                    "Cross-catalogue aliases and transformed duplicates remain unresolved"]}
    (OUT / "jamendo_selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("tracks", "artists", "candidate_tracks", "estimated_mp3_bytes_at_192kbps")}), flush=True)


def download() -> None:
    """Acquire originals under disk limits; journal each full decode and checksum."""
    selected = json.loads((BANK / "selection.json").read_text())
    if not (BANK / "DOWNLOAD_RELEASED").exists():
        raise DataIntegrityError("Jamendo downloading is held pending root's global under-20GB budget release")
    locked = json.loads((OUT / "locked_groups.json").read_text())
    reserved = set(locked["current_locked_eval"]["jamendo_artist_ids"])
    pilot = json.loads((DATA / "additional_training/jamendo/selection.json").read_text())
    pilot_ids = {str(row["id"]) for row in pilot}
    reserved_names = {normalized(row["artist_name"]) for row in pilot if str(row["artist_id"]) in reserved}
    original_dir = BANK / "originals"
    original_dir.mkdir(parents=True, exist_ok=True)
    for path in (DATA / "additional_training/jamendo/originals").glob("*.mp3"):
        target = original_dir / path.name
        if not target.exists():
            os.link(path, target)
    journal = BANK / "sources.jsonl"
    prior = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    done = {row["track_id"] for row in prior if row["status"] in {"retained", "quality_flagged"}}
    accounted = sum(path.stat().st_size for path in original_dir.iterdir())
    lock = Lock()
    started = time.time()
    def acquire(row):
        nonlocal accounted
        identifier = str(row["id"])
        path = original_dir / f"jamendo_{identifier}.mp3"
        reused = path.exists()
        reserved_artist = str(row["artist_id"]) in reserved or normalized(row["artist_name"]) in reserved_names
        base = {"track_id": identifier, "artist_id": str(row["artist_id"]), "artist_name": row["artist_name"],
                "name": row["name"], "source_url": row["audiodownload"], "license_url": row["license_ccurl"],
                "release_date": row["releasedate"], "metadata_file": row["metadata_file"],
                "metadata_duration_seconds": row["duration"], "musicinfo": row.get("musicinfo", {}),
                "revision": REVISION, "split": "unassigned", "reserved_evaluation_artist": reserved_artist,
                "train_eligible_by_group": not reserved_artist, "reused_pilot_original": identifier in pilot_ids,
                "path": str(path.relative_to(ROOT)), "FILE_FAKE": "0", "MUSIC_FAKE": "0", "MUSIC_PRESENT": "1",
                "VOICE_FAKE": "", "VOICE_PRESENT": "", "label_status": "pre2016_release_metadata_only; voice_unknown; no_independent_listening"}
        try:
            if not reused:
                partial = path.with_suffix(".mp3.part")
                before = partial.stat().st_size if partial.exists() else 0
                with lock:
                    if accounted + RESERVE > LIMIT or shutil.disk_usage(BANK).free < FREE + WORKERS * RESERVE:
                        return base | {"status": "budget_deferred"}
                    accounted += RESERVE
                try:
                    retrieve(row["audiodownload"], path)
                finally:
                    after = path.stat().st_size if path.exists() else (partial.stat().st_size if partial.exists() else 0)
                    with lock:
                        accounted += after - before - RESERVE
            info = sf.info(path)
            audio = decode_audio(path, limit_seconds=None)
            magnitude = np.abs(audio)
            duration = len(audio) / 16000
            rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
            silent = float(np.mean(magnitude < 1e-4))
            clipped = float(np.mean(magnitude >= 0.999))
            flags = []
            if not 25 <= duration <= 500:
                flags.append("actual_duration_outside_gate")
            if rms < 1e-4 or silent > 0.95:
                flags.append("mostly_silent")
            if clipped > 0.05:
                flags.append("high_clip_fraction")
            return base | {"status": "quality_flagged" if flags else "retained", "quality_flags": flags,
                           "source_sha256": sha256_file(path), "pcm_sha256": pcm_sha256(audio),
                           "pcm_hash_spec": "ffmpeg_f32le_mono_16000_full_duration", "original_bytes": path.stat().st_size,
                           "duration_seconds": duration, "metadata_duration_delta_seconds": duration - float(row["duration"]),
                           "codec": info.subtype, "sample_rate": info.samplerate, "channels": info.channels,
                           "finite": True, "rms": rms, "silent_fraction_at_1e_4": silent, "clip_fraction_at_0_999": clipped,
                           "peak": float(magnitude.max())}
        except (OSError, subprocess.CalledProcessError, sf.LibsndfileError, ArithmeticError) as exc:
            return base | {"status": "failed", "error": str(exc)[-600:]}
    pending = [row for row in selected if str(row["id"]) not in done]
    results = {row["track_id"]: row for row in prior}
    with journal.open("a") as stream, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for record in pool.map(acquire, pending):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            results[record["track_id"]] = record
            if len(results) % 100 == 0:
                progress = {"processed": len(results), "selected": len(selected), "status_counts": dict(Counter(r["status"] for r in results.values())),
                            "original_bytes": accounted, "elapsed_seconds": time.time() - started, "free_bytes": shutil.disk_usage(BANK).free}
                (OUT / "jamendo_progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                print(json.dumps(progress), flush=True)
    hashes = Counter(row.get("pcm_sha256") for row in results.values() if row.get("pcm_sha256"))
    receipt = {"status": "acquisition_finished", "processed": len(results), "status_counts": dict(Counter(r["status"] for r in results.values())),
               "unique_pcm_hashes": len(hashes), "duplicate_pcm_groups": {key: value for key, value in hashes.items() if value > 1},
               "original_bytes": accounted, "source_journal_sha256": sha256_file(journal), "script_sha256": sha256_file(Path(__file__)),
               "elapsed_seconds": time.time() - started, "free_bytes": shutil.disk_usage(BANK).free, "training_started": False}
    (OUT / "jamendo_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    BANK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    {"metadata": metadata, "select": select, "download": download}[sys.argv[1]]()
