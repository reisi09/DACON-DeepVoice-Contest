#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg", "soundfile"]
# ///
# ─── How to run ───
# Use the project's installed decoder environment:
# .venv/bin/python new_panel_data_speech.py
# Archive acquisition: official OpenSLR URLs below, curl --max-filesize 379800000.
"""Prepare a narrator-disjoint, source-documented real-speech presence panel."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import soundfile as sf

from v2_data import DataIntegrityError, decode_audio, pcm_sha256, sha256_file, stable_key

ROOT: Final = Path(__file__).resolve().parent
DATA: Final = ROOT / "_workspace/local_sampler_20260918/new_data/speech"
OUT: Final = ROOT / "results/local_sampler_20260918"
URL: Final = "https://openslr.trmal.net/resources/12/test-clean.tar.gz"
MD5_URL: Final = "https://openslr.trmal.net/resources/12/md5sum.txt"
LICENSE_URL: Final = "https://creativecommons.org/licenses/by/4.0/"
SHA_PATTERN: Final = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", re.I)


@dataclass(frozen=True, slots=True)
class Utterance:
    member: str
    speaker: str
    seconds: float
    payload: bytes


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def main() -> None:
    """Check archive integrity and reject overlap before retaining full utterances."""
    archive = DATA / "downloads/test-clean.tar.gz"
    checksum_file = DATA / "downloads/md5sum.txt"
    expected = next(line.split()[0] for line in checksum_file.read_text().splitlines()
                    if line.split()[-1] == "test-clean.tar.gz")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != expected or digest != "32fa31d27d2e1cad72775fee3f4849a9":
        raise DataIntegrityError("Official archive MD5 mismatch")
    downloaded_bytes = sum(path.stat().st_size for path in (DATA / "downloads").iterdir())
    if downloaded_bytes > 380_000_000:
        raise DataIntegrityError("Speech acquisition exceeded the 380 MB budget")
    OUT.mkdir(parents=True, exist_ok=True)
    history_path = OUT / "data_preparation_speech_history.json"
    hashes_path = OUT / "data_preparation_speech_prior_hashes.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        previous_hashes = set(json.loads(hashes_path.read_text()))
    else:
        command = ["rg", "--files", "--hidden", "--no-ignore", "-g", "*.csv", "-g", "*.json"]
        for excluded in ("**/.git/**", "**/.venv/**", "**/vendor/**", "**/node_modules/**",
                         "_workspace/local_sampler_20260918/**", "results/local_sampler_20260918/**"):
            command.extend(["-g", "!" + excluded])
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        history = []
        previous_hashes: set[str] = set()
        for filename in sorted(result.stdout.splitlines()):
            path = ROOT / filename
            text = path.read_text(errors="replace")
            hashes = {value.casefold() for value in SHA_PATTERN.findall(text)}
            previous_hashes.update(hashes)
            history.append({"path": filename, "sha256": sha256_file(path),
                            "known_hash_count": len(hashes),
                            "narrators": sorted(set(re.findall(r"mlaad_[A-Za-z_]+_speaker:([A-Za-z0-9_]+)", text))),
                            "librispeech_mention": bool(re.search(r"librispeech|openslr[^\s]*[/]12[/]|test-clean", text, re.I))})
        history_path.write_text(json.dumps(history, indent=2))
        hashes_path.write_text(json.dumps(sorted(previous_hashes)))
    previous_names = {value for row in history for value in row["narrators"]}
    previous_names.update(("elizabeth_klett", "elliot_miller", "judy_bieber", "mary_ann"))
    named_narrators: set[str] = set()
    mixed_books: set[str] = set()
    catalogs = sorted((ROOT / "_workspace/v4_data/sources/metadata").glob("mlaad_*.csv"))
    catalogs += sorted((ROOT / "_workspace/v2_sources/originals").rglob("meta.csv"))
    for catalog in catalogs:
        with catalog.open() as stream:
            for row in csv.DictReader(stream, delimiter="|"):
                paths = " ".join(row.get(key, "") for key in ("original_file", "reference_speaker"))
                named_narrators.update(re.findall(r"/by_book/(?:female|male)/([^/]+)/", paths))
                mixed_books.update(re.findall(r"/by_book/mix/([^/]+)/", paths))
    previous_names.update(named_narrators)
    excluded_names = {normalized_name(value) for value in previous_names}
    prior_mentions = [row["path"] for row in history if row["librispeech_mention"]]
    if prior_mentions:
        raise DataIntegrityError(f"Review prior LibriSpeech metadata matches: {prior_mentions}")
    candidates: dict[str, list[Utterance]] = {}
    names: dict[str, str] = {}
    transcripts: dict[str, str] = {}
    metadata = DATA / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise DataIntegrityError(f"Unreadable archive member: {member.name}")
            with stream:
                payload = stream.read()
            path = Path(member.name)
            if path.suffix == ".flac":
                info = sf.info(io.BytesIO(payload))
                if info.samplerate != 16000 or info.channels != 1:
                    raise DataIntegrityError(f"Unexpected LibriSpeech audio format: {member.name}")
                if info.duration >= 8:
                    speaker = path.parent.parent.name
                    candidates.setdefault(speaker, []).append(Utterance(member.name, speaker, info.duration, payload))
                    candidates[speaker] = sorted(candidates[speaker], key=lambda row: (-row.seconds, row.member))[:2]
                continue
            if path.name == "SPEAKERS.TXT":
                (metadata / path.name).write_bytes(payload)
                for line in payload.decode().splitlines():
                    if line and not line.startswith(";"):
                        identifier, _, subset, _, name = [part.strip() for part in line.split("|", 4)]
                        if subset == "test-clean":
                            names[identifier] = name
                continue
            if path.name.endswith(".trans.txt"):
                (metadata / path.name).write_bytes(payload)
                transcripts.update(line.split(" ", 1) for line in payload.decode().splitlines() if line)
                continue
            if len(path.parts) == 2:
                (metadata / path.name).write_bytes(payload)
    speakers = sorted((speaker for speaker, rows in candidates.items() if len(rows) == 2
                       and normalized_name(names[speaker]) not in excluded_names),
                      key=lambda speaker: (min(row.seconds for row in candidates[speaker]) < 16, stable_key(speaker)))[:8]
    if len(speakers) != 8:
        raise DataIntegrityError("Fewer than eight eligible narrators")
    rows = []
    seen_hashes: set[str] = set()
    for speaker in speakers:
        for candidate in candidates[speaker]:
            original = Path(candidate.member)
            destination = DATA / "audio" / original.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(candidate.payload)
            source_hash = sha256_file(destination)
            audio = decode_audio(destination, limit_seconds=None)
            pcm_hash = pcm_sha256(audio)
            if len(audio) / 16000 != candidate.seconds or len(audio) < 8 * 16000:
                raise DataIntegrityError("Decoded duration differs from FLAC metadata or is too short")
            if {source_hash, pcm_hash} & (previous_hashes | seen_hashes):
                raise DataIntegrityError(f"Duplicate source or decoded PCM: {candidate.member}")
            seen_hashes.update((source_hash, pcm_hash))
            rows.append({"ID": "new_librispeech_" + original.stem, "path": str(destination.relative_to(ROOT)),
                         "group_id": "librispeech_narrator_" + normalized_name(names[speaker]),
                         "split": "test", "source": "LibriSpeech/test-clean", "provider": "human_read_speech",
                         "source_url": URL, "archive_member": candidate.member, "source_sha256": source_hash,
                         "source_pcm_sha256": pcm_hash, "pcm_sha256": pcm_hash, "duration_seconds": len(audio) / 16000,
                         "original_duration_seconds": len(audio) / 16000, "speaker_id": speaker,
                         "narrator_name": names[speaker], "chapter_id": original.parent.name,
                         "transcript": transcripts[original.stem], "license": "CC-BY-4.0", "license_url": LICENSE_URL,
                         "FILE_FAKE": "0", "VOICE_FAKE": "0", "MUSIC_FAKE": "", "VOICE_PRESENT": "1",
                         "MUSIC_PRESENT": "0", "label_status": "source_documented", "independent_listening_review": "false"})
    manifest = OUT / "data_preparation_speech_manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["group_id"] for row in rows)
    if len(rows) != 16 or len(counts) != 8 or set(counts.values()) != {2}:
        raise DataIntegrityError("The panel must contain two utterances from each of eight distinct narrator names")
    report = {"source_page": "https://www.openslr.org/12/", "archive_url": URL, "md5_url": MD5_URL,
              "downloaded_bytes": downloaded_bytes, "archive_md5": digest, "archive_sha256": sha256_file(archive),
              "archive_checksum_verified": True, "manifest": str(manifest.relative_to(ROOT)),
              "manifest_sha256": sha256_file(manifest), "rows": len(rows), "distinct_narrators": len(counts),
              "previous_metadata_files_scanned": len(history), "previous_known_hashes": len(previous_hashes),
              "previous_speaker_identifiers": sorted(previous_names), "previous_narrator_names": sorted(named_narrators),
              "previous_mixed_book_groups": sorted(mixed_books),
              "raw_speaker_catalog_paths": [str(path.relative_to(ROOT)) for path in catalogs],
              "prior_librispeech_metadata_matches": prior_mentions,
              "narrator_names": {speaker: names[speaker] for speaker in speakers}, "hash_overlap_count": 0,
              "normalized_narrator_name_overlap_count": 0, "duration_min_seconds": min(row["duration_seconds"] for row in rows),
              "duration_max_seconds": max(row["duration_seconds"] for row in rows), "label_status": "source_documented",
              "limitations": ["No independent listening review", "Exact narrator-name normalization cannot rule out aliases",
                              "Individual identities within prior mixed-book narration are unresolved",
                              "Speech-only real-audio presence panel; not a fake-speech benchmark",
                              "Full native utterances, without repetition, padding, concatenation or a 60-second claim"]}
    (OUT / "data_preparation_speech.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
