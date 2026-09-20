#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# .venv/bin/python new_panel_data_haim.py --expand-originals
"""Acquire unassigned HAIM A2 originals without changing the frozen pilot."""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf

from new_panel_data_haim import BASE, DATA, LICENSE, REVISION, Candidate
from new_panel_data_inventory import OUT, ROOT
from v2_data import DataIntegrityError, decode_audio, pcm_sha256, sha256_file, stable_key

BANK: Final = ROOT / "_workspace/data_expansion_20260918/haim"
RESULTS: Final = ROOT / "results/data_expansion_20260918"
LIMIT: Final = 2_500_000_000
TARGET_PER_PROVIDER: Final = 800
FILE_LIMIT: Final = 32_000_000
RESERVE: Final = 70 * 2**30
WORKERS: Final = 8


def check_capacity(*, retained: int, free: int, pending: int) -> None:
    """Reserve the worst case for all simultaneous responses before requesting."""
    reservation = pending * FILE_LIMIT
    if retained + reservation > LIMIT or free - reservation < RESERVE:
        raise DataIntegrityError("HAIM 2.5GB budget or global 70GiB free-space reserve reached")


def reuse_original(source: Path, target: Path, expected: str) -> None:
    """Only hardlink an unchanged, already audited original into the new bank."""
    if sha256_file(source) != expected:
        raise DataIntegrityError(f"Pilot source changed: {source}")
    if target.exists() and sha256_file(target) != expected:
        raise DataIntegrityError(f"Existing reused target changed: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.hardlink_to(source)


@dataclass(frozen=True, slots=True)
class Acquired:
    candidate: Candidate
    path: Path
    error: str = ""


def acquire(candidate: Candidate, *, allow_downloads: bool = False) -> Acquired:
    """Use pinned public HTTPS with bounded transfer and retry transient failures."""
    path = BANK / "originals" / candidate.provider / candidate.file_name
    if path.exists():
        return Acquired(candidate, path)
    if not allow_downloads:
        raise DataIntegrityError("New HAIM downloads are disabled")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    result = subprocess.run(["curl", "-fsSL", "--doh-url", "https://1.1.1.1/dns-query",
                             "--connect-timeout", "20", "--max-time", "180", "--retry", "2",
                             "--max-filesize", str(FILE_LIMIT), "-o", str(temporary), candidate.url],
                            capture_output=True, text=True)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        return Acquired(candidate, path, f"curl {result.returncode}: {result.stderr[-800:]}")
    temporary.replace(path)
    return Acquired(candidate, path)


def main(*, allow_downloads: bool = False) -> None:
    """Audit existing originals offline; download the bounded subset only on explicit request."""
    BANK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    metadata = BANK / "metadata"
    metadata.mkdir(exist_ok=True)
    candidates = []
    metadata_hashes = {}
    for provider in ("acestep", "musicgen"):
        path = DATA / "metadata" / f"{provider}.csv"
        metadata_hashes[provider] = sha256_file(path)
        reuse_original(path, metadata / path.name, metadata_hashes[provider])
        with path.open() as stream:
            for row in csv.DictReader(stream):
                filename = Path(row["file_name"])
                if filename.is_absolute() or ".." in filename.parts or not re.fullmatch(r"[A-Za-z0-9_-]+", row["track_id"]):
                    raise DataIntegrityError("Unsafe provider metadata identity or path")
                candidates.append(Candidate(provider, row["track_id"], row["file_name"], row))
    reuse_original(DATA / "metadata/README.md", metadata / "README.md", sha256_file(DATA / "metadata/README.md"))
    candidates.sort(key=lambda row: stable_key(f"haim-expansion-20260918|{row.provider}|{row.track_id}"))
    if len({row.identifier for row in candidates}) != 11000:
        raise DataIntegrityError("Pinned ACE6000/MusicGen5000 metadata identity contract changed")
    if len({(row.provider, row.file_name) for row in candidates}) != len(candidates):
        raise DataIntegrityError("Provider metadata reuses an output path")
    selection = {"revision": REVISION, "metadata_sha256": metadata_hashes,
                 "split": "unassigned", "rows": [asdict(row) for row in candidates]}
    serialized = json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    selection_path = BANK / "selection.json"
    if selection_path.exists() and selection_path.read_text() != serialized:
        raise DataIntegrityError("Frozen expansion candidate selection changed")
    selection_path.write_text(serialized)
    history = json.loads((OUT / "data_preparation_history.json").read_text())
    excluded = set(history["recorded_hashes"])
    excluded.update(json.loads((OUT / "data_preparation_speech_prior_hashes.json").read_text()))
    for path in (OUT / "data_preparation_music_sources.csv", OUT / "data_preparation_confirmation_manifest.csv",
                 OUT / "data_preparation_additional_training_sources.csv"):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if row.get("split", "").startswith("confirmation") or row.get("source") != "HAIM":
                    excluded.update(value for key, value in row.items() if "sha256" in key and value)
    pilot = {}
    with (OUT / "data_preparation_haim_sources.csv").open() as stream:
        pilot = {row["ID"]: row for row in csv.DictReader(stream)}
    for candidate in candidates:
        if candidate.identifier in pilot:
            row = pilot[candidate.identifier]
            reuse_original(ROOT / row["path"], BANK / "originals" / candidate.provider / candidate.file_name, row["source_sha256"])
    records_path = RESULTS / "haim_sources.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    done = {row["ID"] for row in records}
    seen = set()
    for row in records:
        if sha256_file(ROOT / row["path"]) != row["source_sha256"]:
            raise DataIntegrityError(f"Previously acquired source changed: {row['ID']}")
        seen.update((row["source_sha256"], row["pcm_sha256"]))
    subset_versions = sorted(BANK.glob("target_selection_1600*.json"))
    subset_path = subset_versions[-1] if subset_versions else BANK / "target_selection_1600.json"
    if not subset_path.exists():
        subset = [row for row in candidates if (BANK / "originals" / row.provider / row.file_name).exists()]
        chosen = {row.identifier for row in subset}
        quarantined = {row["ID"] for row in records if row["status"] == "quarantined"}
        for provider in ("acestep", "musicgen"):
            needed = TARGET_PER_PROVIDER - sum(row.provider == provider and row.identifier not in quarantined for row in subset)
            extras = [row for row in candidates if row.provider == provider and row.identifier not in chosen][:max(0, needed)]
            subset.extend(extras)
            chosen.update(row.identifier for row in extras)
        subset_path.write_text(json.dumps({"revision": REVISION, "candidate_selection_sha256": sha256_file(selection_path),
                              "rule": "Preserve originals; stable-rank fill toward 800 per provider, excluding known quarantine from that count",
                              "original_byte_limit": LIMIT, "target_valid_per_provider": TARGET_PER_PROVIDER,
                              "split": "unassigned", "planned_IDs": [row.identifier for row in subset]}, indent=2) + "\n")
    subset = json.loads(subset_path.read_text())
    if subset["candidate_selection_sha256"] != sha256_file(selection_path) or subset["original_byte_limit"] != LIMIT:
        raise DataIntegrityError("Bounded HAIM selection no longer matches its pinned candidates or budget")
    selected_ids = set(subset["planned_IDs"])
    candidates = [row for row in candidates if row.identifier in selected_ids]
    failures_path = RESULTS / "haim_failures.jsonl"
    failures = [json.loads(line) for line in failures_path.read_text().splitlines()] if failures_path.exists() else []
    failure_counts = Counter(row["ID"] for row in failures)
    prior_failures = len(failures)
    pending = [row for row in candidates if row.identifier not in done and failure_counts[row.identifier] < 3
               and (allow_downloads or (BANK / "originals" / row.provider / row.file_name).exists())]
    retained = sum(path.stat().st_size for path in (BANK / "originals").rglob("*") if path.is_file())
    started = datetime.now(timezone.utc).isoformat()
    receipt_path = RESULTS / "haim_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    with records_path.open("a") as stream, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(pending), WORKERS):
            batch = pending[start:start + WORKERS]
            new_count = sum(not (BANK / "originals" / row.provider / row.file_name).exists() for row in batch)
            check_capacity(retained=retained, free=shutil.disk_usage(BANK).free, pending=new_count)
            for item in pool.map(partial(acquire, allow_downloads=allow_downloads), batch):
                candidate, path = item.candidate, item.path
                if item.error:
                    failures.append({"ID": candidate.identifier, "error": item.error, "utc": datetime.now(timezone.utc).isoformat()})
                    with failures_path.open("a") as failure_stream:
                        failure_stream.write(json.dumps(failures[-1]) + "\n")
                    continue
                try:
                    info = sf.info(path)
                    audio = decode_audio(path, limit_seconds=None)
                except (sf.LibsndfileError, subprocess.CalledProcessError, ArithmeticError) as error:
                    failures.append({"ID": candidate.identifier, "error": str(error), "utc": datetime.now(timezone.utc).isoformat()})
                    with failures_path.open("a") as failure_stream:
                        failure_stream.write(json.dumps(failures[-1]) + "\n")
                    continue
                digest, pcm = sha256_file(path), pcm_sha256(audio)
                duration = len(audio) / 16000
                peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
                rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0
                flags = []
                if not len(audio):
                    flags.append("empty_audio")
                if duration < 8 or duration > 600:
                    flags.append("duration_outside_8_600_seconds")
                if rms < 1e-5:
                    flags.append("near_silent")
                if {digest, pcm} & excluded:
                    flags.append("historical_or_confirmation_exact_duplicate")
                if {digest, pcm} & seen:
                    flags.append("within_bank_exact_duplicate")
                if float(np.mean(np.abs(audio) >= .999)) > .01:
                    flags.append("more_than_one_percent_clipped_samples")
                hard_failures = [flag for flag in flags if flag in {"empty_audio", "historical_or_confirmation_exact_duplicate", "within_bank_exact_duplicate"}]
                seen.update((digest, pcm))
                row = {"ID": candidate.identifier, "path": str(path.relative_to(ROOT)), "source": "HAIM",
                       "parent_id": candidate.identifier, "group_id": f"reference:haim:{candidate.provider}:{candidate.track_id}",
                       "split": "unassigned", "generator": candidate.provider, "track_id": candidate.track_id,
                       "generator_version": "1.5" if candidate.provider == "acestep" else "",
                       "model_variant": candidate.file_name.split('/')[0] if candidate.provider == "musicgen" else "",
                       "source_url": candidate.url, "revision": REVISION, "source_sha256": digest, "pcm_sha256": pcm,
                       "original_bytes": path.stat().st_size, "duration_seconds": duration, "sample_rate": info.samplerate,
                       "channels": info.channels, "peak": peak, "rms": rms, "quality_flags": flags,
                       "hard_failures": hard_failures, "status": "quarantined" if hard_failures else "integrity_checked", "metadata": candidate.metadata,
                       "FILE_FAKE": "1", "MUSIC_FAKE": "1", "MUSIC_PRESENT": "1", "VOICE_FAKE": "", "VOICE_PRESENT": "",
                       "license": "CC-BY-NC-4.0", "license_url": LICENSE, "label_status": "source_documented",
                       "pilot_original_reused": candidate.identifier in pilot, "independent_listening": False}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(row)
            retained = sum(path.stat().st_size for path in (BANK / "originals").rglob("*") if path.is_file())
            receipt = {"started_utc": started, "updated_utc": datetime.now(timezone.utc).isoformat(),
                       "status": "running", "revision": REVISION, "target": len(candidates), "retained_sources": len(records),
                       "provider_counts": Counter(row["generator"] for row in records), "status_counts": Counter(row["status"] for row in records),
                       "retained_original_bytes": retained, "failures": failures, "free_bytes": shutil.disk_usage(BANK).free,
                       "split": "unassigned", "pilot_originals_reused": len(pilot), "selection_sha256": sha256_file(selection_path),
                       "target_selection_sha256": sha256_file(subset_path), "original_byte_limit": LIMIT,
                       "downloads_enabled": allow_downloads, "candidate_inventory_count": 11000,
                       "metadata_sha256": metadata_hashes, "source_ledger": str(records_path.relative_to(ROOT)),
                       "trained": False, "independent_listening": False,
                       "limitations": ["Integrity flags are not perceptual quality ratings", "Missing MusicGen prompts and unknown ACE lyric provenance",
                                       "Exact hashes do not establish absence of partial overlap or artist aliases"]}
            (RESULTS / "haim_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if start % 64 == 0:
                print(json.dumps({key: receipt[key] for key in ("retained_sources", "target", "retained_original_bytes", "status_counts", "free_bytes")}), flush=True)
            if len(failures) - prior_failures >= 32:
                raise DataIntegrityError("32 source failures reached; inspect haim_receipt.json before retrying")
    valid_counts = Counter(row["generator"] for row in records if row["status"] == "integrity_checked")
    quota_met = all(valid_counts[provider] >= TARGET_PER_PROVIDER for provider in ("acestep", "musicgen"))
    receipt.update(status="complete" if len(records) == len(candidates) and quota_met else "awaiting_download_authorization" if not allow_downloads else "partial",
                   target=len(candidates), downloads_enabled=allow_downloads, original_byte_limit=LIMIT,
                   valid_provider_counts=valid_counts, valid_target_met=quota_met,
                   target_selection_path=str(subset_path.relative_to(ROOT)), target_selection_sha256=sha256_file(subset_path))
    (RESULTS / "haim_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
