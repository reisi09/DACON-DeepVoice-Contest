# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_fma_bank.py (index only); --download-gb 1.5 after release
# noqa: SIZE_OK -- one bounded acquisition transaction; split if another acquisition policy is added.
"""Reuse local FMA originals and build a bounded, artist-screened full-audio bank."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Final
from urllib.parse import quote, urlparse

import numpy as np

from v2_data import decode_audio, pcm_sha256, sha256_file, stable_key

ROOT: Final = Path(__file__).resolve().parent
BANK: Final = ROOT / "_workspace/data_expansion_20260918/fma"
OUT: Final = ROOT / "results/data_expansion_20260918"
METADATA: Final = ROOT / "_workspace/v2_discovery/archive_probes/fma_raw_tracks.csv"
LOCKS: Final = OUT / "locked_groups.json"
MIN_FREE: Final = 70 * 1024 ** 3
MAX_TRANSFER: Final = 1_500_000_000
MAX_RETAINED: Final = 1_550_000_000
METADATA_RESERVE: Final = 2_000_000


class BankError(RuntimeError):
    """Stop acquisition without changing existing evidence."""


def append_receipt(path: Path, record: Mapping) -> None:
    """Persist reservations before network requests so restart cannot forget their cost."""
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def charged_bytes(reservations: list[Mapping], transfers: list[Mapping]) -> int:
    """Unresolved attempts keep their full byte reservation after interruption."""
    actual = {row["track_id"]: row["transferred_bytes"] for row in transfers}
    reserved = {row["track_id"]: row["reserved_bytes"] for row in reservations}
    if len(actual) != len(transfers) or len(reserved) != len(reservations):
        raise BankError("Duplicate transfer/reservation track IDs would undercount acquisition bytes")
    return sum(actual.values()) + sum(cap for track, cap in reserved.items() if track not in actual)


def retained_bytes() -> int:
    """Count all bank files and FMA receipts, including partials, once per inode."""
    files = list(BANK.rglob("*")) + list(OUT.glob("fma*"))
    sizes = {(stat.st_dev, stat.st_ino): stat.st_size for path in files if path.is_file() for stat in [path.stat()]}
    return sum(sizes.values())


def permitted(row: Mapping[str, str]) -> bool:
    """Admit exactly the requested CC families; ND and custom licenses stay excluded."""
    url = urlparse(row["license_url"])
    return url.hostname in {"creativecommons.org", "www.creativecommons.org"} and url.path.startswith(
        tuple(f"/licenses/{name}/" for name in ("by", "by-sa", "by-nc", "by-nc-sa")))


def declared_seconds(row: Mapping[str, str]) -> int:
    parts = row["track_duration"].split(":")
    return int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 and all(part.isdigit() for part in parts) else 0


def local_originals() -> Mapping[str, list[str]]:
    """Use the existing full-repository filename inventory, never derived short WAV views."""
    paths = subprocess.run(["rg", "--files", "--hidden", "--no-ignore", str(ROOT)], check=True,
                           capture_output=True, text=True).stdout.splitlines()
    found: dict[str, list[str]] = {}
    for text in paths:
        path = Path(text)
        if "fma" not in text.lower() or BANK in path.parents:
            continue
        match = re.fullmatch(r"(?:new_fma_|fresh_fma_|fma_)?(\d+)\.mp3", path.name)
        if match:
            found.setdefault(str(int(match[1])), []).append(text)
    return found


def inspect_audio(row: Mapping[str, str], path: Path, origin: str, locks: Mapping) -> Mapping:
    """Decode the entire original; quality flags are technical gates, not listening judgments."""
    record = dict(row, ID="bank_fma_" + row["track_id"], path=str(path.relative_to(ROOT)), origin=origin,
                  source="FMA", source_url="https://files.freemusicarchive.org/storage-freemusicarchive-org/"
                  + quote(row["track_file"], safe="/"), source_sha256=sha256_file(path), source_bytes=path.stat().st_size,
                  group_id="artist:fma:" + row["artist_id"], license_allowed=permitted(row),
                  declared_duration_seconds=declared_seconds(row), FILE_FAKE="0", MUSIC_FAKE="0", MUSIC_PRESENT="1",
                  VOICE_FAKE="", VOICE_PRESENT="", train_eligible=False, independent_listening=False,
                  current_eval_artist=row["artist_id"] in locks["current_locked_eval"]["fma_artist_ids"],
                  prior_seen_artist=row["artist_id"] in locks["all_prior_seen"]["fma_artist_ids"],
                  historical_eval_artist=row["artist_id"] in locks["historical_evaluation_seen"]["fma_artist_ids"])
    try:
        audio = decode_audio(path, limit_seconds=None)
        if not len(audio):
            raise ArithmeticError("Decoded zero samples")
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        blocks = audio[:len(audio) // 16000 * 16000].reshape(-1, 16000)
        quiet = float(np.mean(np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1)) < 1e-3)) if len(blocks) else 1.0
        pcm = pcm_sha256(audio)
        locked = locks["current_locked_eval"]
        prior = locks["all_prior_seen"]
        duplicate_eval = record["source_sha256"] in locked["source_sha256"] or pcm in locked["pcm_sha256"]
        duplicate_prior = record["source_sha256"] in prior["source_sha256"] or pcm in prior["pcm_sha256"]
        duration = len(audio) / 16000
        duration_ok = origin == "local_reuse" or 60 <= duration <= 180.1
        technical_quality = len(audio) >= 30 * 16000 and rms >= 1e-4 and quiet < .9
        quality = technical_quality and duration_ok
        record.update(decode_status="ok", decoder_failure="", pcm_sha256=pcm, duration_seconds=len(audio) / 16000,
                      peak_abs=float(np.abs(audio).max()), rms_dbfs=20 * np.log10(max(rms, 1e-12)),
                      near_silence_fraction_1s=quiet, very_short=len(audio) < 30 * 16000,
                      silence_flag=rms < 1e-4 or quiet >= .9, clipping_fraction=float(np.mean(np.abs(audio) >= .999)),
                      new_duration_in_range=duration_ok,
                      technical_audio_quality_pass=technical_quality,
                      duration_selection_review=origin == "new_download" and technical_quality and not duration_ok,
                      locked_hash_overlap=duplicate_eval, prior_hash_overlap=duplicate_prior, quality_pass=quality,
                      train_eligible=quality and permitted(row) and not record["current_eval_artist"]
                      and not record["historical_eval_artist"] and not duplicate_eval
                      and (origin == "local_reuse" or not duplicate_prior))
    except (subprocess.CalledProcessError, ArithmeticError) as error:
        detail = error.stderr.decode(errors="replace") if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        record.update(decode_status="failed", decoder_failure=detail[-4000:], quality_pass=False)
    return record


def download(row: Mapping[str, str], limit: int, lock_fd: int) -> Mapping:
    """Use the repository's native curl pattern with a per-request byte cap and no retry overspend."""
    path = BANK / "originals" / f"fma_{row['track_id']}.mp3"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://files.freemusicarchive.org/storage-freemusicarchive-org/" + quote(row["track_file"], safe="/")
    part = path.with_suffix(".mp3.part")
    if path.exists() or part.exists():
        raise BankError(f"Preserve existing original/partial file instead of overwriting: {path}")
    headers = BANK / "headers" / f"fma_{row['track_id']}.headers"
    headers.parent.mkdir(parents=True, exist_ok=True)
    command = ["curl", "-fLsS", "--max-time", "120", "--max-filesize", str(max(1, limit - 65536)),
               "-D", str(headers), "-w", "%{http_code}\t%{size_download}\t%{url_effective}", url, "-o", str(part)]
    result = subprocess.run(command, capture_output=True, text=True, check=False, pass_fds=(lock_fd,))
    details = result.stdout.strip().split("\t", 2)
    status, size, resolved = details if len(details) == 3 else ("0", str(limit), url)
    record = {"track_id": row["track_id"], "url": url, "http_status": int(status), "transferred_bytes": int(float(size)),
              "curl_exit": result.returncode, "resolved_url": resolved, "error": result.stderr[-2000:], "path": str(path)}
    if result.returncode == 0 and part.exists():
        part.replace(path)
    return record


def main(download_gb: float = 0, workers: int = 4) -> None:
    """Default to indexing only; serialize all bank writes across processes."""
    if not 0 <= download_gb <= 1.5 or not 1 <= workers <= 8:
        raise BankError("Authorized limits: 0–1.5 GB and 1–8 download workers")
    BANK.mkdir(parents=True, exist_ok=True)
    with (BANK / ".acquisition.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BankError("Another FMA acquisition process holds the bank lock") from error
        acquire(download_gb, workers, lock.fileno())


def acquire(download_gb: float, workers: int, lock_fd: int) -> None:
    """Freeze artist-first selection and preserve originals without assigning splits; main owns the lock."""
    OUT.mkdir(parents=True, exist_ok=True)
    with METADATA.open(newline="") as stream:
        metadata = {row["track_id"]: row for row in csv.DictReader(stream)}
    snapshot = BANK / "locked_groups.json"
    if not snapshot.exists():
        shutil.copy2(LOCKS, snapshot)
    locks = json.loads(snapshot.read_text())
    inventory = {key: values for key, values in local_originals().items() if key in metadata}
    log = OUT / "fma_bank_records.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    finished = {row["track_id"] for row in records}
    seen_artists = set(locks["all_prior_seen"]["fma_artist_ids"])
    candidates = [row for row in metadata.values() if permitted(row) and row["artist_id"] and row["track_file"]
                  and row["artist_id"] not in seen_artists and 60 <= declared_seconds(row) <= 180
                  and row["track_bit_rate"].isdigit() and int(row["track_bit_rate"]) >= 128000]
    candidates.sort(key=lambda row: (-int(row["track_bit_rate"]), stable_key("fma-bank-20260918|" + row["track_id"])))
    by_artist = {row["artist_id"]: row for row in reversed(candidates)}
    selected = sorted(by_artist.values(), key=lambda row: stable_key("fma-bank-artist-20260918|" + row["artist_id"]))
    selection = BANK / "selection_1500mb.json"
    encoded = json.dumps({"metadata_sha256": sha256_file(METADATA), "maximum_transfer_bytes": MAX_TRANSFER,
                          "declared_duration_range": [60, 180], "minimum_recorded_bitrate": 128000,
                          "track_ids": [row["track_id"] for row in selected]}, indent=2) + "\n"
    if selection.exists() and selection.read_text() != encoded:
        raise BankError("Existing FMA bank selection differs; preserve the frozen order")
    selection.write_text(encoded)
    transfer_log = OUT / "fma_bank_transfers.jsonl"
    transfers = [json.loads(line) for line in transfer_log.read_text().splitlines()] if transfer_log.exists() else []
    reservation_log = OUT / "fma_bank_reservations.jsonl"
    reservations = [json.loads(line) for line in reservation_log.read_text().splitlines()] if reservation_log.exists() else []
    attempted = {row["track_id"] for row in transfers + reservations}
    transferred = sum(row["transferred_bytes"] for row in transfers)
    charged = charged_bytes(reservations, transfers)
    limit = int(download_gb * 1_000_000_000)

    def persist(status: str) -> None:
        fields = sorted({key for row in records for key in row})
        with (OUT / "fma_bank_manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        summary = {"status": status, "records": len(records), "origins": dict(Counter(row["origin"] for row in records)),
                   "decode_status": dict(Counter(row["decode_status"] for row in records)),
                   "distinct_artists": len({row["artist_id"] for row in records}),
                   "new_artists": len({row["artist_id"] for row in records if not row["prior_seen_artist"]}),
                   "train_eligible": sum(bool(row["train_eligible"]) for row in records),
                   "eligible_new": sum(bool(row["train_eligible"]) and row["origin"] == "new_download" for row in records),
                   "current_eval_artists_retained_but_excluded": len({row["artist_id"] for row in records if row["current_eval_artist"]}),
                   "network_attempts": len(transfers), "transferred_bytes": transferred, "download_budget_bytes": limit,
                   "charged_bytes_including_unresolved": charged,
                   "unresolved_attempts": len({row["track_id"] for row in reservations} - {row["track_id"] for row in transfers}),
                   "network_failures": sum(row["curl_exit"] != 0 for row in transfers),
                   "retained_bank_and_metadata_bytes": retained_bytes(), "retained_limit_bytes": MAX_RETAINED,
                   "free_gib": shutil.disk_usage(ROOT).free / 1024 ** 3, "candidate_artists": len(selected),
                   "metadata_sha256": sha256_file(METADATA), "locked_groups_sha256": sha256_file(snapshot),
                   "script_sha256": sha256_file(Path(__file__)), "manifest_sha256": sha256_file(OUT / "fma_bank_manifest.csv"),
                   "updated_epoch": time.time(), "trained": False, "independent_listening": False,
                   "quality_gate": "Full finite decode; >=30s; RMS>=-80dBFS; <90% one-second blocks below-60dBFS; new originals60–180s (+0.1s MP3 padding).",
                   "limitations": ["Technical audio quality only, not human listening or genre verification.",
                                   "Per-track licenses from the archived FMA metadata; NC/SA obligations retained.",
                                   "Artist aliases, recoded duplicates, and pretraining overlap unresolved."]}
        (OUT / "fma_bank_receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({key: summary[key] for key in ("status", "records", "new_artists", "eligible_new", "transferred_bytes", "free_gib")}), flush=True)

    bank_hashes = {row[key] for row in records for key in ("source_sha256", "pcm_sha256") if row.get(key)}

    def retain(record: Mapping) -> None:
        record = dict(record)
        duplicate = any(record.get(key) in bank_hashes for key in ("source_sha256", "pcm_sha256") if record.get(key))
        record["bank_hash_overlap"] = duplicate
        if duplicate:
            record["train_eligible"] = False
        bank_hashes.update(record[key] for key in ("source_sha256", "pcm_sha256") if record.get(key))
        records.append(record)
        finished.add(record["track_id"])
        append_receipt(log, record)

    for index, (track, paths) in enumerate(sorted(inventory.items())):
        if track not in finished and permitted(metadata[track]):
            retain(inspect_audio(metadata[track], Path(paths[0]), "local_reuse", locks))
        if index % 32 == 0:
            persist("indexing_local")
    persist("local_complete")
    for path in sorted((BANK / "originals").glob("fma_*.mp3*")):
        track = path.name.removeprefix("fma_").split(".")[0]
        if track not in attempted:
            raise BankError(f"Unaccounted existing bank original: {path}; preserve and reconcile its transfer before acquiring")
        if path.suffix == ".mp3" and track not in finished:
            retain(inspect_audio(metadata[track], path, "new_download", locks))
    candidates = [row for row in selected if row["track_id"] not in attempted and row["track_id"] not in finished]
    stop_reason = "complete_candidates"
    no_progress_batches = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(candidates), workers):
            if limit - charged < 1_000_000:
                stop_reason = "complete_budget"
                break
            batch = candidates[start:start + workers]
            cap = min(30_000_000, (limit - charged) // len(batch))
            unresolved = charged - transferred
            headroom = MAX_RETAINED - retained_bytes() - unresolved - METADATA_RESERVE
            if headroom < cap * len(batch):
                stop_reason = "stopped_retention_cap"
                break
            if shutil.disk_usage(ROOT).free - unresolved - cap * len(batch) - METADATA_RESERVE < MIN_FREE:
                stop_reason = "stopped_storage_floor"
                break
            for row in batch:
                reservation = {"track_id": row["track_id"], "reserved_bytes": cap, "epoch": time.time()}
                append_receipt(reservation_log, reservation)
                reservations.append(reservation)
            charged = charged_bytes(reservations, transfers)
            completed_before = len(records)
            for row, transfer in zip(batch, executor.map(download, batch, [cap] * len(batch), [lock_fd] * len(batch)), strict=True):
                transferred += transfer["transferred_bytes"]
                transfers.append(transfer)
                append_receipt(transfer_log, transfer)
                path = Path(transfer["path"])
                if transfer["curl_exit"] == 0:
                    retain(inspect_audio(row, path, "new_download", locks))
            charged = charged_bytes(reservations, transfers)
            persist("acquiring")
            if charged > limit or retained_bytes() > MAX_RETAINED:
                raise BankError("Recorded network transfer exceeded the authorized budget")
            no_progress_batches = no_progress_batches + 1 if len(records) == completed_before else 0
            if no_progress_batches >= 8:
                stop_reason = "stopped_repeated_no_progress"
                break
    persist(stop_reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-gb", type=float, default=0, help="0 indexes only; maximum1.5 after coordinator release")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(args.download_gb, args.workers)
