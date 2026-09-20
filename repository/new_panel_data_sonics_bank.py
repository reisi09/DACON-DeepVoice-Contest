#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_sonics_bank.py --retain-local
"""Retain a frozen, balanced, <=2 GB SONICS bank from already downloaded archives; no network."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import imageio_ffmpeg
import numpy as np

from v2_data import DataIntegrityError, sha256_file

ROOT: Final = Path(__file__).resolve().parent
DATA: Final = ROOT / "_workspace/data_expansion_20260918/sonics"
OUT: Final = ROOT / "results/data_expansion_20260918"
REVISION: Final = "3788dca9f9f11ad92e9097ef4b58eee247661e7f"
AUDIO_CAP: Final = 2_000_000_000
BANK_CAP: Final = 2_200_000_000
SELECTION: Final = OUT / "sonics_retention_selection.json"


def save(path: Path, record: dict) -> None:
    """Publish complete receipts atomically so an interrupted write cannot look complete."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_sources() -> dict:
    """Freeze source-ID exclusion, equal provider counts and exact original-byte costs before decoding."""
    locks_path = DATA / "metadata/locked_groups.json"
    metadata_path = DATA / "metadata/fake_songs.csv"
    binding = {"locks_sha256": sha256_file(locks_path), "metadata_sha256": sha256_file(metadata_path)}
    if SELECTION.exists():
        frozen = json.loads(SELECTION.read_text())
        if any(frozen[key] != value for key, value in binding.items()):
            raise DataIntegrityError("Frozen SONICS selection source metadata changed")
        return frozen
    if binding["metadata_sha256"] != "fb2dde96c8c57975f8840e4a7d01422b542b755543ffde85194a590963d1bba9":
        raise DataIntegrityError("Cached official CSV failed the pinned LFS SHA-256")
    locks = json.loads(locks_path.read_text())
    blocked = set(locks["all_prior_seen"]["sonics_ids"]) | set(locks["current_locked_eval"]["sonics_ids"])
    with metadata_path.open() as stream:
        records = {"fake_songs/" + row["filename"].removesuffix(".mp3") + ".mp3": row for row in csv.DictReader(stream)}
    tree = json.loads((DATA / "metadata/archive_tree.json").read_text())
    official = {Path(row["path"]).name: row for row in tree}
    grouped = {}
    archives = []
    for path in sorted((DATA / "archives").glob("*.zip")):
        source = official[path.name]
        if path.stat().st_size != source["size"] or sha256_file(path) != source["lfs"]["oid"]:
            raise DataIntegrityError(f"Local ZIP failed the official archive digest: {path.name}")
        archives.append({"name": path.name, "bytes": source["size"], "sha256": source["lfs"]["oid"]})
        with zipfile.ZipFile(path) as bundle:
            for entry in sorted(bundle.infolist(), key=lambda item: item.filename):
                if entry.is_dir():
                    continue
                row = records[entry.filename]
                identifier = str(int(row["id"]))
                if identifier in blocked or identifier in grouped:
                    continue
                if (row["target"] != "1" or row["source"] not in {"suno", "udio"}
                        or row["label"] not in {"half fake", "mostly fake", "full fake"}
                        or entry.flag_bits & 1 or not 0 < entry.file_size <= 30_000_000):
                    raise DataIntegrityError(f"Unexpected source or unsafe member: {entry.filename}")
                grouped[identifier] = {"ID": Path(entry.filename).stem, "source_id": identifier,
                    "provider": row["source"], "algorithm": row["algorithm"], "metadata_label": row["label"],
                    "metadata_duration_seconds": float(row["duration"]), "archive": path.name,
                    "member": entry.filename, "bytes": entry.file_size, "crc32": entry.CRC,
                    "group_id": "reference:sonics:" + identifier}
    cohorts = {provider: sorted((row for row in grouped.values() if row["provider"] == provider),
               key=lambda row: hashlib.sha256(("20260918|retain|" + row["source_id"]).encode()).hexdigest())
               for provider in ("suno", "udio")}
    chosen = []
    used = 0
    for pair_id, pair in enumerate(zip(cohorts["suno"], cohorts["udio"])):
        cost = sum(row["bytes"] for row in pair)
        if used + cost > AUDIO_CAP:
            break
        for row in pair:
            chosen.append({**row, "balance_pair": pair_id})
        used += cost
    if not chosen or len({row["source_id"] for row in chosen}) != len(chosen):
        raise DataIntegrityError("No source-disjoint balanced selection fits the budget")
    frozen = {**binding, "revision": REVISION, "network_bytes_this_phase": 0, "audio_cap_bytes": AUDIO_CAP,
              "selection_rule": "one lexicographically first variant per unused SONICS ID; provider-paired SHA256 rank seed20260918",
              "selected_source_files": len(chosen), "selected_source_groups": len(chosen), "selected_bytes": used,
              "provider_counts": dict(Counter(row["provider"] for row in chosen)), "archives": archives, "rows": chosen,
              "selected_using_model_scores": False, "partition": "unassigned_bank"}
    save(SELECTION, frozen)
    return frozen


def retain_source(task: tuple[dict, bytes], known: set[str]) -> dict:
    """Validate exact original bytes, seekable full decode, finite PCM and prior-hash exclusion."""
    candidate, payload = task
    path = DATA / "originals" / (candidate["ID"] + ".mp3")
    source_hash = hashlib.sha256(payload).hexdigest()
    if len(payload) != candidate["bytes"]:
        raise DataIntegrityError("Selected member decompressed beyond its frozen size")
    temporary = path.with_suffix(".mp3.part")
    temporary.write_bytes(payload)
    pcm = ""
    duration = 0.0
    rms = None
    reason = ""
    try:
        result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror", "-threads", "1",
                                 "-i", str(temporary), "-map", "0:a:0", "-t", "3601", "-ac", "1", "-ar", "16000",
                                 "-f", "f32le", "-"], capture_output=True, timeout=120)
        audio = np.frombuffer(result.stdout, dtype="<f4")
        duration = len(audio) / 16_000
        finite = bool(np.isfinite(audio).all())
        rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2))) if len(audio) and finite else None
        pcm = hashlib.sha256(result.stdout).hexdigest()
        if result.returncode or not finite or not 4 <= duration < 3600 or rms is None or rms <= 1e-5:
            reason = "decoder_or_finite_duration_energy_gate:" + result.stderr.decode(errors="replace")[:600]
        elif {source_hash, pcm} & known:
            reason = "previously_seen_or_locked_exact_hash"
    except subprocess.TimeoutExpired:
        reason = "decoder_timeout_120s"
    if reason:
        temporary.unlink()
    else:
        temporary.replace(path)
    return {**candidate, "path": str(path.relative_to(ROOT)) if not reason else "", "source": "SONICS",
            "source_sha256": source_hash, "pcm16k_sha256": pcm, "duration_seconds": duration, "rms": rms,
            "zip_crc_verified": True, "status": "rejected" if reason else "retained", "rejection_reason": reason,
            "license": "CC-BY-NC-4.0", "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "revision": REVISION, "split": "unassigned_bank", "FILE_FAKE": "1", "MUSIC_FAKE": "1",
            "MUSIC_PRESENT": "1", "VOICE_FAKE": "", "VOICE_PRESENT": "", "independent_listening": False,
            "source_url": f"https://huggingface.co/datasets/awsaf49/sonics/resolve/{REVISION}/fake_songs/{candidate['archive']}::{candidate['member']}"}


def retain_archive(archive: dict, frozen: dict, known: set[str]) -> list[dict]:
    """Finish and receipt each archive's selected members before deleting that acquisition archive."""
    path = DATA / "archives" / archive["name"]
    receipt_path = OUT / ("sonics_retained_" + path.stem + ".json")
    binding = {"selection_sha256": sha256_file(SELECTION), "verifier_sha256": sha256_file(Path(__file__)),
               "archive_sha256": archive["sha256"]}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if any(receipt[key] != value for key, value in binding.items()):
            raise DataIntegrityError("Cannot reuse a retention receipt with different verifier or selection")
        for row in receipt["rows"]:
            if row["status"] == "retained" and sha256_file(ROOT / row["path"]) != row["source_sha256"]:
                raise DataIntegrityError("Previously retained source changed")
            if row["status"] == "rejected" and row.get("discarded_path"):
                (ROOT / row["discarded_path"]).unlink(missing_ok=True)
        return receipt["rows"]
    if not path.exists() or sha256_file(path) != archive["sha256"]:
        raise DataIntegrityError(f"Required original archive missing or changed: {path.name}")
    selected = [row for row in frozen["rows"] if row["archive"] == path.name]
    rows = []
    with zipfile.ZipFile(path) as bundle, ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(selected), 8):
            tasks = []
            for candidate in selected[start:start+8]:
                entry = bundle.getinfo(candidate["member"])
                if entry.file_size != candidate["bytes"] or entry.CRC != candidate["crc32"]:
                    raise DataIntegrityError("ZIP directory changed after selection was frozen")
                tasks.append((candidate, bundle.read(entry)))
            rows.extend(pool.map(lambda task: retain_source(task, known), tasks))
            print(json.dumps({"archive": path.name, "verified": len(rows), "selected": len(selected)}), flush=True)
    receipt = {**binding, "archive": archive, "rows": rows, "selected_rows": len(rows),
               "archive_discarded_bytes": archive["bytes"], "discard_reason": "User-authorized selected-source retention under global20GB cap",
               "unselected_archive_members_preserved": False}
    save(receipt_path, receipt)
    path.unlink()
    print(json.dumps({"archive_deleted": path.name, "bytes_returned": archive["bytes"]}), flush=True)
    return rows


def main() -> None:
    """Retain only complete good pairs; preserve provenance while relinquishing unselected archive audio."""
    (DATA / "originals").mkdir(exist_ok=True)
    frozen = select_sources()
    locks = json.loads((DATA / "metadata/locked_groups.json").read_text())
    known = {value for scope in ("all_prior_seen", "current_locked_eval") for field in ("source_sha256", "pcm_sha256", "audio_sha256") for value in locks[scope][field]}
    rows = [row for archive in frozen["archives"] for row in retain_archive(archive, frozen, known)]
    bad_pairs = {row["balance_pair"] for row in rows if row["status"] != "retained"}
    seen_hashes = set()
    for row in rows:
        hashes = {row["source_sha256"], row["pcm16k_sha256"]}
        if row["status"] == "retained" and hashes & seen_hashes:
            bad_pairs.add(row["balance_pair"])
        if row["status"] == "retained":
            seen_hashes.update(hashes)
    changed_archives = set()
    for row in rows:
        if row["balance_pair"] in bad_pairs and row["status"] == "retained":
            row.update(status="rejected", rejection_reason="balance_pair_quality_or_exact_duplicate",
                       discarded_path=row["path"], path="")
            changed_archives.add(row["archive"])
    for archive in changed_archives:
        receipt_path = OUT / ("sonics_retained_" + Path(archive).stem + ".json")
        receipt = json.loads(receipt_path.read_text())
        receipt["rows"] = [row for row in rows if row["archive"] == archive]
        receipt["rejection_update_verifier_sha256"] = sha256_file(Path(__file__))
        save(receipt_path, receipt)
    for row in rows:
        if row["status"] == "rejected" and row.get("discarded_path"):
            (ROOT / row["discarded_path"]).unlink(missing_ok=True)
    kept = [row for row in rows if row["status"] == "retained"]
    counts = Counter(row["provider"] for row in kept)
    source_bytes = sum(row["bytes"] for row in kept)
    bank_bytes = sum(path.stat().st_size for path in DATA.rglob("*") if path.is_file())
    if (counts["suno"] != counts["udio"] or source_bytes > AUDIO_CAP or bank_bytes > BANK_CAP
            or len({row["source_id"] for row in kept}) != len(kept)):
        raise DataIntegrityError("Final balance, uniqueness or retained-byte budget failed")
    inventory = OUT / "sonics_retained_sources.jsonl"
    inventory.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    save(OUT / "sonics_retention.json", {"status": "complete", "network_bytes_this_phase": 0,
         "source_files": len(kept), "source_groups": len(kept), "provider_counts": dict(counts),
         "source_bytes": source_bytes, "retained_bank_bytes": bank_bytes, "bank_cap_bytes": BANK_CAP,
         "selected_rows": len(rows), "rejected_rows": len(rows)-len(kept), "metadata_labels": dict(Counter(row["metadata_label"] for row in kept)),
         "discarded_complete_archive_bytes": sum(row["bytes"] for row in frozen["archives"]),
         "selection_sha256": sha256_file(SELECTION), "inventory_sha256": sha256_file(inventory),
         "locks_sha256": frozen["locks_sha256"], "verifier_sha256": sha256_file(Path(__file__)),
         "partition": "unassigned_bank", "old_and_locked_id_overlap": 0, "old_and_locked_exact_hash_overlap": 0,
         "within_bank_source_or_pcm_exact_duplicates": 0, "new_generator_family": False,
         "decoder": "seekable MP3; ffmpeg float32 little endian mono16k, complete duration; no view/padding",
         "limitations": ["Source-documented generated music; voice unknown; no independent listening",
                         "Half/mostly fake are source prompting categories, not audio segment percentages",
                         "No model split assigned; no general near-duplicate or unknown-reference-independence proof",
                         "Unselected originals in deleted acquisition archives are not retained"]})


if __name__ == "__main__":
    if sys.argv[1:] == ["--retain-local"]:
        main()
    else:
        raise DataIntegrityError("Network acquisition disabled by user global20GB cap; only --retain-local is supported")
