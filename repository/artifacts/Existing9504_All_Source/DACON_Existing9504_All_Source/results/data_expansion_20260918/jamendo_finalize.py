#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# How to run: PYTHONPATH=. .venv/bin/python results/data_expansion_20260918/jamendo_finalize.py
"""Close a stopped Jamendo acquisition with byte, lineage and quality evidence."""
import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone

import new_panel_data_jamendo_bank as bank


def main() -> None:
    """Verify every retained original against its complete-decode journal."""
    processes = subprocess.check_output(["ps", "-axo", "pid=,args="], text=True).splitlines()
    active = [row for row in processes if row.strip().endswith(".venv/bin/python new_panel_data_jamendo_bank.py download")]
    if active:
        raise bank.DataIntegrityError("Acquisition must finish before finalizing")
    acquisition = json.loads((bank.OUT / "jamendo_receipt.json").read_text())
    rows = {row["track_id"]: row for row in map(json.loads, (bank.BANK / "sources.jsonl").read_text().splitlines())}
    partials = list((bank.BANK / "originals").glob("*.part"))
    removed = [{"path": str(path.relative_to(bank.ROOT)), "bytes": path.stat().st_size,
                "sha256": bank.sha256_file(path), "reason": "Incomplete response after bounded acquisition finished"}
               for path in partials]
    cleanup = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "count": len(removed),
               "bytes": sum(row["bytes"] for row in removed), "items": removed}
    (bank.OUT / "jamendo_final_partial_cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
    for path in partials:
        path.unlink()
    originals = list((bank.BANK / "originals").glob("*.mp3"))
    for path in originals:
        row = rows[path.stem.removeprefix("jamendo_")]
        if row.get("source_sha256") != bank.sha256_file(path):
            raise bank.DataIntegrityError(f"Original differs from decoded journal: {path}")
    original_bytes = sum(path.stat().st_size for path in originals)
    metadata_bytes = sum(path.stat().st_size for path in (bank.BANK / "metadata").glob("*"))
    if original_bytes > bank.LIMIT or metadata_bytes > bank.METADATA_LIMIT:
        raise bank.DataIntegrityError("Final acquisition exceeds assigned byte cap")
    source_groups, pcm_groups = defaultdict(list), defaultdict(list)
    for row in rows.values():
        if row.get("source_sha256"):
            source_groups[row["source_sha256"]].append(row["track_id"])
            pcm_groups[row["pcm_sha256"]].append(row["track_id"])
    duplicates = set()
    for group in pcm_groups.values():
        ordered = sorted(group, key=lambda identifier: (rows[identifier]["status"] != "retained", identifier))
        duplicates.update(group if any(rows[identifier]["reserved_evaluation_artist"] for identifier in group) else ordered[1:])
    metadata_hashes = {}
    locked = json.loads((bank.OUT / "locked_groups.json").read_text())["current_locked_eval"]
    locked_hashes = set(locked["source_sha256"]) | set(locked["pcm_sha256"]) | set(locked["audio_sha256"])
    records = []
    for identifier, row in sorted(rows.items()):
        path = bank.ROOT / row["metadata_file"]
        if row["metadata_file"] not in metadata_hashes:
            metadata_hashes[row["metadata_file"]] = bank.sha256_file(path)
        record = row | {"metadata_sha256": metadata_hashes[row["metadata_file"]],
                        "duplicate_pcm_for_training": identifier in duplicates,
                        "reserved_evaluation_exact_hash": bool({row.get("source_sha256"), row.get("pcm_sha256")} & locked_hashes),
                        "training_candidate_after_technical_checks": row["status"] == "retained"
                        and row["train_eligible_by_group"] and identifier not in duplicates
                        and not ({row.get("source_sha256"), row.get("pcm_sha256")} & locked_hashes)}
        records.append({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                        for key, value in record.items()})
    candidate_pcm = {row["pcm_sha256"] for row in records if row["training_candidate_after_technical_checks"]}
    reserved_pcm = {row["pcm_sha256"] for row in records if row["reserved_evaluation_artist"] and row.get("pcm_sha256")}
    if candidate_pcm & (reserved_pcm | locked_hashes):
        raise bank.DataIntegrityError("A technical training candidate overlaps locked evaluation PCM")
    destination = bank.OUT / "jamendo_sources.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in records for key in row)))
        writer.writeheader()
        writer.writerows(records)
    retained = [row for row in rows.values() if row.get("source_sha256")]
    new = [row for row in retained if not row["reused_pilot_original"]]
    pilot_artists = {row["artist_id"] for row in retained if row["reused_pilot_original"]}
    summary = {"status": "collection_finished_and_stopped", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "selected_tracks": len(rows), "verified_originals": len(originals), "original_bytes": original_bytes,
               "metadata_bytes": metadata_bytes, "new_originals_excluding_pilot": len(new),
               "new_original_bytes_excluding_pilot": sum(row["original_bytes"] for row in new),
               "reused_pilot_originals": len(retained) - len(new), "artist_groups": len({row["artist_id"] for row in retained}),
               "new_artist_groups": len({row["artist_id"] for row in new}),
               "artist_groups_additional_to_pilot": len({row["artist_id"] for row in new} - pilot_artists),
               "status_counts": dict(Counter(row["status"] for row in rows.values())),
               "new_status_counts": dict(Counter(row["status"] for row in new)),
               "quality_flag_counts": dict(Counter(flag for row in retained for flag in row["quality_flags"])),
               "reserved_evaluation_artist_tracks": sum(row["reserved_evaluation_artist"] for row in retained),
               "reserved_evaluation_artist_groups": len({row["artist_id"] for row in retained if row["reserved_evaluation_artist"]}),
               "technical_training_candidates": sum(row["training_candidate_after_technical_checks"] for row in records),
               "distinct_source_hashes": len(source_groups), "distinct_pcm_hashes": len(pcm_groups),
               "duplicate_source_groups": {key: value for key, value in source_groups.items() if len(value) > 1},
               "duplicate_pcm_groups": {key: value for key, value in pcm_groups.items() if len(value) > 1},
               "genre_counts": dict(Counter(tag for row in retained for tag in row["musicinfo"].get("tags", {}).get("genres", []))),
               "license_url_counts": dict(Counter(row["license_url"] for row in retained)),
               "vocal_instrumental_metadata_counts": dict(Counter(row["musicinfo"].get("vocalinstrumental", "missing") for row in retained)),
               "download_processes": 0, "gpu_started": False, "training_started": False,
               "original_cap_bytes": bank.LIMIT, "metadata_cap_bytes": bank.METADATA_LIMIT,
               "source_csv_sha256": bank.sha256_file(destination), "source_journal_sha256": bank.sha256_file(bank.BANK / "sources.jsonl"),
               "selection_sha256": bank.sha256_file(bank.BANK / "selection.json"),
               "last_acquisition_invocation_elapsed_seconds": acquisition["elapsed_seconds"],
               "limitations": ["Technical checks do not establish perceptual quality or human authorship",
                               "Pre-2016 release dates and instrumental tags are source metadata only; voice labels remain unknown",
                               "Exact duplicates excluded from technical candidates; aliases, re-encodings and excerpts remain unresolved",
                               "Source and artist-group checks do not remove dataset-domain confounding"]}
    marker = bank.BANK / "DOWNLOAD_RELEASED"
    if marker.exists():
        marker.replace(bank.BANK / "DOWNLOAD_RELEASED.completed.json")
    (bank.OUT / "jamendo_final_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "verified_originals", "original_bytes",
                      "new_originals_excluding_pilot", "artist_groups", "status_counts", "download_processes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
