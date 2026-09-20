#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# .venv/bin/python new_panel_data_haim_finalize.py
"""Replay retained source hashes and export the unassigned HAIM source catalog."""
from __future__ import annotations

import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from new_panel_data_haim_expansion import BANK, LIMIT, RESULTS, ROOT
from v2_data import DataIntegrityError, decode_audio, pcm_sha256, sha256_file


def verify_source(expected: tuple[Path, str, str]) -> None:
    """Verify original bytes and the complete decoded waveform independently."""
    path, byte_hash, pcm_hash = expected
    if sha256_file(path) != byte_hash or pcm_sha256(decode_audio(path, limit_seconds=None)) != pcm_hash:
        raise DataIntegrityError(f"Retained source replay mismatch: {path}")


def main() -> None:
    """Require the balanced quota, unchanged pilot, and exact full-source replay."""
    ledger = RESULTS / "haim_sources.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    for row in rows:
        row.setdefault("hard_failures", [])
        row["model_variant"] = row["metadata"]["file_name"].split('/')[0] if row["generator"] == "musicgen" else ""
        if row["generator"] == "musicgen":
            row["generator_version"] = ""
    if len({row["ID"] for row in rows}) != len(rows):
        raise DataIntegrityError("Repeated source identity in HAIM ledger")
    valid = [row for row in rows if row["status"] == "integrity_checked"]
    counts = Counter(row["generator"] for row in valid)
    if counts != {"acestep": 800, "musicgen": 800}:
        raise DataIntegrityError(f"Balanced valid quota is incomplete: {counts}")
    originals = [path for path in (BANK / "originals").rglob("*") if path.is_file()]
    if any(path.suffix == ".part" for path in originals):
        raise DataIntegrityError("Inspect and receipt partial files before finalizing")
    if {path.resolve() for path in originals} != {(ROOT / row["path"]).resolve() for row in rows}:
        raise DataIntegrityError("Source ledger does not cover every retained original")
    byte_count = sum(path.stat().st_size for path in originals)
    if byte_count > LIMIT:
        raise DataIntegrityError("HAIM original byte cap exceeded")
    prior = json.loads((RESULTS / "haim_pilot_before.json").read_text())
    if any(sha256_file(ROOT / path) != digest for path, digest in prior.items()):
        raise DataIntegrityError("A frozen pilot artifact changed")
    expected = [(ROOT / row["path"], row["source_sha256"], row["pcm_sha256"]) for row in rows]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify_source, expected))
    if len({row["source_sha256"] for row in valid}) != len(valid) or len({row["pcm_sha256"] for row in valid}) != len(valid):
        raise DataIntegrityError("Repeated bytes or decoded waveform in the valid source subset")
    fields = sorted({key for row in rows for key in row})
    csv_path = RESULTS / "haim_sources.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if key in {"metadata", "quality_flags", "hard_failures"} else value
                             for key, value in row.items()})
    reused = [row for row in rows if row["pilot_original_reused"]]
    receipt = json.loads((RESULTS / "haim_receipt.json").read_text())
    receipt.update(status="complete", downloads_enabled=False, valid_provider_counts=counts, valid_target_met=True,
                   source_catalog_csv=str(csv_path.relative_to(ROOT)), source_catalog_sha256=sha256_file(csv_path),
                   source_ledger_sha256=sha256_file(ledger), full_source_byte_replays=len(rows), full_source_pcm_replays=len(rows),
                   retained_original_bytes=byte_count, pilot_originals_reused=len(reused),
                   net_new_originals=len(rows)-len(reused), net_new_original_bytes=byte_count-sum(row["original_bytes"] for row in reused),
                   quarantined=[{"ID": row["ID"], "reasons": row["quality_flags"]} for row in rows if row["status"] == "quarantined"],
                   catalog_normalization="MusicGen small/medium/large is model_variant; undocumented generator_version remains blank",
                   retained_partial_files=0, trained=False, evaluated=False, final_verified_utc=datetime.now(timezone.utc).isoformat())
    (RESULTS / "haim_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (RESULTS / "haim_verification.json").write_text(json.dumps({"originals": len(rows), "valid_counts": counts,
        "full_byte_and_pcm_replays": len(rows), "original_bytes": byte_count, "pilot_artifacts_unchanged": prior,
        "source_catalog_sha256": sha256_file(csv_path), "source_ledger_sha256": sha256_file(ledger),
        "partial_cleanup": [], "partial_cleanup_bytes": 0, "split": "unassigned", "trained": False,
        "acquisition_script_sha256": sha256_file(ROOT / "new_panel_data_haim_expansion.py"),
        "finalizer_script_sha256": sha256_file(Path(__file__))}, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
