#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["soundfile", "numpy"]
# ///
# Run: .venv/bin/python candidate_data_catalog.py
"""Index verified original archives; collection never implies training admission."""
from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Final

import numpy as np
import soundfile as sf

from candidate_data_acquire import BANK, OUT

AUDIO: Final = {".wav", ".flac", ".mp3", ".ogg"}


def safe_member(name: str) -> bool:
    """Reject parent traversal and platform-specific path escapes before indexing."""
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def probe(payload: bytes) -> dict[str, str | float | int | bool]:
    """Fully decode one sampled member, rejecting empty, corrupt or non-finite audio."""
    with sf.SoundFile(io.BytesIO(payload)) as audio:
        rate, channels = audio.samplerate, audio.channels
        samples = audio.read(dtype="float32", always_2d=True)
    if not samples.size or not np.isfinite(samples).all():
        raise RuntimeError("Decoded sample is empty or non-finite")
    return {"sample_rate": rate, "channels": channels, "duration": len(samples) / rate,
            "peak": float(np.max(np.abs(samples))), "full_sample_decode": True}


def archive_index(path: Path, output: Path) -> dict:
    """Count original members without extracting or executing archive contents."""
    seen: set[str] = set()
    sizes = 0
    duplicates = 0
    samples = []
    sample_errors = []
    suffixes: Counter[str] = Counter()
    with output.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["member", "bytes", "crc32", "training_admitted"])
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    if not safe_member(item.filename):
                        raise RuntimeError(f"Unsafe archive member: {item.filename}")
                    if item.is_dir() or "__MACOSX" in item.filename or Path(item.filename).name.startswith("._"):
                        continue
                    if Path(item.filename).suffix.lower() not in AUDIO:
                        continue
                    duplicates += item.filename in seen
                    seen.add(item.filename)
                    sizes += item.file_size
                    suffixes[Path(item.filename).suffix.lower()] += 1
                    writer.writerow([item.filename, item.file_size, f"{item.CRC:08x}", False])
                    if len(samples) + len(sample_errors) < 3:
                        try:
                            samples.append({"member": item.filename, **probe(archive.read(item))})
                        except (sf.LibsndfileError, RuntimeError) as error:
                            sample_errors.append({"member": item.filename, "error": str(error)})
        else:
            with tarfile.open(path, "r|*") as archive:
                for item in archive:
                    if not safe_member(item.name):
                        raise RuntimeError(f"Unsafe archive member: {item.name}")
                    if not item.isfile() or Path(item.name).suffix.lower() not in AUDIO:
                        continue
                    duplicates += item.name in seen
                    seen.add(item.name)
                    sizes += item.size
                    suffixes[Path(item.name).suffix.lower()] += 1
                    writer.writerow([item.name, item.size, "", False])
                    if len(samples) + len(sample_errors) < 3:
                        source = archive.extractfile(item)
                        if source is None:
                            raise RuntimeError(f"Missing regular tar member: {item.name}")
                        with source:
                            try:
                                samples.append({"member": item.name, **probe(source.read())})
                            except (sf.LibsndfileError, RuntimeError) as error:
                                sample_errors.append({"member": item.name, "error": str(error)})
    return {"audio_members": sum(suffixes.values()), "unique_member_names": len(seen),
            "duplicate_member_names": duplicates, "uncompressed_audio_bytes": sizes,
            "formats": dict(suffixes), "decoded_samples": samples, "sample_errors": sample_errors,
            "all_audio_decoded": False, "content_deduplicated": False, "training_admitted": False}


def main() -> None:
    """Publish only completed indices; unfinished transfers remain pending."""
    receipts = {}
    for file in [OUT / "receipts.jsonl", *OUT.glob("*_receipts.jsonl")]:
        if file.exists():
            for line in file.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A concurrent writer may not have completed its last line.
                if row.get("status") == "original_checksum_verified":
                    receipts[row["path"]] = row
    verified = {name: row for name, row in receipts.items()
                if (BANK / name).is_file() and (BANK / name).stat().st_size == row["size"]}
    indices = OUT / "indices"
    indices.mkdir(exist_ok=True)
    datasets = {}
    for name, row in sorted(verified.items()):
        dataset = row["dataset"]
        totals = datasets.setdefault(dataset, {"verified_files": 0, "verified_bytes": 0, "standalone_audio": 0})
        totals["verified_files"] += 1
        totals["verified_bytes"] += row["size"]
        totals["standalone_audio"] += Path(name).suffix in AUDIO
        if not name.endswith((".zip", ".tar", ".tar.gz")):
            continue
        report = indices / (name.replace("/", "__") + ".json")
        if not report.exists():
            index = report.with_suffix(".csv")
            temporary = index.with_suffix(".csv.part")
            result = archive_index(BANK / name, temporary)
            result.update({"archive": name, "archive_sha256": row["sha256"],
                           "source_split": row["source_split"]})
            temporary.replace(index)
            report.write_text(json.dumps(result, indent=2) + "\n")
            print(name, result["audio_members"], "audio members", flush=True)
    plans = [json.loads(p.read_text()) for p in OUT.glob("*_plan.json")]
    expected = {r["path"]: r for plan in plans for r in plan}
    summary = {"checked_utc": datetime.now(timezone.utc).isoformat(), "datasets": datasets,
               "planned_files": len(expected), "verified_files": len(verified),
               "pending_files": len(set(expected) - set(verified)),
               "planned_bytes": sum(r["size"] for r in expected.values()),
               "verified_bytes": sum(r["size"] for r in verified.values()),
               "training_admitted_rows": 0, "gpu_started": False,
               "all_audio_quality_audit_complete": False}
    (OUT / "collection_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
