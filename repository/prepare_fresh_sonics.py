#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
# How to run: uv run prepare_fresh_sonics.py
"""Acquire a fixed, unscored SONICS fake-song pilot using bounded ZIP ranges."""

from __future__ import annotations

import csv
import io
import json
import re
import struct
import urllib.request
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

from v2_data import DataIntegrityError, decode_audio, extract_local_member, pcm_sha256, sha256_file, stable_key

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "_workspace/fresh_sonics_20260917"
EVIDENCE = ROOT / "results/leaderboard_audit_20260917"
REVISION = "3788dca9f9f11ad92e9097ef4b58eee247661e7f"
BASE = f"https://huggingface.co/datasets/awsaf49/sonics/resolve/{REVISION}/"
LIMIT = 250_000_000


def main(count_per_provider: int = 6, exclude_manifest: Path | None = None) -> None:
    """Keep one source per ID; fail visibly instead of replacing failed downloads."""
    if not 1 <= count_per_provider <= 39:
        raise ValueError("Each provider must contribute between 1 and 39 distinct IDs")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "revision": REVISION, "script_sha256": sha256_file(Path(__file__)),
        "download_limit_bytes_including_metadata": LIMIT, "downloaded_bytes_this_run": 0,
        "requests": [], "selected": [], "completed": [], "evaluated": False,
        "trained": False, "official_score": None,
        "license": "CC-BY-NC-4.0", "license_source": BASE + "README.md",
        "limitations": [
            "Full-fake source labels come from SONICS valid.csv, not independent audio annotation.",
            "Clip-level voice presence and voice authenticity are unverified and left blank.",
            "One ID per song; artist, prompt, musical-content and pretraining overlaps remain unknown.",
            "Fake-only pilot cannot estimate EER or the official total by itself.",
            "No model outputs were used to select or replace sources.",
        ],
    }

    def download(url: str, maximum: int, span: tuple[int, int, int] | None = None) -> bytes:
        if receipt["downloaded_bytes_this_run"] + maximum > LIMIT:
            raise DataIntegrityError("Download budget would be exceeded")
        headers = {"User-Agent": "sonics-fixed-pilot/1.0"}
        if span:
            start, end, total = span
            headers["Range"] = f"bytes={start}-{end}"
            url += f"?range={start}-{end}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
            record = {"url": url, "status": response.status, "body_bytes": 0,
                      "content_range": response.headers.get("Content-Range")}
            receipt["requests"].append(record)
            if span and (response.status != 206 or record["content_range"] != f"bytes {start}-{end}/{total}"):
                raise DataIntegrityError("Server did not honor the exact ZIP range; body was not downloaded")
            payload = bytearray()
            while chunk := response.read(min(65_536, maximum + 1 - len(payload))):
                payload.extend(chunk)
                record["body_bytes"] += len(chunk)
                receipt["downloaded_bytes_this_run"] += len(chunk)
                if len(payload) > maximum:
                    raise DataIntegrityError("Response exceeded its byte limit")
            if span and len(payload) != end - start + 1:
                raise DataIntegrityError("Truncated ZIP range")
            return bytes(payload)

    try:
        metadata = OUTPUT / "metadata"
        metadata.mkdir(exist_ok=True)
        metadata_sizes = {"README.md": 5503, "metadata.json": 3239343, "valid.csv": 9814768}
        for name, size in metadata_sizes.items():
            path = metadata / name
            if not path.exists():
                path.write_bytes(download(BASE + name, size))
            if path.stat().st_size != size:
                raise DataIntegrityError(f"Pinned metadata size mismatch: {name}")
        if "license: cc-by-nc-4.0" not in (metadata / "README.md").read_text():
            raise DataIntegrityError("Expected source license declaration was not found")
        tree_path = metadata / "archive_tree.json"
        if not tree_path.exists():
            api = f"https://huggingface.co/api/datasets/awsaf49/sonics/tree/{REVISION}/fake_songs?recursive=false&expand=false"
            tree_path.write_bytes(download(api, 100_000))
        archive_sizes = {row["path"]: row["size"] for row in json.loads(tree_path.read_text())}
        mapping = json.loads((metadata / "metadata.json").read_text())["file_mapping"]
        with (metadata / "valid.csv").open() as stream:
            candidates = [row for row in csv.DictReader(stream)
                          if row["label"] == "full fake" and row["target"] == "1"]
        excluded = set()
        if exclude_manifest is not None:
            excluded = {row['id'] for row in csv.DictReader(exclude_manifest.open())}
            candidates = [row for row in candidates if row['id'] not in excluded]
            receipt['excluded_prior_song_ids'] = sorted(excluded)
        selected, used_ids = [], set()
        for provider in ("suno", "udio"):
            cohort = sorted((row for row in candidates if row["source"] == provider),
                            key=lambda row: stable_key(f"fresh-sonics-v1|{row['id']}|{row['filename']}"))
            for row in cohort:
                if row["id"] in used_ids:
                    continue
                used_ids.add(row["id"])
                selected.append(row)
                if sum(item["source"] == provider for item in selected) == count_per_provider:
                    break
        if len(selected) != 2 * count_per_provider:
            raise DataIntegrityError("Insufficient independent IDs for the fixed provider quota")
        receipt["selected"] = [{key: row[key] for key in ("id", "filename", "source")} for row in selected]
        (OUTPUT / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
        receipt["metadata_sha256"] = {path.name: sha256_file(path) for path in metadata.iterdir() if path.is_file()}
        prior_rows, prior_inputs = [], {}
        for relative in ("_workspace/v2_sources/source_inventory.json", "recovery_validation/provenance.json",
                         "v4_validation/provenance.json", "_workspace/validation_sources/music_provenance.json"):
            path = ROOT / relative
            value = json.loads(path.read_text())
            prior_rows.extend(value if isinstance(value, list) else value["sources"])
            prior_inputs[relative] = sha256_file(path)
        receipt["prior_source_inputs"] = prior_inputs
        prior_hashes = {row.get(key, "") for row in prior_rows
                        for key in ("source_sha256", "sha256", "source_pcm_sha256", "pcm_sha256", "short_pcm_sha256")}
        indexes, rows, source_hashes, pcm_hashes = {}, [], set(), set()
        for row in selected:
            if not re.fullmatch(r"fake_\d+_(suno|udio)_\d+", row["filename"]):
                raise DataIntegrityError("Unexpected archive member name")
            member = "fake_songs/" + row["filename"] + ".mp3"
            archive = mapping[member]
            total = archive_sizes[archive]
            if archive not in indexes:
                tail = download(BASE + archive, 65_557, (total - 65_557, total - 1, total))
                offset = tail.rfind(b"PK\x05\x06")
                if offset < 0:
                    raise DataIntegrityError("ZIP end record was not found")
                end = list(struct.unpack("<4s4H2LH", tail[offset:offset + 22]))
                _, disk, cd_disk, entries_disk, entries, size, start, comment_size = end
                if disk or cd_disk or entries_disk != entries or entries == 65_535 or not 0 < size < 5_000_000:
                    raise DataIntegrityError("Unsupported split/ZIP64 archive or unreasonable directory")
                end_position = total - len(tail) + offset
                if start + size != end_position:
                    zip64 = struct.unpack("<4sQ2H2L4Q", tail[offset - 76:offset - 20])
                    locator = struct.unpack("<4sLQL", tail[offset - 20:offset])
                    if (start + size != end_position - 76 or zip64[:2] != (b"PK\x06\x06", 44)
                            or zip64[4:] != (0, 0, entries, entries, size, start)
                            or locator != (b"PK\x06\x07", 0, start + size, 1)):
                        raise DataIntegrityError("ZIP64 end records disagree with central directory")
                if offset + 22 + comment_size != len(tail):
                    raise DataIntegrityError("ZIP central directory boundaries do not agree")
                central = download(BASE + archive, size, (start, start + size - 1, total))
                end[6] = 0
                with zipfile.ZipFile(io.BytesIO(central + struct.pack("<4s4H2LH", *end) + tail[offset + 22:])) as archive_index:
                    indexes[archive] = {info.filename: info for info in archive_index.infolist()}
                if len(indexes[archive]) != entries:
                    raise DataIntegrityError("ZIP entries are missing or duplicated")
            info = indexes[archive][member]
            if info.flag_bits & 1 or not 0 < info.file_size <= 50_000_000:
                raise DataIntegrityError("Encrypted or oversized audio member")
            path = OUTPUT / "originals" / (row["filename"] + ".mp3")
            if not path.exists():
                start = info.header_offset
                header = download(BASE + archive, 30, (start, start + 29, total))
                name_size, extra_size = struct.unpack_from("<2H", header, 26)
                size = 30 + name_size + extra_size + info.compress_size
                blob = download(BASE + archive, size, (start, start + size - 1, total))
                if blob[30:30 + name_size].decode("utf-8") != member:
                    raise DataIntegrityError("ZIP local member name differs from selected source")
                entry = {key: getattr(info, key) for key in ("compress_type", "compress_size", "file_size", "CRC")}
                extract_local_member(blob, entry, path)
            if path.stat().st_size != info.file_size or zlib.crc32(path.read_bytes()) != info.CRC:
                raise DataIntegrityError("Retained audio no longer matches archive CRC/size")
            audio = decode_audio(path, limit_seconds=None)
            digest, pcm = sha256_file(path), pcm_sha256(audio)
            seconds = len(audio) / 16_000
            if seconds < 4 or abs(seconds - float(row["duration"])) > 1:
                raise DataIntegrityError("Decoded duration differs from source metadata")
            if digest in prior_hashes | source_hashes or pcm in prior_hashes | pcm_hashes:
                raise DataIntegrityError("Exact source or decoded audio overlaps prior sources or this cohort")
            source_hashes.add(digest)
            pcm_hashes.add(pcm)
            retained = {key: row[key] for key in ("id", "filename", "source", "algorithm", "style", "label", "target", "no_vocal", "split", "duration", "skip_time")}
            retained.update(ID=f"fresh_sonics_{row['id']}", group_id=f"sonics_id_{row['id']}",
                            path=str(path.relative_to(ROOT)), archive_url=BASE + archive, archive_member=member,
                            revision=REVISION, license="CC-BY-NC-4.0", source_bytes=path.stat().st_size,
                            source_sha256=digest, decoded_mono16k_sha256=pcm, duration_seconds=seconds,
                            FILE_FAKE="1", MUSIC_FAKE="1", MUSIC_PRESENT="1", VOICE_FAKE="", VOICE_PRESENT="",
                            label_scope="SONICS full-fake source; voice labels unverified")
            rows.append(retained)
            receipt["completed"].append(row["filename"])
            print(f"Retained {len(rows)}/{len(selected)}: {row['filename']} ({seconds:.2f}s)", flush=True)
        with (EVIDENCE / "fresh_sonics_manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        receipt.update(status="complete", rows=len(rows), independent_song_ids=len(used_ids),
                       source_bytes=sum(row["source_bytes"] for row in rows),
                       duration_seconds=sum(row["duration_seconds"] for row in rows),
                       prior_exact_source_or_pcm_overlap=0, intra_cohort_exact_duplicates=0,
                       retained_metadata_bytes=sum(path.stat().st_size for path in metadata.iterdir() if path.is_file()))
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (EVIDENCE / "fresh_sonics_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
