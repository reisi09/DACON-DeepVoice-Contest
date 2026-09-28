#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_music.py
"""Acquire the frozen 32 real / 32 fake confirmation music parents."""
from __future__ import annotations

import csv
import io
import json
import re
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path
from typing import Final
from urllib.parse import quote

from new_panel_data_inventory import DATA, OUT, ROOT
from v2_data import DataIntegrityError, decode_audio, extract_local_member, pcm_sha256, sha256_file

REVISION: Final = "3788dca9f9f11ad92e9097ef4b58eee247661e7f"
BASE: Final = f"https://huggingface.co/datasets/awsaf49/sonics/resolve/{REVISION}/"
MAX_BYTES: Final = 400_000_000


def main() -> None:
    """Verify original bytes, duration and recorded-history independence before admission."""
    selection = json.loads((DATA / "selection_music.json").read_text())
    history = json.loads((OUT / "data_preparation_history.json").read_text())
    old_hashes = set(history["recorded_hashes"])
    metadata = ROOT / "_workspace/fresh_sonics_20260917/metadata"
    mapping = json.loads((metadata / "metadata.json").read_text())["file_mapping"]
    sizes = {row["path"]: row["size"] for row in json.loads((metadata / "archive_tree.json").read_text())}
    downloads = []
    rows = []
    seen = set(old_hashes)
    receipt = {"status": "running", "selection_sha256": sha256_file(DATA / "selection_music.json"),
               "history_sha256": sha256_file(OUT / "data_preparation_history.json"),
               "script_sha256": sha256_file(Path(__file__)), "downloads": downloads,
               "download_limit_this_script_bytes": MAX_BYTES, "selection_metadata_bytes": 57_003_096,
               "archive_metadata_sha256": {p.name: sha256_file(p) for p in (metadata / "metadata.json", metadata / "archive_tree.json")},
               "source_annotation_only": True, "independent_listening": False, "evaluated": False}
    transferred = 0

    def download(url: str, target: Path, span: tuple[int, int, int] | None = None) -> None:
        nonlocal transferred
        if target.exists():
            return
        maximum = min(30_000_000, MAX_BYTES - transferred)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        headers = DATA / "metadata" / (target.name + ".headers")
        command = ["curl", "-fLsS", "--max-time", "180", "--max-filesize", str(maximum), "-D", str(headers)]
        if span:
            start, end, total = span
            maximum = end - start + 1
            if maximum > MAX_BYTES - transferred:
                raise DataIntegrityError("Music download budget exhausted")
            command += ["--range", f"{start}-{end}"]
            url += f"?range={start}-{end}"
        subprocess.run([*command, url, "-o", str(part)], check=True)
        size = part.stat().st_size
        transferred += size
        downloads.append({"url": url, "bytes": size, "range": span})
        if span:
            actual = re.findall(r"(?im)^content-range:\s*([^\r\n]+)", headers.read_text())
            if not actual or actual[-1].strip() != f"bytes {start}-{end}/{total}" or size != maximum:
                raise DataIntegrityError("Server did not return the exact bounded ZIP range")
        if transferred > MAX_BYTES:
            raise DataIntegrityError("Music download budget exceeded")
        part.replace(target)

    indexes = {}
    try:
        for row in selection:
            real = row["dataset"] == "FMA"
            identifier = "new_fma_" + row["track_id"] if real else "new_sonics_" + row["id"]
            path = DATA / "music" / (identifier + ".mp3")
            if real:
                url = "https://files.freemusicarchive.org/storage-freemusicarchive-org/" + quote(row["track_file"], safe="/")
                download(url, path)
                group, reference, generator = "artist:fma:" + row["artist_id"], "fma:" + row["track_id"], "original"
                license_url = row["license_url"]
            else:
                member = "fake_songs/" + row["filename"] + ".mp3"
                archive = mapping[member]
                total = sizes[archive]
                if archive not in indexes:
                    tail_path = DATA / "metadata" / (Path(archive).stem + ".tail")
                    download(BASE + archive, tail_path, (total - 65_557, total - 1, total))
                    tail = tail_path.read_bytes()
                    offset = tail.rfind(b"PK\x05\x06")
                    if offset < 0:
                        raise DataIntegrityError("ZIP end record missing")
                    end = list(struct.unpack("<4s4H2LH", tail[offset:offset + 22]))
                    _, disk, cd_disk, entries_disk, entries, size, start, comment = end
                    if disk or cd_disk or entries_disk != entries or entries == 65_535 or not 0 < size < 5_000_000:
                        raise DataIntegrityError("Unsupported central directory")
                    end_position = total - len(tail) + offset
                    if start + size != end_position:
                        zip64 = struct.unpack("<4sQ2H2L4Q", tail[offset - 76:offset - 20])
                        locator = struct.unpack("<4sLQL", tail[offset - 20:offset])
                        if (start + size != end_position - 76 or zip64[:2] != (b"PK\x06\x06", 44)
                                or zip64[4:] != (0, 0, entries, entries, size, start)
                                or locator != (b"PK\x06\x07", 0, start + size, 1)):
                            raise DataIntegrityError("ZIP64 central directory disagrees")
                    if offset + 22 + comment != len(tail):
                        raise DataIntegrityError("ZIP boundary mismatch")
                    directory = DATA / "metadata" / (Path(archive).stem + ".directory")
                    download(BASE + archive, directory, (start, start + size - 1, total))
                    end[6] = 0
                    with zipfile.ZipFile(io.BytesIO(directory.read_bytes() + struct.pack("<4s4H2LH", *end) + tail[offset + 22:])) as index:
                        indexes[archive] = {info.filename: info for info in index.infolist()}
                    if len(indexes[archive]) != entries:
                        raise DataIntegrityError("Duplicate/missing ZIP entries")
                info = indexes[archive][member]
                if info.flag_bits & 1 or not 0 < info.file_size <= 30_000_000:
                    raise DataIntegrityError("Encrypted or oversized source")
                if not path.exists():
                    header = DATA / "metadata" / (identifier + ".header")
                    download(BASE + archive, header, (info.header_offset, info.header_offset + 29, total))
                    name_size, extra_size = struct.unpack_from("<2H", header.read_bytes(), 26)
                    size = 30 + name_size + extra_size + info.compress_size
                    blob = DATA / "metadata" / (identifier + ".member")
                    download(BASE + archive, blob, (info.header_offset, info.header_offset + size - 1, total))
                    if blob.read_bytes()[30:30 + name_size].decode("utf-8") != member:
                        raise DataIntegrityError("Local ZIP member name mismatch")
                    entry = {key: getattr(info, key) for key in ("compress_type", "compress_size", "file_size", "CRC")}
                    extract_local_member(blob.read_bytes(), entry, path)
                if path.stat().st_size != info.file_size or zlib.crc32(path.read_bytes()) != info.CRC:
                    raise DataIntegrityError("Retained SONICS source failed CRC/size")
                url = BASE + archive + "::" + member
                group, reference, generator = "reference:sonics:" + row["id"], "sonics:" + row["id"], row["source"]
                license_url = "https://creativecommons.org/licenses/by-nc/4.0/"
            audio = decode_audio(path, limit_seconds=None)
            digest, pcm = sha256_file(path), pcm_sha256(audio)
            if digest in seen or pcm in seen or len(audio) < 60 * 16_000:
                raise DataIntegrityError(f"Historical/within-panel duplicate or insufficient duration: {identifier}")
            if not real and abs(len(audio) / 16_000 - float(row["duration"])) > 1:
                raise DataIntegrityError(f"Source metadata duration mismatch: {identifier}")
            seen.update((digest, pcm))
            rows.append({"ID": identifier, "path": str(path.relative_to(ROOT)), "parent_id": identifier,
                         "group_id": group, "reference_id": reference, "artist_id": row.get("artist_id", ""),
                         "speaker_id": "", "source": row["dataset"], "generator": generator,
                         "split": "confirmation", "source_url": url, "source_sha256": digest,
                         "pcm_sha256": pcm, "duration_seconds": len(audio) / 16_000,
                         "license_url": license_url, "FILE_FAKE": str(int(not real)),
                         "MUSIC_FAKE": str(int(not real)), "MUSIC_PRESENT": "1", "VOICE_FAKE": "", "VOICE_PRESENT": "",
                         "label_status": "source_documented_music_authenticity; voice_unverified; no_independent_listening"})
            print(f"Retained {len(rows)}/64 {identifier} {len(audio) / 16000:.2f}s", flush=True)
        with (OUT / "data_preparation_music_sources.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        receipt.update(status="complete", rows=len(rows), source_bytes=sum((ROOT / r["path"]).stat().st_size for r in rows),
                       prior_recorded_hash_overlap=0, intra_panel_exact_duplicates=0,
                       artist_overlap=0, sonics_id_overlap=0,
                       manifest_sha256=sha256_file(OUT / "data_preparation_music_sources.csv"))
    finally:
        receipt["downloaded_bytes_this_run"] = transferred
        receipt["retained_rows"] = len(rows)
        (OUT / "data_preparation_music_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
