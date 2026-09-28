#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_guitarset.py [--verify]
"""Acquire a fixed, source-documented acoustic-guitar presence panel."""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Final

from v2_data import DataIntegrityError, decode_audio, extract_local_member, pcm_sha256, sha256_file

ROOT: Final = Path(__file__).resolve().parent
OUTPUT: Final = ROOT / "_workspace/local_sampler_20260918/new_data/guitarset"
REPORT: Final = ROOT / "results/local_sampler_20260918/data_preparation_guitarset.json"
MANIFEST: Final = REPORT.with_suffix(".csv")
ARCHIVE: Final = "https://zenodo.org/records/3371780/files/audio_mono-pickup_mix.zip?download=1"
API: Final = "https://zenodo.org/api/records/3371780"
WEBSITE: Final = "https://guitarset.weebly.com/"
LICENSE: Final = "https://creativecommons.org/licenses/by/4.0/legalcode.en"
ARCHIVE_BYTES: Final = 683_145_360
BUDGET: Final = 100_000_000
PREVIOUS_TRANSFER_BOUND: Final = 4_200_000
SELECTION: Final = tuple(f"{player:02d}_{piece}_comp_mix.wav" for player in range(6)
                         for piece in ("Jazz1-130-D", "Funk1-97-C"))


class DirectResponseOnly(urllib.request.HTTPRedirectHandler):
    """Prevent urllib from consuming unmetered redirect response bodies."""

    def redirect_request(self, req, fp, code, msg, headers, newurl) -> None:
        return None


def fetch(url: str, byte_range: tuple[int, int] | None, ledger: list) -> bytes:
    """Refuse ignored or mismatched ranges before reading the response body."""
    headers = {"Accept-Encoding": "identity", "User-Agent": "curl/8.7.1", "Accept": "*/*"}
    expected = 1_000_000 if byte_range is None else byte_range[1] - byte_range[0] + 1
    if PREVIOUS_TRANSFER_BOUND + sum(item["bytes"] for item in ledger) + expected > BUDGET:
        raise DataIntegrityError("Transfer budget would be exceeded")
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    opener = urllib.request.build_opener(DirectResponseOnly())
    with opener.open(urllib.request.Request(url, headers=headers), timeout=45) as response:
        content_range = response.headers.get("Content-Range", "")
        if byte_range is not None and (response.status != 206 or content_range !=
                f"bytes {byte_range[0]}-{byte_range[1]}/{ARCHIVE_BYTES}"):
            raise DataIntegrityError("Server ignored or changed the requested byte range")
        payload = response.read(expected + 1)
        if len(payload) > expected or (byte_range is not None and len(payload) != expected):
            raise DataIntegrityError("Response body length exceeded limit or mismatched range")
        ledger.append({"url": url, "status": response.status, "content_range": content_range,
                       "bytes": len(payload)})
    return payload


def verify() -> None:
    """Re-decode retained parents to detect mutation, duplicate content, or label drift."""
    report = json.loads(REPORT.read_text())
    with MANIFEST.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 and len({row["parent_id"] for row in rows}) == 12
    assert Counter(row["group_id"] for row in rows) == {
        f"guitarset_player_{player:02d}": 2 for player in range(6)}
    assert len({row["source_sha256"] for row in rows}) == 12
    assert len({row["decoded_mono16k_sha256"] for row in rows}) == 12
    for row in rows:
        path = ROOT / row["path"]
        assert sha256_file(path) == row["source_sha256"], path
        audio = decode_audio(path, limit_seconds=None)
        assert pcm_sha256(audio) == row["decoded_mono16k_sha256"], path
        assert len(audio) / 16000 == float(row["duration_seconds"]), path
        assert (row["FILE_FAKE"], row["MUSIC_FAKE"], row["VOICE_PRESENT"],
                row["MUSIC_PRESENT"], row["VOICE_FAKE"]) == ("0", "0", "0", "1", "")
        assert row["voice_labels_status"] == "SOURCE_DOCUMENTED"
    assert report["old_source_hash_overlap"] == report["old_recorded_pcm_hash_overlap"] == 0
    assert not report["prior_guitarset_name_hits"]
    assert sum(item["bytes"] for item in report["transfers"]) == report["transferred_bytes"] <= BUDGET
    assert all(item["status"] == 206 for item in report["transfers"] if item["url"] == ARCHIVE)
    assert sha256_file(MANIFEST) == report["manifest_sha256"]
    print(json.dumps({"verified_parents": len(rows), "performer_groups": 6,
                      "transferred_bytes": report["transferred_bytes"]}))


def main() -> None:
    """Record source selection and licensing before acquiring fixed parent recordings."""
    if MANIFEST.exists() or (OUTPUT / "originals").exists():
        raise DataIntegrityError("Output already exists; use --verify instead of replacing evidence")
    paths = subprocess.check_output([
        "rg", "--files", "--hidden", "--no-ignore", "-g", "*.csv", "-g", "*.json",
        "-g", "!.git/**", "-g", "!.venv/**", "-g", "!**/node_modules/**"],
        cwd=ROOT, text=True).splitlines()
    prior_hashes, name_hits, inputs = set(), [], []
    for name in paths:
        if "local_sampler_20260918/new_data/guitarset/" in name or "data_preparation_guitarset" in name:
            continue
        path = ROOT / name
        data = path.read_bytes()
        prior_hashes.update(item.decode().lower() for item in re.findall(rb"\b[0-9a-fA-F]{64}\b", data))
        if b"guitarset" in data.lower():
            name_hits.append(name)
        inputs.append({"path": name, "bytes": len(data), "sha256": sha256_file(path)})
    if name_hits:
        raise DataIntegrityError(f"Existing GuitarSet references need lineage review: {name_hits}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ledger = []
    metadata = fetch(API, None, ledger)
    record = json.loads(metadata)
    assert record["metadata"]["license"]["id"] == "cc-by-4.0"
    archive_meta = next(item for item in record["files"] if item["key"] == "audio_mono-pickup_mix.zip")
    assert archive_meta["size"] == ARCHIVE_BYTES
    (OUTPUT / "zenodo_record.json").write_bytes(metadata)
    for name, url in (("official_description.html", WEBSITE), ("CC-BY-4.0.html", LICENSE)):
        (OUTPUT / name).write_bytes(fetch(url, None, ledger))
    tail_start = ARCHIVE_BYTES - 65536
    tail = fetch(ARCHIVE, (tail_start, ARCHIVE_BYTES - 1), ledger)
    (OUTPUT / "archive_tail.bin").write_bytes(tail)
    with zipfile.ZipFile(io.BytesIO(tail)) as archive:
        entries = {item.filename: item for item in archive.infolist()}
    selected = []
    for name in SELECTION:
        item = entries[name]
        selected.append({"filename": name, "header_offset": item.header_offset + tail_start,
                         "compress_type": item.compress_type, "compress_size": item.compress_size,
                         "file_size": item.file_size, "CRC": item.CRC})
    (OUTPUT / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    (OUTPUT / "prior_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    rows, seen = [], set()
    for entry in selected:
        start = entry["header_offset"]
        header = fetch(ARCHIVE, (start, start + 29), ledger)
        name_length = int.from_bytes(header[26:28], "little")
        extra_length = int.from_bytes(header[28:30], "little")
        body = fetch(ARCHIVE, (start + 30, start + 29 + name_length + extra_length + entry["compress_size"]), ledger)
        assert body[:name_length].decode() == entry["filename"]
        original = extract_local_member(header + body, entry, OUTPUT / "originals" / entry["filename"])
        audio = decode_audio(original, limit_seconds=None)
        source_hash, pcm_hash = sha256_file(original), pcm_sha256(audio)
        assert source_hash not in prior_hashes and pcm_hash not in prior_hashes, original
        assert pcm_hash not in seen, original
        seen.add(pcm_hash)
        player, performance = entry["filename"].split("_", 1)
        parent = f"guitarset_{player}_{performance.removesuffix('_comp_mix.wav')}"
        rows.append({"ID": parent + "_comp", "path": str(original.relative_to(ROOT)),
                     "source": "GuitarSet", "parent_id": parent, "performance_group": parent,
                     "group_id": f"guitarset_player_{player}", "performer_id": player,
                     "archive_member": entry["filename"], "source_url": ARCHIVE,
                     "revision": "zenodo.3371780", "source_sha256": source_hash,
                     "decoded_mono16k_sha256": pcm_hash, "source_bytes": original.stat().st_size,
                     "duration_seconds": len(audio) / 16000, "license": "CC-BY-4.0",
                     "license_url": "https://creativecommons.org/licenses/by/4.0/",
                     "attribution": "Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye, Juan P. Bello; GuitarSet (2018)",
                     "FILE_FAKE": "0", "MUSIC_FAKE": "0", "VOICE_PRESENT": "0",
                     "MUSIC_PRESENT": "1", "VOICE_FAKE": "", "voice_labels_status": "SOURCE_DOCUMENTED",
                     "voice_label_evidence_url": WEBSITE, "independent_listening": "false",
                     "role": "new_source_documented_presence_panel"})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"sources": rows, "parents": 12, "performer_groups": 6,
              "source_selection": "Six performers; Jazz1-130-D and Funk1-97-C comp performances only",
              "prior_csv_json_files_scanned": len(inputs), "prior_recorded_hash_count": len(prior_hashes),
              "prior_guitarset_name_hits": name_hits, "old_source_hash_overlap": 0,
              "old_recorded_pcm_hash_overlap": 0, "transfers": ledger,
              "transferred_bytes": sum(item["bytes"] for item in ledger), "transfer_budget_bytes": BUDGET,
              "previous_metadata_only_attempts": "Two CC URL requests returned HTTP 403 before audio acquisition; explicit User-Agent succeeded. Upper bound also reserves the discovery tail and license probe.",
              "previous_transfer_upper_bound_bytes": PREVIOUS_TRANSFER_BOUND,
              "archive_upstream_metadata": archive_meta, "archive_full_checksum_verified": False,
              "script_sha256": sha256_file(Path(__file__)), "manifest_sha256": sha256_file(MANIFEST),
              "prior_inputs_sha256": sha256_file(OUTPUT / "prior_inputs.json"),
              "source_evidence": {name: sha256_file(OUTPUT / name) for name in
                  ("zenodo_record.json", "official_description.html", "CC-BY-4.0.html", "selection.json", "archive_tail.bin")},
              "limitations": ["Voice absence is source-documented from acoustic-guitar pickup recording, not independently listened.",
                  "Clips are naturally below 60 seconds; no padding, looping, concatenation, or length claim is applied.",
                  "Only six performer groups exist; the twelve recordings are not twelve independent performers.",
                  "Different performers share lead sheets; group by performer and preserve performance identifiers.",
                  "Historical recorded hashes may represent crops or different PCM quantization; partial duplicates and encoder pretraining exposure remain unexcluded.",
                  "This real-only panel cannot estimate EER or an official score."]}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    verify()


if __name__ == "__main__":
    if sys.argv[1:] == ["--verify"]:
        verify()
    else:
        assert not sys.argv[1:], "Only --verify is supported"
        main()
