#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_fakemusiccaps.py [--prepare-selected|--acquire-selected|--self-check]
"""Recover and acquire a capped, prompt-grouped FakeMusicCaps original WAV bank."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import struct
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from typing import Final

import numpy as np
import soundfile as sf
import imageio_ffmpeg

from v2_data import DataIntegrityError, decode_audio, pcm_sha256, sha256_file

ROOT: Final = Path(__file__).resolve().parent
BANK: Final = ROOT / "_workspace/data_expansion_20260918/fakemusiccaps"
OUT: Final = ROOT / "results/data_expansion_20260918"
TOTAL: Final = 12_889_873_014
MD5: Final = "db418dc95ab7dc378a55f29d6021fd66"
URL: Final = "https://zenodo.org/api/records/15063698/files/FakeMusicCaps.zip/content"
GENERATORS: Final = frozenset(("MusicGen_medium", "audioldm2", "musicldm", "mustango", "stable_audio_open"))
ARCHIVE: Final = BANK / "FakeMusicCaps.zip"
PART: Final = ARCHIVE.with_suffix(".zip.part")
BANK_CAP: Final = 1_750_000_000
SELECTED_GROUPS: Final = 512
REQUEST_LOCK: Final = Lock()
next_request_at = 0.0


def fetch_range(span: tuple[int, int]) -> Path:
    """Reject ignored ranges and partial transfers before adding a contiguous segment."""
    global next_request_at
    start, end = span
    path = BANK / f"chunk_{start}_{end}.part"
    headers = path.with_suffix(".headers")
    command = ["curl", "-fLsS", "--doh-url", "https://1.1.1.1/dns-query",
               "--connect-timeout", "30", "--max-time", "240", "--max-filesize", str(end - start + 1),
               "--range", f"{start}-{end}", "-D", str(headers),
               f"{URL}?range={start}-{end}", "-o", str(path)]
    for attempt in range(4):
        with REQUEST_LOCK:
            time.sleep(max(0.0, next_request_at - time.monotonic()))
            next_request_at = time.monotonic() + 0.8
            with (BANK / "request_starts.jsonl").open("a") as ledger:
                ledger.write(json.dumps({"epoch": time.time(), "range": span, "attempt": attempt + 1}) + "\n")
        process = subprocess.run(command, check=False)
        if process.returncode == 0:
            break
        if attempt == 3:
            raise subprocess.CalledProcessError(process.returncode, command)
        raw = headers.read_text() if headers.exists() else ""
        status = re.findall(r"(?im)^HTTP/\S+\s+(\d+)", raw)
        delay = 5.0
        if status and status[-1] == "429":
            retry = re.findall(r"(?im)^retry-after:\s*([^\r\n]+)", raw)
            reset = re.findall(r"(?im)^x-ratelimit-reset:\s*(\d+)", raw)
            retry_seconds = (float(retry[-1]) if retry[-1].isdigit()
                             else parsedate_to_datetime(retry[-1]).timestamp() - time.time()) if retry else 0
            delay = max(60.0, retry_seconds, float(reset[-1]) - time.time() if reset else 0)
            with REQUEST_LOCK:
                next_request_at = max(next_request_at, time.monotonic() + delay)
            print(json.dumps({"stage": "server_rate_limit_backoff", "seconds": delay, "attempt": attempt + 1}), flush=True)
        time.sleep(delay)
    ranges = re.findall(r"(?im)^content-range:\s*([^\r\n]+)", headers.read_text())
    if not ranges or ranges[-1] != f"bytes {start}-{end}/{TOTAL}" or path.stat().st_size != end-start+1:
        raise DataIntegrityError("FakeMusicCaps range response differs from requested bytes")
    return path


def download() -> None:
    """The old full-archive path is prohibited under the user's global 20GB cap."""
    raise DataIntegrityError("Full 12.9GB archive download disabled; use the capped selected-member bank")


def inspect_audio(payload: bytes) -> tuple[int, int, int, float, float, float, str, list[str]]:
    """Measure original PCM without denoising, level normalization or content filtering."""
    audio, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    frames, channels = audio.shape
    finite = bool(np.isfinite(audio).all())
    peak = float(np.max(np.abs(audio))) if frames else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if frames else 0.0
    clipped = float(np.mean(np.abs(audio) >= 0.999)) if frames else 0.0
    flags = []
    if not finite:
        flags.append("nonfinite")
    if not frames or peak <= 1e-7:
        flags.append("silent")
    if frames / rate < 4:
        flags.append("too_short_lt4s")
    if abs(frames / rate - 10) > 0.5:
        flags.append("duration_differs_from_10s_gt0.5s")
    native_hash = hashlib.sha256(f"{rate}:{channels}:".encode() + audio.astype("<f4").tobytes()).hexdigest()
    return rate, channels, frames, peak, rms, clipped, native_hash, flags


def canonical_hash(payload: bytes) -> str:
    """Match v2_data's full FFmpeg mono16k float32 contract using a WAV byte stream."""
    decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", "pipe:0",
                              "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                             input=payload, check=True, capture_output=True).stdout
    audio = np.frombuffer(decoded, dtype="<f4")
    if not np.isfinite(audio).all():
        raise DataIntegrityError("Canonical decoder returned nonfinite PCM")
    return pcm_sha256(audio)


def member_index() -> dict[str, zipfile.ZipInfo]:
    """Read the retained official remote ZIP central directory without the full archive."""
    record = json.loads((BANK / "zenodo_record.json").read_text())
    if (record["metadata"]["license"]["id"] != "cc-by-nc-4.0" or record["files"][0]["size"] != TOTAL
            or record["files"][0]["checksum"] != "md5:" + MD5):
        raise DataIntegrityError("Official source license/size/checksum metadata changed")
    directory = (BANK / "archive.central").read_bytes()
    location = json.loads((BANK / "central_location.json").read_text())
    end = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, location["entries"], location["entries"], len(directory), 0, 0)
    with zipfile.ZipFile(io.BytesIO(directory + end)) as archive:
        return {item.filename: item for item in archive.infolist()
                if item.filename.endswith(".wav") and not item.filename.startswith("__MACOSX/")}


def unpack_member(block: bytes, item: zipfile.ZipInfo) -> bytes:
    """Validate local header identity and original WAV CRC before retaining any source."""
    header = struct.unpack("<4s5H3L2H", block[:30])
    name_size, extra_size = header[-2:]
    start = 30 + name_size + extra_size
    if (header[0] != b"PK\x03\x04" or header[2] != item.flag_bits or header[3] != item.compress_type
            or block[30:30 + name_size].decode("utf-8") != item.filename
            or start + item.compress_size > len(block) or extra_size > 4096):
        raise DataIntegrityError(f"ZIP local header mismatch: {item.filename}")
    compressed = block[start:start + item.compress_size]
    decoded = zlib.decompress(compressed, -15) if item.compress_type == 8 else compressed
    if len(decoded) != item.file_size or zlib.crc32(decoded) != item.CRC:
        raise DataIntegrityError(f"ZIP member size/CRC mismatch: {item.filename}")
    return decoded


def prepare_selected() -> None:
    """Recover 512 complete local AudioLDM2 originals, then delete only superseded owned partials."""
    if (BANK / "selection.json").exists():
        raise DataIntegrityError("Selection already frozen; use verification or selected acquisition")
    index = member_index()
    locked = json.loads((OUT / "locked_groups.json").read_text())
    excluded = set(locked["all_prior_seen"]["musiccaps_ids"])
    excluded.update(locked["current_locked_eval"]["musiccaps_ids"])
    excluded.update(locked["historical_evaluation_seen"]["musiccaps_ids"])
    prior_hashes = set(locked["all_prior_seen"]["source_sha256"])
    prior_pcm = set(locked["all_prior_seen"]["pcm_sha256"])
    candidates = [item for item in index.values() if item.filename.startswith("audioldm2/")
                  and Path(item.filename).stem not in excluded]
    candidates.sort(key=lambda item: hashlib.sha256(("fmc_cap20260918:" + item.filename).encode()).hexdigest())
    selected, recovered, rejected = [], [], []
    with PART.open("rb") as source:
        for item in candidates:
            if len(selected) == SELECTED_GROUPS:
                break
            ytid = Path(item.filename).stem
            names = [f"{generator}/{ytid}.wav" for generator in sorted(GENERATORS)]
            if not all(name in index for name in names) or item.header_offset + 30 > PART.stat().st_size:
                continue
            source.seek(item.header_offset)
            header = source.read(30)
            name_size, extra_size = struct.unpack_from("<2H", header, 26)
            length = 30 + name_size + extra_size + item.compress_size
            if item.header_offset + length > PART.stat().st_size:
                continue
            block = header + source.read(length - 30)
            payload = unpack_member(block, item)
            stats, canonical, digest = inspect_audio(payload), canonical_hash(payload), hashlib.sha256(payload).hexdigest()
            if stats[-1] or digest in prior_hashes or canonical in prior_pcm:
                rejected.append({"member": item.filename, "flags": stats[-1], "prior_hash": digest in prior_hashes or canonical in prior_pcm})
                continue
            target = BANK / "originals" / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            if sha256_file(target) != digest:
                raise DataIntegrityError("Recovered original write did not preserve source bytes")
            selected.append(ytid)
            recovered.append({"member": item.filename, "bytes": len(payload), "source_sha256": digest,
                              "pcm_sha256": canonical, "crc32": f"{item.CRC:08x}"})
    if len(selected) != SELECTED_GROUPS:
        raise DataIntegrityError("Insufficient quality-passing local AudioLDM2 prompt groups")
    members = [f"{generator}/{ytid}.wav" for ytid in selected for generator in sorted(GENERATORS)]
    expected = sum(index[name].file_size for name in members)
    partials = [PART, *BANK.glob("chunk_*.part"), *BANK.glob("chunk_*.headers")]
    metadata_bytes = sum(p.stat().st_size for p in BANK.rglob("*") if p.is_file()
                         and p not in partials and "originals" not in p.relative_to(BANK).parts)
    if expected + metadata_bytes + 10_000_000 > BANK_CAP:
        raise DataIntegrityError("Selection exceeds the 1.75GB bank cap including metadata reserve")
    selection = {"prompt_ids": selected, "members": members, "expected_original_bytes": expected,
                 "bank_cap_bytes": BANK_CAP, "score_used": False, "group_selection": "sha256 order among locally recoverable clean AudioLDM2; all five siblings preserved",
                 "locked_groups_sha256": sha256_file(OUT / "locked_groups.json"),
                 "central_sha256": sha256_file(BANK / "archive.central"), "recovered": recovered, "rejected_local_candidates": rejected}
    (BANK / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    deleted = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in partials if p.exists()]
    for row in deleted:
        (ROOT / row["path"]).unlink()
    cleanup = {"recovered_originals": len(recovered), "recovered_bytes": sum(row["bytes"] for row in recovered),
               "deleted_owned_partials": deleted, "deleted_bytes": sum(row["bytes"] for row in deleted),
               "selected_original_bytes": expected, "retained_bank_bytes": sum(p.stat().st_size for p in BANK.rglob("*") if p.is_file()),
               "network_bytes_this_preparation": 0, "whole_archive_md5_verified": False,
               "selection_sha256": sha256_file(BANK / "selection.json")}
    (OUT / "fakemusiccaps_cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
    print(json.dumps({key: value for key, value in cleanup.items() if key != "deleted_owned_partials"}))


def acquire_selected() -> None:
    """Acquire only the frozen bounded members, after the root releases the global budget hold."""
    selection = json.loads((BANK / "selection.json").read_text())
    index = member_index()
    if selection["central_sha256"] != sha256_file(BANK / "archive.central"):
        raise DataIntegrityError("Frozen central directory changed")
    if len(selection["members"]) != SELECTED_GROUPS * 5 or selection["expected_original_bytes"] > BANK_CAP - 20_000_000:
        raise DataIntegrityError("Frozen selection no longer meets the capped population")
    if sum(p.stat().st_size for p in BANK.rglob("*") if p.is_file()) + 10_000_000 > BANK_CAP:
        raise DataIntegrityError("Existing retained bank leaves insufficient transfer headroom")

    def acquire_one(name: str) -> None:
        item, target = index[name], BANK / "originals" / name
        if target.exists():
            if target.stat().st_size != item.file_size or zlib.crc32(target.read_bytes()) != item.CRC:
                raise DataIntegrityError("Existing selected source was corrupted")
            return
        if (sum(p.stat().st_size for p in BANK.rglob("*") if p.is_file() and not p.name.startswith("chunk_"))
                + 8 * item.file_size + 32768 > BANK_CAP
                or shutil.disk_usage(BANK).free < 70 * 1024 ** 3):
            raise DataIntegrityError("Selected bank cap reached before the next transfer")
        end = min(TOTAL - 1, item.header_offset + 30 + len(name.encode()) + 4096 + item.compress_size - 1)
        block_path = fetch_range((item.header_offset, end))
        payload = unpack_member(block_path.read_bytes(), item)
        inspect_audio(payload)
        canonical_hash(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".wav.tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
        block_path.unlink()
        block_path.with_suffix(".headers").unlink()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, _ in enumerate(pool.map(acquire_one, selection["members"]), 1):
            if count % 100 == 0:
                print(json.dumps({"stage": "selected_acquisition", "verified_members": count}), flush=True)
    print(json.dumps({"stage": "selected_acquisition_complete", "members": len(selection["members"])}))


def verify() -> None:
    """Record checksums, decode quality, generator groups and frozen-prompt exclusions."""
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads((BANK / "selection.json").read_text())
    index = member_index()
    lock_path = OUT / "locked_groups.json"
    locked = json.loads(lock_path.read_text())
    eval_groups = set(locked["current_locked_eval"]["musiccaps_ids"])
    eval_groups.update(locked["historical_evaluation_seen"]["musiccaps_ids"])
    prior = set(locked["all_prior_seen"]["musiccaps_ids"])
    prior_hashes = set(locked["all_prior_seen"]["source_sha256"])
    prior_pcm = set(locked["all_prior_seen"]["pcm_sha256"])
    rows, group_generators, source_hashes, pcm_hashes = [], defaultdict(set), set(), set()
    flag_counts, generator_counts, failures = Counter(), Counter(), []
    manifest = OUT / "fakemusiccaps_sources.csv"
    for name in selection["members"]:
        item = index[name]
        parts = Path(item.filename).parts
        if (len(parts) != 2 or parts[0] not in GENERATORS or not re.fullmatch(r"[A-Za-z0-9_-]{11}\.wav", parts[1])
                or item.flag_bits & 1 or not 0 < item.file_size < 20_000_000):
            raise DataIntegrityError(f"Unexpected audio member: {item.filename}")
        generator, ytid = parts[0], Path(parts[1]).stem
        path = BANK / "originals" / item.filename
        if not path.exists():
            continue
        payload = path.read_bytes()
        if len(payload) != item.file_size or zlib.crc32(payload) != item.CRC:
            raise DataIntegrityError(f"Retained original CRC mismatch: {item.filename}")
        digest = hashlib.sha256(payload).hexdigest()
        try:
            rate, channels, frames, peak, rms, clipped, pcm, flags = inspect_audio(payload)
            canonical = canonical_hash(payload)
        except (sf.LibsndfileError, ValueError, subprocess.CalledProcessError, DataIntegrityError) as error:
            failures.append({"member": item.filename, "error": str(error)})
            continue
        if digest in source_hashes:
            flags.append("duplicate_source_bytes")
        if pcm in pcm_hashes:
            flags.append("duplicate_native_pcm")
        if digest in prior_hashes:
            flags.append("prior_source_bytes")
        if canonical in prior_pcm:
            flags.append("prior_mono16k_pcm")
        source_hashes.add(digest)
        pcm_hashes.add(pcm)
        generator_counts[generator] += 1
        group_generators[ytid].add(generator)
        flag_counts.update(flags)
        rows.append({"ID": f"fakemusiccaps_{generator}_{ytid}", "path": str(path.relative_to(ROOT)),
                     "archive_member": item.filename, "source": "FakeMusicCaps", "generator": generator,
                     "group_id": "reference:musiccaps:" + ytid, "reference_id": "musiccaps:" + ytid,
                     "source_url": URL, "source_sha256": digest, "native_pcm_sha256": pcm,
                     "pcm_sha256": canonical,
                     "source_bytes": len(payload), "crc32": f"{item.CRC:08x}", "sample_rate": rate,
                     "channels": channels, "frames": frames, "duration_seconds": frames / rate,
                     "peak": peak, "rms": rms, "near_full_scale_fraction": clipped,
                     "quality_flags": ";".join(flags), "overlaps_locked_musiccaps_prompt": ytid in eval_groups,
                     "overlaps_prior_musiccaps_prompt": ytid in prior,
                     "future_training_eligible": ytid not in eval_groups and ytid not in prior and not flags,
                     "split": "bank_unassigned", "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
                     "attribution": "Luca Comanducci; Paolo Bestagini; Stefano Tubaro, FakeMusicCaps v2 (2025)",
                     "FILE_FAKE": 1, "MUSIC_FAKE": 1, "MUSIC_PRESENT": 1,
                     "VOICE_FAKE": "", "VOICE_PRESENT": "", "label_status": "source_documented; voice_unknown"})
        if len(rows) % 1000 == 0:
            print(json.dumps({"stage": "decode", "tracks": len(rows)}), flush=True)
    if not rows:
        raise DataIntegrityError("No acquired original WAVs available for verification")
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    incomplete = {key: sorted(value) for key, value in group_generators.items() if value != GENERATORS}
    receipt = {"status": "verified" if len(rows) == SELECTED_GROUPS * 5 and not failures and not incomplete else "partial_acquisition",
               "audio_tracks": len(rows), "prompt_groups": len(group_generators), "generator_counts": generator_counts,
               "official_full_archive_bytes": TOTAL, "official_full_archive_md5": MD5,
               "whole_archive_md5_verified": False, "verification_scope": "original member CRC, source SHA and decode; official central directory provenance",
               "bank_bytes": sum(p.stat().st_size for p in BANK.rglob("*") if p.is_file()),
               "bank_cap_bytes": BANK_CAP, "selection_sha256": sha256_file(BANK / "selection.json"),
               "manifest_sha256": sha256_file(manifest), "locked_groups_sha256": sha256_file(lock_path),
               "script_sha256": sha256_file(Path(__file__)), "decode_failures": failures,
               "incomplete_prompt_groups": incomplete, "quality_flag_counts": flag_counts,
               "overlap_locked_prompt_groups": sorted(set(group_generators) & eval_groups),
               "overlap_prior_prompt_groups": sorted(set(group_generators) & prior),
               "future_training_eligible_tracks": sum(row["future_training_eligible"] for row in rows),
               "rights_evidence": "https://zenodo.org/api/records/15063698",
               "license": "CC-BY-NC-4.0", "split_assigned": False, "independent_listening": False,
               "no_audio_modification": True, "all_audio_crc_verified": True,
               "native_pcm_hash_contract": "SHA256(UTF8(sample_rate:channels:) + interleaved float32 little-endian PCM)",
               "historical_canonical_pcm_comparison": "full FFmpeg mono16k float32, same settings as v2_data.decode_audio(limit_seconds=None)",
               "canonical_decoder": imageio_ffmpeg.get_ffmpeg_exe(),
               "canonical_decoder_sha256": sha256_file(Path(imageio_ffmpeg.get_ffmpeg_exe())),
               "near_duplicate_check": "not performed", "youtube_human_audio_downloaded": False}
    (OUT / "fakemusiccaps_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "audio_tracks", "prompt_groups", "generator_counts", "quality_flag_counts", "future_training_eligible_tracks")}))


def self_check() -> None:
    """Protect the actual quality gate against silent/nonfinite/short WAVs and CRC corruption."""
    payload = io.BytesIO()
    sf.write(payload, np.zeros(16000, dtype=np.float32), 16000, format="WAV", subtype="FLOAT")
    assert set(inspect_audio(payload.getvalue())[-1]) == {"silent", "too_short_lt4s", "duration_differs_from_10s_gt0.5s"}
    payload = io.BytesIO()
    sf.write(payload, np.array([np.nan, 0.1], dtype=np.float32), 16000, format="WAV", subtype="FLOAT")
    assert "nonfinite" in inspect_audio(payload.getvalue())[-1]
    with tempfile.TemporaryDirectory() as folder:
        wave = Path(folder) / "reference.wav"
        sf.write(wave, np.sin(np.arange(441000) * 0.1).astype(np.float32), 44100, subtype="PCM_24")
        assert canonical_hash(wave.read_bytes()) == pcm_sha256(decode_audio(wave, limit_seconds=None))
        path = Path(folder) / "bad.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("source.wav", b"known audio bytes")
            item = archive.getinfo("source.wav")
        assert unpack_member(path.read_bytes(), item) == b"known audio bytes"
        path.write_bytes(path.read_bytes().replace(b"known audio bytes", b"wrong audio bytes"))
        try:
            unpack_member(path.read_bytes(), item)
        except DataIntegrityError:
            print("Silent/short/nonfinite PCM, archive CRC corruption and path-vs-stream mono16k hash parity verified")
        else:
            raise AssertionError("CRC corruption was not rejected")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    elif "--prepare-selected" in sys.argv:
        prepare_selected()
        verify()
    elif "--acquire-selected" in sys.argv:
        if "--release-network" not in sys.argv:
            raise DataIntegrityError("Network acquisition remains held by the global budget coordinator")
        acquire_selected()
        verify()
    elif "--download" in sys.argv:
        download()
    else:
        verify()
