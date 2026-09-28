"""Index licensed speech archives without extracting an additional audio tree.

Run with the existing environment:
  .venv/bin/python new_panel_data_speech_bank.py test-clean
  .venv/bin/python new_panel_data_speech_bank.py --prepare-asv-selected
  .venv/bin/python new_panel_data_speech_bank.py --acquire-asv-selected --release-network

Outputs are an unassigned training/diagnostic bank, not a new evaluation split.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Final, Iterator

import numpy as np
import soundfile as sf

from v2_data import DataIntegrityError, decode_audio, extract_local_member, pcm_sha256, sha256_file

ROOT: Final = Path(__file__).resolve().parent
DATA: Final = ROOT / "_workspace/data_expansion_20260918/speech"
OUT: Final = ROOT / "results/data_expansion_20260918"
HELDOUT_RECEIPT: Final = ROOT / "results/local_sampler_20260918/data_preparation_speech.json"
DF_CARD: Final = "https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1/blob/fb6ce85/README.md"
ASV_URL: Final = "https://zenodo.org/api/records/6906306/files/LA.zip/content"
ASV_TOTAL: Final = 7_640_952_520
ASV_CAP: Final = 300_000_000
ASV_MAX_SPAN: Final = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    archive: Path
    url: str
    md5: str
    license: str
    license_url: str
    corpus_seen_by_df_arena: str


@dataclass(frozen=True, slots=True)
class ArchiveSpan:
    start: int
    end: int
    members: tuple[zipfile.ZipInfo, ...]


SOURCES: Final = {
    "test-clean": Source(
        "librispeech_test-clean",
        ROOT / "_workspace/local_sampler_20260918/new_data/speech/downloads/test-clean.tar.gz",
        "https://openslr.trmal.net/resources/12/test-clean.tar.gz",
        "32fa31d27d2e1cad72775fee3f4849a9", "CC-BY-4.0",
        "https://www.openslr.org/12/", "not_explicitly_listed; shared_LibriVox_possible"),
    "train-clean-100": Source(
        "librispeech_train-clean-100", DATA / "downloads/train-clean-100.tar.gz",
        "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
        "2a93770f6d5c6c964bc36631d331a522", "CC-BY-4.0",
        "https://www.openslr.org/12/", "not_explicitly_listed; shared_LibriVox_possible"),
    "asvspoof2019-la": Source(
        "asvspoof2019_la", DATA / "downloads/ASVspoof2019_LA.zip",
        "https://zenodo.org/records/6906306/files/LA.zip?download=1",
        "30c98f11d8b2bc21f2c257bfd78bb5c5", "ODC-By-1.0",
        "https://zenodo.org/records/6906306", "listed_training_corpus; exact_recording_unknown"),
}


@dataclass(frozen=True, slots=True)
class BankRow:
    ID: str
    source: str
    archive_path: str
    archive_member: str
    source_url: str
    official_partition: str
    utterance_id: str
    speaker_id: str
    group_id: str
    source_sha256: str
    pcm_sha256: str
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    source_bytes: int
    path: str = ""
    native_pcm_sha256: str = ""
    crc32: str = ""
    source_reference_id: str = ""
    target_speaker_id: str = ""
    vc_input_reference_id: str = ""
    vc_input_speaker_id: str = ""
    source_relationships_verified: bool = False
    speaker_name: str = ""
    gender: str = "unknown"
    chapter_id: str = ""
    transcript: str = ""
    attack_id: str = ""
    generator: str = "unknown"
    label: str = "unknown"
    FILE_FAKE: int | str = ""
    VOICE_FAKE: int | str = ""
    MUSIC_FAKE: str = ""
    VOICE_PRESENT: int = 1
    MUSIC_PRESENT: int = 0
    FILE_FAKE_mask: int = 0
    VOICE_FAKE_mask: int = 0
    MUSIC_FAKE_mask: int = 0
    VOICE_PRESENT_mask: int = 1
    MUSIC_PRESENT_mask: int = 1
    frame_labels_available: bool = False
    split: str = "unassigned"
    train_eligible: bool = True
    exclusion_reason: str = ""
    duplicate_of: str = ""
    license: str = ""
    license_url: str = ""
    label_status: str = "source_documented"
    df_arena_pretraining: str = ""
    evaluation_scope: str = "training_or_diagnostic_only; not_pretrained_independent"
    independent_listening_review: bool = False
    decode_status: str = "finite_complete_native_pcm"


def asv_index() -> dict[str, zipfile.ZipInfo]:
    """Rebase only ZIP64 directory bookkeeping, preserving original member offsets."""
    metadata = DATA / "metadata"
    tail = (metadata / "ASV_LA_zip_eocd.bin").read_bytes()
    directory = (metadata / "ASV_LA_zip_central.bin").read_bytes()
    ranges = re.findall(r"(?im)^content-range:\s*([^\r\n]+)",
                        (metadata / "ASV_LA_zip_central.headers").read_text())
    if ranges != [f"bytes 7627949105-7640952363/{ASV_TOTAL}"] or len(directory) != 13_003_259:
        raise DataIntegrityError("ASV central directory range/size mismatch")
    offset = tail.rfind(b"PK\x05\x06")
    end = list(struct.unpack_from("<4s4H2LH", tail, offset))
    large = list(struct.unpack_from("<4sQ2H2L4Q", tail, offset - 76))
    locator = list(struct.unpack_from("<4sLQL", tail, offset - 20))
    if large != [b"PK\x06\x06", 44, 62, 45, 0, 0, 122328, 122328, 13003259, 7627949105]:
        raise DataIntegrityError("ASV ZIP64 directory identity differs")
    large[-1], locator[2], end[6] = 0, len(directory), 0
    synthetic = (directory + struct.pack("<4sQ2H2L4Q", *large)
                 + struct.pack("<4sLQL", *locator) + struct.pack("<4s4H2LH", *end) + tail[offset + 22:])
    with zipfile.ZipFile(io.BytesIO(synthetic)) as archive:
        infos = archive.infolist()
    result = {entry.filename: entry for entry in infos}
    if len(result) != 122328:
        raise DataIntegrityError("ASV ZIP directory has missing or duplicate entries")
    return result


def fetch_asv_member(info: zipfile.ZipInfo, output: Path) -> Path:
    """Fetch a bounded original member, requiring exact ranges, name and CRC."""
    if output.exists():
        payload = output.read_bytes()
        if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
            raise DataIntegrityError(f"Cached ASV member CRC mismatch: {output}")
        return output
    if info.flag_bits & 1 or not 0 < info.file_size < 5_000_000:
        raise DataIntegrityError(f"Unsupported ASV source member: {info.filename}")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = info.header_offset
    length = 30 + len(info.filename.encode("utf-8")) + 128 + info.compress_size
    end = start + length - 1
    blob = output.with_suffix(output.suffix + ".range")
    headers = output.with_suffix(output.suffix + ".headers")
    subprocess.run(["curl", "-fLsS", "--connect-timeout", "20", "--max-time", "180",
                    "--max-filesize", str(length),
                    "--range", f"{start}-{end}", "-D", str(headers),
                    f"{ASV_URL}?range={start}-{end}", "-o", str(blob)], check=True)
    actual = re.findall(r"(?im)^content-range:\s*([^\r\n]+)", headers.read_text())
    if not actual or actual[-1].strip() != f"bytes {start}-{end}/{ASV_TOTAL}" or blob.stat().st_size != length:
        raise DataIntegrityError("ASV server did not return the exact bounded member range")
    raw = blob.read_bytes()
    name_size, extra_size = struct.unpack_from("<2H", raw, 26)
    if raw[30:30 + name_size].decode("utf-8") != info.filename or extra_size > 128:
        raise DataIntegrityError("ASV local header identity/extra length differs")
    entry = {key: getattr(info, key) for key in ("compress_type", "compress_size", "file_size", "CRC")}
    extract_local_member(raw, entry, output)
    blob.unlink()
    return output


def prepare_asv_selection() -> None:
    """Lock balanced CM-protocol members and all source-voice relationships before audio download."""
    index = asv_index()
    metadata = DATA / "metadata"
    relationships = {}
    with (metadata / "ASVspoof2019_LA_VCTK_MetaInfo.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            relationships[row["ASVspoof_ID"]] = row
    candidates = []
    protocol_counts = Counter()
    for name, info in index.items():
        if not ("_cm_protocols/" in name and name.endswith(".txt")):
            continue
        path = fetch_asv_member(info, metadata / "cm_protocols" / Path(name).name)
        partition = next(part for part in ("train", "dev", "eval") if f".{part}." in name)
        for line in path.read_text().splitlines():
            speaker, utterance, _, attack, label = line.split()
            member = f"LA/ASVspoof2019_LA_{partition}/flac/{utterance}.flac"
            native = index[member]
            relation = relationships[utterance]
            real = label == "bonafide"
            if (label not in {"bonafide", "spoof"}
                    or real != (relation["VCTK_ID"] != "-")
                    or (not real and attack not in {f"A{i:02d}" for i in range(1, 20)})):
                raise DataIntegrityError(f"ASV authenticity/provenance disagreement: {utterance}")
            protocol_counts[label] += 1
            candidates.append({"member": member, "utterance_id": utterance,
                               "partition": partition, "protocol_speaker_id": speaker,
                               "attack_id": "" if real else attack, "label": label,
                               "source_bytes": native.file_size, "compressed_bytes": native.compress_size,
                               "VCTK_ID": relation["VCTK_ID"],
                               "TTS_VC_target_speaker": relation["TTS_VC_target_speaker"],
                               "VC_source_VCTK_ID": relation["VC_source_VCTK_ID"],
                               "TTS_text": relation["TTS_text"]})
    if protocol_counts != {"bonafide": 12483, "spoof": 108978}:
        raise DataIntegrityError(f"ASV official CM coverage mismatch: {protocol_counts}")
    selected = []
    for label in ("bonafide", "spoof"):
        buckets = defaultdict(lambda: defaultdict(list))
        for row in candidates:
            if row["label"] == label:
                person = row["VCTK_ID"].split("_")[0] if label == "bonafide" else row["TTS_VC_target_speaker"]
                buckets[row["attack_id"]][person].append(row)
        for people in buckets.values():
            for pool in people.values():
                pool.sort(key=lambda row: hashlib.sha256(("20260918:" + row["utterance_id"]).encode()).hexdigest())
        cursors = Counter()
        class_rows = []
        while len(class_rows) < 2000:
            for attack in sorted(buckets):
                people = buckets[attack]
                available = sorted(person for person in people if people[person])
                if not available:
                    continue
                person = available[cursors[attack] % len(available)]
                cursors[attack] += 1
                class_rows.append(people[person].pop())
                if len(class_rows) == 2000:
                    break
        selected.extend(class_rows)
    source_bytes = sum(row["source_bytes"] for row in selected)
    if source_bytes > ASV_CAP:
        raise DataIntegrityError(f"Frozen balanced ASV selection exceeds 300MB: {source_bytes}")
    target = DATA / "asv_selection.json"
    document = {"status": "selected_not_acquired", "max_original_bytes": ASV_CAP,
                "original_bytes": source_bytes, "rows": len(selected),
                "official_cm_pool_rows": sum(protocol_counts.values()), "class_counts": dict(protocol_counts),
                "selected_labels": dict(Counter(row["label"] for row in selected)),
                "selected_attacks": dict(Counter(row["attack_id"] or "bonafide" for row in selected)),
                "source_relationships": "original VCTK / target VCTK / VC input VCTK kept separately",
                "central_sha256": sha256_file(metadata / "ASV_LA_zip_central.bin"),
                "relationships_sha256": sha256_file(metadata / "ASVspoof2019_LA_VCTK_MetaInfo.tsv"),
                "selection_method": "fixed SHA256 order, round robin attack and actual source/target speaker, 2000 real + 2000 fake",
                "members": sorted(selected, key=lambda row: row["member"])}
    if target.exists() and json.loads(target.read_text()) != document:
        raise DataIntegrityError("Existing frozen ASV selection differs; refusing replacement")
    target.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({key: value for key, value in document.items() if key != "members"}, indent=2))


def coalesced_asv_spans(infos: list[zipfile.ZipInfo]) -> list[ArchiveSpan]:
    """Group nearby selected entries within the authorized response-size limit."""
    spans: list[ArchiveSpan] = []
    for info in sorted(infos, key=lambda item: item.header_offset):
        path = DATA / "asv_originals" / Path(info.filename).name
        if path.exists():
            payload = path.read_bytes()
            if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
                raise DataIntegrityError(f"Cached selected ASV file failed CRC: {path}")
            continue
        start = info.header_offset
        end = start + 30 + len(info.filename.encode("utf-8")) + 128 + info.compress_size - 1
        if end - start + 1 > ASV_MAX_SPAN:
            raise DataIntegrityError("Selected ASV member exceeds the authorized span limit")
        if spans and end - spans[-1].start + 1 <= ASV_MAX_SPAN:
            spans[-1] = replace(spans[-1], end=end, members=spans[-1].members + (info,))
        else:
            spans.append(ArchiveSpan(start, end, (info,)))
    return spans


def fetch_asv_span(span: ArchiveSpan) -> int:
    """Discard non-selected gap bytes in memory; admit only CRC-verified originals."""
    length = span.end - span.start + 1
    if length > ASV_MAX_SPAN:
        raise DataIntegrityError("ASV coalesced response exceeds the authorized span limit")
    headers = DATA / "metadata/asv_ranges" / f"{span.start}-{span.end}.headers"
    headers.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["curl", "-fLsS", "--connect-timeout", "20", "--max-time", "180",
         "--max-filesize", str(length), "--range", f"{span.start}-{span.end}",
         "-D", str(headers), f"{ASV_URL}?range={span.start}-{span.end}"],
        check=True, capture_output=True)
    raw = completed.stdout
    response = headers.read_text()
    actual = re.findall(r"(?im)^content-range:\s*([^\r\n]+)", response)
    if len(raw) != length or not actual or actual[-1].strip() != f"bytes {span.start}-{span.end}/{ASV_TOTAL}":
        raise DataIntegrityError("Coalesced ASV HTTP response range/length differs")
    for info in span.members:
        local = raw[info.header_offset - span.start:]
        if len(local) < 30:
            raise DataIntegrityError("Coalesced ASV member header is truncated")
        name_size, extra_size = struct.unpack_from("<2H", local, 26)
        if local[30:30 + name_size].decode("utf-8") != info.filename or extra_size > 128:
            raise DataIntegrityError("Coalesced ASV member identity differs")
        entry = {key: getattr(info, key) for key in ("compress_type", "compress_size", "file_size", "CRC")}
        output = DATA / "asv_originals" / Path(info.filename).name
        temporary = output.with_suffix(output.suffix + ".member.part")
        try:
            extract_local_member(local, entry, temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return len(raw)


def probe_asv_span() -> None:
    """Check one real multi-member response before the full coalesced acquisition."""
    if "--release-network" not in sys.argv:
        raise DataIntegrityError("ASV probe remains held by the global budget coordinator")
    selection = json.loads((DATA / "asv_selection.json").read_text())
    index = asv_index()
    candidates = coalesced_asv_spans([index[row["member"]] for row in selection["members"]])
    span = next(candidate for candidate in candidates if len(candidate.members) > 1)
    transferred = fetch_asv_span(span)
    items = {row["member"]: row for row in selection["members"]}
    checked = []
    for info in span.members:
        item = items[info.filename]
        path = DATA / "asv_originals" / Path(info.filename).name
        row = audio_row(SOURCES["asvspoof2019-la"], info.filename, path.read_bytes(), item["partition"],
                        item["protocol_speaker_id"], item["label"], item["attack_id"])
        if row.pcm_sha256 != pcm_sha256(decode_audio(path, limit_seconds=None)):
            raise DataIntegrityError("First coalesced ASV source has decoder disagreement")
        checked.append({"member": info.filename, "source_bytes": info.file_size,
                        "source_sha256": row.source_sha256, "pcm_sha256": row.pcm_sha256})
    receipt = {"status": "verified_real_multi_member_range", "range": [span.start, span.end],
               "expected_and_received_bytes": transferred, "original_archive_total_bytes": ASV_TOTAL,
               "selected_members": checked, "member_crc_verified": True,
               "full_native_and_canonical_decoder_parity": True, "unselected_gap_bytes_retained": False}
    (OUT / "speech_asv_coalesced_probe.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


def asv_retry_delay(response: str, now: float) -> float:
    """Honor the later valid server reset, Retry-After and conservative60s floor."""
    requested = re.findall(r"(?im)^retry-after:\s*(\d+)\s*$", response)
    resets = re.findall(r"(?im)^x-ratelimit-reset:\s*(\d+(?:\.\d+)?)\s*$", response)
    reset = float(resets[-1]) if resets else 0.0
    reset_delay = max(0, reset - now) if np.isfinite(reset) else 0.0
    return max(60, int(requested[-1]) if requested else 60, reset_delay)


def acquire_asv_selection() -> None:
    """Acquire only the frozen 300MB selection after explicit network release."""
    if "--release-network" not in sys.argv:
        raise DataIntegrityError("ASV audio acquisition remains held by the global budget coordinator")
    selection = json.loads((DATA / "asv_selection.json").read_text())
    if selection["original_bytes"] > ASV_CAP or selection["rows"] > 4000:
        raise DataIntegrityError("ASV selection exceeds the authorized cap")
    index = asv_index()
    selected_infos = [index[row["member"]] for row in selection["members"]]
    if (len(selected_infos) != selection["rows"] or len({info.filename for info in selected_infos}) != len(selected_infos)
            or sum(info.file_size for info in selected_infos) != selection["original_bytes"]
            or sha256_file(DATA / "metadata/ASV_LA_zip_central.bin") != selection["central_sha256"]):
        raise DataIntegrityError("Frozen ASV selection size/identity differs")
    tasks = coalesced_asv_spans(selected_infos)
    planned = {"stage": "coalesced_plan", "requests": len(tasks),
               "remaining_selected_members": sum(len(span.members) for span in tasks),
               "planned_transfer_bytes": sum(span.end - span.start + 1 for span in tasks),
               "selected_original_bytes": selection["original_bytes"], "max_span_bytes": ASV_MAX_SPAN,
               "unselected_gap_bytes_retained": False}
    (OUT / "speech_asv_transfer_plan.json").write_text(json.dumps(planned, indent=2) + "\n")
    print(json.dumps(planned), flush=True)
    request_lock = threading.Lock()
    last_request = [0.0]

    def acquire(span: ArchiveSpan) -> int:
        for attempt in range(6):
            while True:
                with request_lock:
                    delay = max(0, 0.8 - (time.monotonic() - last_request[0]))
                    if delay == 0:
                        last_request[0] = time.monotonic()
                        break
                time.sleep(min(delay, 60))
            try:
                return fetch_asv_span(span)
            except subprocess.CalledProcessError as error:
                headers = DATA / "metadata/asv_ranges" / f"{span.start}-{span.end}.headers"
                response = headers.read_text() if headers.exists() else ""
                limited = " 429 " in response
                transport = error.returncode in {6, 7, 18, 28, 35, 52, 55, 56}
                transient_http = bool(re.search(r"(?m)^HTTP/\S+ (408|500|502|503|504)\b", response))
                if attempt == 5 or not (limited or transport or transient_http):
                    raise
                delay = asv_retry_delay(response, time.time()) if limited else min(60, 5 * 2**attempt)
                with request_lock:
                    last_request[0] = max(last_request[0], time.monotonic() + delay)
                print(json.dumps({"stage": "shared_rate_limit_cooldown" if limited else "shared_transport_cooldown",
                                  "seconds": delay, "curl_exit": error.returncode,
                                  "stderr": (error.stderr or b"").decode("utf-8", errors="replace")[-600:],
                                  "range": [span.start, span.end], "retry": attempt + 1}), flush=True)
        raise DataIntegrityError(f"ASV retries exhausted: {span.start}-{span.end}")

    completed_members = selection["rows"] - sum(len(span.members) for span in tasks)
    transferred = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        for number, (span, size) in enumerate(zip(tasks, executor.map(acquire, tasks)), 1):
            transferred += size
            completed_members += len(span.members)
            if number % 100 == 0:
                print(json.dumps({"acquired_asv_members": completed_members, "selected": selection["rows"],
                                  "completed_requests": number, "transferred_bytes": transferred}), flush=True)
    planned.update(status="acquired", completed_requests=len(tasks), transferred_bytes=transferred,
                   acquired_members=completed_members)
    (OUT / "speech_asv_transfer_plan.json").write_text(json.dumps(planned, indent=2) + "\n")


def verify_asv_selection() -> None:
    """Decode the frozen originals; retain source/target/input IDs for later leakage-safe splitting."""
    selection_path = DATA / "asv_selection.json"
    selection = json.loads(selection_path.read_text())
    index = asv_index()
    source = SOURCES["asvspoof2019-la"]
    rows = []
    duplicates: dict[str, str] = {}
    for item in selection["members"]:
        info = index[item["member"]]
        path = DATA / "asv_originals" / Path(info.filename).name
        if not path.exists():
            raise DataIntegrityError(f"Frozen ASV source missing: {path}")
        payload = path.read_bytes()
        if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
            raise DataIntegrityError(f"Original ASV source CRC mismatch: {path}")
        row = audio_row(source, info.filename, payload, item["partition"],
                        item["protocol_speaker_id"], item["label"], item["attack_id"])
        canonical = pcm_sha256(decode_audio(path, limit_seconds=None))
        if canonical != row.pcm_sha256:
            raise DataIntegrityError(f"Native versus canonical ASV decoder differs: {path}")
        original = "" if item["VCTK_ID"] == "-" else item["VCTK_ID"]
        target = original.split("_")[0] if original else item["TTS_VC_target_speaker"]
        vc_input = "" if item["VC_source_VCTK_ID"] == "-" else item["VC_source_VCTK_ID"]
        duplicate = duplicates.get(canonical, "")
        duplicates[canonical] = row.ID
        row = replace(row, archive_path="", path=str(path.relative_to(ROOT)),
                      native_pcm_sha256=row.pcm_sha256, crc32=f"{info.CRC:08x}",
                      source_reference_id=original, target_speaker_id=target,
                      vc_input_reference_id=vc_input, vc_input_speaker_id=vc_input.split("_")[0] if vc_input else "",
                      source_relationships_verified=True,
                      transcript="" if item["TTS_text"] == "-" else item["TTS_text"],
                      train_eligible=not bool(duplicate), duplicate_of=duplicate,
                      exclusion_reason="exact_pcm_duplicate" if duplicate else "")
        rows.append(row)
    manifest = OUT / "speech_bank_asv_selected.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(BankRow)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    receipt = {"status": "verified", "rows": len(rows), "source_bytes": sum(row.source_bytes for row in rows),
               "max_original_bytes": ASV_CAP, "whole_archive_md5_verified": False,
               "verification_scope": "exact HTTP ranges; official ZIP member CRC; source SHA256; full native and FFmpeg mono16k PCM parity",
               "manifest_sha256": sha256_file(manifest), "selection_sha256": sha256_file(selection_path),
               "labels": dict(Counter(row.label for row in rows)), "attacks": dict(Counter(row.attack_id or "bonafide" for row in rows)),
               "protocol_speakers": len({row.speaker_id for row in rows}),
               "actual_original_or_target_vctk_speakers": len({row.target_speaker_id for row in rows}),
               "vc_input_speakers": len({row.vc_input_speaker_id for row in rows} - {""}),
               "vc_input_link_rows": sum(bool(row.vc_input_reference_id) for row in rows),
               "exact_pcm_duplicates": sum(bool(row.duplicate_of) for row in rows),
               "df_arena_pretraining": source.corpus_seen_by_df_arena, "df_arena_model_card": DF_CARD,
               "split": "unassigned", "independent_listening_review": False,
               "license": source.license, "source_url": ASV_URL,
               "canonical_decoder": "v2_data.decode_audio(limit_seconds=None)",
               "limitations": ["Individual gender unknown", "Corpus-listed DF-Arena overlap; exact pretrained recording membership unknown",
                               "No frame labels or listening review", "Target and VC-input identities must both constrain any future split"]}
    (OUT / "speech_bank_asv_selected.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def verified_hashes(source: Source) -> tuple[str, str]:
    """Verify official bytes before publishing any bank rows."""
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with source.archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != source.md5:
        raise DataIntegrityError(f"Official MD5 mismatch: {source.archive}")
    return md5.hexdigest(), sha.hexdigest()


def audio_row(source: Source, member: str, payload: bytes, partition: str,
              speaker: str, label: str, attack: str = "") -> BankRow:
    """Decode each complete original utterance and hash native float32 PCM."""
    with sf.SoundFile(io.BytesIO(payload)) as stream:
        rate, channels, expected_frames = stream.samplerate, stream.channels, stream.frames
        audio = stream.read(dtype="float32", always_2d=True)
    if rate != 16000 or channels != 1 or len(audio) != expected_frames or len(audio) == 0:
        raise DataIntegrityError(f"Invalid native speech metadata/decoded frames: {member}")
    if not np.isfinite(audio).all():
        raise DataIntegrityError(f"Nonfinite decoded speech: {member}")
    known = label in {"bonafide", "spoof"}
    fake = int(label == "spoof") if known else ""
    utterance = Path(member).stem
    return BankRow(
        ID=f"{source.name}:{utterance}", source=source.name,
        archive_path=str(source.archive.relative_to(ROOT)), archive_member=member,
        source_url=source.url, official_partition=partition, utterance_id=utterance,
        speaker_id=speaker, group_id=f"{source.name.split('_')[0]}:{speaker}" if speaker else "",
        source_sha256=hashlib.sha256(payload).hexdigest(),
        pcm_sha256=hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes()).hexdigest(),
        sample_rate=rate, channels=channels, frames=expected_frames,
        duration_seconds=expected_frames / rate, source_bytes=len(payload),
        attack_id=attack, generator=attack or ("original" if label == "bonafide" else "unknown"),
        label=label, FILE_FAKE=fake, VOICE_FAKE=fake,
        FILE_FAKE_mask=int(known), VOICE_FAKE_mask=int(known),
        train_eligible=known, exclusion_reason="" if known else "missing_cm_protocol_label",
        label_status="source_documented" if known else "authenticity_unknown",
        license=source.license, license_url=source.license_url,
        df_arena_pretraining=source.corpus_seen_by_df_arena,
    )


def libri_rows(source: Source) -> Iterator[BankRow]:
    """Keep official speaker IDs, names, genders, chapters and transcripts."""
    speakers: dict[str, list[str]] = {}
    transcripts: dict[str, str] = {}
    metadata = DATA / "metadata" / source.name
    metadata.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source.archive, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".txt"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise DataIntegrityError(f"Cannot read metadata: {member.name}")
            with stream:
                raw = stream.read()
            text = raw.decode("utf-8")
            target = metadata / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            if member.name.endswith("SPEAKERS.TXT"):
                for line in text.splitlines():
                    if line.strip() and not line.startswith(";"):
                        values = [part.strip() for part in line.split("|")]
                        speakers[values[0]] = values
            if member.name.endswith(".trans.txt"):
                for line in text.splitlines():
                    utterance, transcript = line.split(" ", 1)
                    transcripts[utterance] = transcript
    heldout = json.loads(HELDOUT_RECEIPT.read_text())["narrator_names"]
    heldout_names = {normalized_name(name) for name in heldout.values()}
    with tarfile.open(source.archive, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".flac"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise DataIntegrityError(f"Cannot read audio: {member.name}")
            with stream:
                payload = stream.read()
            utterance = Path(member.name).stem
            speaker, chapter, _ = utterance.split("-")
            info = speakers[speaker]
            excluded = speaker in heldout or normalized_name(info[4]) in heldout_names
            row = audio_row(source, member.name, payload, info[2], speaker, "bonafide")
            yield replace(row, speaker_name=info[4], gender=info[1], chapter_id=chapter,
                          transcript=transcripts[utterance], train_eligible=not excluded,
                          exclusion_reason="reserved_panel_narrator" if excluded else "")


def asv_rows(source: Source) -> Iterator[BankRow]:
    """Join original countermeasure labels; retain other audio as unknown."""
    protocol: dict[str, tuple[str, str, str, str]] = {}
    metadata = DATA / "metadata" / source.name
    metadata.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source.archive) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".txt"):
                continue
            if "__MACOSX" in member.filename:
                continue
            raw = archive.read(member)
            target = metadata / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            if "_cm_protocols/" not in member.filename:
                continue
            partition = next((part for part in ("train", "dev", "eval")
                              if f".{part}." in member.filename), "unknown")
            for line in raw.decode("utf-8").splitlines():
                speaker, utterance, _, attack, label = line.split()
                if label not in {"bonafide", "spoof"} or utterance in protocol:
                    raise DataIntegrityError(f"Unexpected/duplicate CM protocol entry: {line}")
                protocol[utterance] = (partition, speaker, attack, label)
        encountered: set[str] = set()
        for member in archive.infolist():
            if member.is_dir() or not member.filename.endswith(".flac") or "__MACOSX" in member.filename:
                continue
            utterance = Path(member.filename).stem
            fallback_partition = next((part for part in ("train", "dev", "eval")
                                       if f"_LA_{part}/" in member.filename), "unknown")
            partition, speaker, attack, label = protocol.get(
                utterance, (fallback_partition, "", "", "unknown"))
            encountered.add(utterance)
            yield audio_row(source, member.filename, archive.read(member), partition,
                            speaker, label, "" if attack == "-" else attack)
        missing = set(protocol) - encountered
        if missing:
            raise DataIntegrityError(f"CM labels without audio: {len(missing)}")


def main() -> None:
    """Publish only a fully decoded, checksum-verified archive index."""
    if "--prepare-asv-selected" in sys.argv:
        prepare_asv_selection()
        return
    if "--probe-asv-span" in sys.argv:
        probe_asv_span()
        return
    if "--acquire-asv-selected" in sys.argv:
        acquire_asv_selection()
        verify_asv_selection()
        return
    if "--verify-asv-selected" in sys.argv:
        verify_asv_selection()
        return
    if len(sys.argv) != 2 or sys.argv[1] not in SOURCES:
        raise SystemExit("Usage: new_panel_data_speech_bank.py {test-clean|train-clean-100|asvspoof2019-la}")
    key = sys.argv[1]
    if key != "test-clean":
        raise DataIntegrityError("Full new speech archives are prohibited by the user's global 20GB cap; use the selected ASV path")
    source = SOURCES[key]
    free = shutil.disk_usage(ROOT).free
    added_bytes = sum(path.stat().st_size for path in (DATA / "downloads").glob("*") if path.is_file())
    if free < 70 * 1024**3 or added_bytes > 16_000_000_000:
        raise DataIntegrityError(f"Speech storage budget exceeded: free={free}, added={added_bytes}")
    started = time.time()
    archive_md5, archive_sha = verified_hashes(source)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"speech_bank_{key}.csv"
    temporary = destination.with_suffix(".partial.csv")
    receipt = OUT / f"speech_bank_{key}.json"
    rows = libri_rows(source) if key in {"test-clean", "train-clean-100"} else asv_rows(source)
    counts: Counter[str] = Counter()
    speakers: set[str] = set()
    seen_ids: set[str] = set()
    pcm_ids: dict[str, str] = {}
    total_seconds = 0.0
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(BankRow)])
        writer.writeheader()
        for row in rows:
            if row.ID in seen_ids:
                raise DataIntegrityError(f"Duplicate utterance ID: {row.ID}")
            seen_ids.add(row.ID)
            duplicate = pcm_ids.get(row.pcm_sha256, "")
            if duplicate:
                row = replace(row, duplicate_of=duplicate, train_eligible=False,
                              exclusion_reason=";".join(filter(None, (row.exclusion_reason, "exact_pcm_duplicate"))))
            else:
                pcm_ids[row.pcm_sha256] = row.ID
            writer.writerow(asdict(row))
            speakers.add(row.group_id)
            total_seconds += row.duration_seconds
            counts["rows"] += 1
            counts[f"label:{row.label}"] += 1
            counts[f"partition:{row.official_partition}"] += 1
            counts[f"gender:{row.gender}"] += 1
            counts[f"attack:{row.attack_id or 'none'}"] += 1
            counts["train_eligible"] += int(row.train_eligible)
            counts["exact_pcm_duplicates"] += bool(duplicate)
            counts["reserved_panel_narrator_rows"] += "reserved_panel_narrator" in row.exclusion_reason
            if counts["rows"] % 5000 == 0:
                stream.flush()
                print(json.dumps({"source": source.name, "decoded_rows": counts["rows"],
                                  "elapsed_seconds": round(time.time() - started, 2)}), flush=True)
    temporary.replace(destination)
    with destination.open("rb") as stream:
        manifest_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt.write_text(json.dumps({
        "source": source.name, "source_url": source.url, "archive_path": str(source.archive.relative_to(ROOT)),
        "archive_bytes": source.archive.stat().st_size, "archive_md5": archive_md5,
        "archive_sha256": archive_sha, "official_checksum_verified": True,
        "license": source.license, "license_url": source.license_url,
        "manifest": str(destination.relative_to(ROOT)), "manifest_sha256": manifest_sha,
        "counts": dict(counts), "distinct_documented_speaker_ids": len(speakers - {""}),
        "hours": total_seconds / 3600, "bank_split": "unassigned", "full_audio_extraction": False,
        "pcm_hash_format": "complete native 16000Hz mono float32 little-endian samples; no resampling or trimming",
        "df_arena_pretraining": source.corpus_seen_by_df_arena, "df_arena_model_card": DF_CARD,
        "evaluation_scope": "training/diagnostic bank; no pretrained-independent evaluation claim",
        "heldout_speaker_receipt": str(HELDOUT_RECEIPT.relative_to(ROOT)),
        "elapsed_seconds": time.time() - started,
        "limitations": ["No independent listening or semantic transcription review",
                        "Narrator alias and cross-corpus person identity are not fully resolved",
                        "Unknown authenticity and generator metadata remain masked",
                        "No frame boundaries or local train/dev/test assignment inferred",
                        "No DACON permission or submission result follows from dataset licensing"],
    }, ensure_ascii=False, indent=2) + "\n")
    print(receipt.read_text(), flush=True)


if __name__ == "__main__":
    main()
