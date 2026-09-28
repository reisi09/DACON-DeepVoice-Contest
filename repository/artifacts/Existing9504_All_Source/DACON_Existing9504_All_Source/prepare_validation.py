#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "imageio-ffmpeg==0.6.0"]
# ///
# How to run: install uv, then `uv run prepare_validation.py`.
# Recheck leakage, audio integrity and telephony behavior: append --check.
"""Prepare a small, provenance-preserving public audio validation bank."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import wave
import zipfile
from pathlib import Path
from typing import Final, TypedDict
from urllib.parse import quote

import imageio_ffmpeg
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

ROOT: Final = Path(__file__).resolve().parent
CACHE: Final = ROOT / "_workspace/validation_sources"
OUT: Final = ROOT / "validation"
REV: Final = "9143e5ea709575ebab6bec52840a1043aada7bb1"
BASE: Final = f"https://huggingface.co/datasets/mueller91/MLAAD-tiny/resolve/{REV}/"
LABELS: Final = ["FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT"]


class Row(TypedDict):
    ID: str
    path: str
    group_id: str
    split: str
    source: str
    FILE_FAKE: str
    VOICE_FAKE: str
    MUSIC_FAKE: str
    VOICE_PRESENT: str
    MUSIC_PRESENT: str


def fetch(url: str, path: Path) -> Path:
    """Use curl's native TLS, redirects and retries; retain immutable originals."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        subprocess.run(["curl", "-fLsS", "--retry", "3", "--max-time", "120", url, "-o", str(temporary)], check=True)
        temporary.replace(path)
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_for(group: str) -> str:
    return "holdout" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 3 == 0 else "dev"


def decode(path: Path) -> np.ndarray:
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"], check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not np.isfinite(audio).all():
        raise ArithmeticError(f"Non-finite decoded audio: {path}")
    start = max(0, (len(audio) - 160000) // 2)
    return audio[start:start + 160000]


def telephony(audio: np.ndarray) -> np.ndarray:
    """300–3400 Hz, 8 kHz resampling, 8-bit mu-law, return at 16 kHz."""
    band = sosfilt(butter(5, [300, 3400], fs=16000, btype="bandpass", output="sos"), audio)
    narrow = np.clip(resample_poly(band, 1, 2), -1, 1)
    encoded = np.sign(narrow) * np.log1p(255 * np.abs(narrow)) / np.log1p(255)
    encoded = np.round((encoded + 1) * 127.5) / 127.5 - 1
    restored = np.sign(encoded) * np.expm1(np.abs(encoded) * np.log1p(255)) / 255
    return resample_poly(restored, 2, 1)[:len(audio)]


def save(audio: np.ndarray, row: Row) -> None:
    path = OUT / row["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio)))
    pcm = np.round(audio / max(1.0, peak / 0.98) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        handle.writeframes(pcm.tobytes())


def speech() -> list[Row]:
    rows: list[Row] = []
    audit = []
    notices = CACHE / "licenses"
    for name in ["LICENSE", "original/LICENSE", "README.md"]:
        fetch(BASE + name, notices / name.replace("/", "_"))
    for generator in ["Chatterbox", "Edge-TTS", "FishTTS", "Kyutai-TTS"]:
        directory = f"fake/en/{generator}"
        listing = fetch(f"https://huggingface.co/api/datasets/mueller91/MLAAD-tiny/tree/{REV}/{directory}?limit=1000", CACHE / "metadata" / f"{generator}.json")
        entries = [item for item in json.loads(listing.read_text()) if item["type"] == "file" and 128000 <= item["size"] <= 768000]
        entries.sort(key=lambda item: hashlib.sha256(item["path"].encode()).hexdigest())
        accepted = 0
        for entry in entries:
            filename = Path(entry["path"]).name
            pair = [(f"original/en/{filename}", "0"), (entry["path"], "1")]
            loaded = [(fetch(BASE + quote(name), CACHE / "originals" / name), label) for name, label in pair]
            decoded = [(path, label, decode(path)) for path, label in loaded]
            if any(len(audio) < 64000 for _, _, audio in decoded):
                continue
            group = "mlaad:" + Path(filename).stem
            for path, label, audio in decoded:
                sample_id = hashlib.sha256(str(path.relative_to(CACHE)).encode()).hexdigest()[:16]
                row: Row = {"ID": sample_id, "path": f"audio/{sample_id}.wav", "group_id": group, "split": split_for(group), "source": f"MLAAD-tiny/{generator}" if label == "1" else "M-AILABS", "FILE_FAKE": label, "VOICE_FAKE": label, "MUSIC_FAKE": "", "VOICE_PRESENT": "1", "MUSIC_PRESENT": "0"}
                if any(existing["ID"] == sample_id for existing in rows):
                    continue
                save(audio, row)
                rows.append(row)
                phone: Row = {**row, "ID": sample_id + "_tel", "path": f"audio/{sample_id}_tel.wav"}
                save(telephony(audio), phone)
                rows.append(phone)
                audit.append({"ID": sample_id, "original": str(path.relative_to(ROOT)), "url": BASE + quote(str(path.relative_to(CACHE / "originals"))), "sha256": sha(path), "group_id": group, "license": "CC-BY-NC-4.0" if label == "1" else "M-AILABS permissive notice", "crop": "center up to 10 seconds; reject if under 4 seconds", "telephony_id": phone["ID"]})
            accepted += 1
            if accepted == 8:
                break
        print(f"speech: {generator}: {accepted} source pairs", flush=True)
    (CACHE / "speech_provenance.json").write_text(json.dumps(audit, indent=2))
    return rows


def music() -> list[Row]:
    """Import only explicit source labels; absent evidence remains blank."""
    candidates = ROOT / "_workspace/music_candidates.json"
    if not candidates.exists():
        print("music candidate manifest not ready; speech bank only", flush=True)
        return []
    rows: list[Row] = []
    audit = []
    real_count = 0
    (CACHE / "metadata/music_candidates.json").write_bytes(candidates.read_bytes())
    fetch("https://huggingface.co/datasets/disco-eth/AIME/resolve/b84d4be5eda830b6eb714998569dba73530f2601/README.md", CACHE / "licenses/AIME_README.md")
    for item in json.loads(candidates.read_text()):
        if int(item["label_real"]):
            real_count += 1
            if real_count > 8:
                continue
        if item.get("head_http_status", 200) != 200 and not item.get("local_path"):
            continue
        url = item["url"]
        sample_id = "music_" + hashlib.sha256((item["source"] + ":" + item["source_id"]).encode()).hexdigest()[:16]
        path = CACHE / "originals/music" / (sample_id + ".audio")
        if item.get("local_path") and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["local_path"], path)
        fetch(url, path)
        if item.get("sha256"):
            assert sha(path) == item["sha256"], f"Source checksum mismatch: {sample_id}"
        audio = decode(path)
        if len(audio) < 64000:
            continue
        group = item.get("group_id", "music_artist:" + item.get("author", item["source_id"]))
        license_url = item["license_url"].replace("http://", "https://").rstrip("/")
        fetch(license_url + "/legalcode", CACHE / "licenses" / (hashlib.sha256(license_url.encode()).hexdigest()[:12] + ".html"))
        fake = str(1 - int(item["label_real"]))
        row: Row = {"ID": sample_id, "path": f"audio/{sample_id}.wav", "group_id": group, "split": split_for(group), "source": item["source"], "FILE_FAKE": fake, "VOICE_FAKE": "", "MUSIC_FAKE": fake, "VOICE_PRESENT": "", "MUSIC_PRESENT": "1"}
        save(audio, row)
        rows.append(row)
        audit.append({**item, "ID": sample_id, "original": str(path.relative_to(ROOT)), "sha256": sha(path)})
    (CACHE / "music_provenance.json").write_text(json.dumps(audit, indent=2))
    return rows


def mixtures(rows: list[Row]) -> list[Row]:
    """Mix only sources already assigned to the same split; mask uncertain voice."""
    result: list[Row] = []
    groups = sorted({row["group_id"] for row in rows if row["group_id"].startswith("mlaad:")})[:8]
    for group in groups:
        voices = [row for row in rows if row["group_id"] == group and not row["ID"].endswith("_tel")]
        for fake in ["0", "1"]:
            options = [row for row in rows if row["MUSIC_FAKE"] == fake and row["split"] == voices[0]["split"]]
            if not options:
                continue
            music_row = options[int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % len(options)]
            background = decode(OUT / music_row["path"])
            for voice in voices[:2]:
                foreground = decode(OUT / voice["path"])
                length = min(len(foreground), len(background))
                foreground, back = foreground[:length], background[:length]
                for snr in [0, 10]:
                    gain = np.sqrt(np.mean(foreground ** 2) / max(float(np.mean(back ** 2)), 1e-12)) / 10 ** (snr / 20)
                    sample_id = voice["ID"] + "_" + music_row["ID"] + f"_snr{snr}"
                    voice_fake = voice["VOICE_FAKE"] if fake == "0" or voice["VOICE_FAKE"] == "1" else ""
                    row: Row = {"ID": sample_id, "path": f"audio/{sample_id}.wav", "group_id": group + "|" + music_row["group_id"], "split": voice["split"], "source": f"mixture:SNR={snr}dB", "FILE_FAKE": str(max(int(voice["FILE_FAKE"]), int(fake))), "VOICE_FAKE": voice_fake, "MUSIC_FAKE": fake, "VOICE_PRESENT": "1", "MUSIC_PRESENT": "1"}
                    save(foreground + gain * back, row)
                    result.append(row)
    return result


def check() -> None:
    """Protect source-disjoint splits, real/fake pairing, and actual band limiting."""
    with (OUT / "manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, set[str]] = {}
    for row in rows:
        for group in row["group_id"].split("|"):
            groups.setdefault(group, set()).add(row["split"])
        with wave.open(str(OUT / row["path"])) as handle:
            assert 4 <= handle.getnframes() / handle.getframerate() <= 12, row["ID"]
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
            assert np.isfinite(pcm).all() and np.std(pcm) > 1, row["ID"]
    assert all(len(splits) == 1 for splits in groups.values()), "source leakage"
    assert len({row["ID"] for row in rows}) == len(rows), "duplicate sample IDs"
    for group in [key for key in groups if key.startswith("mlaad:")]:
        assert {row["FILE_FAKE"] for row in rows if row["group_id"] == group} == {"0", "1"}, group
    time = np.arange(32000) / 16000
    low = np.sin(2 * np.pi * 1000 * time) * 0.2
    high = np.sin(2 * np.pi * 5000 * time) * 0.2
    assert np.std(telephony(high)[1600:]) < np.std(telephony(low)[1600:]) * 0.1, "telephony fails to reject out-of-band signal"
    assert {row["split"] for row in rows} == {"dev", "holdout"}
    assert len(rows) <= 300
    print(json.dumps({"clips": len(rows), "source_groups": len(groups), "splits": {split: sum(row["split"] == split for row in rows) for split in ["dev", "holdout"]}, "checks": ["source separation", "real/fake pairing", "decodable 4–12-second PCM", "measured telephony band rejection"]}), flush=True)


def main() -> None:
    if "--check" in sys.argv:
        check()
        return
    rows = speech() + music()
    rows += mixtures(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        writer.writerows(rows)
    check()
    for license_id in ["by-nc/4.0", "by-nc-sa/4.0"]:
        fetch(f"https://creativecommons.org/licenses/{license_id}/legalcode", CACHE / "licenses" / (license_id.replace("/", "_") + ".html"))
    with zipfile.ZipFile(ROOT / "_workspace/validation.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(OUT / "manifest.csv", "validation/manifest.csv")
        archive.write(ROOT / "DATA_SOURCES.md", "validation/DATA_SOURCES.md")
        for row in rows:
            archive.write(OUT / row["path"], "validation/" + row["path"])
        for path in CACHE.rglob("*"):
            if path.is_file() and "originals" not in path.parts:
                archive.write(path, "validation/provenance/" + str(path.relative_to(CACHE)))
    assert (ROOT / "_workspace/validation.zip").stat().st_size < 500_000_000
    print(f"Packaged {ROOT / '_workspace/validation.zip'}", flush=True)


if __name__ == "__main__":
    main()
