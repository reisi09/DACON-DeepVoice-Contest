"""Run bounded format, channel, duration, and memory QA on an unpacked submission."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf

PREDICTION_COLUMNS: Final = (
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)


class QaError(RuntimeError):
    """Supplemental runtime QA cannot produce trustworthy evidence."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, payload: dict) -> None:  # noqa: DICT_OK
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_sample(path: Path, identifiers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ID", *PREDICTION_COLUMNS])
        writer.writeheader()
        for identifier in identifiers:
            writer.writerow({"ID": identifier, **{name: "" for name in PREDICTION_COLUMNS}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    if args.output.exists() or sha256(args.source_audio) != args.source_sha256:
        raise QaError("Fresh output and the exact pinned real source are required")
    if not (args.package_root / "script.py").is_file():
        raise QaError("Unpacked submission entrypoint is missing")
    remaining = args.deadline_epoch - time.time() - 120
    if remaining <= 60:
        raise QaError("Original run deadline leaves insufficient bounded QA time")

    test_dir = args.output / "test"
    test_dir.mkdir(parents=True)
    audio, rate = sf.read(args.source_audio, dtype="float32", always_2d=True)
    if rate != 16_000 or not len(audio) or not np.isfinite(audio).all():
        raise QaError("Pinned source must contain finite audio at 16 kHz")
    source_seconds = len(audio) / rate
    if len(audio) >= 60 * rate:
        long_audio = audio[: 60 * rate]
        long_case = "first_60s_real_crop"
    else:
        long_audio = np.tile(audio, (math.ceil(60 * rate / len(audio)), 1))[: 60 * rate]
        long_case = "repeated_canonical_real_synthetic_stress"
    if long_audio.shape[1] == 1:
        stereo = np.repeat(long_audio, 2, axis=1)
    else:
        stereo = long_audio[:, :2]
    silence = test_dir / "qa_silence_4s.wav"
    real_flac = test_dir / "qa_real_stereo_60s.flac"
    sf.write(silence, np.zeros(4 * rate, dtype=np.float32), rate, subtype="PCM_16")
    sf.write(real_flac, stereo, rate, subtype="PCM_16")
    identifiers = [silence.stem, real_flac.stem]
    ffmpeg = shutil.which("ffmpeg")
    mp3_status = "skipped_ffmpeg_unavailable"
    if ffmpeg is not None:
        mp3 = test_dir / "qa_codec_mp3_6s.mp3"
        encoded = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(real_flac),
             "-t", "6", "-ac", "2", "-ar", "16000", "-b:a", "128k", str(mp3)],
            capture_output=True, text=True, check=False,
        )
        if encoded.returncode != 0 or not mp3.is_file():
            raise QaError(f"Existing ffmpeg failed MP3 generation: {encoded.stderr[-500:]}")
        identifiers.append(mp3.stem)
        mp3_status = "tested"

    sample = args.output / "sample_submission.csv"
    write_sample(sample, identifiers)
    generated = {path.name: sha256(path) for path in sorted(test_dir.iterdir())}
    atomic_json(args.output / "input_receipt.json", {
        "source_audio": str(args.source_audio), "source_sha256": args.source_sha256,
        "cases": {"silence_wav": "4s mono 16k", "real_flac": f"60s stereo 16k {long_case}",
                  "mp3": mp3_status}, "generated_sha256": generated,
        "source_seconds": source_seconds, "long_case": long_case,
        "quality_claim": None, "purpose": "finite probabilities and channel/codec/runtime/memory behavior",
    })

    submission = args.output / "submission.csv"
    command = [str(args.runtime_python), str(args.package_root / "script.py"),
               "--test-dir", str(test_dir), "--sample-submission", str(sample),
               "--output", str(submission), "--device", "cuda"]
    began = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=remaining)
    elapsed = time.monotonic() - began
    (args.output / "runtime.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise QaError(f"Packaged supplemental inference failed with exit code {completed.returncode}")

    with submission.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row.get("ID") for row in rows] != identifiers:
        raise QaError("Supplemental prediction IDs or order differ from the requested cases")
    probabilities = [float(row[name]) for row in rows for name in PREDICTION_COLUMNS]
    if not probabilities or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise QaError("Supplemental runtime emitted invalid probabilities")
    runtime_path = args.output / "runtime_receipt.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if (runtime.get("files") != len(identifiers) or runtime.get("device") != "cuda"
            or runtime.get("submission_sha256") != sha256(submission)
            or runtime.get("peak_gpu_allocated_bytes", 0) <= 0
            or runtime.get("peak_gpu_reserved_bytes", 0) <= 0
            or runtime.get("peak_rss_bytes", 0) <= 0):
        raise QaError("Runtime timing, GPU/CPU memory, or submission binding is incomplete")
    atomic_json(args.output / "supplemental_qa.json", {
        "status": "passed", "files": len(identifiers), "ids": identifiers,
        "mp3_status": mp3_status, "long_case": long_case, "finite_probabilities": True,
        "probability_range": [min(probabilities), max(probabilities)],
        "wall_seconds": elapsed, "command": command,
        "submission_sha256": sha256(submission), "runtime_receipt_sha256": sha256(runtime_path),
        "peak_gpu_allocated_bytes": runtime["peak_gpu_allocated_bytes"],
        "peak_gpu_reserved_bytes": runtime["peak_gpu_reserved_bytes"],
        "peak_rss_bytes": runtime["peak_rss_bytes"], "quality_claim": None,
    })
    print(json.dumps({"status": "passed", "files": len(identifiers), "seconds": elapsed,
                      "mp3_status": mp3_status}))


if __name__ == "__main__":
    main()
