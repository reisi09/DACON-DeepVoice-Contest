#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
# How to run: uv run prepare_fresh_music_crops.py
"""Freeze nested duration probes before model inference; retain source-level labels."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from v2_data import decode_audio, pcm_sha256, sha256_file

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results/leaderboard_audit_20260917"
OUTPUT = ROOT / "_workspace/fresh_music_crops_20260917"


def main() -> None:
    """Use one fixed central-60s start and retain every available duration probe."""
    inputs = [EVIDENCE / f"fresh_{name}_manifest.csv" for name in ("fma", "sonics")]
    rows = [row for path in inputs for row in csv.DictReader(path.open())]
    assert len({row["ID"] for row in rows}) == len(rows)
    clips = []
    skipped = []
    for row in rows:
        source = ROOT / row["path"]
        assert sha256_file(source) == row["source_sha256"], source
        audio = decode_audio(source, limit_seconds=None)
        if len(audio) < 15 * 16_000:
            raise ValueError(f"Minimum fixed 15-second probe unavailable: {source}")
        start = max(0, (len(audio) - 60 * 16_000) // 2)
        for seconds in (4, 15, 60):
            if len(audio) < seconds * 16_000:
                skipped.append({"parent_id": row["ID"], "duration_seconds": seconds,
                                "reason": "source shorter than requested probe; no padding or replacement"})
                continue
            identifier = f"{row['ID']}_{seconds}s"
            path = OUTPUT / f"{identifier}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            clip = audio[start:start + seconds * 16_000]
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
                "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
                "-c:a", "pcm_f32le", str(path),
            ], input=clip.astype("<f4").tobytes(), check=True, capture_output=True)
            # Validate actual saved samples against the selected source, with no gain change.
            decoded = decode_audio(path, limit_seconds=None)
            assert np.array_equal(decoded, clip), path
            clips.append({
                "ID": identifier, "path": str(path.relative_to(ROOT)),
                "parent_id": row["ID"], "group_id": row["group_id"],
                "source_sha256": row["source_sha256"], "audio_sha256": sha256_file(path),
                "pcm_sha256": pcm_sha256(decoded), "start_sample_16k": start,
                "duration_seconds": seconds,
                **{name: row[name] for name in
                   ("FILE_FAKE", "MUSIC_FAKE", "MUSIC_PRESENT", "VOICE_FAKE", "VOICE_PRESENT")},
                "label_status": "inherited_source_metadata; clip_audibility_and_presence_not_verified",
            })
    with (EVIDENCE / "fresh_music_crops.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(clips[0]))
        writer.writeheader()
        writer.writerows(clips)
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "input_sha256": {path.name: sha256_file(path) for path in inputs},
        "source_files": len(rows), "source_groups": len({row["group_id"] for row in rows}),
        "clips": len(clips), "durations": [4, 15, 60],
        "crop_rule": "central 60-second start (zero for shorter files), then available same-start 4/15/60-second prefixes",
        "unavailable_views": skipped,
        "saved_samples_exact": True, "sample_rate": 16_000, "channels": 1,
        "encoding": "float32 WAV without amplitude normalization",
        "evaluated": False, "trained": False, "official_score": None,
        "limitations": [
            "These are grouped duration diagnostics; three crops are not independent samples.",
            "Duration changes content and end-window alignment, not just pooling count.",
            "Compare duration effects only on the same parent subset available at all requested lengths.",
            "Source-level labels need clip audibility/presence review before scored evaluation.",
            "Mono WAV probes do not validate stereo/codec/telephone robustness.",
        ],
    }
    (EVIDENCE / "fresh_music_crops_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("source_files", "source_groups", "clips", "saved_samples_exact")}))


if __name__ == "__main__":
    main()
