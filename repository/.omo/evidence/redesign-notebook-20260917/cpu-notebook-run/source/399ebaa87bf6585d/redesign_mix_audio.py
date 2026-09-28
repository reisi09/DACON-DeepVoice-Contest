# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile"]
# ///
# How to run: imported by `redesign_data.py`; it has no standalone command.
"""Deterministic paired pure/mixed audio construction for redesign experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf

LABELS: Final = ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _order(rows: list[dict[str, str]], seed: int, tag: str) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}|{tag}|{row['ID']}".encode()).hexdigest())


def _eligible(rows: list[dict[str, str]], component: str, fake: str) -> list[dict[str, str]]:
    presence = "VOICE_PRESENT" if component == "voice" else "MUSIC_PRESENT"
    other = "MUSIC_PRESENT" if component == "voice" else "VOICE_PRESENT"
    label = "VOICE_FAKE" if component == "voice" else "MUSIC_FAKE"
    return [row for row in rows if row[presence] == "1" and row[other] == "0" and row[label] == fake]


def _read_mono(row: dict[str, str]) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(row["path"], dtype="float32", always_2d=True)
    return audio.mean(axis=1, dtype=np.float32), int(sample_rate)


def _generated_row(
    identifier: str,
    path: Path,
    *,
    pair_id: str,
    group_id: str,
    split: str,
    view: str,
    parents: list[dict[str, str]],
    labels: tuple[str, str, str, str, str],
    duration: float,
    events: list[dict[str, str | float]],
    mode: str,
) -> dict[str, str]:
    parent_ids = [row["ID"] for row in parents]
    return {
        "ID": identifier,
        "path": str(path.resolve()),
        "audio_sha256": _digest(path),
        "source": "generated/paired-redesign",
        "license": " + ".join(dict.fromkeys(row["license"] for row in parents)),
        "license_url": json.dumps(list(dict.fromkeys(row["license_url"] for row in parents))),
        "source_url": json.dumps(list(dict.fromkeys(row["source_url"] for row in parents))),
        "parent_ids": json.dumps(parent_ids),
        "artist_id": "",
        "speaker_id": "",
        "generator": "",
        "kind": "instrumental" if view == "pure" else "mixed",
        "domain": f"{parents[-1]['domain']}/{mode}",
        **dict(zip(LABELS, labels, strict=True)),
        "group_id": group_id,
        "split": split,
        "pair_id": pair_id,
        "view": view,
        "event_intervals": json.dumps(events, separators=(",", ":")),
        "duration_seconds": f"{duration:.9f}",
    }


def make_paired_views(
    rows: list[dict[str, str]], output: Path, seed: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Build equal-exposure pure/mixed alternatives for four component states."""
    output.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, str]] = []
    state_counts: dict[str, int] = {"skipped_sample_rate_mismatch": 0}
    for split in ("train", "validation", "sealed"):
        split_rows = [row for row in rows if row["split"] == split]
        buckets = {
            (component, fake): _order(_eligible(split_rows, component, fake), seed, f"{split}:{component}:{fake}")
            for component in ("voice", "music") for fake in ("0", "1")
        }
        modes = ("simultaneous", "sequential", "sparse")
        slots = min((len(bucket) // (2 * len(modes)) for bucket in buckets.values()), default=0)
        state_counts[split] = 0
        for slot in range(slots):
            for mode_index, mode in enumerate(modes):
                for voice_fake, music_fake in (("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")):
                    state = f"v{voice_fake}m{music_fake}"
                    voice_index = 6 * slot + 2 * mode_index + int(music_fake)
                    music_index = 6 * slot + 2 * mode_index + int(voice_fake)
                    voice = buckets[("voice", voice_fake)][voice_index]
                    music = buckets[("music", music_fake)][music_index]
                    voice_audio, voice_rate = _read_mono(voice)
                    music_audio, music_rate = _read_mono(music)
                    if voice_rate != music_rate:
                        state_counts["skipped_sample_rate_mismatch"] += 1
                        continue
                    samples = min(len(voice_audio), len(music_audio))
                    if samples == 0:
                        continue
                    voice_audio, music_audio = voice_audio[:samples], music_audio[:samples]
                    voice_start, voice_end, music_start, music_end = 0, samples, 0, samples
                    if mode == "sequential":
                        midpoint = samples // 2
                        voice_start, music_end = midpoint, midpoint
                    elif mode == "sparse":
                        sparse = min(samples, round(voice_rate * 0.5))
                        if voice_fake == "1":
                            voice_start = samples - sparse
                        if music_fake == "1":
                            music_end = sparse
                    voice_view = np.zeros(samples, dtype=np.float32)
                    music_view = np.zeros(samples, dtype=np.float32)
                    voice_view[voice_start:voice_end] = voice_audio[voice_start:voice_end] * 0.5
                    music_view[music_start:music_end] = music_audio[music_start:music_end] * 0.5
                    mixed = (voice_view + music_view).astype(np.float32)
                    duration = samples / voice_rate
                    token = hashlib.sha256(f"{seed}|{split}|{slot}|{mode}|{state}|{voice['ID']}|{music['ID']}".encode()).hexdigest()[:16]
                    pair_id = f"pair-{token}"
                    group_id = "|".join(sorted({*voice["group_id"].split("|"), *music["group_id"].split("|")}))
                    pure_path, mixed_path = output / f"{pair_id}-pure.wav", output / f"{pair_id}-mixed.wav"
                    sf.write(pure_path, music_view, voice_rate, subtype="PCM_16")
                    sf.write(mixed_path, mixed, voice_rate, subtype="PCM_16")
                    music_event = {"component": "music", "start": music_start / voice_rate, "end": music_end / voice_rate, "label": int(music_fake)}
                    voice_event = {"component": "voice", "start": voice_start / voice_rate, "end": voice_end / voice_rate, "label": int(voice_fake)}
                    pure_labels = (music_fake, "", music_fake, "0", "1")
                    mixed_labels = (str(max(int(voice_fake), int(music_fake))), voice_fake, music_fake, "1", "1")
                    generated.append(_generated_row(
                        f"{pair_id}-pure", pure_path, pair_id=pair_id, group_id=group_id, split=split,
                        view="pure", parents=[music], labels=pure_labels, duration=duration, events=[music_event], mode=mode,
                    ))
                    generated.append(_generated_row(
                        f"{pair_id}-mixed", mixed_path, pair_id=pair_id, group_id=group_id, split=split,
                        view="mixed", parents=[voice, music], labels=mixed_labels, duration=duration,
                        events=[voice_event, music_event], mode=mode,
                    ))
                    state_counts[split] += 1
    return generated, state_counts
