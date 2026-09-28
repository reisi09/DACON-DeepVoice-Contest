"""Deterministic mono-16k audio construction used by eight_data."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Final, Literal, assert_never

import imageio_ffmpeg
import numpy as np
import soundfile as sf

from eight_contract import ManifestRow, sha256_file

SR: Final = 16000
Codec = Literal["telephone", "mp3_32k", "opus_telephone"]


class AudioError(RuntimeError):
    """Audio decoding or construction failed a semantic check."""


def load(path: str | Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate == SR:
        mono = audio.mean(axis=1, dtype=np.float32)
    else:
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
                   "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
        result = subprocess.run(command, check=True, capture_output=True)
        mono = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(mono) == 0 or not np.isfinite(mono).all():
        raise AudioError(f"empty or non-finite audio: {path}")
    return mono


def write(path: Path, audio: np.ndarray) -> None:
    value = np.asarray(audio, dtype=np.float32)
    if len(value) == 0 or not np.isfinite(value).all() or float(np.max(np.abs(value))) >= 1:
        raise AudioError(f"invalid or clipping output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, value, SR, subtype="FLOAT")
    if not np.array_equal(load(path), value):
        raise AudioError(f"lossless write changed samples: {path}")


def core(path: Path, seconds: int = 8) -> np.ndarray:
    audio = load(path)
    size = min(len(audio), seconds * SR)
    start = max(0, (len(audio) - size) // 2)
    value = audio[start:start + size]
    peak = float(np.max(np.abs(value)))
    return value * min(1.0, 0.6 / max(peak, 1e-8))


def transform(audio: np.ndarray, codec: Codec) -> np.ndarray:
    match codec:
        case "telephone":
            args, suffix = ["-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-c:a", "pcm_mulaw"], ".wav"
        case "mp3_32k":
            args, suffix = ["-c:a", "libmp3lame", "-b:a", "32k"], ".mp3"
        case "opus_telephone":
            args, suffix = ["-af", "highpass=f=300,lowpass=f=3400", "-c:a", "libopus", "-b:a", "24k"], ".opus"
        case unreachable:
            assert_never(unreachable)
    with tempfile.TemporaryDirectory() as temporary:
        source, encoded = Path(temporary) / "source.wav", Path(temporary) / ("encoded" + suffix)
        write(source, audio)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(source), *args, str(encoded)],
                       check=True, capture_output=True)
        changed = load(encoded)
    if len(changed) < len(audio):
        changed = np.pad(changed, (0, len(audio) - len(changed)))
    changed = changed[:len(audio)]
    if float(np.max(np.abs(changed))) >= 1:
        raise AudioError(f"{codec} clipped after fixed-gain transform")
    return changed


def repeat_to(parts: list[np.ndarray], samples: int) -> np.ndarray:
    if not parts:
        raise AudioError("cannot concatenate an empty within-speaker cohort")
    return np.concatenate([parts[index % len(parts)] for index in range((samples // min(map(len, parts))) + len(parts) + 1)])[:samples]


def rms_match(segment: np.ndarray, reference: np.ndarray) -> np.ndarray:
    rms = lambda value: float(np.sqrt(np.mean(np.square(value), dtype=np.float64)))
    return segment * (rms(reference) / max(rms(segment), 1e-8))


def replace_event(bed: np.ndarray, insert: np.ndarray, start: int, fade_samples: int = 160) -> np.ndarray:
    length = len(insert)
    target = bed[start:start + length]
    changed = bed.copy()
    value = rms_match(insert, target)
    fade = min(fade_samples, length // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, endpoint=False, dtype=np.float32)
        value[:fade] = target[:fade] * (1 - ramp) + value[:fade] * ramp
        value[-fade:] = value[-fade:] * (1 - ramp) + target[-fade:] * ramp
    changed[start:start + length] = value
    peak = float(np.max(np.abs(changed)))
    return changed * min(1.0, 0.95 / max(peak, 1e-8))


def deterministic_noise(samples: int, key: str) -> np.ndarray:
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).standard_normal(samples).astype(np.float32)


def manifest_row(identifier: str, path: Path, parents: list[str], atoms: set[str], split: str, source: str,
                 generator: str, kind: str, variant: str, family: str, pair: str, labels: tuple[str, ...],
                 events: list[dict[str, str | float | int]], dataset: Path) -> ManifestRow:
    names = ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")
    return ManifestRow(**dict(zip(names, labels, strict=True)), ID=identifier, path=str(path.relative_to(dataset)),
                       parent_id=parents[0], parent_ids=json.dumps(parents, separators=(",", ":")),
                       group_id="|".join(sorted(atoms)), split=split, source=source, generator=generator,
                       kind=kind, variant=variant, probe_family=family, pair_id=pair,
                       audio_sha256=sha256_file(path), duration_seconds=f"{len(load(path)) / SR:.8f}",
                       event_intervals=json.dumps(events, separators=(",", ":")))


def partial_rows(rows: list[ManifestRow], dataset: Path) -> list[ManifestRow]:
    base = [row for row in rows if row["variant"] == "base" and row["kind"] == "speech"]
    by_speaker: dict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for row in base:
        speaker = next(atom for atom in row["group_id"].split("|") if atom.startswith("speaker:"))
        by_speaker[(row["split"], speaker)].append(row)
    chosen = [item for split, count in (("train", 2), ("calibration", 3), ("dev", 2), ("test", 2))
              for item in sorted((key for key in by_speaker if key[0] == split), key=lambda key: key[1])[:count]]
    additions: list[ManifestRow] = []
    for split, speaker in chosen:
        cohort = by_speaker[(split, speaker)]
        real = [row for row in cohort if row["VOICE_FAKE"] == "0"]
        fake = [row for row in cohort if row["VOICE_FAKE"] == "1"]
        if not real or not fake:
            continue
        parents = sorted({row["ID"] for row in real + fake})
        atoms = {atom for row in real + fake for atom in row["group_id"].split("|")}
        for seconds in (12, 30, 60):
            samples = seconds * SR
            bed = repeat_to([load(dataset / row["path"]) for row in real], samples)
            fake_stream = repeat_to([load(dataset / row["path"]) for row in fake], samples)
            control_stream = repeat_to([load(dataset / row["path"]) for row in real[1:] + real[:1]], samples)
            for event_seconds in (.25, .5, 1, 2):
                length = int(event_seconds * SR)
                for position, start in (("start", 0), ("mid", (samples - length) // 2), ("end", samples - length)):
                    pair = f"e5:{speaker}:{seconds}:{event_seconds}:{position}"
                    for label, stream in ((1, fake_stream), (0, control_stream)):
                        audio = replace_event(bed, stream[start:start + length], start)
                        variant = f"partial_{'fake' if label else 'real'}_{seconds}s_{event_seconds:g}s_{position}"
                        identifier = "e5_" + hashlib.sha256((pair + str(label)).encode()).hexdigest()[:24]
                        path = dataset / "audio" / (identifier + ".wav")
                        write(path, audio)
                        event = [{"component": "voice", "start": start / SR, "end": (start + length) / SR, "label": label}]
                        additions.append(manifest_row(identifier, path, parents, atoms, split, "constructed:E5", "", "speech",
                                                      variant, "partial", pair, (str(label), str(label), "", "1", "0"), event, dataset))
    return additions


def semantic_audio_check(dataset: Path, rows: list[ManifestRow]) -> dict[str, int | float]:
    by_id = {row["ID"]: row for row in rows}
    codecs: dict[str, list[ManifestRow]] = defaultdict(list)
    swaps: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    partials: dict[str, list[ManifestRow]] = defaultdict(list)
    peak = 0.0
    for row in rows:
        match row["probe_family"]:
            case "core":
                codecs[row["pair_id"]].append(row)
            case "swap":
                vocal = load(dataset / by_id[json.loads(row["parent_ids"])[0]]["path"])
                mixture = load(dataset / row["path"])
                background, level = row["variant"].rsplit("_", 1)
                swaps[(row["parent_id"], background)][level] = float(np.sqrt(np.mean((mixture - vocal) ** 2, dtype=np.float64)))
                peak = max(peak, float(np.max(np.abs(mixture))))
            case "partial":
                partials[row["pair_id"]].append(row)
                peak = max(peak, float(np.max(np.abs(load(dataset / row["path"])))))
            case unreachable:
                assert_never(unreachable)
    for group in codecs.values():
        variants = {row["variant"] for row in group}
        expected = {"base", "telephone", "mp3_32k"} | ({"opus_telephone"} if group[0]["split"] != "train" else set())
        if variants != expected or len({row["duration_seconds"] for row in group}) != 1:
            raise AudioError(f"codec exposure/timing mismatch: {group[0]['pair_id']}")
    if any(abs(levels["minus6dB"] / levels["0dB"] - .501187) >= 2e-5 for levels in swaps.values()):
        raise AudioError("E2 vocal-fixed gain ratio changed")
    for group in partials.values():
        events = [json.loads(row["event_intervals"])[0] for row in group]
        if len(group) != 2 or {row["VOICE_FAKE"] for row in group} != {"0", "1"} or len({(e["start"], e["end"]) for e in events}) != 1:
            raise AudioError(f"E5 matched boundary control changed: {group[0]['pair_id']}")
    if peak >= 1:
        raise AudioError("constructed probe clips")
    return {"codec_pairs_checked": len(codecs), "e2_background_pairs_checked": len(swaps),
            "e5_pairs_checked": len(partials), "constructed_max_abs_peak": peak}
