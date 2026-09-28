"""File-local, ordered raw/stem evidence for the V5 temporal comparison."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from v5_dataset import Example, compose_mixture, read_audio, training_crop


@dataclass(frozen=True, slots=True)
class AudioViews:
    raw: Path
    voice: Path
    music: Path
    offset_seconds: float


@dataclass(frozen=True, slots=True)
class EncodedFile:
    identifier: str
    features: np.ndarray
    times: np.ndarray
    labels: tuple[float, ...]


def load_baseline(path: Path):
    """Import the archived official module without executing its CLI."""
    spec = importlib.util.spec_from_file_location("v5_official_baseline", path / "script.py")
    if spec is None or spec.loader is None:
        message = f"Cannot import the preserved baseline: {path}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def window_starts(length: int, width: int) -> tuple[int, ...]:
    """Cover the complete file, including a tail that does not fit the stride."""
    if length <= 0 or width <= 0:
        message = "Waveform and window lengths must be positive"
        raise ValueError(message)
    return tuple(sorted({*range(0, max(1, length - width + 1), width), max(0, length - width)}))


def prepare_views(
    examples: list[Example], baseline: Path, cache: Path, deadline: float, *, training: bool,
) -> dict[str, AudioViews]:
    """Separate one deterministic <=20s training context or every full dev file."""
    cache.mkdir(parents=True, exist_ok=True)
    base = load_baseline(baseline)
    device = torch.device("cuda")
    separator = base.load_htdemucs_model().to(device)
    views = {}
    try:
        for index, example in enumerate(examples):
            if time.time() >= deadline:
                raise TimeoutError("Full raw/stem preparation exceeded its stage budget")
            waveform = read_audio(example.path)
            offset = 0
            if training and len(waveform) > 320_000:
                offset = int(hashlib.sha256(example.id.encode()).hexdigest()[:16], 16) % (len(waveform) - 320_000 + 1)
                waveform = waveform[offset:offset + 320_000]
            raw = cache / f"{example.id}_raw.wav"
            voice = cache / f"{example.id}_voice.wav"
            music = cache / f"{example.id}_music.wav"
            sf.write(raw, waveform, 16_000, subtype="FLOAT")
            vocal_audio, music_audio = base.separate_voice_and_music(raw, separator, device)
            sf.write(voice, vocal_audio, 16_000, subtype="FLOAT")
            sf.write(music, music_audio, 16_000, subtype="FLOAT")
            views[example.id] = AudioViews(raw, voice, music, offset / 16_000)
            if index % 50 == 0:
                print(f"SEPARATE {'train' if training else 'dev'} {index + 1}/{len(examples)}", flush=True)
    finally:
        del separator
        gc.collect()
        torch.cuda.empty_cache()
    return views


def encode_view(model: torch.nn.Module, path: Path, starts: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Encode each ordered start with a fixed-weight model, never across files."""
    waveform = read_audio(path)
    width = model.window_samples
    logits, embeddings = [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in starts:
            clip = waveform[start:start + width]
            clip = np.pad(clip, (0, max(0, width - len(clip))))
            logit, embedding = model(torch.from_numpy(clip).unsqueeze(0).to(device))
            logits.append(logit.float().cpu().numpy()[0])
            embeddings.append(embedding.float().cpu().numpy()[0])
    return np.stack(logits), np.stack(embeddings)


def encode_expert_views(
    model: torch.nn.Module, examples: list[Example], views: dict[str, AudioViews],
    kind: str, output: Path, deadline: float,
) -> None:
    """Save raw/stem embeddings on the same 4.0375s timeline; model version owns output."""
    output.mkdir(parents=True, exist_ok=False)
    for index, example in enumerate(examples):
        if time.time() >= deadline:
            raise TimeoutError(f"{kind} feature extraction exceeded its stage budget")
        view = views[example.id]
        length = sf.info(view.raw).frames
        starts = window_starts(length, 64_600)
        raw_logits, raw_embedding = encode_view(model, view.raw, starts)
        if kind == "presence":
            stem_logits, stem_embedding = raw_logits, raw_embedding
        else:
            path = view.voice if kind == "voice" else view.music
            stem_logits, stem_embedding = encode_view(model, path, starts)
        # Raw/stem disagreement remains evidence for the learned fusion module.
        features = np.concatenate((raw_embedding, stem_embedding, raw_logits, stem_logits, raw_logits - stem_logits), axis=1)
        np.savez_compressed(output / f"{example.id}.npz", features=features,
                            times=np.asarray(starts, dtype=np.float32) / 16_000 + view.offset_seconds)
        if index % 100 == 0:
            print(f"ENCODE {kind} {index + 1}/{len(examples)}", flush=True)


def assemble_features(examples: list[Example], roots: list[Path]) -> list[EncodedFile]:
    """Reject misaligned expert timelines before combining evidence."""
    result = []
    for example in examples:
        blocks, times = [], None
        for root in roots:
            with np.load(root / f"{example.id}.npz", allow_pickle=False) as data:
                current = data["times"]
                if times is not None and not np.array_equal(times, current):
                    message = f"Expert window alignment differs for {example.id}"
                    raise ValueError(message)
                times = current.copy()
                blocks.append(data["features"].copy())
        result.append(EncodedFile(example.id, np.concatenate(blocks, axis=1), times, example.labels))
    return result


def add_compositions(examples: list[Example], output: Path) -> tuple[list[Example], list[dict]]:
    """Add same-split mixtures plus matched partial-fake/real-real speech edits."""
    output.mkdir(parents=True, exist_ok=True)
    speech = [row for row in examples if row.panel == "speech"]
    music = [row for row in examples if row.labels[4] == 1 and row.panel in {"music_real", "music_seen"}]
    real = [row for row in speech if row.labels[1] == 0]
    fake = [row for row in speech if row.labels[1] == 1]
    derived, ledger = [], []
    for index in range(min(160, len(speech))):
        voice, background = speech[index], music[(index * 17) % len(music)]
        composition = compose_mixture(voice, background, 0, 160_000, 20260913)
        identifier = f"v5_mix_{index:04d}"
        path = output / f"{identifier}.wav"
        sf.write(path, composition.waveform, 16_000, subtype="FLOAT")
        derived.append(Example(identifier, path, composition.metadata["parent_groups"], composition.labels, "mixture"))
        ledger.append({"ID": identifier, **composition.metadata})
    for index in range(min(40, len(real), len(fake))):
        base = real[index]
        for tag, donor, target in (("rf", fake[index], 1.), ("rr", real[(index + 1) % len(real)], 0.)):
            first, _ = training_crop(base, 0, 64_600, False, 20260913)
            second, _ = training_crop(donor, 0, 64_600, False, 20260913)
            waveform = np.concatenate((first, second))
            identifier = f"v5_splice_{tag}_{index:04d}"
            path = output / f"{identifier}.wav"
            sf.write(path, waveform, 16_000, subtype="FLOAT")
            groups = tuple(sorted({*base.groups, *donor.groups}))
            labels = (target, target, float("nan"), 1., 0.)
            derived.append(Example(identifier, path, groups, labels, "splice"))
            ledger.append({"ID": identifier, "kind": f"speech_splice_{tag}", "parents": [base.id, donor.id],
                           "parent_groups": groups, "boundary_seconds": 64_600 / 16_000,
                           "segment_voice_fake": [0, int(target)]})
    return [*examples, *derived], ledger
