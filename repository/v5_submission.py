"""Faithful offline inference for the four completed V5 arms."""

# allow: SIZE_OK - one evaluator entrypoint keeps model loading and inference contract together.

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Literal

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.dont_write_bytecode = True

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import nn

from v5_experts import MusicExpert, PresenceExpert, VoiceExpert
from v5_fusion import TemporalFusion

Arm = Literal["B1", "B2", "B3", "B4"]
ExpertModels = Mapping[str, nn.Module]
SAMPLE_RATE: Final = 16_000
TIMELINE_SAMPLES: Final = 64_600
WINDOW_BATCH_SIZE: Final = 4
SUBMISSION_COLUMNS: Final = (
    "ID",
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
)
SUPPORTED_AUDIO_SUFFIXES: Final = frozenset(
    {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
)


class InferenceError(ValueError):
    """Evaluator input or packaged model assets violate the V5 contract."""


def window_starts(samples: int, width: int) -> tuple[int, ...]:
    """Return the non-overlapping grid plus its tail anchor."""
    if samples < 1 or width < 1:
        raise InferenceError("Audio and window lengths must be positive")
    if samples <= width:
        return (0,)
    final = samples - width
    starts = list(range(0, final + 1, width))
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def read_audio(path: Path) -> np.ndarray:
    """Decode one file through the same pinned ffmpeg path used by V5 training."""
    try:
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
             "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise InferenceError(f"Cannot decode audio: {path.name}") from error
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise InferenceError(f"Invalid audio: {path.name}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def read_sample_ids(path: Path) -> list[str]:
    """Parse the evaluator-owned order at the CSV boundary."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(SUBMISSION_COLUMNS):
                raise InferenceError("Sample submission must use the official six-column schema")
            identifiers = [str(row.get("ID", "")).strip() for row in reader]
    except OSError as error:
        raise InferenceError("Cannot read data/sample_submission.csv") from error
    if not identifiers or any(not value for value in identifiers):
        raise InferenceError("Sample submission IDs must be non-empty")
    if len(identifiers) != len(set(identifiers)):
        raise InferenceError("Sample submission IDs must be unique")
    return identifiers


def ordered_audio_files(test_dir: Path, identifiers: Sequence[str]) -> list[Path]:
    """Resolve every evaluator ID to exactly one supported file."""
    if not test_dir.is_dir():
        raise InferenceError("Evaluator data/test directory is missing")
    files = [
        path for path in test_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    ]
    by_id = {path.stem: path for path in files}
    if len(by_id) != len(files):
        raise InferenceError("Evaluator audio IDs must be unique")
    requested = set(identifiers)
    if set(by_id) != requested:
        missing = sorted(requested - set(by_id))[:3]
        extra = sorted(set(by_id) - requested)[:3]
        raise InferenceError(f"Evaluator audio ID mismatch: missing={missing}, extra={extra}")
    return [by_id[identifier] for identifier in identifiers]


def _windows(audio: np.ndarray, width: int, starts: Sequence[int]) -> torch.Tensor:
    result = np.zeros((len(starts), width), dtype=np.float32)
    for index, start in enumerate(starts):
        clip = audio[start : start + width]
        result[index, : clip.size] = clip
    return torch.from_numpy(result)


def _encode(
    model: nn.Module, audio: np.ndarray, starts: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    logits: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    model.eval()
    values = _windows(audio, int(model.window_samples), starts)
    with torch.inference_mode(), torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
    ):
        for start in range(0, len(values), WINDOW_BATCH_SIZE):
            output, embedding = model(values[start : start + WINDOW_BATCH_SIZE].to(device))
            logits.extend(output.float().cpu().numpy())
            embeddings.extend(embedding.float().cpu().numpy())
    return np.stack(logits), np.stack(embeddings)


def predict_raw(models: ExpertModels, audio: np.ndarray) -> np.ndarray:
    """Apply independent whole-file maxima and the V5 FILE marginal."""
    pooled: dict[str, np.ndarray] = {}
    for kind in ("voice", "music", "presence"):
        model = models[kind]
        logits, _ = _encode(model, audio, window_starts(len(audio), int(model.window_samples)))
        pooled[kind] = torch.sigmoid(torch.from_numpy(logits.max(axis=0))).numpy()
    voice = float(pooled["voice"][0])
    music = float(pooled["music"][0])
    voice_present, music_present = map(float, pooled["presence"])
    return np.asarray(
        [max(voice_present * voice, music_present * music), voice, music,
         voice_present, music_present],
        dtype=np.float32,
    )


def encode_temporal(
    models: ExpertModels,
    raw: np.ndarray,
    voice: np.ndarray,
    music: np.ndarray,
    *,
    timeline_samples: int = TIMELINE_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact B3/B4 ordered raw/stem feature matrix for one file."""
    starts = window_starts(len(raw), timeline_samples)
    blocks: list[np.ndarray] = []
    for kind, stem in (("voice", voice), ("music", music), ("presence", raw)):
        model = models[kind]
        raw_logits, raw_embeddings = _encode(model, raw, starts)
        if kind == "presence":
            stem_logits, stem_embeddings = raw_logits, raw_embeddings
        else:
            stem_logits, stem_embeddings = _encode(model, stem, starts)
        blocks.append(np.concatenate(
            (raw_embeddings, stem_embeddings, raw_logits, stem_logits,
             raw_logits - stem_logits), axis=1,
        ))
    return (
        np.concatenate(blocks, axis=1),
        np.asarray(starts, dtype=np.float32) / SAMPLE_RATE,
    )


def predict_temporal(
    models: ExpertModels,
    fusion: TemporalFusion,
    raw: np.ndarray,
    voice: np.ndarray,
    music: np.ndarray,
) -> np.ndarray:
    """Infer one file through its B3 or B4 temporal fusion checkpoint."""
    features, times = encode_temporal(models, raw, voice, music)
    device = next(fusion.parameters()).device
    values = torch.from_numpy(features).unsqueeze(0).to(device)
    positions = torch.from_numpy(times).unsqueeze(0).to(device)
    padding = torch.zeros(positions.shape, dtype=torch.bool, device=device)
    fusion.eval()
    with torch.inference_mode():
        result = fusion(values, padding, positions).probabilities
    return result.float().cpu().numpy()[0]


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _PerExampleAmplitudeToDB(nn.Module):
    def __init__(self, transform: nn.Module) -> None:
        super().__init__()
        self.transform = transform

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.transform(spectrogram.unsqueeze(1)).squeeze(1)


def _load_state(model: nn.Module, path: Path) -> None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as error:
        raise InferenceError(f"Cannot load packaged checkpoint: {path.name}") from error
    if not isinstance(state, dict) or not state:
        raise InferenceError(f"Packaged checkpoint is empty: {path.name}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise InferenceError(f"Checkpoint architecture mismatch: {path.name}") from error


def _load_voice(model_root: Path) -> VoiceExpert:
    df_root = model_root / "df_arena_1b"
    sys.path.insert(0, str(model_root))
    try:
        from df_arena_1b.configuration_antispoofing import DF_Arena_1B_Config
        from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

        config = DF_Arena_1B_Config.from_pretrained(str(df_root), local_files_only=True)
        with _working_directory(df_root):
            wrapper = DF_Arena_1B_Antispoofing(config)
    finally:
        sys.path.remove(str(model_root))
    return VoiceExpert(wrapper)


def _load_music(model_root: Path) -> MusicExpert:
    vendor = model_root / "sonics_vendor"
    sys.path.insert(0, str(vendor))
    try:
        from sonics.models.hf_model import HFAudioClassifier

        config = json.loads((model_root / "sonics" / "config.json").read_text(encoding="utf-8"))
        model = HFAudioClassifier(config)
    finally:
        sys.path.remove(str(vendor))
    model.ft_extractor.amplitude_to_db = _PerExampleAmplitudeToDB(
        model.ft_extractor.amplitude_to_db,
    )
    return MusicExpert(model)


def _load_presence(model_root: Path) -> PresenceExpert:
    labels = model_root / "panns" / "class_labels_indices.csv"
    installed_labels = Path.home() / "panns_data" / labels.name
    installed_labels.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(labels, installed_labels)
    from panns_inference.models import Cnn14

    with labels.open(encoding="utf-8", newline="") as stream:
        classes = sum(1 for _ in csv.DictReader(stream))
    cnn = Cnn14(
        sample_rate=32_000, window_size=1024, hop_size=320, mel_bins=64,
        fmin=50, fmax=14_000, classes_num=classes,
    )
    return PresenceExpert(cnn, nn.Linear(2048, 2))


def load_models(root: Path, device: torch.device) -> dict[str, nn.Module]:
    """Instantiate local architectures and strictly restore only final refit states."""
    model_root = root / "model"
    constructors = {"voice": _load_voice, "music": _load_music, "presence": _load_presence}
    result: dict[str, nn.Module] = {}
    for kind in ("voice", "music", "presence"):
        model = constructors[kind](model_root)
        _load_state(model, model_root / "weights" / f"{kind}.pt")
        result[kind] = model.to(device).eval()
    return result


def load_fusion(root: Path, device: torch.device, joint: bool) -> TemporalFusion:
    """Infer the trained feature width from a strict B3/B4 state dictionary."""
    path = root / "model" / "weights" / "fusion.pt"
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        projection = state["input_projection.weight"]
    except (OSError, RuntimeError, EOFError, KeyError, TypeError) as error:
        raise InferenceError("Cannot determine temporal fusion architecture") from error
    if not isinstance(projection, torch.Tensor) or projection.ndim != 2:
        raise InferenceError("Temporal fusion projection has an invalid shape")
    model = TemporalFusion(int(projection.shape[1]), joint)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise InferenceError("Temporal fusion checkpoint architecture mismatch") from error
    return model.to(device).eval()


def load_separator(root: Path, device: torch.device) -> nn.Module:
    """Load the packaged HTDemucs state without any model-repository access."""
    from demucs.pretrained import get_model

    original = torch.load

    def trusted(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = trusted
    try:
        separator = get_model("htdemucs", repo=root / "model" / "htdemucs")
    finally:
        torch.load = original
    return separator.to(device).eval()


def separate_views(
    raw: np.ndarray, separator: nn.Module, device: torch.device, temporary_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Match V5 preparation: write mono 16 kHz FLOAT, then run the baseline separator."""
    from demucs.apply import apply_model
    from demucs.separate import load_track

    with tempfile.NamedTemporaryFile(suffix=".wav", dir=temporary_root, delete=False) as stream:
        raw_path = Path(stream.name)
    try:
        sf.write(raw_path, raw, SAMPLE_RATE, subtype="FLOAT")
        waveform = load_track(raw_path, separator.audio_channels, separator.samplerate).float()
        mono = waveform.mean(0)
        mean, std = mono.mean(), mono.std()
        if float(std) < 1e-8:
            length = max(1, round(waveform.shape[-1] * SAMPLE_RATE / separator.samplerate))
            silence = np.zeros(length, dtype=np.float32)
            return silence, silence.copy()
        with torch.inference_mode():
            sources = apply_model(
                separator, ((waveform - mean) / std)[None], device=device, shifts=0,
                split=True, overlap=0.25, progress=False,
            )[0]
        sources = sources * std + mean
        vocals = sources[separator.sources.index("vocals")].mean(0, keepdim=True)
        music = torch.stack([
            source for index, source in enumerate(sources)
            if separator.sources[index] != "vocals"
        ]).sum(0).mean(0, keepdim=True)
        return tuple(
            torchaudio.functional.resample(value, separator.samplerate, SAMPLE_RATE)[0]
            .cpu().numpy().astype(np.float32)
            for value in (vocals, music)
        )
    finally:
        raw_path.unlink(missing_ok=True)


def _arm(root: Path) -> Arm:
    try:
        value = json.loads((root / "model" / "runtime" / "arm.json").read_text())["arm"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise InferenceError("Cannot read packaged V5 arm") from error
    match value:
        case "B1" | "B2" | "B3" | "B4":
            return value
        case unsupported:
            raise InferenceError(f"Unsupported V5 arm: {unsupported}")


def generate_submission(root: Path) -> Path:
    """Run file-local inference and atomically write output/submission.csv."""
    arm = _arm(root)
    identifiers = read_sample_ids(root / "data" / "sample_submission.csv")
    paths = ordered_audio_files(root / "data" / "test", identifiers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(root, device)
    fusion = load_fusion(root, device, joint=arm == "B4") if arm in {"B3", "B4"} else None
    separator = load_separator(root, device) if fusion is not None else None
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    for index, (identifier, path) in enumerate(zip(identifiers, paths, strict=True), 1):
        raw = read_audio(path)
        if fusion is None:
            probabilities = predict_raw(models, raw)
        else:
            if separator is None:
                raise InferenceError("Temporal arm separator is unavailable")
            voice, music = separate_views(raw, separator, device, output)
            probabilities = predict_temporal(models, fusion, raw, voice, music)
        if probabilities.shape != (5,) or not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise InferenceError(f"Invalid probabilities for evaluator ID: {identifier}")
        rows.append(dict(zip(SUBMISSION_COLUMNS, [identifier, *map(float, probabilities)], strict=True)))
        print(f"PREDICT {index}/{len(paths)} {identifier}", flush=True)
    destination = output / "submission.csv"
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination


def main(root: Path | None = None) -> None:
    generate_submission(root or Path.cwd())


if __name__ == "__main__":
    main()
