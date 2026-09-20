"""Offline S2 EAT inference with the original voice/presence models."""
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchaudio", "transformers", "timm", "demucs"]
# ///
# Run the packaged script.py, which sets the bundled runtime import path.

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model

from fullft_model import MusicClassifier
from graph_scoring import SUBMISSION_COLUMNS, validate_predictions


def evaluation_bag(music: np.ndarray) -> torch.Tensor:
    """Reproduce the training worker's eight fixed five-second evaluation crops."""
    if not music.size or not np.isfinite(music).all():
        message = "Empty or nonfinite separated music"
        raise ValueError(message)
    width = 80_000
    starts = sorted(set(list(range(0, max(1, len(music) - width + 1), width))
                        + [max(0, len(music) - width)]))
    selected = np.linspace(0, len(starts) - 1, 8).round().astype(int)
    clips = [music[starts[i]:starts[i] + width] for i in selected]
    return torch.from_numpy(np.stack([
        np.tile(clip, math.ceil(width / len(clip)))[:width] for clip in clips
    ]).astype("float32"))[None]


def separate(audio: np.ndarray, model: torch.nn.Module, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Match the experiment's 16 kHz -> native HTDemucs -> 16 kHz audio path."""
    wave = torch.from_numpy(audio).to(device)[None].repeat(model.audio_channels, 1)
    wave = torchaudio.functional.resample(wave, 16000, model.samplerate)
    mean, std = wave.mean(0).mean(), wave.mean(0).std()
    if std.item() < 1e-8:
        silence = np.zeros(len(audio), dtype=np.float32)
        return silence, silence.copy()
    with torch.inference_mode():
        sources = apply_model(model, ((wave - mean) / std)[None], device=device,
                              shifts=0, split=True, overlap=.25, progress=False)[0] * std + mean
        voice = sources[model.sources.index("vocals")].mean(0)
        music = sources[[i for i, name in enumerate(model.sources) if name != "vocals"]].sum(0).mean(0)
        streams = torchaudio.functional.resample(torch.stack([voice, music]), model.samplerate, 16000)
    return tuple(streams.cpu().numpy().astype("float32"))


def main(root: Path) -> None:
    """Write all five probabilities for new evaluator audio, without historical predictions."""
    assets = root / "model/assets"
    spec = importlib.util.spec_from_file_location("s2_baseline", assets / "baseline/script.py")
    if spec is None or spec.loader is None:
        message = "Cannot import packaged baseline"
        raise ImportError(message)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    args = base.parse_arguments()
    device = base.select_device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((root / "model/manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            message = "Unsafe manifest path"
            raise ValueError(message)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                message = f"Packaged file changed: {name}"
                raise ValueError(message)
    columns, rows = base.read_sample_submission(args.sample_submission)
    if columns != list(SUBMISSION_COLUMNS):
        message = "Unexpected evaluator CSV columns"
        raise ValueError(message)
    paths = base.order_audio_files(base.find_audio_files(args.test_dir), rows)
    presence = base.predict_presence_for_all_files(paths, device)
    voice_model, fake_index = base.load_df_arena_model(device)
    separator = base.load_htdemucs_model().to(device)
    music_model = MusicClassifier("eat", False, assets).to(device).eval().requires_grad_(False)
    music_model.head.load_state_dict(torch.load(root / "model/trained/head.pt", map_location=device,
                                              weights_only=True), strict=True)
    for index, (row, path) in enumerate(zip(rows, paths, strict=True), 1):
        voice_audio, music_audio = separate(base.load_audio(path), separator, device)
        voice = base.predict_fake(voice_model, fake_index, voice_audio, device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                    enabled=device.type == "cuda"):
            music = float(music_model(evaluation_bag(music_audio).to(device)).float().softmax(-1)[0, 1])
        vp, mp = presence[path.stem]
        values = [max(vp * voice, mp * music), voice, music, vp, mp]
        row.update({key: round(value, 10) for key, value in zip(base.PREDICTION_COLUMNS, values, strict=True)})
        print(f"S2_EAT {index}/{len(paths)} {path.stem}", flush=True)
    validate_predictions(rows, [row["ID"] for row in rows])
    temporary = args.output.with_suffix(".csv.partial")
    base.save_submission(temporary, columns, rows)
    temporary.replace(args.output)

