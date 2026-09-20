"""Offline, file-local inference for the self-contained v2 AST submission."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as soundfile
import torch
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor

from ast_multitask import PREDICTION_COLUMNS, SAMPLE_RATE, ASTMultiTask, load_checkpoint, predict_file

SUBMISSION_COLUMNS: Final = ("ID", *PREDICTION_COLUMNS)
SUPPORTED_AUDIO_SUFFIXES: Final = frozenset({".wav", ".flac", ".mp3"})


class InferenceContractError(ValueError):
    """The evaluator input or local inference asset violates the submission contract."""


def load_audio(path: Path) -> np.ndarray:
    """Decode one evaluator file to finite mono 16 kHz float32 PCM without a network codec."""
    try:
        waveform, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        raise InferenceContractError(f"Cannot decode audio file: {path.name}") from error
    if waveform.size == 0 or sample_rate <= 0 or not np.isfinite(waveform).all():
        raise InferenceContractError(f"Invalid audio file: {path.name}")
    mono = waveform.mean(axis=1, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise InferenceContractError(f"Invalid resampled audio file: {path.name}")
    return mono


def _sample_ids(path: Path) -> list[str]:
    """Parse the evaluator-owned ID order once at the CSV trust boundary."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(SUBMISSION_COLUMNS):
                raise InferenceContractError("Sample submission columns differ from the official schema")
            identifiers = [row.get("ID", "") for row in reader]
    except OSError as error:
        raise InferenceContractError("Cannot read sample submission") from error
    if not identifiers or any(not identifier for identifier in identifiers) or len(identifiers) != len(set(identifiers)):
        raise InferenceContractError("Sample submission IDs must be non-empty and unique")
    return identifiers


def ordered_audio_files(test_dir: Path, identifiers: list[str]) -> list[Path]:
    """Map each requested ID to exactly one supported evaluator file."""
    files = [path for path in test_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES]
    by_id = {path.stem: path for path in files}
    if len(by_id) != len(files):
        raise InferenceContractError("Evaluator audio IDs are not unique")
    missing = set(identifiers) - set(by_id)
    extra = set(by_id) - set(identifiers)
    if missing or extra:
        raise InferenceContractError(f"Evaluator audio IDs mismatch: missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}")
    return [by_id[identifier] for identifier in identifiers]


def load_packaged_model(root: Path) -> tuple[ASTMultiTask, ASTFeatureExtractor]:
    """Load one local checkpoint, selecting only a declared training mode."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = root / "model" / "ast"
    try:
        metadata = json.loads((checkpoint / "multitask_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InferenceContractError("Cannot read local AST checkpoint metadata") from error
    mode = metadata.get("mode")
    match mode:
        case "frozen" | "last_block":
            model, processor = load_checkpoint(checkpoint, mode)
        case _:
            raise InferenceContractError("Unsupported local AST checkpoint mode")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.eval().to(device), processor


def predict_audio_file(model: ASTMultiTask, processor: ASTFeatureExtractor, path: Path) -> list[float]:
    """Predict one file, rejecting malformed output before it can reach the CSV."""
    probabilities = predict_file(model, processor, load_audio(path)).tolist()
    if len(probabilities) != len(PREDICTION_COLUMNS) or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities
    ):
        raise InferenceContractError(f"Invalid probability output for: {path.name}")
    return probabilities


def generate_submission(root: Path) -> Path:
    """Run file-local inference and atomically write evaluator-ordered submission.csv."""
    identifiers = _sample_ids(root / "data" / "sample_submission.csv")
    paths = ordered_audio_files(root / "data" / "test", identifiers)
    model, processor = load_packaged_model(root)
    rows = []
    for index, path in enumerate(paths, 1):
        probabilities = predict_audio_file(model, processor, path)
        rows.append(dict(zip(SUBMISSION_COLUMNS, [path.stem, *probabilities], strict=True)))
        print(f"PREDICT {index}/{len(paths)} {path.stem}", flush=True)
    destination = root / "output" / "submission.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination


def main() -> None:
    """Run against an evaluator-layout directory passed as the working directory."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    generate_submission(Path.cwd())


if __name__ == "__main__":
    main()
