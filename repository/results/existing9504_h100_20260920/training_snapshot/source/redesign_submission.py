"""Offline full-data B ensemble using the unchanged frozen-feature extractor."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import numpy as np
import torch

from graph_scoring import PREDICTION_COLUMNS, SUBMISSION_COLUMNS, validate_predictions
from redesign_checkpoint import load_checkpoint, sha256
from redesign_features_cache import FeatureRecord, validate_record
from redesign_features_runtime import FrozenEncoders, _load_baseline
from redesign_models import MusicResidual

SEEDS: Final = (41, 42, 43)
EPOCHS: Final = (17, 17, 8)


class SubmissionError(ValueError):
    """The artifact violates the locked B ensemble or submission contract."""


class Choice(TypedDict):
    candidate: str
    seeds: list[int]
    epochs: list[int]
    checkpoints: list[str]
    file_fusion: str


class Manifest(TypedDict):
    schema_version: int
    choice: Choice
    sha256: dict[str, str]


def verify_package(root: Path) -> Manifest:
    """Check all manifest-covered bytes before loading trusted local model code."""
    manifest = json.loads((root / 'model/manifest.json').read_text(encoding='utf-8'))
    choice = manifest['choice']
    if (choice['seeds'] != list(SEEDS) or choice['epochs'] != list(EPOCHS)
            or choice['file_fusion'] != 'noisy_or' or len(choice['checkpoints']) != 3
            or any(name not in manifest['sha256'] for name in choice['checkpoints'])):
        raise SubmissionError('Package does not contain the locked three-seed B ensemble')
    for name, expected in manifest['sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise SubmissionError(f'Packaged asset or runtime digest mismatch: {name}')
    return manifest


def load_heads(checkpoints: Sequence[Path], device: str = 'cpu') -> list[MusicResidual]:
    """Reload exactly the completed fixed-epoch B heads in seed order."""
    if len(checkpoints) != len(SEEDS):
        raise SubmissionError('The submission requires exactly three B checkpoints')
    heads = []
    for checkpoint, seed, epochs in zip(checkpoints, SEEDS, EPOCHS, strict=True):
        model, report = load_checkpoint(checkpoint, device=device)
        if (not isinstance(model, MusicResidual) or report.get('arm') != 'B'
                or report.get('seed') != seed or report.get('status') != 'complete'
                or report.get('mode') != 'fixed_epochs' or report.get('fixed_epochs') is not True
                or report.get('selected_epoch') != epochs or report.get('epochs_completed') != epochs
                or report.get('epochs') != epochs or report.get('earlystop_ids') != []
                or report.get('validation_loss') != [] or report.get('raw_equal_stem') != 'False'):
            raise SubmissionError(f'Checkpoint is not the completed B seed{seed}/{epochs}-epoch refit')
        heads.append(model.eval())
    return heads


def score_ensemble(heads: Sequence[MusicResidual], record: FeatureRecord, device: str) -> list[float]:
    """Average seed music probabilities, then fuse frozen evidence by noisy-OR."""
    if not heads:
        raise SubmissionError('At least one music classifier is required')
    validate_record(record, 'inference')
    stem = torch.as_tensor(record['stem'], device=device).float()[None]
    raw = torch.as_tensor(record['raw'], device=device).float()[None]
    aux = torch.as_tensor(record['aux'], device=device).float()[None]
    usable = np.asarray(record['valid'], dtype=bool)
    valid = torch.as_tensor(usable & record['stem_audible'], device=device)[None]
    with torch.inference_mode():
        music = float(np.mean([float(head(stem, raw, aux, valid).bag_logits.softmax(-1)[0, 1])
                               for head in heads]))
    voice = float(record['voice_probability'][usable].max()) if usable.any() else 0.0
    vp = float(record['presence'][usable, 0].max()) if usable.any() else 0.0
    mp = float(record['presence'][usable, 1].max()) if usable.any() else 0.0
    return [1 - (1 - vp * voice) * (1 - mp * music), voice, music, vp, mp]


def main(root: Path) -> None:
    """Use official ID matching and infer each file without cross-file statistics."""
    began = time.monotonic()
    root = root.resolve()
    manifest = verify_package(root)
    assets = root / 'model/assets'
    base = _load_baseline(assets / 'baseline/script.py')
    args = base.parse_arguments()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SubmissionError('CUDA was requested but is unavailable')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    columns, rows = base.read_sample_submission(args.sample_submission)
    if columns != list(SUBMISSION_COLUMNS):
        raise SubmissionError('Unexpected sample-submission columns')
    files = base.order_audio_files(base.find_audio_files(args.test_dir), rows)
    heads = load_heads([root / name for name in manifest['choice']['checkpoints']], args.device)
    engine = FrozenEncoders(assets, args.device)
    windows = 0
    for index, (row, path) in enumerate(zip(rows, files, strict=True), 1):
        record = engine.extract(path)
        values = score_ensemble(heads, record, args.device)
        windows += len(record['starts'])
        row.update({key: round(value, 10) for key, value in zip(PREDICTION_COLUMNS, values, strict=True)})
        print(f'B ensemble {index}/{len(files)} {path.stem}', flush=True)
    validate_predictions(rows, [row['ID'] for row in rows])
    temporary = args.output.with_suffix('.csv.partial')
    base.save_submission(temporary, columns, rows)
    temporary.replace(args.output)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = dict(files=len(rows), windows=windows, device=args.device,
        elapsed_seconds=time.monotonic() - began,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == 'cuda' else 0,
        peak_rss_bytes=int(peak_rss if sys.platform == 'darwin' else peak_rss * 1024),
        submission_sha256=sha256(args.output), manifest_sha256=sha256(root / 'model/manifest.json'),
        official_l4_validated=False)
    (args.output.parent / 'runtime_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Saved {len(rows)} predictions to {args.output}', flush=True)
