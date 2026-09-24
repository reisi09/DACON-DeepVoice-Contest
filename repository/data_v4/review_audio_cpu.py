"""Offline AST screening evidence; never human listening or authenticity labels.

Run from repository: python data_v4/review_audio_cpu.py ROWS.jsonl ASSETS OUT.jsonl
Rows need ID and absolute path (or --root); optional sha256 pins original bytes.
The centered 15 seconds are covered by overlapping 10-second AST windows.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

for _name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'

import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2_data import decode_audio, pcm_sha256, sha256_file

PINS = {
    'model.safetensors': 'ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d',
    'config.json': 'a93d525511d77e8ecc933d09674b85099815bbbb417c228a4edd655e252fb9ff',
    'preprocessor_config.json': '8d04ba5a9c6fca5d39d0de2b1fd05ecf79deb589fbba279728bbebac39934231',
}
CLASSES = {0: 'Speech', 27: 'Singing', 137: 'Music'}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rows', type=Path)
    ap.add_argument('assets', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--root', type=Path)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--threads', type=int, choices=(4, 8), default=4)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    started = time.monotonic()
    if (a.output.exists() and not a.resume) or a.limit < 0:
        raise ValueError('Existing output requires --resume; limit must be nonnegative')
    for name, expected in PINS.items():
        if sha256_file(a.assets / name) != expected:
            raise ValueError(f'Pinned AST asset mismatch: {name}')
    config = json.loads((a.assets / 'config.json').read_text())
    labels = config['id2label']
    if any(labels[str(i)] != name for i, name in CLASSES.items()):
        raise ValueError('AudioSet class identity mismatch')
    rows = [json.loads(line) for line in a.rows.open()]
    if a.limit:
        rows = rows[:a.limit]
    if not rows or len({r['ID'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty rows with unique IDs')
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    processor = ASTFeatureExtractor.from_pretrained(a.assets, local_files_only=True)
    model, info = ASTForAudioClassification.from_pretrained(
        a.assets, local_files_only=True, use_safetensors=True,
        output_loading_info=True, attn_implementation='sdpa')
    if any(info.values()):
        raise ValueError(f'AST loading mismatch: {info}')
    model = model.to('cpu').eval().requires_grad_(False)
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise RuntimeError('CPU-only model invariant violated')
    runtime = dict(device='cpu', gpu_used=False, torch_threads=a.threads,
                   torch_version=torch.__version__, model_sha256=PINS['model.safetensors'],
                   code_sha256=sha256_file(Path(__file__)), input_sha256=sha256_file(a.rows),
                   score_aggregation='minimum across 10-second windows covering centered 15 seconds')
    previous = [json.loads(line) for line in a.output.open()] if a.resume and a.output.exists() else []
    done = {r['ID']: r for r in previous}
    if len(done) != len(previous) or not set(done).issubset({r['ID'] for r in rows}):
        raise ValueError('Resume output has duplicate or foreign IDs')
    if any(any(r.get(k) != v for k, v in runtime.items()) for r in previous):
        raise ValueError('Resume output belongs to different code, input, model, or runtime')
    setup_seconds = time.monotonic() - started
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('a' if a.resume else 'x') as out, torch.inference_mode():
        for r in rows:
            row_started = time.monotonic()
            path = Path(r['path'])
            if not path.is_absolute():
                if a.root is None:
                    raise ValueError('Relative audio path requires --root')
                path = a.root / path
            path = path.resolve()
            if a.root and not path.is_relative_to(a.root.resolve()):
                raise ValueError('Audio path escapes configured root')
            if not path.is_file() or not 0 < path.stat().st_size <= 100_000_000:
                raise ValueError('Audio missing or exceeds 100 MB review bound')
            digest = sha256_file(path)
            expected = r.get('audio_sha256', r.get('sha256'))
            if expected is not None and digest != expected:
                raise ValueError(f'Audio identity mismatch: {r["ID"]}')
            if r['ID'] in done:
                if done[r['ID']]['audio_sha256'] != digest:
                    raise ValueError(f'Resume audio changed: {r["ID"]}')
                continue
            audio = decode_audio(path, limit_seconds=None)
            if not 0 < len(audio) <= 600 * 16000:
                raise ValueError('Audio duration outside (0, 600] seconds')
            start = max(0, (len(audio) - 15 * 16000) // 2)
            clip = audio[start:start + 15 * 16000]
            starts = sorted({0, max(0, len(clip) - 10 * 16000)})
            waves = [clip[s:s + 10 * 16000] for s in starts]
            inputs = processor(waves, sampling_rate=16000, return_tensors='pt')
            if any(v.device.type != 'cpu' for v in inputs.values()):
                raise RuntimeError('CPU-only input invariant violated')
            scores = model(**inputs).logits.sigmoid().cpu().numpy()
            if not np.isfinite(scores).all():
                raise ArithmeticError('Nonfinite AST output')
            maximum = scores.max(axis=0)
            minimum = scores.min(axis=0)
            result = dict(ID=r['ID'], audio_sha256=digest, review_pcm_sha256=pcm_sha256(clip),
                input_duration_seconds=len(audio) / 16000,
                review_start_seconds=start / 16000, review_seconds=len(clip) / 16000,
                window_starts_seconds=[s / 16000 for s in starts],
                window_durations_seconds=[len(w) / 16000 for w in waves],
                reviewed_crop_coverage_fraction=1.0,
                class_max={name: float(maximum[i]) for i, name in CLASSES.items()},
                class_min={name: float(minimum[i]) for i, name in CLASSES.items()},
                speech_score=float(minimum[0]), singing_score=float(minimum[27]),
                music_score=float(minimum[137]),
                class_by_window={name: scores[:, i].tolist() for i, name in CLASSES.items()},
                top_labels=[dict(label=labels[str(i)], score=float(maximum[i]))
                            for i in np.argsort(maximum)[-8:][::-1]],
                evidence_kind='automatic_audio_event_screening', review_model='AST-AudioSet',
                review_method='overlapping 10-second windows covering centered 15 seconds',
                listening_verified=False,
                authenticity_inferred=False, seconds=time.monotonic() - row_started, **runtime)
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
            out.flush()
            print(json.dumps(dict(ID=r['ID'], class_max=result['class_max'], seconds=result['seconds'])), flush=True)
    print(json.dumps(dict(rows=len(rows), reused_rows=len(done), setup_seconds=setup_seconds,
                          total_seconds=time.monotonic() - started, **runtime)), flush=True)


if __name__ == '__main__':
    main()
