# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
# Run: .venv/bin/python prepare_full_bank.py
"""CPU-only full-duration decode audit and zero-copy MIL window preparation."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from expanded_data import ROOT, BANK, Row, csv_rows
from full_bank_catalog import selection
from eight_contract import sha256_file

OUTPUT = ROOT / 'results/full_bank_mil_20260918'
WINDOW, HOP = 64600, 32300


def decode(row: Row) -> np.ndarray:
    path = (ROOT / row['path']).resolve()
    if not path.is_relative_to(ROOT) or sha256_file(path) != row['source_sha256']:
        raise ValueError('Source identity changed: ' + row['ID'])
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(path),
               '-f', 'f32le', '-ac', '1', '-ar', '16000', '-']
    data = subprocess.run(command, check=True, capture_output=True).stdout
    values = np.frombuffer(data, dtype='<f4').copy()
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid decoded audio: ' + row['ID'])
    return values


def starts_for(samples: int) -> list[int]:
    if samples <= 0:
        raise ValueError('Empty audio cannot become a MIL bag')
    tail = max(0, samples - WINDOW)
    starts = list(range(0, tail + 1, HOP))
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def process(row: Row) -> Row:
    audio = decode(row)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms < 1e-6:
        raise ValueError('Near-silent source requires review: ' + row['ID'])
    peak = float(np.abs(audio).max())
    gain = min(1.0, .6 / max(peak, 1e-8)) if row['input_kind'] == 'original_full_duration' else 1.0
    starts = starts_for(len(audio))
    empty = sum(float(np.max(np.abs(audio[s:s + WINDOW]))) < 1e-6 for s in starts)
    # Keep source-level weak labels distinct from verified partial interval labels.
    local_truth = 'legacy_event_intervals' if row['input_kind'] != 'original_full_duration' else 'unknown'
    return dict(row, decoded_samples=str(len(audio)), processed_duration_seconds=f'{len(audio)/16000:.8f}',
                normalized_gain=repr(gain), decoded_mono16k_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
                window_starts=json.dumps(starts, separators=(',', ':')), window_count=str(len(starts)),
                silent_window_count=str(empty), source_peak=repr(peak), source_rms=repr(rms),
                local_target_scope=local_truth, normalization='full_record_peak_ceiling_0.6' if gain != 1 else 'unchanged')


def write_csv(path: Path, rows: list[Row]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    began = time.monotonic()
    OUTPUT.mkdir(exist_ok=True)
    admitted, ledger = selection()
    write_csv(OUTPUT / 'admission_ledger.csv', ledger)
    results: list[Row] = []
    # Only four CPU decoders run at once; no GPU or cloud operation is imported/called.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(process, admitted), 1):
            results.append(result)
            if index % 250 == 0:
                print(json.dumps({'decoded': index, 'total': len(admitted), 'elapsed_seconds': round(time.monotonic()-began,1)}), flush=True)
    write_csv(OUTPUT / 'train_full_duration.csv', results)
    speech = []
    for name in ('speech_bank_asv_selected.csv', 'speech_bank_test-clean.csv'):
        for row in csv_rows(name):
            speech.append(dict(row, admission='diagnostic_only' if row['train_eligible'] == 'True' else 'protected_or_review',
                               reason='music_authenticity_unknown; frozen_voice_branch', catalogue_file=name))
    write_csv(OUTPUT / 'speech_catalogue.csv', speech)
    original = json.loads((ROOT / 'results/expanded_h100_20260918/selection.json').read_text())
    write_csv(OUTPUT / 'frozen_evaluation_sources.csv', [row for row in original['new_originals'] if row['split'] != 'train'])
    rows_by_source = Counter(row['source'] for row in results)
    report = dict(status='cpu_preparation_complete', gpu_started=False, cloud_modified=False,
                  train_rows=len(results), full_original_rows=sum(row['input_kind']=='original_full_duration' for row in results),
                  legacy_rows=sum(row['input_kind']!='original_full_duration' for row in results),
                  duration_hours=sum(int(row['decoded_samples'])/16000 for row in results)/3600,
                  windows=sum(int(row['window_count']) for row in results),
                  original_audio_bytes_copied=0, rows_by_source=dict(rows_by_source),
                  labels=dict(Counter(row['MUSIC_FAKE'] for row in results)),
                  admission_reasons=dict(Counter(row['reason'] or 'train_music' for row in ledger)),
                  speech_catalogue_rows=len(speech), frozen_evaluation_rows=len(original['new_originals'])-2700,
                  files_sha256={p.name:sha256_file(p) for p in sorted(OUTPUT.glob('*.csv'))},
                  elapsed_seconds=time.monotonic()-began,
                  cache_status='No HTDemucs stems or MERT embeddings computed; GPU remains off',
                  storage='Originals retained; normalized waveform and windows are reconstructed from this manifest on demand')
    (OUTPUT/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
