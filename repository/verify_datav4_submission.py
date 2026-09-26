"""Run the actual data v4 submit.zip offline (network audit guard, empty HF cache) on real audio.

  .venv/bin/python verify_datav4_submission.py
Inputs: the 10 parity originals of the Existing9504 handoff (speech, music, songs; wav/flac/mp3). Checks: the
runtime's own manifest hash gate passes, ID order and [0,1] finiteness (validate_predictions), per-file
independence (a reversed 2-file subset reproduces the same rows), and that the packaged decision layer is live
(presence heads move PRESENT away from the legacy PANNs max; FILE is not the plain max rule everywhere).
CPU/macOS only: this is not L4/CUDA throughput or Linux FFmpeg evidence.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'submissions/DataV4_DurationBalanced'
QA = OUT / 'qa'
HANDOFF = ROOT / 'results/existing9504_h100_20260920/completed/DACON_Existing9504_H100_Handoff.zip'
sys.path.insert(0, str(ROOT))
from graph_scoring import PREDICTION_COLUMNS, SUBMISSION_COLUMNS, validate_predictions  # noqa: E402
from v6_data import sha256  # noqa: E402


def main() -> None:
    receipt = json.loads((OUT / 'submit.json').read_text())
    if sha256(OUT / 'submit.zip') != receipt['sha256']:
        raise ValueError('submit.zip changed since packaging')
    runtime = QA / 'extracted'
    shutil.rmtree(runtime, ignore_errors=True)
    with zipfile.ZipFile(OUT / 'submit.zip') as z:
        z.extractall(runtime)
    with zipfile.ZipFile(HANDOFF) as h:
        samples = json.loads(h.read('parity/parity_receipt.json'))['samples']
        for s in samples:
            member = 'parity/audio/' + s['file_stem'] + Path(s['audio_path']).suffix
            s['local'] = QA / 'originals' / (s['ID'] + Path(member).suffix)
            s['local'].parent.mkdir(parents=True, exist_ok=True)
            s['local'].write_bytes(h.read(member))
            if sha256(s['local']) != s['audio_sha256']:
                raise ValueError('parity audio hash mismatch')
    guard = QA / 'guard'
    guard.mkdir(exist_ok=True)
    (guard / 'sitecustomize.py').write_text("import sys\ndef deny(event,args):\n    if event in {'socket.connect','socket.getaddrinfo'}: raise RuntimeError('OFFLINE_NETWORK_BLOCKED')\nsys.addaudithook(deny)\nprint('OFFLINE_GUARD_ACTIVE',flush=True)\n")
    env = os.environ | {'PYTHONPATH': str(guard), 'HF_HOME': str(QA / 'empty_hf_cache'), 'HF_HUB_OFFLINE': '1',
                        'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}

    def cli(selected, name):
        folder = QA / name
        shutil.rmtree(folder, ignore_errors=True)
        (folder / 'inputs').mkdir(parents=True)
        for s in selected:
            os.link(s['local'], folder / 'inputs' / s['local'].name)
        with (folder / 'sample.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS); w.writeheader()
            w.writerows({'ID': s['ID'], **dict.fromkeys(PREDICTION_COLUMNS, 0)} for s in selected)
        with (folder / 'runtime.log').open('w') as log:
            subprocess.run([sys.executable, str(runtime / 'script.py'), '--test-dir', str(folder / 'inputs'),
                            '--sample-submission', str(folder / 'sample.csv'), '--output', str(folder / 'predictions.csv'),
                            '--device', 'cpu'], cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if 'OFFLINE_GUARD_ACTIVE' not in (folder / 'runtime.log').read_text():
            raise ValueError('offline guard not loaded')
        rows = list(csv.DictReader((folder / 'predictions.csv').open()))
        validate_predictions(rows, [s['ID'] for s in selected])
        return rows, json.loads((folder / 'runtime_receipt.json').read_text())

    rows, timing = cli(samples, 'all')
    subset = samples[1::-1]
    again, _ = cli(subset, 'reversed_subset')
    ref = {r['ID']: r for r in rows}
    independence = max(abs(float(r[k]) - float(ref[r['ID']][k])) for r in again for k in PREDICTION_COLUMNS)
    legacy = {s['ID']: s['reference_predictions'] for s in samples}   # Existing9504 H100 rows (legacy presence)
    presence_shift = max(abs(float(r[k]) - float(legacy[r['ID']][k])) for r in rows
                         for k in ('VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB'))
    not_max_rule = sum(abs(float(r['FILE_FAKE_PROB']) - max(float(r['VOICE_PRESENT_PROB']) * float(r['VOICE_FAKE_PROB']),
                           float(r['MUSIC_PRESENT_PROB']) * float(r['MUSIC_FAKE_PROB']))) > 1e-6 for r in rows)
    report = {'submission_sha256': receipt['sha256'], 'device': 'cpu', 'files': len(rows),
              'seconds': timing['seconds'], 'network_connections_forbidden': True,
              'file_independence_max_abs_error': independence, 'presence_shift_vs_legacy_max': presence_shift,
              'files_where_file_score_is_not_plain_max_rule': not_max_rule, 'official_l4_validated': False,
              'predictions': rows}
    assert independence == 0.0, 'per-file prediction depends on other files'
    assert presence_shift > 1e-3 and not_max_rule > 0, 'decision layer not active'
    (QA / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    shutil.rmtree(runtime)
    print(json.dumps({k: v for k, v in report.items() if k != 'predictions'}, indent=2))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT, help='directory holding submit.zip and submit.json')
    OUT = ap.parse_args().out.resolve()
    QA = OUT / 'qa'
    main()
