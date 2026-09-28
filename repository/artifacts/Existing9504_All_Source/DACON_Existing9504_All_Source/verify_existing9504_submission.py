"""Exercise the actual offline submission CLI and compare it with locked H100 predictions.

Run: .venv/bin/python verify_existing9504_submission.py
CPU validation does not establish CUDA/L4 throughput or Linux decoder compatibility.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results/existing9504_h100_20260920'
OUTPUT = ROOT / 'submissions/Existing9504_MIL'
QA = OUTPUT / 'qa'
TOLERANCE = 2e-3
sys.path.insert(0, str(RUN / 'author_code.zip'))
from dacon9504a100.common import checked_members, extract_member
from graph_scoring import PREDICTION_COLUMNS, SUBMISSION_COLUMNS, validate_predictions
from v6_data import sha256


def read_rows(path: Path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    """Check real inference, offline operation, file order and legacy gain policy."""
    QA.mkdir(parents=True, exist_ok=True)
    package = json.loads((OUTPUT / 'submit.json').read_text())
    if sha256(OUTPUT / 'submit.zip') != package['sha256']:
        raise ValueError('Submission ZIP changed before runtime QA')
    runtime = QA / 'extracted'
    with zipfile.ZipFile(OUTPUT / 'submit.zip') as archive:
        for name, info in checked_members(archive).items():
            extract_member(archive, info, runtime / name, expected_sha256=package['member_sha256'][name])
    handoff = RUN / 'completed/DACON_Existing9504_H100_Handoff.zip'
    with zipfile.ZipFile(handoff) as archive:
        receipt = json.loads(archive.read('parity/parity_receipt.json'))
        members = checked_members(archive)
        for sample in receipt['samples']:
            member = 'parity/audio/' + sample['file_stem'] + Path(sample['audio_path']).suffix
            target = QA / 'originals' / (sample['ID'] + Path(member).suffix)
            extract_member(archive, members[member], target, expected_sha256=sample['audio_sha256'])
            sample['local_audio'] = str(target)
    regular = [s for s in receipt['samples'] if not s['input_kind'].startswith(('legacy', 'constructed'))]
    legacy = [s for s in receipt['samples'] if s not in regular]
    guard = QA / 'guard'
    guard.mkdir(exist_ok=True)
    (guard / 'sitecustomize.py').write_text("import sys\ndef deny(event,args):\n    if event in {'socket.connect','socket.getaddrinfo'}: raise RuntimeError('OFFLINE_NETWORK_BLOCKED')\nsys.addaudithook(deny)\nprint('OFFLINE_GUARD_ACTIVE',flush=True)\n")
    env = os.environ | {'PYTHONPATH': str(guard), 'HF_HOME': str(QA / 'empty_hf_cache'),
                        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
                        'MKL_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'}
    report = {'submission_sha256': package['sha256'], 'device': 'cpu', 'absolute_tolerance': TOLERANCE,
              'relative_tolerance': 0, 'network_connections_forbidden': True,
              'official_l4_validated': False, 'started_epoch': time.time()}

    def check_predictions(actual, selected):
        validate_predictions(actual, [s['ID'] for s in selected])
        differences = {s['ID']: {k: abs(float(row[k]) - float(s['reference_predictions'][k]))
                                for k in PREDICTION_COLUMNS} for row, s in zip(actual, selected, strict=True)}
        worst = max(value for entry in differences.values() for value in entry.values())
        return {'files': len(selected), 'max_abs_error': worst, 'differences': differences,
                'passed': worst <= TOLERANCE}

    def cli(selected, name):
        folder = QA / name
        inputs = folder / 'inputs'
        inputs.mkdir(parents=True, exist_ok=True)
        for sample in selected:
            src = Path(sample['local_audio'])
            dst = inputs / src.name
            if not dst.exists():
                os.link(src, dst)
        with (folder / 'sample.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
            writer.writeheader()
            writer.writerows({'ID': s['ID'], **dict.fromkeys(PREDICTION_COLUMNS, 0)} for s in selected)
        with (folder / 'runtime.log').open('w') as log:
            subprocess.run([sys.executable, str(runtime / 'script.py'), '--test-dir', str(inputs),
                            '--sample-submission', str(folder / 'sample.csv'), '--output', str(folder / 'predictions.csv'),
                            '--device', 'cpu'], cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if 'OFFLINE_GUARD_ACTIVE' not in (folder / 'runtime.log').read_text():
            raise ValueError('Offline guard was not loaded')
        return read_rows(folder / 'predictions.csv')

    actual = cli(regular, 'regular')
    report['regular'] = check_predictions(actual, regular)
    (QA / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
    print('regular', report['regular']['max_abs_error'], flush=True)
    selected = regular[1::-1]
    reordered = cli(selected, 'reordered_subset')
    report['reordered_subset'] = check_predictions(reordered, selected)
    reference = {row['ID']: row for row in actual}
    report['file_independence_max_abs_error'] = max(abs(float(row[k]) - float(reference[row['ID']][k]))
                                                   for row in reordered for k in PREDICTION_COLUMNS)
    (QA / 'legacy_inputs.json').write_text(json.dumps(legacy))
    child = """import json, sys, os
from pathlib import Path
import torch
root, qa = map(Path, sys.argv[1:])
sys.path.insert(0,str(root/'model/runtime'))
from full_bank_submission import Engine
torch.set_num_threads(4)
os.environ['PATH']=str(root/'model/runtime/bin')+os.pathsep+os.environ['PATH']
engine=Engine(root/'model/assets',root/'model/trained','cpu')
samples=json.loads((qa/'legacy_inputs.json').read_text())
out=[dict(ID=s['ID'],**engine.predict(Path(s['local_audio']),preserve_gain=True)) for s in samples]
(qa/'legacy_predictions.json').write_text(json.dumps(out))
"""
    with (QA / 'legacy_runtime.log').open('w') as log:
        subprocess.run([sys.executable, '-c', child, str(runtime), str(QA)], env=env,
                       cwd=runtime, stdout=log, stderr=subprocess.STDOUT, check=True)
    report['legacy_preserve_gain'] = check_predictions(json.loads((QA / 'legacy_predictions.json').read_text()), legacy)
    report['passed'] = all(report[k]['passed'] for k in ('regular', 'reordered_subset', 'legacy_preserve_gain'))
    report['passed'] &= report['file_independence_max_abs_error'] <= 1e-7
    report['finished_epoch'] = time.time()
    (QA / 'smoke_result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('regular', 'reordered_subset', 'legacy_preserve_gain')}, indent=2))
    if not report['passed']:
        raise ValueError('Packaged inference exceeds the predeclared parity tolerance; inspect qa/smoke_result.json')
    shutil.rmtree(runtime)


if __name__ == '__main__':
    main()
