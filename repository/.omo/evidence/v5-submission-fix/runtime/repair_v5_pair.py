"""Repair dependency pins without changing weights, then execute both real ZIPs offline."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import soundfile as sf

WORK = Path('/content/v5_repair')
DRIVE = Path('/content/drive/MyDrive/DACON_236749/runs_final_v5/20260913_215338/submissions_B1_B2')
PYTHON = '/content/v5_qa_env/bin/python'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    evidence = WORK / 'evidence'
    evidence.mkdir(exist_ok=True)
    # Loader/CLI smoke probes, not an estimate of full evaluation throughput.
    for path in (WORK / 'data/test').glob('*.wav'):
        audio, rate = sf.read(path, frames=64_000, dtype='float32')
        sf.write(path, audio, rate, subtype='FLOAT')
    guard = WORK / 'guard'
    guard.mkdir(exist_ok=True)
    (guard / 'sitecustomize.py').write_text('''import socket
_connect = socket.socket.connect
def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('Offline QA blocked network: ' + str(address))
    return _connect(self, address)
socket.socket.connect = connect
''')
    receipts = []
    for arm in ('B1', 'B2'):
        original = WORK / f'{arm}_original.zip'
        archive = WORK / f'V5_{arm}_fixed.zip'
        shutil.copyfile(DRIVE / f'V5_{arm}.zip', original)
        print('COPIED', arm, flush=True)
        shutil.copyfile(original, archive)
        subprocess.run(['zip', '-u', str(archive), 'requirements.txt'], cwd=WORK, check=True)
        with zipfile.ZipFile(original) as before, zipfile.ZipFile(archive) as after:
            assert len(after.namelist()) == len(set(after.namelist()))
            assert set(before.namelist()) == set(after.namelist())
            preserved = {}
            for entry in before.infolist():
                if entry.filename == 'requirements.txt':
                    continue
                with before.open(entry) as left, after.open(entry.filename) as right:
                    old = hashlib.file_digest(left, 'sha256').hexdigest()
                    new = hashlib.file_digest(right, 'sha256').hexdigest()
                assert old == new, entry.filename
                if entry.filename.endswith('.pt'):
                    preserved[entry.filename] = new
            assert after.read('requirements.txt') == (WORK / 'requirements.txt').read_bytes()
            root = WORK / arm
            after.extractall(root)
        shutil.copytree(WORK / 'data', root / 'data', dirs_exist_ok=True)
        environment = dict(os.environ, PYTHONPATH=str(guard), HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        start = time.monotonic()
        with (evidence / f'{arm}.log').open('w') as log:
            subprocess.run([PYTHON, 'script.py'], cwd=root, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        elapsed = time.monotonic() - start
        output = root / 'output' / 'submission.csv'
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        with (root / 'data/sample_submission.csv').open() as stream:
            expected = list(csv.DictReader(stream))
        assert [r['ID'] for r in rows] == [r['ID'] for r in expected]
        assert list(rows[0]) == list(expected[0])
        for row in rows:
            assert all(math.isfinite(float(v)) and 0 <= float(v) <= 1
                       for k, v in row.items() if k != 'ID')
            assert abs(float(row['FILE_FAKE_PROB']) - max(
                float(row['VOICE_FAKE_PROB']) * float(row['VOICE_PRESENT_PROB']),
                float(row['MUSIC_FAKE_PROB']) * float(row['MUSIC_PRESENT_PROB']))) < 1e-6
        shutil.copyfile(output, evidence / f'{arm}.csv')
        receipts.append(dict(arm=arm, bytes=archive.stat().st_size, sha256=digest(archive),
                             original_sha256=digest(original), preserved_weights=preserved,
                             cpu_seconds=elapsed, rows=len(rows), offline_cli_passed=True))
        (evidence / 'receipt.json').write_text(json.dumps(receipts, indent=2))
        print('OFFLINE_CLI_PASSED', arm, elapsed, flush=True)
        shutil.rmtree(root)
    # Publish only after both actual entrypoints have passed; originals remain local.
    for arm in ('B1', 'B2'):
        shutil.copyfile(WORK / f'V5_{arm}_fixed.zip', DRIVE / f'V5_{arm}.zip')
    shutil.copytree(evidence, DRIVE / 'dependency_fix_qa', dirs_exist_ok=True)
    print('PAIR_FIXED_AND_COPIED_TO_DRIVE', flush=True)


if __name__ == '__main__':
    main()
