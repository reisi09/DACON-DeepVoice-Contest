# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run with the prepared offline Python: panns_submission_verify.py --help
"""Run the packaged entry point on a small hash-verified panel in a fresh process."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile

from graph_scoring import PREDICTION_COLUMNS, SUBMISSION_COLUMNS, validate_predictions
from mert12_submission import SubmissionError
from v6_data import sha256


def verify(args: argparse.Namespace) -> None:
    started = time.monotonic()
    deadline = args.deadline_epoch or time.time()+2*args.timeout
    if not 1 <= args.limit <= 8 or not 1 <= args.timeout <= 1800:
        raise SubmissionError('QA must use 1–8 files and a 1–1800 second process limit')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads(args.package.with_suffix('.json').read_text())
    if sha256(args.package) != receipt['sha256']:
        raise SubmissionError('Submission ZIP differs from its packaging receipt')
    package = output/'package'
    package.mkdir()
    with zipfile.ZipFile(args.package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(receipt['member_sha256']):
            raise SubmissionError('Unexpected or duplicate ZIP members')
        if sum(member.file_size for member in archive.infolist()) > 32_000_000_000:
            raise SubmissionError('Expanded archive exceeds the submission limit')
        for name in names:
            member = PurePosixPath(name)
            path = package/name
            if member.is_absolute() or '..' in member.parts or '\\' in name or not path.resolve().is_relative_to(package):
                raise SubmissionError('Unsafe ZIP member')
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, path.open('xb') as destination:
                shutil.copyfileobj(source, destination, 8 << 20)
            if sha256(path) != receipt['member_sha256'][name]:
                raise SubmissionError('Extracted package member changed')
    config = json.loads(args.run.read_text())
    cache_path = Path(config['output'])/'cache/index.json'
    cache = json.loads(cache_path.read_text())
    rows = {row['ID']: row for row in map(json.loads, Path(config['rows']).read_text().splitlines())}
    entries = sorted((entry for entry in cache['entries'] if entry['role'] == 'selection'),
                     key=lambda entry: (entry['segments'], entry['ID']))
    expected = {row['ID']: row for row in json.loads(args.expected.read_text())} if args.expected else None
    chosen, patterns = [], set()
    for entry in entries:
        pattern = tuple(entry['targets'])
        if expected is not None and entry['ID'] not in expected:
            continue
        if pattern in patterns and expected is None:
            continue
        chosen.append(entry)
        patterns.add(pattern)
        if len(chosen) == args.limit:
            break
    if not chosen:
        raise SubmissionError('No matching selection audio for package QA')
    audio_root = output/'audio'
    audio_root.mkdir()
    ids, staged = [], []
    with Path(config['archive']).open('rb') as stream:
        for index, entry in enumerate(chosen):
            row = rows[entry['ID']]
            audio = row['verified_audio']
            if 'local_path' in audio:
                local = Path(audio['local_path']).resolve()
                if not local.is_relative_to(Path(config['recovered_audio_root']).resolve()):
                    raise SubmissionError('Recovered QA audio is outside the approved root')
                payload = local.read_bytes()
            else:
                stream.seek(audio['tar_offset'])
                payload = stream.read(audio['bytes'])
            if hashlib.sha256(payload).hexdigest() != audio['sha256']:
                raise SubmissionError('QA audio changed')
            sample_id = f'qa_{index:02}'
            (audio_root/(sample_id+Path(row['path']).suffix)).write_bytes(payload)
            ids.append(sample_id)
            staged.append(dict(ID=entry['ID'], sample_id=sample_id, audio_sha256=audio['sha256']))
    sample = output/'sample_submission.csv'
    with sample.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(dict(ID=sample_id, **{name: 0. for name in PREDICTION_COLUMNS}) for sample_id in ids)
    for name in ('home', 'hf', 'tmp', 'mpl', 'numba'):
        (output/name).mkdir()
    env = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=str(output/'home'),
        HF_HOME=str(output/'hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', TMPDIR=str(output/'tmp'),
        MPLCONFIGDIR=str(output/'mpl'), NUMBA_CACHE_DIR=str(output/'numba'))
    prediction = output/'prediction.csv'
    command = [sys.executable, str(package/'script.py'), '--test-dir', str(audio_root),
               '--sample-submission', str(sample), '--output', str(prediction), '--device', args.device]
    remaining = deadline-time.time()
    if remaining <= 0:
        raise TimeoutError('Submission QA deadline reached before entry point')
    with (output/'entrypoint.log').open('w') as log:
        completed = subprocess.run(command, cwd=package, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=min(args.timeout, remaining), check=False)
    if completed.returncode:
        raise SubmissionError('Packaged entry point failed; inspect entrypoint.log')
    with prediction.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(SUBMISSION_COLUMNS):
            raise SubmissionError('Packaged output columns differ from the official contract')
        predictions = [{key: value if key == 'ID' else float(value) for key, value in row.items()} for row in reader]
    validate_predictions(predictions, ids)
    direct_program = '''import json,sys
from pathlib import Path
import torch
root,audio,out=map(Path,sys.argv[1:4])
sys.path.insert(0,str(root/'model/runtime'))
from panns_submission import Engine
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.use_deterministic_algorithms(True)
engine=Engine(root/'model/assets',root/'model/trained',sys.argv[4])
rows=[dict(ID=p.stem,**engine.predict(p)) for p in sorted(audio.iterdir())]
out.write_text(json.dumps(rows))
'''
    direct_path = output/'fresh_engine_predictions.json'
    remaining = deadline-time.time()
    if remaining <= 0:
        raise TimeoutError('Submission QA deadline reached before independent fresh Engine replay')
    with (output/'fresh_engine.log').open('w') as log:
        subprocess.run([sys.executable, '-c', direct_program, str(package), str(audio_root),
            str(direct_path), args.device], cwd=package, env=env, stdout=log, stderr=subprocess.STDOUT,
            timeout=min(args.timeout, remaining), check=True)
    direct = {row['ID']: row for row in validate_predictions(json.loads(direct_path.read_text()), ids)}
    errors = {row['ID']: max(abs(row[name]-direct[row['ID']][name]) for name in PREDICTION_COLUMNS)
              for row in predictions}
    cached_errors = {}
    if expected is not None:
        for entry, row in zip(staged, predictions, strict=True):
            cached_errors[entry['ID']] = max(abs(row[name]-expected[entry['ID']][name]) for name in PREDICTION_COLUMNS)
    report = dict(package_sha256=receipt['sha256'], files=staged, device=args.device,
        actual_entrypoint=True, fresh_process=True, offline_environment=True,
        fresh_engine_compared=True, probability_max_abs=errors,
        cached_expected_compared=expected is not None, cached_probability_max_abs=cached_errors,
        tolerance=1e-5, within_tolerance=all(value <= 1e-5 for value in errors.values()),
        seconds=time.monotonic()-started, model_quality_or_official_runtime_verified=False)
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    if errors and not report['within_tolerance']:
        raise SubmissionError('Packaged probabilities differ from expected predictions; see verification.json')
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--deadline-epoch', type=float)
    verify(parser.parse_args())


if __name__ == '__main__':
    main()
