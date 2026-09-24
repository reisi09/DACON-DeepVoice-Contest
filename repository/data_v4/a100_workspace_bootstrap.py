"""One bounded A100 run using the verified bank and existing environment lock."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_checked(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        while block := incoming.read(16 * 1024**2):
            outgoing.write(block)
            digest.update(block)
    if digest.hexdigest() != expected:
        target.unlink()
        raise ValueError('Copied input hash mismatch: ' + source.name)


def extract_regular(archive, target):
    """Validate all member paths and types before writing into a fresh directory."""
    with tarfile.open(archive) as packed:
        members = packed.getmembers()
        names = set()
        for item in members:
            name = PurePosixPath(item.name)
            if (not item.isfile() or not name.parts or name.is_absolute() or '..' in name.parts
                    or str(name) != item.name or '\\' in item.name or item.name in names):
                raise ValueError('Unsafe or duplicate archive member')
            names.add(item.name)
        target.mkdir(parents=True, exist_ok=False)
        for item in members:
            path = target / item.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with packed.extractfile(item) as incoming, path.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 16 * 1024**2)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def run(args):
    work, shared = args.work, args.shared
    work.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(work).free < 60 * 1024**3:
        raise RuntimeError('Need 60 GiB free for inputs, isolated environment and outputs')
    source = shared / 'lb-improvement-20260924/a100-preparation-v1'
    remote = shared / args.prefix
    write_json(remote / 'run_started.json', dict(started_epoch=time.time(),
               pause_deadline_epoch=args.deadline, expected_gpu='A100', work=str(work)))
    print('STAGING_VERIFIED_INPUTS', flush=True)
    copy_checked(source / 'a100-preparation.tar.gz', work / 'prepared.tar.gz',
                 'ff0f79538af12dbc518fe57d77847d222beb78a235f872bfe26ce4f4a16d93b6')
    prepared = work / 'prepared'
    extract_regular(work / 'prepared.tar.gz', prepared)
    if sha(prepared / 'SHA256SUMS.json') != 'a6f6380ef2c7e7bcedac09cf0bf42104b961739f90775da43c4342419852d9ac':
        raise ValueError('Prepared manifest differs')
    for name, expected in json.loads((prepared / 'SHA256SUMS.json').read_text()).items():
        if sha(prepared / name) != expected:
            raise ValueError('Prepared member differs: ' + name)
    copy_checked(source / 'audio/portable-audio.tar', work / 'audio.tar',
                 '030ff2d8109cfcb39aa0b7c298d68882a829ea4e21dafd41b4343347d038cfa9')
    extract_regular(work / 'audio.tar', work / 'input')
    hashes = json.loads((prepared / 'inputs/asset_hashes.json').read_text())
    for name, expected in hashes.items():
        copy_checked(shared / 'lb-improvement-20260924/weights/baseline-assets' / name,
                     work / 'assets' / name, expected)
    # The imports below reuse the reviewed environment helpers, without changing HOME.
    sys.path.insert(0, str(Path(__file__).parent / 'env_support'))
    from dacon9504a100 import environment as setup
    env_dir = work / 'env'
    uv = setup.bootstrap_uv(env_dir)
    interpreter, interpreter_source = setup.install_python(uv, env_dir)
    python = env_dir / 'venv/bin/python'
    setup.run([str(uv), 'venv', '--python', str(interpreter), str(python.parent.parent)],
              env=setup.uv_env(env_dir))
    lock = Path(__file__).parent / 'requirements.lock.txt'
    setup.run([str(uv), 'pip', 'install', '--python', str(python), '--require-hashes',
               '--no-deps', '--strict', '-r', str(lock)], env=setup.uv_env(env_dir))
    installed = setup.freeze(uv, python, env_dir)
    if installed != setup.lock_pins(lock):
        raise RuntimeError('Environment freeze differs from the existing lock')
    labels = Path.home() / 'panns_data/class_labels_indices.csv'
    source_labels = work / 'assets/baseline/model/panns/class_labels_indices.csv'
    if labels.exists():
        if sha(labels) != sha(source_labels):
            raise ValueError('Existing PANNs labels differ')
    else:
        copy_checked(source_labels, labels, sha(source_labels))
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'HF_DATASETS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1', 'MPLBACKEND': 'Agg',
           'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    report = setup.probe(python, env, 'A100')
    if not report['torchaudio_torchcodec_decode']['ok']:
        # Same FFmpeg package as the existing workflow, confined to this new container.
        setup.run(['apt-get', '-o', 'Acquire::Retries=2', 'update'],
                  env={**env, 'DEBIAN_FRONTEND': 'noninteractive'}, timeout=180)
        setup.run(['apt-get', 'install', '-y', '--no-install-recommends', 'ffmpeg'],
                  env={**env, 'DEBIAN_FRONTEND': 'noninteractive'}, timeout=300)
        report = setup.probe(python, env, 'A100')
    if (any(value != 'ok' for value in report['imports'].values())
            or not report['torchaudio_torchcodec_decode']['ok']
            or 'A100' not in report.get('gpu', {}).get('name', '')):
        write_json(work / 'environment.json', report)
        raise RuntimeError('A100 environment/import/audio-decoder probe failed')
    setup.link_static_ffmpeg(work, report['imageio_ffmpeg']['exe'])
    env['PATH'] = str(work / 'bin') + os.pathsep + str(python.parent) + os.pathsep + env['PATH']
    write_json(work / 'environment.json', dict(probe=report, lock_sha256=sha(lock),
               interpreter_source=interpreter_source, packages=installed))
    shutil.copyfile(work / 'environment.json', remote / 'environment.json')
    remaining = int(args.deadline - 360 - time.time())
    if remaining <= 0:
        raise TimeoutError('No experiment time remains before export reserve')
    print('STARTING_E00_E04_PIPELINE', flush=True)
    command = [str(python), str(prepared / 'code/data_v4/gpu_experiments.py'), 'all',
               '--gpu-name', 'A100', '--max-seconds', str(remaining)]
    for key, value in dict(feature_manifest=prepared / 'inputs/feature_union.csv',
                           e00_manifest=prepared / 'inputs/E00.csv', e04_manifest=prepared / 'inputs/E04.csv',
                           config=prepared / 'inputs/seed41.json', input=work / 'input',
                           assets=work / 'assets', asset_hashes=prepared / 'inputs/asset_hashes.json',
                           cache=work / 'cache', output=work / 'runs').items():
        command.extend(['--' + key.replace('_', '-'), str(value)])
    subprocess.run(command, env=env, check=True)


def export(args):
    """Only completed indexed features, results and runtime receipts leave ephemeral storage."""
    work, remote = args.work, args.shared / args.prefix
    files = {}
    for name in ('environment.json',):
        if (work / name).is_file():
            files[name] = work / name
    if (work / 'runs').exists():
        for path in (work / 'runs').rglob('*'):
            if path.is_file() and not path.is_symlink() and '.tmp' not in path.name:
                files[path.relative_to(work).as_posix()] = path
    index_path = work / 'cache/index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        files['cache/index.json'] = index_path
        for entry in index['items'].values():
            name = entry['path']
            if Path(name).name != name:
                raise ValueError('Unsafe cache export path')
            path = work / 'cache' / name
            if path.is_symlink() or sha(path) != entry['sha256']:
                raise ValueError('Completed cache changed before export')
            files['cache/' + name] = path
    classifier = work / 'cache/voice_classifier_initial.pt'
    if classifier.is_file() and not classifier.is_symlink():
        files['cache/voice_classifier_initial.pt'] = classifier
    if sum(path.stat().st_size for path in files.values()) > 5 * 1024**3:
        raise RuntimeError('Export exceeds the bounded 5 GiB first-run allowance')
    archive = work / 'results.zip'
    hashes = {name: sha(path) for name, path in files.items()}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as packed:
        for name, path in files.items():
            packed.write(path, name)
        packed.writestr('EXPORT_SHA256.json', json.dumps(hashes, sort_keys=True))
    digest = sha(archive)
    copy_checked(archive, remote / 'results.zip', digest)
    if sha(remote / 'results.zip') != digest:
        raise ValueError('Remote results readback differs')
    progress = work / 'runs/features/progress.json'
    receipt = dict(status='exported', archive_sha256=digest, archive_bytes=archive.stat().st_size,
                   export_member_count=len(files), archive_verified=True,
                   feature_progress=json.loads(progress.read_text()) if progress.exists() else None,
                   pipeline_complete=(work / 'runs/compare_exit.json').exists(),
                   pause_deadline_epoch=args.deadline)
    write_json(remote / 'export_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'export'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--shared', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    if args.deadline <= time.time():
        raise SystemExit('Original pause deadline has already passed')
    (run if args.action == 'run' else export)(args)
