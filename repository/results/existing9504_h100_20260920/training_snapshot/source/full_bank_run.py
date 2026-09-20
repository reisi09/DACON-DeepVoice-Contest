"""Verified full-bank H100 job; no wall-clock training limit."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile

OBJECT = Path(os.environ.get('FULL_BANK_OBJECT', '/shared/full-bank-mil-8020-20260918'))
JOB = Path(os.environ.get('FULL_BANK_JOB', '/root/full_bank_mil_8020_20260918'))
ASSETS = Path('/root/redesign_h100_20260917/assets')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def extract(path, destination, members=None):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        for item in archive.infolist():
            if not (destination/item.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe archive member')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlink')
        archive.extractall(destination, members=None if members is None else [name for name in names if name in members])


def audio_valid(root, row):
    path = (root/row['path']).resolve()
    return path.is_relative_to(root.resolve()) and path.is_file() and digest(path) == row['audio_sha256']


def decoder_path(directory: Path) -> str:
    """Expose the existing offline FFmpeg binary to librosa's fallback decoder."""
    import imageio_ffmpeg
    executable = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    link = directory/'ffmpeg'
    if link.exists():
        if link.resolve() != executable:
            raise RuntimeError('Unexpected FFmpeg executable in job directory')
    else:
        link.symlink_to(executable)
    return str(directory)+os.pathsep+os.environ['PATH']


def main():
    output = JOB/'output'
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status, error = 'failed', None
    stop = threading.Event()
    def progress():
        while not stop.wait(30):
            current = {'status': 'running', 'started_at': started, 'updated_at': time.time()}
            for stage in ('features', 'train'):
                path = output/stage/'progress.json'
                if path.exists():
                    current[stage] = json.loads(path.read_text())
            save(OBJECT/'progress.json', current)
    thread = threading.Thread(target=progress, daemon=True)
    thread.start()
    try:
        deployment = json.loads((OBJECT/'deployment.json').read_text())
        if deployment['workspace'] != 'wsp-9683432r72m7' or deployment['wall_time_limit'] is not None:
            raise ValueError('Unexpected run authorization')
        for name, record in deployment['files'].items():
            if digest(OBJECT/name) != record['sha256']:
                raise ValueError('Deployment bytes changed: '+name)
        extract(OBJECT/'input.zip', JOB/'input')
        incoming = JOB/'input'
        uploads = json.loads((incoming/'audio_upload.json').read_text())
        if uploads.get('all_sources_uploaded') is not True:
            raise ValueError('Incomplete dataset upload')
        needed = sum(x['bytes'] for x in uploads['shards']) + 15_000_000_000
        if shutil.disk_usage(JOB).free < needed:
            raise RuntimeError(f'Insufficient root storage: need {needed} free bytes')
        manifest = incoming/'dataset/manifest.csv'
        rows = list(csv.DictReader(manifest.open()))
        if len(rows) != deployment['rows'] or len({r['ID'] for r in rows}) != len(rows):
            raise ValueError('Dataset accounting changed')
        if digest(manifest) != deployment['runtime_manifest_sha256']:
            raise ValueError('Runtime manifest changed')
        if 'split_counts' in deployment and deployment['split_counts'] != {
                split: sum(r['split'] == split for r in rows) for split in ('train', 'val')}:
            raise ValueError('Train/validation accounting changed')
        reused = 0
        if deployment.get('reuse_audio_root'):
            source = Path(deployment['reuse_audio_root'])
            for row in rows:
                target = incoming/'dataset'/row['path']
                if not target.resolve().is_relative_to((incoming/'dataset').resolve()):
                    raise ValueError('Audio path escaped dataset')
                if not target.exists() and audio_valid(source, row):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source/row['path'], target)
                    reused += 1
        previous = output/'storage_preflight.json'
        already_extracted = (previous.exists() and json.loads(previous.read_text()).get('manifest_sha256') == digest(manifest)
                             and all(audio_valid(incoming/'dataset', row) for row in rows))
        if not already_extracted:
            missing = {row['path'] for row in rows if not audio_valid(incoming/'dataset', row)}
            for shard in uploads['shards'] if missing else []:
                path = Path('/shared')/shard['key']
                local_shard = JOB/'current_audio_shard.zip'
                shutil.copyfile(path, local_shard)
                if local_shard.stat().st_size != shard['bytes'] or digest(local_shard) != shard['sha256']:
                    raise ValueError('Audio shard changed: '+shard['key'])
                extract(local_shard, incoming/'dataset', missing)
                local_shard.unlink()
        for row in rows:
            if not audio_valid(incoming/'dataset', row):
                raise ValueError('Audio bytes changed: '+row['ID'])
        hashes = json.loads((incoming/'asset_hashes.json').read_text())
        for name, expected in hashes.items():
            if digest(ASSETS/name) != expected:
                raise ValueError('Model asset changed: '+name)
        shutil.copytree(incoming/'source', output/'source', dirs_exist_ok=True)
        for name in ('audio_upload.json', 'asset_hashes.json', 'split_verification.json'):
            shutil.copyfile(incoming/name, output/name)
        shutil.copyfile(manifest, output/'dataset_manifest.csv')
        save(output/'storage_preflight.json', {'free_bytes': shutil.disk_usage(JOB).free,
             'rows_verified': len(rows), 'audio_hardlinks_reused': reused,
             'assets_verified': len(hashes), 'manifest_sha256': digest(manifest)})
        environment = dict(os.environ, PYTHONPATH=str(incoming/'source')+':/root/expanded_h100_20260918_retry1/runtime',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        environment['PATH'] = decoder_path(JOB/'bin')
        version = subprocess.run(['ffmpeg', '-version'], env=environment, capture_output=True,
                                 text=True, check=True).stdout.splitlines()[0]
        save(output/'decoder_runtime.json', {'ffmpeg_sha256': digest(JOB/'bin/ffmpeg'), 'version': version})
        common = ['--manifest', str(manifest), '--cache', str(JOB/'cache')]
        for stage, script, extra in (
            ('features', 'full_bank_features.py', ['--input', str(incoming/'dataset'), '--assets', str(ASSETS),
                                                  '--asset-hashes', str(incoming/'asset_hashes.json')]),
            ('train', 'full_bank_train.py', [])):
            destination = output/stage
            destination.mkdir(exist_ok=True)
            with (output/(stage+'.log')).open('a') as log:
                subprocess.run([sys.executable, str(incoming/'source'/script), *common,
                                '--output', str(destination), *extra], env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        if not (output/'train/complete.json').exists():
            raise RuntimeError('Training did not publish completion evidence')
        status = 'complete'
    except Exception:
        error = traceback.format_exc()
        (output/'failure.txt').write_text(error)
    finally:
        stop.set()
        thread.join(timeout=35)
        save(output/'run_state.json', {'status': status, 'error': error, 'started_at': started, 'finished_at': time.time()})
        archive = JOB/'results.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as packed:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    packed.write(path, 'output/'+str(path.relative_to(output)))
        shutil.copyfile(archive, OBJECT/'results.zip.uploading')
        if digest(OBJECT/'results.zip.uploading') != digest(archive):
            raise RuntimeError('Result export differs from local archive')
        (OBJECT/'results.zip.uploading').replace(OBJECT/'results.zip')
        save(OBJECT/'run_state.json', {'status': status, 'archive_ready': True,
             'sha256': digest(archive), 'bytes': archive.stat().st_size,
             'started_at': started, 'finished_at': time.time(), 'error': error})
    return 0 if status == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
