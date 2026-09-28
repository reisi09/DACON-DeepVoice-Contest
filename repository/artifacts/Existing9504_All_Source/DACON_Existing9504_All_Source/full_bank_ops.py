"""Prepare and launch the existing H100 only after complete verified uploads."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

import expanded_h100_ops
from full_bank_upload import ROOT, OUTPUT, client, upload
from v6_data import sha256
import redesign_h100_watch as cloud

PREFIX = 'full-bank-mil-8020-20260918/'
WORKSPACE = 'wsp-9683432r72m7'
WORK = ROOT/'_workspace/full_bank_mil_8020_20260918/deploy'
MANIFEST = OUTPUT/'all_sources.csv'
UPLOAD_RECEIPT = OUTPUT/'audio_upload.json'
SPLIT_VERIFICATION = OUTPUT/'split_verification.json'
REUSE_AUDIO_ROOT = None
LEGACY_STARTUP = b'#!/usr/bin/env bash\nset -euo pipefail\nexec bash /shared/full-bank-mil-8020-20260918/bootstrap.sh\n'
STARTUP = b'#!/usr/bin/env bash\nset -euo pipefail\nnohup bash /shared/full-bank-mil-8020-20260918/bootstrap.sh >/shared/full-bank-mil-8020-20260918/dispatch.log 2>&1 </dev/null &\necho "full-bank job dispatched; initialization complete"\n'
BOOTSTRAP = '''#!/usr/bin/env bash
set -euo pipefail
exec 9>/root/full_bank_mil_8020_20260918.lock
flock -n 9 || exit 1
exec >/shared/full-bank-mil-8020-20260918/bootstrap.log 2>&1
exec /root/fullft8/.venv/bin/python /shared/full-bank-mil-8020-20260918/full_bank_run.py
'''


def configure_balanced():
    """Reuse the verified audio objects with the separately approved 7,144-row split."""
    global PREFIX, OUTPUT, WORK, MANIFEST, SPLIT_VERIFICATION, REUSE_AUDIO_ROOT, STARTUP, BOOTSTRAP
    PREFIX = 'balanced7144-h100-20260919/'
    OUTPUT = ROOT/'results/balanced7144_h100_20260919'
    WORK = ROOT/'_workspace/balanced7144_h100_20260919/deploy'
    MANIFEST = ROOT/'results/balanced7144_20260919/selected.csv'
    SPLIT_VERIFICATION = MANIFEST.parent/'verification.json'
    REUSE_AUDIO_ROOT = '/root/full_bank_mil_8020_20260918/input/dataset'
    STARTUP = STARTUP.replace(b'full-bank-mil-8020-20260918', PREFIX.rstrip('/').encode())
    BOOTSTRAP = BOOTSTRAP.replace('full-bank-mil-8020-20260918', PREFIX.rstrip('/'))
    BOOTSTRAP = BOOTSTRAP.replace('exec /root/fullft8',
        'export FULL_BANK_OBJECT=/shared/'+PREFIX.rstrip('/')+'\n'
        'export FULL_BANK_JOB=/root/balanced7144_h100_20260919\nexec /root/fullft8')


def sources():
    pending = ['full_bank_features', 'full_bank_train', 'full_bank_evaluate']
    found = {}
    while pending:
        name = pending.pop()
        path = ROOT/(name+'.py')
        if name in found or not path.exists():
            continue
        found[name] = path
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                pending.append(node.module.split('.')[0])
            elif isinstance(node, ast.Import):
                pending.extend(alias.name.split('.')[0] for alias in node.names)
    return found.values()


def remote_verify(records):
    s3, bucket, base = client()
    for record in records:
        head = s3.head_object(Bucket=bucket, Key=base+record['key'])
        if (head['ContentLength'] != record['bytes'] or head['ETag'] != record['etag']
                or head['Metadata'].get('sha256') != record['sha256']):
            raise ValueError('Remote object changed: '+record['key'])


def prepare():
    expanded_h100_ops.configure_profile()
    expanded_h100_ops.detail(stopped=True)
    receipt = json.loads(UPLOAD_RECEIPT.read_text())
    if receipt.get('all_sources_uploaded') is not True:
        raise ValueError('Audio upload is still incomplete')
    remote_verify(receipt['shards'])
    rows = list(csv.DictReader(MANIFEST.open()))
    if REUSE_AUDIO_ROOT:
        from balanced_bank_split import verify
        original = list(csv.DictReader((UPLOAD_RECEIPT.parent/'all_sources.csv').open()))
        verify(rows, original)
    mapping = {row['ID']: row for row in receipt['items']}
    if len(mapping) != len(receipt['items']) or not {r['ID'] for r in rows} <= set(mapping):
        raise ValueError('Uploaded audio does not match final split')
    receipt['items'] = [mapping[row['ID']] for row in rows]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'audio_upload.json').write_text(json.dumps(receipt, indent=2)+'\n')
    for row in rows:
        if mapping[row['ID']]['audio_sha256'] != row['source_sha256']:
            raise ValueError('Upload identity does not match split')
        row['original_path'] = row['path']
        row.update(mapping[row['ID']])
    with (WORK/'manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with zipfile.ZipFile(WORK/'input.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(WORK/'manifest.csv', 'dataset/manifest.csv')
        archive.write(OUTPUT/'audio_upload.json', 'audio_upload.json')
        archive.write(SPLIT_VERIFICATION, 'split_verification.json')
        archive.write(ROOT/'results/expanded_h100_20260918/downloaded/unpacked/output/asset_hashes.json', 'asset_hashes.json')
        for path in sources():
            archive.write(path, 'source/'+path.name)
    (WORK/'bootstrap.sh').write_text(BOOTSTRAP)
    records = {}
    for path in (WORK/'input.zip', WORK/'bootstrap.sh', ROOT/'full_bank_run.py'):
        records[path.name] = upload(path, PREFIX+path.name)
    deployment = {'workspace': WORKSPACE, 'wall_time_limit': None, 'files': records,
                  'all_sources_sha256': sha256(MANIFEST),
                  'runtime_manifest_sha256': sha256(WORK/'manifest.csv'),
                  'rows': len(rows), 'audio_bytes': sum(x['bytes'] for x in receipt['shards']),
                  'prepared_at': time.time()}
    deployment['split_counts'] = {split: sum(r['split'] == split for r in rows) for split in ('train', 'val')}
    deployment['reuse_audio_root'] = REUSE_AUDIO_ROOT
    (WORK/'deployment.json').write_text(json.dumps(deployment, indent=2)+'\n')
    deployment_record = upload(WORK/'deployment.json', PREFIX+'deployment.json')
    remote_verify([*records.values(), deployment_record])
    (OUTPUT/'deployment.json').write_text(json.dumps(deployment, indent=2)+'\n')
    expanded_h100_ops.detail(stopped=True)
    print(json.dumps({'ready': True, 'rows': len(rows), 'audio_bytes': deployment['audio_bytes']}))


def launch():
    expanded_h100_ops.configure_profile()
    expanded_h100_ops.detail(stopped=True)
    deployment = json.loads((OUTPUT/'deployment.json').read_text())
    if deployment['all_sources_sha256'] != sha256(MANIFEST):
        raise ValueError('Split changed after deployment')
    remote_verify(list(deployment['files'].values()))
    remote_verify(json.loads((OUTPUT/'audio_upload.json').read_text())['shards'])
    with tempfile.TemporaryDirectory() as folder:
        previous = expanded_h100_ops.read_init(Path(folder))
    if previous not in (expanded_h100_ops.IDLE, STARTUP, LEGACY_STARTUP):
        raise ValueError('Unknown active experiment startup')
    path = WORK/expanded_h100_ops.INIT_NAME
    path.write_bytes(STARTUP)
    upload(path, expanded_h100_ops.INIT_PREFIX+path.name)
    with tempfile.TemporaryDirectory() as folder:
        if expanded_h100_ops.read_init(Path(folder)) != STARTUP:
            raise ValueError('Startup readback failed')
    cloud._call(['workspace', 'start', WORKSPACE], timeout=180)
    (OUTPUT/'launch.json').write_text(json.dumps({'workspace': WORKSPACE, 'started_at': time.time(),
         'deployment_sha256': sha256(OUTPUT/'deployment.json'), 'wall_time_limit': None}, indent=2)+'\n')
    print('H100 start requested for verified deployment: '+PREFIX)


def watch():
    """Five-minute polling; download and verify before stopping GPU billing."""
    previous = None
    while True:
        s3, bucket, base = client()
        state = None
        for name in ('progress.json', 'run_state.json'):
            try:
                data = json.loads(s3.get_object(Bucket=bucket, Key=base+PREFIX+name)['Body'].read())
            except s3.exceptions.NoSuchKey:
                continue
            (OUTPUT/name).write_text(json.dumps(data, indent=2)+'\n')
            if name == 'run_state.json':
                state = data
            else:
                meaningful = {k: v for k, v in data.items() if k != 'updated_at'}
                if meaningful != previous:
                    print(json.dumps(meaningful), flush=True)
                    previous = meaningful
        if state and state.get('archive_ready'):
            destination = OUTPUT/'downloaded'
            destination.mkdir(exist_ok=True)
            archive = destination/'results.zip'
            s3.download_file(bucket, base+PREFIX+'results.zip', str(archive))
            if archive.stat().st_size != state['bytes'] or sha256(archive) != state['sha256']:
                raise ValueError('Downloaded result archive differs from exported bytes')
            from full_bank_run import extract
            extract(archive, destination/'unpacked')
            with zipfile.ZipFile(archive) as packed:
                if packed.testzip() is not None:
                    raise ValueError('Downloaded result archive CRC failure')
            (destination/'verification.json').write_text(json.dumps({'sha256':sha256(archive),
                'bytes':archive.stat().st_size, 'crc_verified':True, 'status':state['status']}, indent=2)+'\n')
            cloud.OUTPUT = OUTPUT
            cloud._pause_and_verify({'workspace':WORKSPACE}, 'verified full-bank result download: '+state['status'])
            idle = WORK/expanded_h100_ops.INIT_NAME
            idle.write_bytes(expanded_h100_ops.IDLE)
            upload(idle, expanded_h100_ops.INIT_PREFIX+idle.name)
            print(json.dumps({'finished':state['status'],'results_verified':True,'gpu_stopped':True}), flush=True)
            return
        time.sleep(300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'launch', 'watch'])
    parser.add_argument('--balanced7144', action='store_true')
    args = parser.parse_args()
    if args.balanced7144:
        configure_balanced()
    if args.action == 'watch':
        from botocore.exceptions import BotoCoreError, ClientError
        while True:
            try:
                watch()
                break
            except (BotoCoreError, ClientError, subprocess.SubprocessError, OSError, RuntimeError) as error:
                print(json.dumps({'monitor_retry': type(error).__name__, 'at': time.time()}), flush=True)
                time.sleep(60)
    else:
        globals()[args.action]()
