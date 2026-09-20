"""Stage the fixed 9,504-row bank on the existing stopped A100's object volume."""
from __future__ import annotations

import csv
import fcntl
import json
import zipfile
from collections import Counter
from pathlib import Path

import full_bank_upload as uploader
from eight_fixed_io import atomic_json as save
from full_bank_ops import remote_verify
from v6_data import sha256 as sha

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'results/existing_only_20260919_final'
OUTPUT = ROOT / 'results/existing_only_20260919_vessl'
PREFIX = 'existing-only-9504-20260919/'
WORKSPACE = 'wsp-5i3knd3gxeox'
SELECTED_SHA = '21bb36fb0e7377d986a7c77a466f24ddf473675596effbc57c4bfaac220429fd'


def stopped_mount() -> None:
    detail = json.loads(uploader.cloud._call(['workspace', 'show', WORKSPACE, '-o', 'json']))
    workspace = detail['workspace']
    assert workspace['state'] == 'stopped', 'A100 must remain stopped during preparation'
    assert workspace['resourceSpecSlug'] == 'resourcespec-a100x1'
    assert workspace['initScript'] == '', 'Unexpected startup script: inspect before continuing'
    assert any(m['objectVolumeSlug'] == 'objvol-a5gxdg7er70i' and m['mountPath'] == '/root-1'
               for m in detail['objectVolumes']), 'Object volume is not mounted at the notebook path'
    save(OUTPUT / 'workspace_binding.json', {
        'workspace': WORKSPACE, 'name': workspace['name'], 'state': workspace['state'],
        'resource': workspace['resourceSpecSlug'], 'object_volume': 'objvol-a5gxdg7er70i',
        'mount': '/root-1', 'dataset_root': '/root-1/' + PREFIX.rstrip('/'),
        'gpu_started': False, 'startup_changed': False,
    })


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stopped_mount()
    assert sha(SNAPSHOT / 'selected.csv') == SELECTED_SHA, 'Approved split changed'
    with (SNAPSHOT / 'selected.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert Counter(r['split'] for r in rows) == {'train': 6652, 'val': 1426, 'test': 1426}
    assert len({r['ID'] for r in rows}) == len({r['source_sha256'] for r in rows}) == 9504
    old = json.loads((ROOT / 'results/full_bank_mil_8020_20260918/audio_upload.json').read_text())
    remote_verify(old['shards'])
    existing = {r['audio_sha256']: r for r in old['items']}
    missing = [r for r in rows if r['source_sha256'] not in existing]
    assert len(missing) == 2619
    new = OUTPUT / 'new_audio'
    new.mkdir(exist_ok=True)
    with (new / 'all_sources.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(missing)
    if (new / 'audio_upload.json').exists():
        remote_verify(json.loads((new / 'audio_upload.json').read_text())['shards'])
    uploader.OUTPUT, uploader.PREFIX = new, PREFIX + 'data/'
    uploader.main()
    added = json.loads((new / 'audio_upload.json').read_text())
    existing.update({r['audio_sha256']: r for r in added['items']})
    items = [{**existing[r['source_sha256']], 'ID': r['ID']} for r in rows]
    shards = old['shards'] + added['shards']
    remote_verify(shards)
    assert {r['audio_sha256'] for r in items} == {r['source_sha256'] for r in rows}
    receipt = dict(shards=shards, items=items, all_sources_uploaded=True, gpu_started=False,
                   reused_rows=9504-len(missing), newly_uploaded_rows=len(missing),
                   selected_source_bytes=sum((ROOT/r['path']).stat().st_size if r['path']
                                             else int(r['source_bytes']) for r in rows))
    assert receipt['selected_source_bytes'] == json.loads((SNAPSHOT/'verification.json').read_text())['selected_audio_integrity']['audio_bytes']
    save(OUTPUT / 'audio_upload.json', receipt)
    for row, item in zip(rows, items, strict=True):
        row['original_path'] = row['path']
        row.update(item)
    with (OUTPUT / 'manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    records_path = OUTPUT / 'files.json'
    records = json.loads(records_path.read_text()) if records_path.exists() else {}

    def stage(path: Path, name: str) -> None:
        if name in records and sha(path) == records[name]['sha256']:
            remote_verify([records[name]])
            return
        records[name] = uploader.upload(path, PREFIX + name)
        save(records_path, records)
        print(json.dumps({'uploaded': name, 'bytes': records[name]['bytes']}), flush=True)

    stage(OUTPUT / 'manifest.csv', 'dataset/manifest.csv')
    stage(OUTPUT / 'audio_upload.json', 'audio_upload.json')
    for name in ('selected.csv', 'train.csv', 'val.csv', 'test.csv', 'README.md', 'verification.json',
                 'audit.json', 'policy_checks.json', 'provenance.json', 'reproducibility.json'):
        stage(SNAPSHOT / name, 'dataset/' + name)
    assets = ROOT / '_workspace/expanded_submission_qa/package/model/assets'
    with zipfile.ZipFile(ROOT / '_workspace/full_bank_mil_8020_20260918/deploy/input.zip') as archive:
        hashes = json.loads(archive.read('asset_hashes.json'))
    save(OUTPUT / 'asset_hashes.json', hashes)
    packed = ROOT / '_workspace/existing9504_assets.zip'
    if 'assets.zip' not in records:
        with zipfile.ZipFile(packed, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, digest in hashes.items():
                assert sha(assets / name) == digest, 'Frozen asset changed: ' + name
                archive.write(assets / name, name)
            archive.write(OUTPUT / 'asset_hashes.json', 'asset_hashes.json')
        stage(packed, 'assets.zip')
    else:
        remote_verify([records['assets.zip']])
    stage(OUTPUT / 'asset_hashes.json', 'asset_hashes.json')
    stage(OUTPUT / 'claude_author_context.zip', 'source.zip')
    stopped_mount()
    deployment = dict(workspace=WORKSPACE, object_volume='objvol-a5gxdg7er70i', mount='/root-1',
                      prefix=PREFIX, files=records, rows=9504,
                      split_counts={'train': 6652, 'val': 1426, 'test': 1426},
                      selected_sha256=SELECTED_SHA, runtime_manifest_sha256=sha(OUTPUT/'manifest.csv'),
                      gpu_started=False, training_ready=False, notebook_review_pending=True)
    save(OUTPUT / 'deployment.json', deployment)
    record = uploader.upload(OUTPUT / 'deployment.json', PREFIX + 'deployment.json')
    remote_verify([*records.values(), record])
    save(OUTPUT / 'deployment_receipt.json', record)
    print(json.dumps({'data_staged': True, 'rows': 9504, 'gpu_started': False}), flush=True)


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'upload.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
