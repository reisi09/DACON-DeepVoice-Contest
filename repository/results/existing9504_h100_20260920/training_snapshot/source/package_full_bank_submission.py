"""Bind the full-bank submission archive to all six selected checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

from full_bank_submission import CANDIDATE, load_heads
from package_expanded_submission import RUNTIME, REQUIREMENTS
from package_redesign_submission import _write_member
from redesign_assets import validate_assets
from v6_data import sha256


def package(root: Path, assets: Path, results: Path, output: Path, ffmpeg: Path):
    hashes = validate_assets(assets)
    if hashes != json.loads((results/'asset_hashes.json').read_text()):
        raise ValueError('Packaged encoders differ from training encoders')
    trained = results/'train'
    complete = json.loads((trained/'complete.json').read_text())
    if complete['results_sha256'] != sha256(trained/'results.json'):
        raise ValueError('Training result changed')
    lock = json.loads((trained/'candidate_lock.json').read_text())
    if lock['validation_manifest_sha256'] != sha256(results/'dataset_manifest.csv'):
        raise ValueError('Checkpoint selection used a different manifest')
    rows = list(csv.DictReader((results/'dataset_manifest.csv').open()))
    counts = {split: sum(row['split'] == split for row in rows) for split in ('train', 'val')}
    if counts != {'train':5716, 'val':1428} or len({row['ID'] for row in rows}) != 7144:
        raise ValueError('Expected the approved balanced 7,144-row split')
    if sha256(ffmpeg) != json.loads((results/'decoder_runtime.json').read_text())['ffmpeg_sha256']:
        raise ValueError('Decoder differs from the verified training decoder')
    load_heads(trained, 'cpu')
    members = {f'model/assets/{name}':assets/name for name in hashes}
    members['model/assets/asset_hashes.json'] = assets/'asset_hashes.json'
    members['model/runtime/bin/ffmpeg'] = ffmpeg
    for name in (*[n for n in RUNTIME if n != 'expanded_submission.py'], 'full_bank_submission.py'):
        members['model/runtime/'+name] = root/name
    for name, expected in lock['checkpoints'].items():
        if sha256(trained/name) != expected:
            raise ValueError('Selected checkpoint changed: '+name)
        members['model/trained/'+name] = trained/name
    if len(lock['checkpoints']) != 6:
        raise ValueError('Expected all six selected checkpoints')
    for name in ('candidate_lock.json', 'run_identity.json', 'validation_metrics.json'):
        members['model/trained/'+name] = trained/name
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    written = {}
    choice = {'candidate':CANDIDATE, 'train_rows':counts['train'], 'validation_rows':counts['val'],
              'seeds':[41,42,43], 'file_fusion':'max', 'selection':'validation_only',
              'independent_test':False, 'voice_head':'backbone.conformer.fc5',
              'htdemucs':'htdemucs', 'mert':'MERT-v1-330M', 'mert_layer':12,
              'inference_gain':'peak capped at 0.6; legacy gain-preserving cache QA is separate',
              'historically_exposed_validation_rows':561}
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, path in sorted(members.items()):
            written[name] = _write_member(archive, name, path)
        entry = "import sys\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parent\nsys.dont_write_bytecode=True\nsys.path.insert(0,str(ROOT/'model/runtime'))\nfrom full_bank_submission import run\nif __name__=='__main__':\n    run(ROOT)\n"
        for name, data in {'script.py':entry.encode(), 'requirements.txt':REQUIREMENTS.encode()}.items():
            archive.writestr(zipfile.ZipInfo(name), data)
            written[name] = hashlib.sha256(data).hexdigest()
        content = (json.dumps({'choice':choice,'sha256':written}, indent=2)+'\n').encode()
        archive.writestr(zipfile.ZipInfo('model/manifest.json'), content)
    expected = {**written, 'model/manifest.json':hashlib.sha256(content).hexdigest()}
    with zipfile.ZipFile(temporary) as archive:
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError('Missing or duplicate package members')
        expanded = sum(x.file_size for x in archive.infolist())
        if temporary.stat().st_size > 10_000_000_000 or expanded > 32_000_000_000:
            raise ValueError('Submission exceeds competition package limits')
        for name, digest in expected.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError('Archive member changed: '+name)
        if archive.testzip() is not None:
            raise ValueError('Archive CRC error')
    temporary.replace(output)
    report = {'file':str(output.resolve()), 'sha256':sha256(output), 'bytes':output.stat().st_size,
              'expanded_bytes':expanded, 'member_sha256':expected, 'choice':choice,
              'crc_verified':True, 'official_l4_validated':False, 'inference_parity_verified':False}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets','results','output','ffmpeg'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report = package(Path(__file__).resolve().parent, args.assets, args.results, args.output, args.ffmpeg)
    print(json.dumps({k:v for k,v in report.items() if k != 'member_sha256'}, indent=2))
