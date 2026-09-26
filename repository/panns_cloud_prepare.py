"""Verify pre-uploaded inputs and materialize a locked A100 run; no downloads."""
from __future__ import annotations
import argparse
import errno
import json
from pathlib import Path
import shutil
import tempfile
from panns_adaptation_data import identity, sha256, write_json

PREFIX='lb-improvement-20260924'
AUDIO=PREFIX+'/a100-preparation-v1/audio/portable-audio.tar'
WEIGHTS=PREFIX+'/weights/baseline-assets'
AUDIO_SHA='030ff2d8109cfcb39aa0b7c298d68882a829ea4e21dafd41b4343347d038cfa9'


def prepare(bundle, volume, work):
    manifest=json.loads((bundle/'bundle_manifest.json').read_text())
    actual={str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file() and not p.is_relative_to(bundle/'noise')}
    if actual != set(manifest['sha256']) | {'bundle_manifest.json'}:
        raise ValueError('Bundle contains missing or unmanifested files')
    for name, expected in manifest['sha256'].items():
        path=(bundle/name).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha256(path)!=expected:
            raise ValueError('Uploaded bundle integrity failure: '+name)
    if sha256(volume/AUDIO)!=AUDIO_SHA:
        raise ValueError('Canonical uploaded audio hash mismatch')
    package=work/'package'
    package.mkdir(parents=True,exist_ok=True)
    reference=json.loads((bundle/'reference.json').read_text())
    for name, expected in reference['members'].items():
        small=bundle/'package'/name
        source=small if small.exists() else volume/WEIGHTS/str(Path(name).relative_to('model/assets'))
        if sha256(source)!=expected:
            raise ValueError('Pre-uploaded reference asset mismatch: '+name)
        destination=package/name
        if destination.is_symlink() or not destination.resolve().is_relative_to(package.resolve()):
            raise ValueError('Unsafe working asset destination: '+name)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if sha256(destination)!=expected:
                raise ValueError('Existing working asset differs: '+name)
        else:
            if shutil.disk_usage(destination.parent).free < source.stat().st_size:
                raise OSError(errno.ENOSPC,'Insufficient free space for reference asset',str(destination))
            with tempfile.NamedTemporaryFile(dir=work,prefix='.asset-',suffix='.tmp',delete=False) as stream:
                temporary=Path(stream.name)
            try:
                shutil.copyfile(source,temporary)
                if sha256(temporary)!=expected:
                    raise ValueError('Copied reference asset mismatch: '+name)
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError('Working asset appeared during copy: '+name)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    actual_package={str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    if actual_package != set(reference['members']):
        raise ValueError('Working reference package contains unmanifested files')
    reference['package']=str(package)
    write_json(work/'reference.json',reference)
    rows=[]
    for row in map(json.loads,(bundle/'presence_rows.jsonl').read_text().splitlines()):
        audio=row['verified_audio']
        if 'local_path' in audio:
            audio['local_path']=str(bundle/'noise'/row['path'])
            if sha256(audio['local_path'])!=audio['sha256']:
                raise ValueError('Uploaded recovered noise hash mismatch')
        rows.append(row)
    (work/'presence_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    config=dict(reference=str(work/'reference.json'),rows=str(work/'presence_rows.jsonl'),archive=str(volume/AUDIO),
        recovered_audio_root=str(bundle/'noise'),output=str(volume/'panns-cnn-v1-20260925/run-v8'),panel_ids=[r['ID'] for r in rows],
        arms=['R1','R2','R3','R5'],seeds=[41,42,43],cache_device='cuda',microbatch=4,
        early_stopping=dict(max_epochs=200,patience=15,min_delta=1e-4,monitor='selection_presence_bce'),
        checkpoint_every=100,fixed_predictions=str(volume/'panns-cnn-v1-20260925/run-v8/fixed_predictions.json'),
        inputs={str(work/'presence_rows.jsonl'):sha256(work/'presence_rows.jsonl'),
                str(work/'reference.json'):sha256(work/'reference.json')})
    for name in ('spec.md','workflow.md','bundle_manifest.json'):
        config['inputs'][str(bundle/name)]=sha256(bundle/name)
    if 'stem_cache.json' in manifest['sha256']:
        config['stem_cache']=str(bundle/'stem_cache.json')
        config['inputs'][config['stem_cache']]=manifest['sha256']['stem_cache.json']
    code_root=Path(__file__).parent
    for name in manifest['sha256']:
        if name.startswith('code/'):
            path=code_root/Path(name).relative_to('code')
            config['inputs'][str(path)]=sha256(path)
    config['identity']=identity(config)
    path=work/'run.json'
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('Existing run configuration differs; use a fresh work namespace')
    write_json(path,config)
    write_json(work/'prepared.json',dict(bundle_identity=identity(manifest),run_identity=config['identity'],
        uploaded_inputs_verified=True,rows=len(rows),training_budget_status='requires_measured_profile',gpu_started=False))
    return config

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--volume-root',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();prepare(a.bundle.resolve(),a.volume_root.resolve(),a.work.resolve())
