"""Build an allowlisted cloud bundle from the audited local PANNs artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from panns_adaptation_data import sha256, write_json


def build(data,output,noise_archive,stem_cache=None):
    source=Path(__file__).resolve().parent
    task=data/'_workspace/panns_cnn_v1'
    reference=json.loads((task/'reference_lock.json').read_text())
    package=Path(reference['package'])
    root=output/'bundle'
    root.mkdir(parents=True,exist_ok=False)
    def copy(path,name):
        destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    runtime=[Path(n).name for n in reference['members'] if n.startswith('model/runtime/') and n.endswith('.py')]
    code=['panns_adaptation.py','panns_adaptation_cache.py','panns_adaptation_data.py','panns_adaptation_run.py',
        'panns_cloud_run.py','panns_cloud_pipeline.py','panns_cloud_prepare.py','panns_cloud_report.py',
        'panns_fixed_predictions.py','panns_training_budget.py','panns_training_state.py',
        'panns_cpu_environment.py','panns_restore_environment.py','panns_a100_launch.sh','panns_launch_guard.py',
        'DACON_PANNs_ADS_CPU_A100.ipynb','data_v4/build_manifest.py',*runtime]
    for name in sorted(set(code)):copy(source/name,'code/'+name)
    if stem_cache is not None:
        copy(stem_cache,'stem_cache.json')
    for name,expected in reference['members'].items():
        path=package/name
        if sha256(path)!=expected:raise ValueError('Reference package changed: '+name)
        if name.startswith('model/assets/') and path.stat().st_size>10*1024**2:
            continue
        copy(path,'package/'+name)
    reference.pop('package');reference.pop('archive')
    write_json(root/'reference.json',reference)
    rows=[]
    for row in map(json.loads,(task/'audit_v3/presence_rows.jsonl').read_text().splitlines()):
        if 'local_path' in row['verified_audio']:
            row['verified_audio']['local_path']='noise/'+row['path']
        rows.append(row)
    (root/'presence_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    previous_rows={row['ID']:row for row in map(json.loads,(task/'audit_v2/presence_rows.jsonl').read_text().splitlines())}
    derived_rows={row['original_id']:row for row in rows if 'original_id' in row}
    gold=json.loads((task/'review_final/cache/index.json').read_text())
    for entry in gold['entries']:
        path=Path(entry['path'])
        if sha256(path)!=entry['sha256']:raise ValueError('CPU gold cache changed')
        if entry['ID'] in derived_rows:
            row=derived_rows[entry['ID']];previous=previous_rows[entry['ID']]
            if (row['verified_audio']['sha256']!=previous['verified_audio']['sha256']
                or row['role']!=entry['role'] or [row['VOICE_PRESENT'],row['MUSIC_PRESENT']]!=entry['targets']):
                raise ValueError('Derived audio no longer matches historical CPU gold')
            entry['original_id']=entry['ID'];entry['ID']=row['ID'];entry['component_id']=row['component_id']
        copy(path,'gold/'+path.name);entry['path']=path.name
    write_json(root/'gold/index.json',gold)
    copy(source.parent/'docs/superpowers/specs/2026-09-25-panns-cnn-expanded-design.md','spec.md')
    copy(source.parent/'docs/superpowers/plans/2026-09-25-panns-cnn-v1-workflow.md','workflow.md')
    copy(data/'_workspace/lb_a100_run_20260924_v1/bundle/requirements.lock.txt','requirements.lock.txt')
    files=sorted(p for p in root.rglob('*') if p.is_file())
    manifest=dict(sha256={str(p.relative_to(root)):sha256(p) for p in files},
        omitted_preuploaded_weights=True,expected_volume='objvol-a5gxdg7er70i',expected_mount='/shared')
    write_json(root/'bundle_manifest.json',manifest)
    with tarfile.open(output/'run_bundle.tar','w') as tar:
        for path in [*files,root/'bundle_manifest.json']:
            tar.add(path,arcname=str(path.relative_to(root)),recursive=False)
    write_json(output/'launch_lock.json',dict(sha256={'run_bundle.tar':sha256(output/'run_bundle.tar'),
        'noise-v2.tar':sha256(noise_archive)},workspace='wsp-nvxqw1jf9zdv',volume='objvol-a5gxdg7er70i'))
    return dict(sha256=sha256(output/'run_bundle.tar'),bytes=(output/'run_bundle.tar').stat().st_size,members=len(files)+1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--noise-archive',type=Path,required=True)
    p.add_argument('--stem-cache',type=Path)
    a=p.parse_args();print(json.dumps(build(a.data_root.resolve(),a.output.resolve(),a.noise_archive.resolve(),
        a.stem_cache.resolve() if a.stem_cache is not None else None)))
