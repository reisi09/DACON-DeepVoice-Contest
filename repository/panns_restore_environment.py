"""Restore the CPU-verified Linux runtime without downloading or installing packages."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_members(root, expected):
    paths={str(p.relative_to(root)):p for p in root.rglob('*')
           if (p.is_file() or p.is_symlink()) and p!=root/'.panns_environment.json'}
    if set(paths)!=set(expected):raise ValueError('Environment member set differs from CPU archive')
    for name,path in paths.items():
        actual={'link':os.readlink(path)} if path.is_symlink() else {'sha256':sha(path)}
        if actual!=expected[name]:raise ValueError('Environment member changed: '+name)


def restore(shared,work):
    source=shared/'panns-cnn-v1-20260925/environment'
    record=json.loads((source/'environment.json').read_text())
    if record['producer_sha256']!=sha(Path(__file__).with_name('panns_cpu_environment.py')):
        raise ValueError('CPU environment producer differs from the reviewed bundle')
    if record['lock_sha256']!=sha(work/'bundle/requirements.lock.txt') or record['actual_gpu_used']:
        raise ValueError('CPU environment receipt does not match the frozen runtime')
    if record['bundle_sha256']!=sha(shared/'panns-cnn-v1-20260925/inputs-v3/run_bundle.tar'):
        raise ValueError('CPU environment was verified against a different run bundle')
    marker=work/'env/.panns_environment.json'
    if (work/'env').exists():
        if not marker.exists() or json.loads(marker.read_text())!=record:
            raise ValueError('Existing environment identity mismatch')
        verify_members(work/'env',record['members'])
        return
    archive=source/'linux-env.tar.gz'
    if archive.stat().st_size!=record['bytes'] or sha(archive)!=record['archive_sha256']:
        raise ValueError('Pre-uploaded environment archive hash mismatch')
    temporary=Path(tempfile.mkdtemp(prefix='environment_unpack_',dir=work))
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)>150000 or sum(m.size for m in members)>30*1024**3:
            raise ValueError('Environment archive exceeds declared resource bounds')
        seen=set()
        for member in members:
            name=PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or '\\' in member.name or str(name)!=member.name
                or name.parts[0] not in ('env','panns_data') or member.name in seen):
                raise ValueError('Unsafe environment member')
            seen.add(member.name)
        tar.extractall(temporary,filter='data')
    verify_members(temporary/'env',record['members'])
    labels=temporary/'panns_data/class_labels_indices.csv'
    target=Path.home()/'panns_data/class_labels_indices.csv'
    if target.exists():
        if sha(target)!=sha(labels):raise ValueError('Existing offline labels differ')
    else:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(labels,target)
    (temporary/'env/.panns_environment.json').write_text(json.dumps(record,sort_keys=True)+'\n')
    (temporary/'env').replace(work/'env')
    shutil.rmtree(temporary)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();restore(a.shared.resolve(),a.work.resolve())
