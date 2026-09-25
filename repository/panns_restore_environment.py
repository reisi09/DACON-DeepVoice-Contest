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
from typing import Final

ENVIRONMENT_ROOT: Final = Path('/root/panns-cnn-v1-20260925')
RECEIPT_SHA256: Final = 'db4755f3b15d449015e8802d531623b1af959c78d045482d978539e602d3eb5a'
HISTORICAL_BUNDLE_SHA256: Final = '9356724b1b5ad13d3581c738e8214b39e8101bb303d7a2eadfe585a617c05c3b'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_members(root, expected):
    paths={str(p.relative_to(root)):p for p in root.rglob('*')
           if (p.is_file() or p.is_symlink()) and p!=root/'.panns_environment.json'}
    if set(paths)!=set(expected):raise ValueError('Environment member set differs from CPU archive')
    for name,path in paths.items():
        actual={'link':os.readlink(path)} if path.is_symlink() else {'sha256':sha(path)}
        if actual!=expected[name]:raise ValueError('Environment member changed: '+name)


def verify_reuse_inputs(shared, bundle):
    source=shared/'panns-cnn-v1-20260925/environment'
    receipt=source/'environment.json'
    if sha(receipt)!=RECEIPT_SHA256:raise ValueError('Historical CPU environment receipt changed')
    record=json.loads(receipt.read_text())
    if record['producer_sha256']!=sha(Path(__file__).with_name('panns_cpu_environment.py')):
        raise ValueError('CPU environment producer differs from the reviewed bundle')
    if record['lock_sha256']!=sha(bundle/'requirements.lock.txt') or record['actual_gpu_used']:
        raise ValueError('CPU environment receipt does not match the frozen runtime')
    if record['python_prefix']!=str(ENVIRONMENT_ROOT/'env'):
        raise ValueError('CPU environment prefix relocation is not verified')
    historical=shared/'panns-cnn-v1-20260925/inputs-v3/run_bundle.tar'
    if record['bundle_sha256']!=HISTORICAL_BUNDLE_SHA256 or sha(historical)!=HISTORICAL_BUNDLE_SHA256:
        raise ValueError('Historical CPU verification bundle changed')
    with tarfile.open(historical,'r:') as tar:
        manifest=json.load(tar.extractfile('bundle_manifest.json'))['sha256']
    def fixed_input(name):
        return name in ('reference.json','requirements.lock.txt') or name.startswith('package/') or (name.startswith('gold/') and name.endswith('.npz'))
    expected={name:digest for name,digest in manifest.items() if fixed_input(name)}
    actual={str(path.relative_to(bundle)) for path in bundle.rglob('*') if path.is_file() and fixed_input(str(path.relative_to(bundle)))}
    if actual!=set(expected):raise ValueError('CPU verification input set differs')
    for name,digest in expected.items():
        path=(bundle/name).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path)!=digest:
            raise ValueError('CPU verification input changed: '+name)
    return record


def restore(shared,work):
    record=verify_reuse_inputs(shared,work/'bundle')
    source=shared/'panns-cnn-v1-20260925/environment'
    environment=ENVIRONMENT_ROOT/'env'
    if environment.resolve()!=environment:raise ValueError('CPU environment prefix relocation is not verified')
    marker=environment/'.panns_environment.json'
    if environment.exists():
        if not marker.exists() or json.loads(marker.read_text())!=record:
            raise ValueError('Existing environment identity mismatch')
        verify_members(environment,record['members'])
        return
    archive=source/'linux-env.tar.gz'
    if archive.stat().st_size!=record['bytes'] or sha(archive)!=record['archive_sha256']:
        raise ValueError('Pre-uploaded environment archive hash mismatch')
    ENVIRONMENT_ROOT.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='environment_unpack_',dir=ENVIRONMENT_ROOT))
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
    (temporary/'env').replace(environment)
    shutil.rmtree(temporary)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();restore(a.shared.resolve(),a.work.resolve())
