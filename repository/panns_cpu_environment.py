"""Build the existing locked Linux environment on a bounded CPU workspace."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import zipfile


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    shared=Path('/shared')
    work=Path('/root/panns-cnn-v1-20260925')
    work.mkdir(exist_ok=False)
    archive=shared/'lb-improvement-20260924/a100-run-v1/bootstrap_bundle.zip'
    if sha(archive)!='357f7e387ae6367f29974a3b582fbcb3e874a68b503b4dc42558030dbf7a5162':
        raise ValueError('Existing environment bootstrap changed')
    support=work/'support';support.mkdir()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            name=PurePosixPath(item.filename)
            if item.is_dir() or name.is_absolute() or '..' in name.parts or '\\' in item.filename:
                raise ValueError('Unsafe bootstrap member')
            destination=support/item.filename;destination.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as source,destination.open('xb') as target:shutil.copyfileobj(source,target)
    # Build backends never inherit the workload API token or user credentials.
    environment={k:v for k,v in os.environ.items() if k in ('HOME','PATH','LANG','LC_ALL','TMPDIR','LD_LIBRARY_PATH')}
    os.environ.clear();os.environ.update(environment)
    sys.path.insert(0,str(support/'env_support'))
    from dacon9504a100 import environment as setup
    env=work/'env'
    uv=setup.bootstrap_uv(env)
    python,_=setup.install_python(uv,env)
    target=env/'venv/bin/python'
    setup.run([str(uv),'venv','--python',str(python),str(target.parent.parent)],env=setup.uv_env(env))
    lock=support/'requirements.lock.txt'
    setup.run([str(uv),'pip','install','--python',str(target),'--require-hashes','--no-deps','--strict','-r',str(lock)],env=setup.uv_env(env),timeout=1200)
    versions=setup.freeze(uv,target,env)
    if versions!=setup.lock_pins(lock):raise ValueError('Installed environment differs from historical lock')
    labels=shared/'lb-improvement-20260924/weights/baseline-assets/baseline/model/panns/class_labels_indices.csv'
    label_copy=Path.home()/'panns_data/class_labels_indices.csv'
    label_copy.parent.mkdir(parents=True,exist_ok=True)
    if label_copy.exists():
        if sha(label_copy)!=sha(labels):raise ValueError('Existing CPU label table differs')
    else:shutil.copyfile(labels,label_copy)
    probe_code = """
import hashlib,io,json,sys,tarfile
from pathlib import Path
import torch,torchaudio,numpy as np,librosa,soundfile,soxr,panns_inference,demucs,torchlibrosa,transformers
from panns_inference.models import Cnn14
assert not torch.cuda.is_available(), 'CPU-only resource required'
torch.set_num_threads(3)
shared=Path('/shared')
base=Cnn14(32000,1024,320,64,50,14000,527).eval()
weights=shared/'lb-improvement-20260924/weights/baseline-assets/baseline/model/panns/Cnn14_mAP=0.431.pth'
base.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['model'],strict=True)
archive=shared/'panns-cnn-v1-20260925/inputs-v3/run_bundle.tar'
lock=json.loads((archive.parent/'launch_lock.json').read_text())
with archive.open('rb') as stream:bundle_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
assert bundle_sha256==lock['sha256']['run_bundle.tar'], 'Final run bundle hash differs'
with tarfile.open(archive,'r:') as tar:
    member=next(m for m in tar if m.name.startswith('gold/') and m.name.endswith('.npz'))
    with np.load(io.BytesIO(tar.extractfile(member).read()),allow_pickle=False) as data:
        features=torch.from_numpy(data['block5']);expected=torch.from_numpy(data['tags'])
with torch.no_grad():
    x=base.conv_block6(features,pool_size=(1,1),pool_type='avg').mean(3)
    tags=torch.sigmoid(base.fc_audioset(torch.relu(base.fc1(x.max(2).values+x.mean(2)))))
error=float((tags-expected).abs().max())
assert error <= 1e-5, error
print(json.dumps(dict(torch=str(torch.__version__),cpu_gold_tag_max_abs=error,gpu_used=False,bundle_sha256=bundle_sha256)))
"""
    probe=subprocess.run([str(target),'-c',probe_code],
        env={**environment,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'},
        capture_output=True,text=True,timeout=180,check=True)
    # Relativize only task-owned links, preserving the installed environment target.
    for directory in (env/'venv',env/'python'):
        for path in directory.rglob('*'):
            if path.is_symlink():
                resolved=path.resolve(strict=True)
                if not resolved.is_relative_to(env):raise ValueError('Environment symlink escapes owned root')
                path.unlink();path.symlink_to(os.path.relpath(resolved,path.parent))
    members={}
    for directory in (env/'venv',env/'python'):
        for path in directory.rglob('*'):
            if path.is_symlink():members[str(path.relative_to(env))]={'link':os.readlink(path)}
            elif path.is_file():members[str(path.relative_to(env))]={'sha256':sha(path)}
    destination=shared/'panns-cnn-v1-20260925/environment'
    destination.mkdir(parents=True,exist_ok=True)
    packed=destination/'linux-env.tar.gz'
    if packed.exists():raise FileExistsError('Refusing to overwrite an existing environment')
    temporary=packed.with_suffix('.gz.tmp')
    with tarfile.open(temporary,'w:gz',compresslevel=1,dereference=False) as tar:
        for name in ('venv','python'):
            tar.add(env/name,arcname='env/'+name)
        tar.add(label_copy,arcname='panns_data/class_labels_indices.csv')
    temporary.replace(packed)
    record=dict(bundle_sha256=json.loads(probe.stdout.splitlines()[-1])['bundle_sha256'],producer_sha256=sha(__file__),archive_sha256=sha(packed),bytes=packed.stat().st_size,lock_sha256=sha(lock),
        packages=versions,members=members,python_prefix=str(env),probe_stdout=probe.stdout,actual_gpu_used=False)
    receipt=destination/'environment.json'
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(archive_sha256=record['archive_sha256'],bytes=record['bytes'],actual_gpu_used=False)),flush=True)

if __name__=='__main__':main()
