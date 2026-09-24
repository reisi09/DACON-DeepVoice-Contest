"""One-off authorized switch from extraction to E08 on completed, unchanged-role rows."""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile

WORK = Path('/root/lb-a100-20260924-v1')
CODE = WORK / 'prepared/code'
OUT = WORK / 'runs/partial_training'
REMOTE = Path('/shared/lb-improvement-20260924/a100-partial-v1')
DEADLINE = 1790257450
PARENT, FEATURES = 2513, 2520

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')

assert time.time() < DEADLINE - 400
# Extraction and original init script were stopped through authenticated workspace API.
OUT.mkdir(exist_ok=False)
try:
    state = Path(f'/proc/{FEATURES}/stat')
    assert not state.exists() or state.read_text().split()[2] == 'Z'
    print('EXTRACTION_STOPPED', flush=True)
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    save('gpu_after_stop.json', {'compute_processes': gpu, 'checked_epoch': time.time()})
    assert str(FEATURES) not in gpu
    index = json.loads((WORK / 'cache/index.json').read_text())
    with (WORK / 'prepared/inputs/feature_union.csv').open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if r['ID'] in index['items']]
    assert len(rows) == len(index['items'])
    selected = [r for r in rows if r['role'] in ('train', 'selection', 'calibration')]
    for route in ('music', 'voice'):
        label = route.upper() + '_FAKE'
        for role in ('train', 'selection'):
            assert {r[label] for r in selected if r['route'] == route and r['role'] == role} == {'0', '1'}
    manifest = OUT / 'available_rows.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(selected)
    coverage = {'completed_features': len(rows), 'full_features': 43516,
                'roles_unchanged': True, 'partial_ordered_subset': True,
                'manifest_sha256': sha(manifest),
                'role_route_label_counts': dict(collections.Counter('|'.join((r['role'], r['route'], r.get('VOICE_FAKE',''), r.get('MUSIC_FAKE',''))) for r in selected))}
    save('coverage.json', coverage)
    files = {'cache/index.json': WORK / 'cache/index.json',
             'cache/voice_classifier_initial.pt': WORK / 'cache/voice_classifier_initial.pt',
             'available_rows.csv': manifest}
    for entry in index['items'].values():
        name = entry['path']
        assert Path(name).name == name
        p = WORK / 'cache' / name
        assert not p.is_symlink() and sha(p) == entry['sha256']
        files['cache/' + name] = p
    assert sum(p.stat().st_size for p in files.values()) < 5 * 1024**3
    archive = OUT / 'completed_cache.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as z:
        for name, p in files.items(): z.write(p, name)
        z.writestr('EXPORT_SHA256.json', json.dumps({name: sha(p) for name,p in files.items()}))
    REMOTE.mkdir(parents=True, exist_ok=True)
    target = REMOTE / archive.name
    with archive.open('rb') as src, target.open('xb') as dst:
        while block := src.read(16 * 1024**2): dst.write(block)
    assert sha(target) == sha(archive)
    save('cache_saved.json', {'path': str(target), 'sha256': sha(target), 'bytes': target.stat().st_size, 'verified': True, **coverage})
    archive.unlink()  # verified external copy; avoid duplicating bulk cache in final results ZIP
    print('CACHE_SAVED', json.dumps(coverage), flush=True)
    python = WORK / 'env/venv/bin/python'
    for name, config in [('E08_topk', {'seeds':[41], 'voice_pooling':'topk', 'voice_topk_fraction':0.3}),
                         ('E00_control', {'seeds':[41]})]:
        remaining = int(DEADLINE - 400 - time.time())
        if remaining < 30: break
        cfg = OUT / (name + '.json'); cfg.write_text(json.dumps(config))
        cmd = [str(python), str(CODE / 'full_bank_train.py'), '--manifest', str(manifest),
               '--feature-manifest', str(WORK / 'prepared/inputs/feature_union.csv'),
               '--cache', str(WORK / 'cache'), '--output', str(OUT / name),
               '--config', str(cfg), '--gpu-name', 'A100', '--max-epochs', '50', '--patience', '5']
        save(name + '_launch.json', {'command':cmd,'timeout_seconds':remaining,'started_epoch':time.time()})
        print('TRAIN_START', name, flush=True)
        with (OUT / (name + '.log')).open('w') as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1'})
            try: rc = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                rc = 124
        save(name + '_exit.json', {'returncode':rc,'finished_epoch':time.time()})
        print('TRAIN_EXIT', name, rc, flush=True)
        if rc: break
finally:
    # Original init shell stays stopped; export only after this training process has finished.
    try:
        rc = subprocess.run(['python', '/tmp/lb-a100-bootstrap-v1/a100_workspace_bootstrap.py',
             'export', '--work', str(WORK), '--shared', '/shared',
             '--prefix', 'lb-improvement-20260924/a100-run-v1/', '--deadline', str(DEADLINE)],
             timeout=max(1, DEADLINE-time.time()-60)).returncode
        save('transition_exit.json', {'export_returncode':rc,'finished_epoch':time.time()})
    finally:
        (Path('/shared/lb-improvement-20260924/a100-run-v1')/'partial_training_exit.json').write_text(
             json.dumps({'finished_epoch':time.time()}))
