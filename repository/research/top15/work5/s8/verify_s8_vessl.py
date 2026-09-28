"""VESSL-side verification of a Mac-built package without moving its 9.7 GB zip: the package tree is rebuilt from the
verified S7 archive plus the uploaded changed members, every member must match the Mac receipt's member_sha256 (so the
tree IS the zip's content), then the packaged script.py runs offline (socket audit guard, empty HF cache) on the 10
parity originals on DEVICE, all files and a reversed 2-file subset (per-file independence), as verify_datav4_submission.py.
  python verify_s8_vessl.py RECEIPT S7_ZIP CHANGED_DIR PARITY_DIR OUT DEVICE"""
import csv, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

RECEIPT, S7, CHANGED, PARITY, OUT = map(Path, sys.argv[1:6])
DEVICE = sys.argv[6]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = json.loads(RECEIPT.read_text())
tree = OUT / 'tree'
if not tree.exists():   # extraction is reused, the member check below runs every time
    shutil.rmtree(tree, ignore_errors=True)
    with zipfile.ZipFile(S7) as z:
        z.extractall(tree)
    for p in sorted(CHANGED.rglob('*')):
        if p.is_file():
            dst = tree / p.relative_to(CHANGED)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst)
files = {str(p.relative_to(tree)) for p in tree.rglob('*') if p.is_file()}
for extra in sorted(files - set(receipt['member_sha256'])):
    (tree / extra).unlink()
missing = sorted(set(receipt['member_sha256']) - files)
bad = [n for n, d in receipt['member_sha256'].items() if (tree / n).exists() and sha(tree / n) != d]
assert not missing and not bad, f'tree differs from the receipt: missing {missing[:5]} bad {bad[:5]}'
(OUT / 'TREE_OK').write_text(f"{len(receipt['member_sha256'])} members match {receipt['sha256']}\n")
print((OUT / 'TREE_OK').read_text().strip(), flush=True)
sys.path.insert(0, str(tree / 'model' / 'runtime'))
from graph_scoring import PREDICTION_COLUMNS, SUBMISSION_COLUMNS, validate_predictions  # noqa: E402
samples = json.loads((PARITY / 'parity' / 'parity_receipt.json').read_text())['samples']
for s in samples:
    s['local'] = PARITY / 'parity' / 'audio' / (s['file_stem'] + Path(s['audio_path']).suffix)
    assert sha(s['local']) == s['audio_sha256'], s['ID']
qa = OUT / f'qa_{DEVICE}'
guard = qa / 'guard'; guard.mkdir(parents=True, exist_ok=True)
(guard / 'sitecustomize.py').write_text("import sys\ndef deny(event,args):\n    if event in {'socket.connect','socket.getaddrinfo'}: raise RuntimeError('OFFLINE_NETWORK_BLOCKED')\nsys.addaudithook(deny)\nprint('OFFLINE_GUARD_ACTIVE',flush=True)\n")
env = os.environ | {'PYTHONPATH': str(guard), 'HF_HOME': str(qa / 'empty_hf_cache'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                    'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}


def cli(selected, name):
    folder = qa / name
    shutil.rmtree(folder, ignore_errors=True)
    (folder / 'inputs').mkdir(parents=True)
    for s in selected:
        shutil.copy(s['local'], folder / 'inputs' / (s['ID'] + s['local'].suffix))
    with (folder / 'sample.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS); w.writeheader()
        w.writerows({'ID': s['ID'], **dict.fromkeys(PREDICTION_COLUMNS, 0)} for s in selected)
    t = time.time()
    with (folder / 'runtime.log').open('w') as log:
        subprocess.run([sys.executable, str(tree / 'script.py'), '--test-dir', str(folder / 'inputs'), '--sample-submission', str(folder / 'sample.csv'),
                        '--output', str(folder / 'predictions.csv'), '--device', DEVICE], cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert 'OFFLINE_GUARD_ACTIVE' in (folder / 'runtime.log').read_text(), 'offline guard not loaded'
    rows = list(csv.DictReader((folder / 'predictions.csv').open()))
    validate_predictions(rows, [s['ID'] for s in selected])
    return rows, time.time() - t


rows, seconds = cli(samples, 'all')
again, _ = cli(samples[1::-1], 'reversed_subset')
ref = {r['ID']: r for r in rows}
independence = max(abs(float(r[k]) - float(ref[r['ID']][k])) for r in again for k in PREDICTION_COLUMNS)
not_max_rule = sum(abs(float(r['FILE_FAKE_PROB']) - max(float(r['VOICE_PRESENT_PROB']) * float(r['VOICE_FAKE_PROB']),
                   float(r['MUSIC_PRESENT_PROB']) * float(r['MUSIC_FAKE_PROB']))) > 1e-6 for r in rows)
report = {'submission_sha256': receipt['sha256'], 'device': DEVICE, 'files': len(rows), 'seconds': round(seconds, 1),
          'network_connections_forbidden': True, 'file_independence_max_abs_error': independence,
          'files_where_file_score_is_not_plain_max_rule': not_max_rule, 'predictions': rows}
assert independence == 0.0 and not_max_rule > 0
(qa / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'predictions'}, indent=2))
