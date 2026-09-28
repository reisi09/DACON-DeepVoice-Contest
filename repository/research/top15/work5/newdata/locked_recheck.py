"""Locked-set feature provenance (review round 5e), no labels and no metric: re-extract a seeded sample of locked rows with
the same packaged Engine (dump_features.py) and compare every stored npz key with the fresh one. Proves the ID -> audio ->
npz link for the sample; score_locked.py requires this record with result PASS and re-hashes the sampled npz before scoring.
Comparison (fixed before use, review round 5f): every value finite in both; float keys np.allclose(rtol=1e-3, atol=1e-3)
(fp16-stored features), presence atol 1e-4; integer/bool keys exactly equal.
  python locked_recheck.py MANIFEST_CSV OUT_JSON [N=30]"""
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

MANIFEST, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 30
WORK = Path('/root/x3/locked_recheck')
FLOAT = ('mert_layers', 'eat_backing', 'sslam_pt_backing', 'wavlm_voice', 'voice_embeddings', 'ad_embeddings', 'presence')
EXACT = ('samples', 'layers', 'eat_layers', 'sslam_layers', 'wavlm_layers', 'voice_silent')


def main():
    rows = sorted(csv.DictReader(MANIFEST.open()), key=lambda r: r['ID'])
    pick = random.Random(20260929).sample(rows, N)
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / 'manifest.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'path'])
        w.writerows([r['ID'], r['path']] for r in pick)
    shutil.rmtree(WORK / 'npz', ignore_errors=True)   # dump_features skips existing npz: always extract fresh
    subprocess.run(['/root/envs/data/bin/python', str(Path(__file__).with_name('dump_features.py')), '/root/s7pkg/x',
                    str(WORK / 'manifest.csv'), str(WORK / 'npz'), '--device', 'cuda:0'], check=True)
    worst, bad = {}, []
    for r in pick:
        with np.load(r['npz']) as a, np.load(WORK / 'npz' / f"{r['ID']}.npz") as b:
            for k in EXACT:
                if not np.array_equal(a[k], b[k]):
                    bad.append(f"{r['ID']}:{k}")
            for k in FLOAT:
                x, y = a[k].astype(np.float64), b[k].astype(np.float64)
                if x.shape != y.shape:
                    bad.append(f"{r['ID']}:{k}:shape")
                    continue
                if not (np.isfinite(x).all() and np.isfinite(y).all()):
                    bad.append(f"{r['ID']}:{k}:nonfinite")
                    continue
                worst[k] = max(worst.get(k, 0.), float(np.abs(x - y).max()))
                tol = dict(rtol=0., atol=1e-4) if k == 'presence' else dict(rtol=1e-3, atol=1e-3)
                if not np.allclose(x, y, **tol):
                    bad.append(f"{r['ID']}:{k}:maxdiff{np.abs(x - y).max():.2e}")
    res = {'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), 'n': N, 'seed': 20260929,
           'ids_sha256': hashlib.sha256('\n'.join(sorted(r['ID'] for r in pick)).encode()).hexdigest(),
           'max_abs_diff': {k: float(v) for k, v in worst.items()}, 'mismatches': bad, 'result': 'PASS' if not bad else 'FAIL',
           'npz_sha256': {r['ID']: {'stored': hashlib.sha256(Path(r['npz']).read_bytes()).hexdigest(),
                                    'fresh': hashlib.sha256((WORK / 'npz' / f"{r['ID']}.npz").read_bytes()).hexdigest()} for r in pick}}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if not bad else 1)


if __name__ == '__main__':
    main()
