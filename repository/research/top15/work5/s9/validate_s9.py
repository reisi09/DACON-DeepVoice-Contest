"""Cache gate for S9: every manifest_s6 id has a finite s9 npz (mixeat (n,6,1024), mixmert (n,8,1024)) with n ==
len(window_layout(samples)) and samples == the stored stems' samples. Exit 1 on any failure."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, '/root/ft/code')
from mert12_submission import window_layout  # noqa: E402
CACHE, STEMS = Path('/root/s9_cache'), Path('/root/stems')


def check(i):
    p = CACHE / f'{i}.npz'
    if not p.exists():
        return i, 'missing'
    try:
        with np.load(p) as d, np.load(STEMS / f'{i}.npz') as s:
            n = len(window_layout(int(d['samples'])))
            if int(d['samples']) != int(s['samples']):
                return i, 'samples != stems'
            if tuple(d['eat_layers']) != (4, 8, 12, 16, 20, 24) or tuple(d['mert_layers']) != (3, 6, 9, 12, 15, 18, 21, 24):
                return i, 'layer lists differ'
            for k, shape in (('mixeat', (6, 1024)), ('mixmert', (8, 1024))):
                if d[k].shape != (n, *shape) or not np.isfinite(d[k].astype(np.float32)).all():
                    return i, f'{k} {d[k].shape}'
    except Exception as e:
        return i, f'{type(e).__name__}: {e}'
    return i, None


ids = Path('/root/s6/manifest_s6.txt').read_text().split()
with ProcessPoolExecutor(48) as ex:
    bad = {i: e for i, e in ex.map(check, ids, chunksize=200) if e}
tmp = sorted(p.name for p in CACHE.glob('*.tmp.npz'))
print(json.dumps({'expected': len(ids), 'bad': len(bad), 'examples': dict(list(bad.items())[:10]), 'tmp_files': tmp[:10]}))
sys.exit(1 if bad or tmp else 0)
