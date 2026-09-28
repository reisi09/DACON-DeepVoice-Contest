"""Cache gate of s8_lock.json: every manifest_s6 id has a complete, finite s8 npz whose window count matches the runtime
layout and whose samples equal the stored stems. Writes /root/s8_logs/validate.json; exit 1 on any failure.
  python validate_s8.py IDS S8_CACHE STEMS CODE_ROOT"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

IDS, CACHE, STEMS, CODE = sys.argv[1:5]
sys.path.insert(0, CODE)
from mert12_submission import window_layout  # noqa: E402
SHAPES = {'mixwavlm': (6, 1024), 'd2v': (6, 1024), 'usat': (6, 1024), 'mixsslam': (6, 768), 'encodec': (256,)}


def check(row_id):
    p = Path(CACHE) / f'{row_id}.npz'
    if not p.exists():
        return row_id, 'missing'
    try:
        with np.load(p) as d, np.load(Path(STEMS) / f'{row_id}.npz') as s:
            n = len(window_layout(int(d['samples'])))
            if int(d['samples']) != int(s['samples']):
                return row_id, 'samples != stems'
            for k, shape in SHAPES.items():
                x = d[k]
                if x.shape != (n, *shape):
                    return row_id, f'{k} shape {x.shape} != {(n, *shape)}'
                if not np.isfinite(x.astype(np.float32)).all():
                    return row_id, f'{k} nonfinite'
    except Exception as e:  # corrupt zip etc.
        return row_id, f'{type(e).__name__}: {e}'
    return row_id, None


ids = Path(IDS).read_text().split()
assert len(ids) == len(set(ids)), 'duplicate ids in manifest'
with ProcessPoolExecutor(48) as ex:
    bad = {i: e for i, e in ex.map(check, ids, chunksize=200) if e}
extra = sorted({p.stem for p in Path(CACHE).glob('*.npz')} - set(ids))
report = {'expected': len(ids), 'bad': len(bad), 'bad_examples': dict(list(bad.items())[:20]), 'unexpected_files': extra[:20]}
Path('/root/s8_logs/validate.json').write_text(json.dumps(report, indent=1))
print(json.dumps(report))
sys.exit(1 if bad else 0)
