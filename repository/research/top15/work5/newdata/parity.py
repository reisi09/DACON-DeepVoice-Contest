"""dump_features.py output vs the training caches for rows that already have cached features (same ID).
  python parity.py DUMP_DIR [ROOT]   -> per key: min cosine over windows/rows, max |diff|"""
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/root/proj')
W = ROOT / '_workspace'
CHECKS = [('ad_embeddings', W / 'backbone_cache', 'ad_embeddings', None, None),
          ('mert_layers', W / 'backbone_cache', 'mert_layers', 'layers', 'layers'),
          ('eat_backing', W / 'newbackbone_cache', 'eat_backing', 'eat_layers', 'eat_layers'),
          ('sslam_pt_backing', W / 's6_cache', 'sslam_pt_backing', 'sslam_layers', 'sslam_layers'),
          ('wavlm_voice', W / 's6_cache', 'wavlm_voice', 'wavlm_layers', 'wavlm_layers'),
          ('voice_embeddings', W / 'cache_v4/cache_tv', 'voice_embeddings', None, None),
          ('presence', W / 'cache_v4/cache_tv', 'presence', None, None)]
stats = {k: [] for k, *_ in CHECKS}
silent = []
for f in sorted(D.glob('*.npz')):
    if f.name.endswith('.tmp.npz'):
        continue
    with np.load(f) as d:
        for key, cache, ckey, lk, clk in CHECKS:
            c = cache / f.name
            if not c.exists() or key not in d.files:
                continue
            with np.load(c) as o:
                x = d[key].astype(np.float32)
                y = o[ckey].astype(np.float32)
                if key == 'presence':
                    x, y = x[None], y[None]
                if lk:
                    idx = [list(o[clk]).index(k) for k in d[lk]]
                    y = y[:, idx]
                if x.shape != y.shape:
                    stats[key].append((f.stem, 'shape', x.shape, y.shape))
                    continue
                x2, y2 = x.reshape(len(x), -1), y.reshape(len(y), -1)
                cos = (x2 * y2).sum(1) / (np.linalg.norm(x2, axis=1) * np.linalg.norm(y2, axis=1) + 1e-12)
                stats[key].append((f.stem, float(cos.min()), float(np.abs(x2 - y2).max()), float(np.abs(y2).max())))
                if key == 'ad_embeddings':
                    silent.append(bool(d['voice_silent']) == bool(o['voice_silent']))
for key, v in stats.items():
    ok = [t for t in v if t[1] != 'shape']
    bad = [t for t in v if t[1] == 'shape']
    if ok:
        print(f"{key:18s} n={len(ok)} min_cos={min(t[1] for t in ok):.5f} median_min_cos={np.median([t[1] for t in ok]):.5f} "
              f"max_abs_diff={max(t[2] for t in ok):.4f} (max |x| {max(t[3] for t in ok):.2f}) shape_mismatch={len(bad)}")
    else:
        print(key, 'no comparable rows', bad[:3])
print('voice_silent agreement', sum(silent), '/', len(silent))
