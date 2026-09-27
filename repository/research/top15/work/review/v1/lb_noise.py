"""Q: how noisy is a paired ADS delta measured on a 1,200-file test (the official LB size), and does the LB S3 gain
(+0.0153) fall inside the held-out sampling spread of the runtime-semantics S3-vs-baseline delta?
Re-runs eval_semantics.py (runtime + screen predictions, same components) via runpy, then draws 1,200 held-out files
(iid files, and whole groups until >=1,200 files) 2,000 times and records the paired ADS delta.
  python lb_noise.py voice_df_mlp voice_ad_resmlp"""
import runpy, sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
g = runpy.run_path(str(HERE / 'eval_semantics.py'), run_name='lb_noise')   # uses sys.argv[1:] extras as eval_semantics
rows, built, arrays, fast_ads = g['rows']['test'], g['built'], g['arrays'], g['fast_ads']
pos = {r['ID']: k for k, r in enumerate(rows)}
pairs = [(('runtime', 'S3'), ('runtime', 'baseline')), (('screen', 'S3'), ('screen', 'baseline'))]
pairs += [(('runtime', c), ('runtime', 'S3')) for (sem, c) in built if sem == 'runtime' and c.startswith('S3+')]
A = {k: arrays(rows, v[2]['test']) for k, v in built.items()}
gid = np.unique([r['group_id'] for r in rows], return_inverse=True)[1]
members = [np.flatnonzero(gid == j) for j in range(gid.max() + 1)]
rng = np.random.default_rng(1)
out = {}
for mode in ('iid_files', 'groups'):
    D = {p: [] for p in pairs}
    for _ in range(2000):
        if mode == 'iid_files':
            idx = rng.integers(0, len(rows), 1200)
        else:
            idx, order = [], rng.permutation(len(members))
            for j in order:
                idx.extend(members[j])
                if len(idx) >= 1200:
                    break
            idx = np.array(idx)
        w = np.bincount(idx, minlength=len(rows)).astype(float)
        c = {}
        for p in pairs:
            for k in p:
                if k not in c:
                    c[k] = fast_ads(A[k], w)
            D[p].append(c[p[0]] - c[p[1]])
    for p, d in D.items():
        d = np.array(d)
        out[f'{mode}: {"|".join(p[0])} - {"|".join(p[1])}'] = {
            'mean': round(float(d.mean()), 5), 'sd': round(float(d.std()), 5),
            'q025_q975': [round(float(np.quantile(d, q)), 5) for q in (.025, .975)],
            'P(delta>=0.0153)': round(float((d >= .0153).mean()), 3)}
for k, v in out.items():
    print(k, v, flush=True)
(HERE / 'lb_noise.json').write_text(json.dumps(out, indent=1))
