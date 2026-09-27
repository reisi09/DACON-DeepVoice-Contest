"""Generator coverage (experiment 3): distinct fake generators per component (voice / music head labels) in train vs
held-out test (R2 row universe), and whether held-out errors of the 100% heads (runs/*__full.json, 2-seed mean, error at
each head's own held-out route-EER threshold) concentrate on generators unseen in train.
Generator key: bank `generator`, resolved for collapsed rows (echoes: from orig_id path; existing:Echoes/<gen>/..;
existing:suno/udio) and for constructed rows (the fake parent's key); normalised to lower-case alphanumerics.
    python coverage.py        (run after choice_lock.json; reads held-out predictions)
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import subsets as S  # noqa: E402
sys.path.insert(0, str(S.ROOT)); sys.path.insert(0, str(S.ROOT / 'data_v4'))
import csv  # noqa: E402
import full_bank_train as T  # noqa: E402

assert (HERE / 'choice_lock.json').exists()
rows = S.load_rows()
allrows = {r['orig_id']: r for f in ('bank_trainval.csv', 'bank_test.csv') for r in csv.DictReader((S.BANK / f).open())}
norm = lambda g: re.sub(r'[^a-z0-9]', '', g.lower())


def gkey(r, route):
    src, gen = r['source'], r['generator']
    if src.startswith('constructed'):   # the parent that carries this component's fake label
        ps = [allrows[p] for p in json.loads(r['parent_ids'] or '[]') if p in allrows]
        fakes = [p for p in ps if T.head_labels(p).get(route) == 1.] or [p for p in ps if p['FILE_FAKE'] == '1']
        return '+'.join(sorted({gkey(p, route) for p in fakes})) or 'constructed:none'
    if src == 'echoes':
        return 'echoes:' + norm(r['orig_id'].split('/')[2])
    if src.startswith('existing:Echoes/'):
        return 'echoes:' + norm(src.split('/')[1])
    if src in ('existing:suno', 'existing:udio'):
        return norm(src.split(':')[1])
    if src.startswith('asvspoof') or src == 'existing:asvspoof2019_la':
        return 'asvspoof:' + norm(gen)
    return norm(gen) if gen else 'src:' + src


def eer(t, s):
    fpr, tpr, thr = roc_curve(t, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2, thr[i]


out = {}
by_id = {r['ID']: r for s in rows for r in rows[s]}
for route, g in (('voice', 'voice_df'), ('music', 'music')):
    keys = {s: Counter(gkey(r, route) for r in rows[s] if T.head_labels(r).get(route) == 1.) for s in ('train', 'test')}
    unseen = {k: n for k, n in keys['test'].items() if k not in keys['train']}
    run = json.loads((HERE / 'runs' / f'{g}__full.json').read_text())
    t, s = np.array(run['test_labels']), np.mean(run['test_preds'], 0)
    e, thr = eer(t, s)
    err = (s >= thr) != t.astype(bool)
    fake = [(by_id[i], bool(x)) for i, lab, x in zip(run['test_ids'], t, err) if lab == 1]
    real = [(by_id[i], bool(x)) for i, lab, x in zip(run['test_ids'], t, err) if lab == 0]
    split = lambda xs, f: [x for r, x in xs if f(r)]
    un, se = split(fake, lambda r: gkey(r, route) in unseen), split(fake, lambda r: gkey(r, route) not in unseen)
    # group bootstrap of miss(unseen) - miss(seen)
    gid = np.array([r['group_id'] for r, _ in fake])
    isun = np.array([gkey(r, route) in unseen for r, _ in fake])
    ex = np.array([x for _, x in fake])
    ug = np.unique(gid)
    mem = [np.flatnonzero(gid == u) for u in ug]
    rng = np.random.default_rng(0)
    d = []
    for _ in range(1000):
        idx = np.concatenate([mem[j] for j in rng.integers(0, len(ug), len(ug))])
        d.append(ex[idx][isun[idx]].mean() - ex[idx][~isun[idx]].mean())
    per_gen = defaultdict(lambda: [0, 0])
    for r, x in fake:
        per_gen[gkey(r, route)][0] += 1
        per_gen[gkey(r, route)][1] += x
    real_src_train = {r['source'] for r in rows['train'] if T.head_labels(r).get(route) == 0.}
    rs = defaultdict(lambda: [0, 0])
    for r, x in real:
        rs[(r['source'], r['source'] in real_src_train)][0] += 1
        rs[(r['source'], r['source'] in real_src_train)][1] += x
    out[route] = {
        'distinct_fake_generators': {'train': len(keys['train']), 'test': len(keys['test']), 'test_unseen_in_train': len(unseen)},
        'fake_rows': {'train': sum(keys['train'].values()), 'test': sum(keys['test'].values()),
                      'test_from_unseen_generators': sum(unseen.values())},
        'unseen_generators_top': sorted(unseen.items(), key=lambda kv: -kv[1])[:15],
        'full_head_route_eer': round(e, 5), 'threshold': float(thr),
        'miss_rate': {'unseen': [len(un), round(float(np.mean(un)), 4)], 'seen': [len(se), round(float(np.mean(se)), 4)],
                      'unseen_minus_seen_ci95': np.percentile(d, [2.5, 97.5]).round(4).tolist()},
        'share_of_fake_errors_from_unseen': round(sum(un) / max(1, sum(un) + sum(se)), 4),
        'share_of_all_route_errors_from_unseen_fakes': round(sum(un) / max(1, int(err.sum())), 4),
        'fake_errors_by_generator': {k: [n, round(m / n, 3), 'unseen' if k in unseen else 'seen']
                                     for k, (n, m) in sorted(per_gen.items(), key=lambda kv: -kv[1][1]) if n >= 20},
        'real_fp_by_source': {f'{k[0]}|{"seen" if k[1] else "unseen"}': [n, round(m / n, 3)]
                              for k, (n, m) in sorted(rs.items(), key=lambda kv: -kv[1][1]) if n >= 20},
        'errors_total': {'fake_miss': int(sum(ex)), 'real_fp': int(sum(x for _, x in real))},
    }
    print(route, json.dumps(out[route], indent=1))
(HERE / 'coverage.json').write_text(json.dumps(out, indent=1))
