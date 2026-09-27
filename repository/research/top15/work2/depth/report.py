"""python report.py val   -> val table + choice_lock.json (rule in PROTOCOL.json; no held-out number is computed)
   python report.py test  -> held-out route EERs + saturation for every run (needs choice_lock.json) -> report.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/choiwuseck/Documents/ChatGPT/데이콘')
from graph_scoring import _roc_metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
INCUMBENT = {'fullvoice_db': 'mlp256', 'voice_ad': 'resmlp', 'music': 'mil128_r2'}
eer = lambda y, p: float(_roc_metrics(y, list(p))[0])


def load():
    runs = {}
    for p in sorted((HERE / 'runs').glob('*.json')):
        r = json.loads(p.read_text())
        runs.setdefault(r['group'], {})[r['variant']] = r
    return runs


def sat(preds, labels):
    """exact-1.0 outputs: per seed and in the 3-seed mean (the runtime seed-means, then clips fused values to [0, 1])."""
    P, y = np.array(preds), np.array(labels)
    m = P.mean(0) == 1.
    return {'per_seed': [int((p == 1.).sum()) for p in P], 'seed_mean': int(m.sum()), 'seed_mean_reals': int((m & (y == 0)).sum()),
            'max_seed_mean_reals': float(P.mean(0)[y == 0].max()), 'n': P.shape[1]}


def row(r, split):
    per = [x['val_eer'] for x in r['seeds'].values()] if split == 'val' else [eer(r['test_labels'], p) for p in r['test_preds']]
    ens = r['val_ens_eer'] if split == 'val' else eer(r['test_labels'], np.mean(r['test_preds'], 0))
    return {'ens': round(ens, 5), 'ens_raw': ens, 'seed_mean': round(float(np.mean(per)), 5),
            'seed_sd': round(float(np.std(per, ddof=1)), 5), 'per_seed': [round(x, 5) for x in per],
            'saturation': sat(r[f'{split}_preds'], r[f'{split}_labels'])}


def val():
    from run import VARIANTS
    runs = load()
    lock = {'written': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'rule': json.loads((HERE / 'PROTOCOL.json').read_text())['choice_rule'],
            'amendment': 'added 15:23:45 KST, after some val results, before any held-out number: equal-size non-incumbent candidates are ordered by lower val ens (the rule left this unspecified)', 'missing': {g: sorted(set(VARIANTS[g]) - set(runs.get(g, {}))) for g in VARIANTS}, 'choice': {}, 'candidates': {}, 'table': {}}
    for g, vs in runs.items():
        for v, r in vs.items():
            s = list(r['seeds'].values())
            lock['table'].setdefault(g, {})[v] = {**row(r, 'val'), 'n_params': s[0]['n_params'], 'best_epochs': [i['best_epoch'] for i in s],
                                                  'epochs_run': [i['epochs_run'] for i in s], 'hit_cap': sum(i['hit_cap'] for i in s),
                                                  'train_s': [i['train_seconds'] for i in s], 'sec_per_epoch': [i['sec_per_epoch'] for i in s],
                                                  'peak_rss_gb': max(i['peak_rss_gb'] for i in s), 'n_train': s[0]['n_train'], 'config': r['config']}
        t = lock['table'][g]
        best = min(x['ens_raw'] for x in t.values())
        cand = [v for v, x in t.items() if x['ens_raw'] <= best + .002]
        lock['candidates'][g] = {'best_val_ens': best, 'threshold': best + .002, 'within': cand}
        lock['choice'][g] = min(cand, key=lambda v: (t[v]['n_params'], v != INCUMBENT[g], t[v]['ens_raw']))
    (HERE / 'choice_lock.json').write_text(json.dumps(lock, indent=1))
    for g, t in lock['table'].items():
        for v, x in sorted(t.items(), key=lambda kv: kv[1]['ens_raw']):
            print(g, v, x['ens'], x['seed_mean'], x['seed_sd'], x['n_params'], x['best_epochs'], x['train_s'], x['saturation'])
    print(json.dumps({k: lock[k] for k in ('written', 'missing', 'candidates', 'choice')}, indent=1))


def test():
    lock = json.loads((HERE / 'choice_lock.json').read_text())
    runs = load()
    out = {'choice_lock_written': lock['written'], 'choice': lock['choice'], 'val': lock['table'], 'test': {}}
    for g, vs in runs.items():
        for v, r in vs.items():
            out['test'].setdefault(g, {})[v] = {**row(r, 'test'), 'n_test': len(r['test_labels'])}
    (HERE / 'report.json').write_text(json.dumps(out, indent=1))
    for g in out['test']:
        for v, x in sorted(out['test'][g].items(), key=lambda kv: lock['table'][g][kv[0]]['ens_raw']):
            print(g, v, 'val', lock['table'][g][v]['ens'], 'test', x['ens'], x['seed_mean'], x['seed_sd'], x['per_seed'], 'sat', x['saturation'])


if __name__ == '__main__':
    {'val': val, 'test': test}[sys.argv[1]]()
