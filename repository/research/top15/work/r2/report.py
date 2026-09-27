"""python report.py val   -> val table + choice_lock.json (pre-declared rule in PROTOCOL.md)
   python report.py test  -> held-out route EERs for every run (needs choice_lock.json), report.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/choiwuseck/Documents/ChatGPT/데이콘')
from graph_scoring import _roc_metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = {'voice_df': 'base', 'voice_ad': 'base', 'music': 'L6_12', 'fullvoice_db': 'base', 'fullvoice_p1': 'base'}
eer = lambda y, p: float(_roc_metrics(y, list(p))[0])


def load():
    runs = {}
    for p in sorted((HERE / 'runs').glob('*.json')):
        r = json.loads(p.read_text())
        runs.setdefault(r['group'], {})[r['variant']] = r
    return runs


def row(r, split):
    s = r['seeds'].values()
    per = [x['val_eer'] for x in s] if split == 'val' else [eer(r['test_labels'], p) for p in r['test_preds']]
    ens = r['val_ens_eer'] if split == 'val' else eer(r['test_labels'], np.mean(r['test_preds'], 0))
    return {'ens': round(ens, 5), 'seed_mean': round(float(np.mean(per)), 5), 'seed_sd': round(float(np.std(per, ddof=1)), 5),
            'per_seed': [round(x, 5) for x in per]}


def val():
    runs, lock = load(), {'written': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'rule': 'PROTOCOL.md', 'choice': {}, 'table': {}}
    for g, vs in runs.items():
        b = row(vs[BASE[g]], 'val')
        passing = {}
        for v, r in vs.items():
            x = row(r, 'val')
            s = r['seeds'].values()
            lock['table'].setdefault(g, {})[v] = {**x, 'best_epochs': [i['best_epoch'] for i in s], 'epochs_run': [i['epochs_run'] for i in s],
                                                  'updates_to_best': [i['updates_to_best'] for i in s], 'hit_cap': sum(i['hit_cap'] for i in s),
                                                  'train_s': [i['train_seconds'] for i in s], 'sec_per_epoch': [i['sec_per_epoch'] for i in s],
                                                  'updates_per_epoch': next(iter(s))['updates_per_epoch'], 'n_train': next(iter(s))['n_train'],
                                                  **({'layer_weights': [i['layer_weights'] for i in s]} if 'layer_weights' in next(iter(s)) else {})}
            if v != BASE[g] and b['ens'] - x['ens'] >= .002 and x['seed_mean'] < b['seed_mean']:
                passing[v] = x['ens']
        lock['choice'][g] = min(passing, key=passing.get) if passing else BASE[g]
    (HERE / 'choice_lock.json').write_text(json.dumps(lock, indent=1))
    print(json.dumps(lock, indent=1))


def test():
    lock = json.loads((HERE / 'choice_lock.json').read_text())
    runs = load()
    out = {'choice_lock_written': lock['written'], 'choice': lock['choice'], 'val': lock['table'], 'test': {}}
    for g, vs in runs.items():
        for v, r in vs.items():
            if 'test_preds' not in r:
                continue
            out['test'].setdefault(g, {})[v] = {**row(r, 'test'), 'n_test': len(r['test_labels'])}
    (HERE / 'report.json').write_text(json.dumps(out, indent=1))
    for g in out['test']:  # noqa
        for v, x in out['test'][g].items():
            print(g, v, 'val', lock['table'][g][v]['ens'], lock['table'][g][v]['seed_mean'], 'test', x)


if __name__ == '__main__':
    {'val': val, 'test': test}[sys.argv[1]]()
