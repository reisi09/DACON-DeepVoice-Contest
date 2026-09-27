"""Official ADS for new-encoder components, on the exact ads_compare.py pipeline (copy of research/top15/work/r2/ads.py:
shared presence heads, FILE ensemble refitted on val per configuration, voice val-CDF fusion), rule B/C of choice_lock.json.
Baseline = R2 all-chosen (voice_df mlp_long + voice_ad resmlp_long @0.5 + music L6_12), read from R2's run files.

  python ads.py MUSIC_RUNS VOICE_RUNS      comma lists of runs/ names ('none' for no candidate), e.g.
  python ads.py music_new__muq6,music_new__mert6_12+muq6+eat12 voice_new__muqv6

Phase 1 (val only): every configuration of the pre-registered grid -> val official ADS -> the choice, written with a
timestamp to ads_lock__<args>.json BEFORE any held-out metric is computed (an existing lock with a different choice is an
error). Phase 2: held-out official ADS for baseline / music-only / voice-only / joint choice, plus a group bootstrap
(test group_id, 1000 reps, models fixed) of the ADS deltas vs baseline -> ads_report__<args>.json.
Rows: R2's pinned backbone rows that also have a $NEWCACHE npz (all rows when the extraction is complete)."""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'data_v4'))
import train_backbone_heads as H  # noqa: E402
import decision_heads as dh  # noqa: E402
from component_fusion import fuse  # noqa: E402
from file_ensemble import fit as fit_file  # noqa: E402
from full_bank_evaluate import score_rows  # noqa: E402
from graph_scoring import official_metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
R2 = ROOT / 'research/top15/work/r2/runs'
NEW = Path(os.environ.get('NEWCACHE', HERE / 'cache'))
PINNED = set(Path(os.environ.get('ROWS', HERE / 'rows_r2.txt')).read_text().split())
GATE = float(os.environ.get('ADS_GATE', .002))   # pre-registered .002; override only for plumbing checks
MUSIC_W, VOICE_W = (.5, 1, 2), (.25, .5, 1)
music_runs, voice_runs = ([x for x in a.split(',') if x != 'none'] for a in sys.argv[1:3])
uses_new = any(not n.endswith(('mert6_12', '__ad')) for n in music_runs + voice_runs)
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and r['ID'] in PINNED
            and (not uses_new or (NEW / f"{r['ID']}.npz").exists())]
        for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}


def scores(path):
    r = json.loads(path.read_text())
    out = dict(zip(r['val_ids'], np.mean(r['val_preds'], 0).tolist()))
    out.update(zip(r['test_ids'], np.mean(r['test_preds'], 0).tolist()))
    return out


comp = {'df': scores(R2 / 'voice_df__mlp_long.json'), 'ad': scores(R2 / 'voice_ad__resmlp_long.json'), 'mert': scores(R2 / 'music__L6_12.json')}
comp |= {n: scores(HERE / 'runs' / f'{n}.json') for n in music_runs + voice_runs}
pres = json.loads(H.PRESENCE.read_text())
P = {}
for s in ('val', 'test'):
    for r, x in zip(rows[s], score_rows(rows[s], H.OLD[s], ROOT / '_keep_best_20260926/train_duration_balanced', 'cpu')):
        P[r['ID']] = dh.presence(pres, x['presence_features'], (x['Pv'], x['Pm']))
val_table = lambda key: dh.quantile_table([comp[key][r['ID']] for r in rows['val'] if r['ID'] in comp[key]])


def component(members, weights):
    """VOICE/MUSIC score per row id: val-CDF fusion mapped to the first member's scale (component_fusion.fuse);
    a single member passes through; a missing member score is 0 (R2/ads_compare convention, also inside the fusion)."""
    spec = {'weights': dict(zip(['ours', *members[1:]], weights)), 'tables': {'ours': val_table(members[0]), **{m: val_table(m) for m in members[1:]}}}
    out = {}
    for s in ('val', 'test'):
        for r in rows[s]:
            i = r['ID']
            values = {'ours': comp[members[0]].get(i, 0.), **{m: comp[m].get(i, 0.) for m in members[1:]}}
            out[i] = values['ours'] if len(members) == 1 else fuse(spec, values)
    return out


def predictions(cfg, split):
    voice, music = component(*cfg['voice']), component(*cfg['music'])
    files = {s: [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fv': voice[r['ID']], 'Fm': music[r['ID']]} for r in rows[s]] for s in ('val', split)}
    keep = [i for i, r in enumerate(rows['val']) if r['FILE_FAKE_mask'] == '1']
    ens, _ = fit_file([files['val'][i] for i in keep], np.array([int(rows['val'][i]['FILE_FAKE']) for i in keep]))
    return [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'],
             'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in files[split]]


def metrics(cfg, split):
    m = official_metrics(rows[split], predictions(cfg, split))['metrics']
    return {k: round(m[k], 5) for k in ('file_eer', 'voice_eer', 'music_eer')} | {
        'ads': round(.5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer']), 5)}


BASE_VOICE, BASE_MUSIC = (['df', 'ad'], [1, .5]), (['mert'], [1])
base = {'voice': BASE_VOICE, 'music': BASE_MUSIC}
grid = {'baseline': base}
for m in music_runs:
    grid[f'music replace {m}'] = {'voice': BASE_VOICE, 'music': ([m], [1])}
    grid |= {f'music fuse {m} w{w}': {'voice': BASE_VOICE, 'music': (['mert', m], [1, w])} for w in MUSIC_W}
for v in voice_runs:
    grid |= {f'voice add {v} w{w}': {'voice': (['df', 'ad', v], [1, .5, w]), 'music': BASE_MUSIC} for w in VOICE_W}
tag = '__'.join(a.replace(',', '-') for a in sys.argv[1:3])
lock_path, report_path = HERE / f'ads_lock__{tag}.json', HERE / f'ads_report__{tag}.json'

# ---- phase 1: val only ----
val = {n: metrics(c, 'val') for n, c in grid.items()}
simpler = lambda n: (('eat' in n) + ('fuse' in n) + ('+' in n))   # tie-break: fewer encoders, replace before fusion


def pick(prefix, families=None):
    names = [n for n in val if n.startswith(prefix) and (families is None or families(n))]
    return max(names, key=lambda n: (val[n]['ads'], -simpler(n))) if names else None


music_best, music_muq_only = pick('music'), pick('music', lambda n: 'eat' not in n)
if music_best and music_muq_only and music_best != music_muq_only and val[music_best]['ads'] < val[music_muq_only]['ads'] + GATE:
    music_best = music_muq_only                                              # rule B3: EAT must add >= GATE over MuQ-only
music_choice = music_best if music_best and val[music_best]['ads'] >= val['baseline']['ads'] + GATE else None
voice_best = pick('voice')
voice_choice = voice_best if voice_best and val[voice_best]['ads'] >= val['baseline']['ads'] + GATE else None
choice = {'music': music_choice, 'voice': voice_choice}
joint = {'voice': grid[voice_choice]['voice'] if voice_choice else BASE_VOICE, 'music': grid[music_choice]['music'] if music_choice else BASE_MUSIC}
if music_choice and voice_choice:
    val['joint'] = metrics(joint, 'val')
lock = {'written': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'rule': 'choice_lock.json rule B (gate %.3f val ADS; EAT only if +%.3f over MuQ-only; ties -> simpler)' % (GATE, GATE),
        'rows': {s: len(v) for s, v in rows.items()}, 'choice': choice, 'val': val}
if lock_path.exists():
    old = json.loads(lock_path.read_text())
    if old['choice'] != choice or old['rows'] != lock['rows']:
        raise SystemExit(f'{lock_path.name} holds a different val choice; not recomputing held-out')
else:
    lock_path.write_text(json.dumps(lock, indent=1))
print('val choice', choice, {n: val[n]['ads'] for n in val}, flush=True)

# ---- phase 2: held-out, once ----
final = {'baseline': base, **({'music-only': grid[music_choice]} if music_choice else {}),
         **({'voice-only': grid[voice_choice]} if voice_choice else {}), **({'joint': joint} if music_choice and voice_choice else {})}
preds = {n: predictions(c, 'test') for n, c in final.items()}
test = {n: metrics(c, 'test') for n, c in final.items()}
lab = lambda r, c: None if r[c + '_mask'] != '1' or r[c] == '' else int(r[c])
el = {'file': np.array([lab(r, 'FILE_FAKE') is not None for r in rows['test']]),
      'voice': np.array([lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') is not None for r in rows['test']]),
      'music': np.array([lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') is not None for r in rows['test']])}
y = {k: np.array([lab(r, c) if e else -1 for r, e in zip(rows['test'], el[k])]) for k, c in (('file', 'FILE_FAKE'), ('voice', 'VOICE_FAKE'), ('music', 'MUSIC_FAKE'))}
S = {n: {k: np.array([p[c] for p in v]) for k, c in (('file', 'FILE_FAKE_PROB'), ('voice', 'VOICE_FAKE_PROB'), ('music', 'MUSIC_FAKE_PROB'))} for n, v in preds.items()}


def eer(t, s):
    fpr, tpr, _ = roc_curve(t, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2


def ads(n, idx):
    e = {k: eer(y[k][idx][el[k][idx]], S[n][k][idx][el[k][idx]]) for k in ('file', 'voice', 'music')}
    return .5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music'])


groups = np.array([r['group_id'] for r in rows['test']])
ug = np.unique(groups)
members = {g: np.flatnonzero(groups == g) for g in ug}
rng, draws = np.random.default_rng(0), {n: [] for n in final if n != 'baseline'}
for _ in range(1000):
    idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
    b = ads('baseline', idx)
    for n in draws:
        draws[n].append(ads(n, idx) - b)
report = {'lock': lock_path.name, 'choice': choice, 'rows': lock['rows'], 'val': {n: val[n] for n in val}, 'test': test, 'n_test_groups': len(ug),
          'delta_vs_baseline_test': {n: {'ads': round(test[n]['ads'] - test['baseline']['ads'], 5), 'ci95': np.percentile(d, [2.5, 97.5]).round(5).tolist(),
                                         'p_le_0': float((np.array(d) <= 0).mean())} for n, d in draws.items()}}
report_path.write_text(json.dumps(report, indent=1))
print(json.dumps({k: report[k] for k in ('choice', 'test', 'delta_vs_baseline_test')}, indent=1))
