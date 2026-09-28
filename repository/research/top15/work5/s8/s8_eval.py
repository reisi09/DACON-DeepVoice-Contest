"""S8 screening under research/top15/work5/s8/s8_lock.json, runtime (RT) semantics as work4/rt_s6enc.py.
  python s8_eval.py select    step A/B + joint on VAL only -> s8_choice.json (refuses to change an existing choice)
  python s8_eval.py heldout   step C for the locked choice -> s8_heldout.json
Base S7_cpu members = work4/runs (CPU); candidates = $S8_RUNS (GPU fp32 run.py jobs with EVERY=1)."""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
TOP = ROOT / 'research/top15'
sys.path[:0] = [str(TOP / 'work2' / 'backbones'), str(TOP / 'work3' / 'integration')]
import run as R  # noqa: E402,F401  (puts ROOT and data_v4 on sys.path)
import fit_ft_fusion as F  # noqa: E402
from component_fusion import fuse  # noqa: E402
import decision_heads as dh  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get('S8_RUNS', '/root/s8_runs'))
ST, W4 = TOP / 'work2' / 'stacking', TOP / 'work4' / 'runs'
GATE, WEIGHTS = .002, (.25, .5, 1)
frozen = set(json.loads((ST / 'frozen_rows.json').read_text())['ids'])
BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
rows = {s: F.bank_rows(BANK, s, frozen) for s in ('val', 'test')}
train = [r for r in csv.DictReader((BANK / 'bank_trainval.csv').open()) if r['split'] == 'train' and r['ID'] in frozen]
P = {**json.loads((ST / 'presence_val.json').read_text()), **json.loads((ST / 'presence_test.json').read_text())}
M, silent, val_eer = {}, {}, {}


def load_every(name, path, keep_silent=False):
    r = json.loads(path.read_text())
    M[name] = {}
    for s in ('val', 'test'):
        M[name].update(zip(r['every'][s]['ids'], np.mean(r['every'][s]['preds'], 0).tolist()))
        if keep_silent:
            silent.update(zip(r['every'][s]['ids'], r['every'][s]['silent']))
    if 'val_ens_eer' in r:
        val_eer[name] = r['val_ens_eer']


load_every('df_L', W4 / 'voice_df__L.json', keep_silent=True)
load_every('ad_L', W4 / 'voice_ad__L.json')
load_every('wavlm8_cpu', W4 / 'voice_wavlm8__every.json')
load_every('music_s7_cpu', W4 / 'music_sslam__every.json')
for p in sorted(RUNS.glob('*_new__*.json')):
    load_every(p.stem.replace('__', ':'), p)
for n, c in M.items():
    missing = [r['ID'] for s in rows for r in rows[s] if r['ID'] not in c]
    assert not missing, f'{n}: {len(missing)} val/test rows without an every-row prediction'
ns = [r['ID'] for r in rows['val'] if not silent[r['ID']]]
tables = {n: dh.quantile_table([c[i] for i in ns]) for n, c in M.items()}
S7_VOICE, S7_MUSIC = {'df_L': 1, 'ad_L': .5, 'wavlm8_cpu': 1}, 'music_s7_cpu'
CONTROL = 'music_new:mert6_12+eat16+sslampt5'


def predictions(voice, music):
    first = next(iter(voice))
    key = lambda n: 'ours' if n == first else n
    spec = {'weights': {key(n): w for n, w in voice.items()}, 'tables': {key(n): tables[n] for n in voice}}
    files = {s: [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fm': M[music][r['ID']],
                  'Fv': 0. if silent[r['ID']] else fuse(spec, {key(n): M[n][r['ID']] for n in voice})}
                 for r in rows[s]] for s in ('val', 'test')}
    return F.file_preds(files, rows['val'])[2]


def val_ads(voice, music):
    return F.ads_of(rows['val'], predictions(voice, music)['val'])['ads']


def select():
    out = {'written': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'lock': 's8_lock.json', 'val_ens_eer': val_eer}
    enc = lambda prefix, name: sorted([n for n in val_eer if n.startswith(f'{prefix}:{name}') and n[len(prefix) + 1 + len(name):].isdigit()],
                                      key=lambda n: (val_eer[n], int(n.rsplit(name, 1)[1])))
    best = {name: enc('voice_new', name)[0] for name in ('hubert', 'd2v', 'usat', 'mixwavlm') if enc('voice_new', name)}
    best_block = enc('music_new', 'mixsslam')[0].split(':')[1] if enc('music_new', 'mixsslam') else None
    out['step_A'] = {'voice': best, 'music_mixsslam': best_block}
    base = val_ads(S7_VOICE, S7_MUSIC)
    out['S7_cpu_val_ads'] = base
    # step B voice: greedy, existing weights fixed, ties -> smaller w then encoder order (strict > keeps the earlier one)
    current, cur, trace, remaining = dict(S7_VOICE), base, [], [best[k] for k in ('hubert', 'd2v', 'usat', 'mixwavlm') if k in best]
    for _ in range(2):
        tried = [(val_ads({**current, n: w}, S7_MUSIC), n, w) for n in remaining for w in WEIGHTS]
        trace.append(tried)
        top = max(tried, key=lambda t: t[0]) if tried else None
        top = next(t for t in tried if t[0] == top[0]) if top else None   # first in (encoder order, w) among equal scores
        if not top or top[0] < cur + GATE:
            break
        cur, current = top[0], {**current, top[1]: top[2]}
        remaining.remove(top[1])
    out['step_B_voice'] = {'trace': trace, 'choice': current, 'val_ads': cur}
    # step B music (voice = S7): vs the GPU control of the S7 music recipe
    ctrl = val_ads(S7_VOICE, CONTROL)
    cands = [f'{CONTROL}+{best_block}', f'{CONTROL}+encodec', f'{CONTROL}+{best_block}+encodec'] if best_block else [f'{CONTROL}+encodec']
    scored = [(val_ads(S7_VOICE, c), c) for c in cands if c in M]
    top = next(t for t in scored if t[0] == max(s for s, _ in scored)) if scored else None
    music = top[1] if top and top[0] >= ctrl + GATE else None
    out['step_B_music'] = {'gpu_control_val_ads': ctrl, 'scored': scored, 'choice': music}
    # joint, then the better passing single change; every final choice must beat S7_cpu val by GATE
    options = []
    if current != S7_VOICE and music:
        options.append(('joint', current, music))
    options += [(n, v, m) for n, v, m in (('voice_only', current, S7_MUSIC), ('music_only', S7_VOICE, music)) if (v != S7_VOICE or m != S7_MUSIC) and m]
    evaluated = [(val_ads(v, m), n, v, m) for n, v, m in options]
    out['options'] = [(a, n) for a, n, _, _ in evaluated]
    final = next(((a, n, v, m) for a, n, v, m in evaluated if n == 'joint' and a >= base + GATE), None) or \
        max(((a, n, v, m) for a, n, v, m in evaluated if n != 'joint' and a >= base + GATE), default=None, key=lambda t: t[0])
    out['choice'] = None if final is None else {'name': final[1], 'val_ads': final[0], 'voice': final[2], 'music': final[3]}
    # exploratory alternative A1 (val only): WavLM h6+h8+h12 concat vs GPU h8 control, replacing the S7 member at w 1
    alt = {}
    for n in ('voice_new:wavlm8', 'voice_new:wavlm6_8_12'):
        if n in M:
            alt[n] = val_ads({'df_L': 1, 'ad_L': .5, n: 1}, S7_MUSIC)
    out['alternative_A1_val'] = alt
    path = HERE / 's8_choice.json'
    if path.exists() and json.loads(path.read_text())['choice'] != out['choice']:
        raise SystemExit(f'existing {path} has a different choice; refusing to overwrite')
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('step_A', 'S7_cpu_val_ads', 'step_B_music', 'options', 'choice', 'alternative_A1_val')}, indent=1))


def heldout():
    choice = json.loads((HERE / 's8_choice.json').read_text())['choice']
    assert choice, 'no S8 choice passed the val gates'
    test, lab = rows['test'], F.lab
    configs = {'S7_cpu': (S7_VOICE, S7_MUSIC), 'gpu_control': (S7_VOICE, CONTROL), 'S8': (choice['voice'], choice['music'])}
    preds = {n: predictions(v, m) for n, (v, m) in configs.items()}
    rep = {'choice': choice, 'test': {n: F.ads_of(test, p['test']) for n, p in preds.items()}}
    assert rep['test']['S7_cpu']['ads'] == .90921, rep['test']['S7_cpu']
    boot = F.Boot(test)
    rep['boot'] = {'S8 vs S7_cpu': boot.delta(preds['S7_cpu']['test'], preds['S8']['test']),
                   'S8 vs gpu_control': boot.delta(preds['gpu_control']['test'], preds['S8']['test']),
                   'gpu_control vs S7_cpu': boot.delta(preds['S7_cpu']['test'], preds['gpu_control']['test'])}
    gen = lambda r: r['generator'].strip().lower()
    train_gens = {gen(r) for r in train if r['FILE_FAKE'] == '1'}
    keep = [i for i, r in enumerate(test) if lab(r, 'FILE_FAKE') == 0 or gen(r) not in train_gens]
    sub = [test[i] for i in keep]
    rep['unseen_gen'] = {n: F.ads_of(sub, [p['test'][i] for i in keep]) for n, p in preds.items()}

    def fpr_guard(comp, prob, keep_row):
        """FPR on the single-class mixture slice at each model's val TPR-95% threshold of that component."""
        res = {}
        for n in ('S7_cpu', 'S8'):
            vy = [(lab(r, f'{comp}_FAKE'), p[prob]) for r, p in zip(rows['val'], preds[n]['val'])
                  if lab(r, f'{comp}_PRESENT') == 1 and lab(r, f'{comp}_FAKE') is not None]
            thr = float(np.quantile([s for y, s in vy if y == 1], .05))
            sl = [p[prob] for r, p in zip(test, preds[n]['test']) if keep_row(r)]
            res[n] = {'threshold': thr, 'n': len(sl), 'fpr': float(np.mean([s >= thr for s in sl]))}
        res['increase'] = res['S8']['fpr'] - res['S7_cpu']['fpr']
        return res
    rep['realV_fakeM_voice_fpr'] = fpr_guard('VOICE', 'VOICE_FAKE_PROB', lambda r: lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') == 0
                                             and lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') == 1)
    rep['fakeV_realM_music_fpr'] = fpr_guard('MUSIC', 'MUSIC_FAKE_PROB', lambda r: lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') == 0
                                             and lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') == 1)
    uses_mix_voice = any(n.startswith('voice_new:mixwavlm') for n in choice['voice'])
    uses_mix_music = 'mixsslam' in (choice['music'] or '')
    d = rep['boot']['S8 vs S7_cpu']
    rep['gates'] = {'dads>=.002': d['delta'] >= .002, 'p_gt_0>=.90': d['p_gt_0'] >= .9,
                    'unseen_dads>=-.002': rep['unseen_gen']['S8']['ads'] - rep['unseen_gen']['S7_cpu']['ads'] >= -.002,
                    'mix_voice_fpr': (not uses_mix_voice) or rep['realV_fakeM_voice_fpr']['increase'] <= .03,
                    'mix_music_fpr': (not uses_mix_music) or rep['fakeV_realM_music_fpr']['increase'] <= .03}
    rep['gates'] = {k: bool(v) for k, v in rep['gates'].items()}
    rep['ship'] = all(rep['gates'].values())
    (HERE / 's8_heldout.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    {'select': select, 'heldout': heldout}[sys.argv[1]]()
