"""Write choice_lock.json (rules + val-only choices) BEFORE any held-out number is computed. Refuses to overwrite.
    python lock.py      (after `python ads.py val` and the pilot fits)"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
out = HERE / 'choice_lock.json'
assert not out.exists(), 'lock already written'
V = json.loads((HERE / 'val_report.json').read_text())
ads = lambda k: V[k]['val_official']['ads']
val = {'full': ads('full')}
for f in (25, 50, 75):
    d = [ads(f'D{f}_d{i}') for i in (1, 2, 3)]
    val[f'{f}%'] = {'Q': ads(f'Q{f}_d0'), 'D_draws': d, 'D_mean': round(float(np.mean(d)), 5),
                    'Q_minus_Dmean': round(ads(f'Q{f}_d0') - float(np.mean(d)), 5), 'full_minus_Q': round(ads('full') - ads(f'Q{f}_d0'), 5)}
lofo = {k[5:]: {'LOFO_val_ads': ads(k), 'CTRL_val_ads': ads('CTRL_' + k[5:]),
                'LOFO_val_route_eer': V[k]['val_route_eer'], 'CTRL_val_route_eer': V['CTRL_' + k[5:]]['val_route_eer']}
        for k in V if k.startswith('LOFO_')}
pilot = {}
for g, base, cand in (('music', 'pilot_music__base', 'pilot_music__plus_arena'), ('voice', 'voice_df__full', 'pilot_voice__plus_arena')):
    a, b = (json.loads((HERE / 'runs' / f'{n}.json').read_text()) for n in (base, cand))
    sa, sb = np.mean([s['val_eer'] for s in a['seeds'].values()]), np.mean([s['val_eer'] for s in b['seeds'].values()])
    better = a['val_ens_eer'] - b['val_ens_eer'] >= .002 and sb < sa
    pilot[g] = {'base': base, 'candidate': cand, 'val_ens_eer': [round(a['val_ens_eer'], 5), round(b['val_ens_eer'], 5)],
                'val_seed_mean_eer': [round(float(sa), 5), round(float(sb), 5)], 'choice': 'plus_arena' if better else 'base'}
lock = {
    'written': time.strftime('%Y-%m-%d %H:%M:%S KST'),
    'protocol': 'DB: R2 val-chosen head designs (voice = DF-Arena fc5 MLP mlp_long alone, music = MERT 6+12 WindowMIL) trained '
                'on train subsets of the frozen R2 row universe (31,270 backbone rows); full val for early stopping, FILE refit '
                'and CDF tables; 2 seeds (41, 42) averaged; held-out test scored once after this lock; group bootstrap 1000 reps '
                'over test group_id with models fixed.',
    'rules': {
        'exp1': 'Primary contrast = 50% rows: diversity is a limiter iff held-out ADS(Q50) - mean over 3 draws ADS(D50) > 0 '
                'with 95% group CI excluding 0; same test at 25%/75% reported as secondary. Quantity is a limiter at f iff '
                'ADS(full) - ADS(Qf) > 0 with CI excluding 0. Val numbers are the locked expectation; held-out can confirm '
                'or contradict them but nothing is re-selected on held-out.',
        'exp2': 'A family depends on its nearest train relatives iff error(LOFO) - error(CTRL) on the family error class '
                '(miss for fakes, FPR for reals, at each head\'s own held-out route-EER threshold) >= +0.05 absolute with the '
                'family-group bootstrap CI excluding 0. Relatives fixed in relatives.json before any held-out prediction was read.',
        'exp3': 'Held-out error concentrates on unseen generators iff miss(unseen) - miss(seen) > 0 with group CI excluding 0 '
                '(per component, 100% heads).',
        'pilot': 'R2 rule: +music_arena chosen iff val route ens EER improves by >= 0.002 AND the per-seed mean val EER improves; '
                 'else base. Held-out reported regardless of the choice.'},
    'val': {'exp1_ads': val, 'exp2': lofo},
    'choice': {'exp1_val_verdict': {f: ('diversity costs more than quantity' if val[f]['Q_minus_Dmean'] > .002 else 'no difference on val')
                                    for f in ('25%', '50%', '75%')},
               'pilot': {g: v['choice'] for g, v in pilot.items()}, 'pilot_detail': pilot},
}
out.write_text(json.dumps(lock, indent=1))
print(json.dumps(lock, indent=1))
