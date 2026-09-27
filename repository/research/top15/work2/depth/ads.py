"""Official ADS (val and held-out) of the depth-study val-chosen heads with the exact r2/ads.py (= ads_compare.py) pipeline:
shared presence heads, FILE ensemble refitted on val, voice CDF fusion ours 1 : ad 0.5, rows = R2's frozen 31,270 backbone rows.
Configurations: A = R2 both-MLP (r2 runs, must reproduce 0.88643), B = depth all-row reference (DF mlp256 on all rows,
AD resmlp, music mil128_r2), C = depth chosen, and one-component swaps of C into B.   python ads.py  (needs choice_lock.json)"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'data_v4'))
import train_backbone_heads as H  # noqa: E402
import decision_heads as dh  # noqa: E402
from component_fusion import fuse  # noqa: E402
from file_ensemble import fit as fit_file  # noqa: E402
from full_bank_evaluate import score_rows  # noqa: E402
from graph_scoring import official_metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
R2 = HERE.parents[1] / 'work/r2/runs'
frozen = set(json.loads((HERE / 'rows_r2.json').read_text())['ids'])
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and r['ID'] in frozen]
        for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}


def run_scores(path):
    r = json.loads(path.read_text())
    out = dict(zip(r['val_ids'], np.mean(r['val_preds'], 0).tolist()))
    out.update(zip(r['test_ids'], np.mean(r['test_preds'], 0).tolist()))
    return out


lock = json.loads((HERE / 'choice_lock.json').read_text())['choice']
comp = {'r2_df': run_scores(R2 / 'voice_df__mlp_long.json'), 'r2_ad': run_scores(R2 / 'voice_ad__resmlp_long.json'),
        'r2_mu': run_scores(R2 / 'music__L6_12.json'),
        'ref_df': run_scores(HERE / 'runs/fullvoice_db__mlp256.json'), 'ref_ad': run_scores(HERE / 'runs/voice_ad__resmlp.json'),
        'ref_mu': run_scores(HERE / 'runs/music__mil128_r2.json'),
        'ch_df': run_scores(HERE / f"runs/fullvoice_db__{lock['fullvoice_db']}.json"),
        'ch_ad': run_scores(HERE / f"runs/voice_ad__{lock['voice_ad']}.json"), 'ch_mu': run_scores(HERE / f"runs/music__{lock['music']}.json")}
r2full = run_scores(R2 / 'fullvoice_db__mlp_long.json')
repro = {'ref_df vs r2 fullvoice_db mlp_long': max(abs(comp['ref_df'][i] - r2full[i]) for i in r2full),
         'ref_ad vs r2 voice_ad resmlp_long': max(abs(comp['ref_ad'][i] - comp['r2_ad'][i]) for i in comp['r2_ad']),
         'ref_mu vs r2 music L6_12': max(abs(comp['ref_mu'][i] - comp['r2_mu'][i]) for i in comp['r2_mu'])}
print('reproduction max |diff|', repro, flush=True)

pres = json.loads(H.PRESENCE.read_text())
P = {}
for s in ('val', 'test'):
    for r, x in zip(rows[s], score_rows(rows[s], H.OLD[s], ROOT / '_keep_best_20260926/train_duration_balanced', 'cpu')):
        P[r['ID']] = dh.presence(pres, x['presence_features'], (x['Pv'], x['Pm']))

PRED = {}


def evaluate(vdf, vad, mkey, w=.5):
    spec = {'weights': {'ours': 1, 'ad': w}, 'tables': {
        'ours': dh.quantile_table([comp[vdf][r['ID']] for r in rows['val'] if r['ID'] in comp[vdf]]),
        'ad': dh.quantile_table([comp[vad][r['ID']] for r in rows['val'] if r['ID'] in comp[vad]])}}

    def files(s):
        out = []
        for r in rows[s]:
            vp, mp = P[r['ID']]
            fv = min(1., fuse(spec, {'ours': comp[vdf].get(r['ID'], 0.), 'ad': comp[vad].get(r['ID'], 0.)}))
            out.append({'ID': r['ID'], 'Pv': vp, 'Pm': mp, 'Fv': fv, 'Fm': comp[mkey].get(r['ID'], 0.)})
        return out
    fv = files('val')
    keep = [i for i, r in enumerate(rows['val']) if r['FILE_FAKE_mask'] == '1']
    ens, _ = fit_file([fv[i] for i in keep], np.array([int(rows['val'][i]['FILE_FAKE']) for i in keep]))
    res = {}
    for s in ('val', 'test'):
        preds = [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'],
                  'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in files(s)]
        if s == 'test':
            PRED[f'{vdf}|{vad}|{mkey}'] = preds
        m = official_metrics(rows[s], preds)['metrics']
        res[s] = {k: round(m[k], 4) for k in ('file_eer', 'voice_eer', 'music_eer')} | {
            'ads': round(.5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer']), 5),
            'n_exact_1': {c: sum(p[c] == 1. for p in preds) for c in ('FILE_FAKE_PROB', 'VOICE_FAKE_PROB', 'MUSIC_FAKE_PROB')}}
    return res


configs = {'A R2 both-MLP': ('r2_df', 'r2_ad', 'r2_mu'), 'B depth all-row reference': ('ref_df', 'ref_ad', 'ref_mu'),
           'C depth chosen': ('ch_df', 'ch_ad', 'ch_mu')}
for i, (g, k) in enumerate((('fullvoice_db', 'df'), ('voice_ad', 'ad'), ('music', 'mu'))):
    if lock[g] != {'fullvoice_db': 'mlp256', 'voice_ad': 'resmlp', 'music': 'mil128_r2'}[g]:
        c = list(configs['B depth all-row reference'])
        c[i] = f'ch_{k}'
        configs[f'B + chosen {g} ({lock[g]})'] = tuple(c)
report = {'reproduction_max_abs_diff': repro, 'choice': lock, 'n_rows': {s: len(v) for s, v in rows.items()}, 'configs': configs}
for name, c in configs.items():
    report[name] = evaluate(*c)
    print(name, c, report[name], flush=True)
(HERE / 'ads_report.json').write_text(json.dumps(report, indent=1))
(HERE / 'ads_test_preds.json').write_text(json.dumps(PRED))
