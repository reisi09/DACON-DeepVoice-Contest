"""Official ADS (val and held-out) for val-chosen R2 heads, with the exact ads_compare.py pipeline
(_workspace/audit_20260927/ads_compare.py: shared presence heads, FILE ensemble refitted on val, voice CDF fusion ours 1 : ad w).
Components come from R2 run files (3-seed mean) or from ads_compare_scores.json. Also checks that R2 'base' runs reproduce
ads_compare_scores.json.   python ads.py"""
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
cache = ROOT / '_workspace/backbone_cache'
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and (cache / f"{r['ID']}.npz").exists()]
        for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
comp = json.loads((ROOT / '_workspace/audit_20260927/ads_compare_scores.json').read_text())


def run_scores(name):
    r = json.loads((HERE / 'runs' / f'{name}.json').read_text())
    out = dict(zip(r['val_ids'], np.mean(r['val_preds'], 0).tolist()))
    out.update(zip(r['test_ids'], np.mean(r['test_preds'], 0).tolist()))
    return out


repro = {}
for key, name in (('voice_df', 'voice_df__base'), ('voice_ad', 'voice_ad__base'), ('music_6_12', 'music__L6_12')):
    mine = run_scores(name)
    repro[key] = max(abs(mine[i] - comp[key][i]) for i in comp[key])
print('max |R2 base - ads_compare| per component', repro, flush=True)

lock = json.loads((HERE / 'choice_lock.json').read_text())['choice']
for g, key in (('voice_df', 'voice_df'), ('voice_ad', 'voice_ad'), ('music', 'music_6_12')):
    comp[f'{key}_r2'] = run_scores(f'{g}__{lock[g]}')

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
            # clip: quantile_table adds i*1e-12, so a head saturating at 1.0 maps to 1+1e-9 (runtime would raise)
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
            'ads': round(.5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer']), 5)}
    return res


report = {'reproduction_max_abs_diff': repro, 'choice': lock,
          'S3-like baseline (ads_compare components)': evaluate('voice_df', 'voice_ad', 'music_6_12'),
          'R2 chosen voice_df': evaluate('voice_df_r2', 'voice_ad', 'music_6_12'),
          'R2 chosen voice_ad': evaluate('voice_df', 'voice_ad_r2', 'music_6_12'),
          'R2 chosen music': evaluate('voice_df', 'voice_ad', 'music_6_12_r2'),
          'R2 all chosen': evaluate('voice_df_r2', 'voice_ad_r2', 'music_6_12_r2')}
for k, v in report.items():
    print(k, v, flush=True)
(HERE / 'ads_report.json').write_text(json.dumps(report, indent=1))
(HERE / 'ads_test_preds.json').write_text(json.dumps(PRED))
