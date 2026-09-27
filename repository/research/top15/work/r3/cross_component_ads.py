"""Q: what is the held-out OFFICIAL ADS effect (FILE refit on val, runtime presence heads, not oracle labels) of adding
the file's own music member to VOICE fusion (song-aware VOICE), vs the S3 configuration (MERT-6+12 music, VOICE =
CDF-mean(DF-Arena heads 1, AD heads 0.5))?  Reuses the screen's DB-protocol member scores
(_workspace/audit_20260927/ads_compare_scores.json) and replicates ads_compare.py evaluate() exactly, adding a gated
member: VOICE = CDF-mean(ours 1, ad a, music b*g), g = runtime music presence Pm (soft gate) or 1[Pm>=.5] (hard gate).
(a, b, gate) chosen on VAL voice EER; FILE ensemble refit on val per configuration (file_ensemble.fit), as in the screen.
Read-only; CPU (reads ~10k cached npz for presence)."""
import csv, json, sys, itertools, numpy as np
from pathlib import Path
R = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘'); sys.path[:0] = [str(R / 'data_v4'), str(R)]
import os; os.chdir(R)
import train_backbone_heads as H
import decision_heads as dh
from file_ensemble import fit as fit_file
from full_bank_evaluate import score_rows
from graph_scoring import official_metrics
cache = Path('_workspace/backbone_cache')
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and (cache / f"{r['ID']}.npz").exists()]
        for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
pres = json.loads(H.PRESENCE.read_text()); P = {}
for s in ('val', 'test'):
    for r, x in zip(rows[s], score_rows(rows[s], H.OLD[s], Path('_keep_best_20260926/train_duration_balanced'), 'cpu')):
        P[r['ID']] = dh.presence(pres, x['presence_features'], (x['Pv'], x['Pm']))
comp = json.load(open('_workspace/audit_20260927/ads_compare_scores.json'))
tab = {k: dh.quantile_table([comp[m][r['ID']] for r in rows['val'] if r['ID'] in comp[m]])
       for k, m in (('ours', 'voice_df'), ('ad', 'voice_ad'), ('mu', 'music_6_12'))}
def fv(i, a, b, gate):
    vals = {'ours': comp['voice_df'].get(i, 0.), 'ad': comp['voice_ad'].get(i, 0.)}
    ws = {'ours': 1., 'ad': a}
    if b and i in comp['music_6_12']:
        pm = P[i][1]; g = pm if gate == 'soft' else float(pm >= .5)
        vals['mu'] = comp['music_6_12'][i]; ws['mu'] = b * g
    c = sum(w * dh.to_cdf(tab[n], vals[n]) for n, w in ws.items() if w) / sum(w for w in ws.values() if w)
    return float(np.interp(c, np.linspace(0, 1, len(tab['ours'])), tab['ours']))
def files(s, a, b, gate):
    return [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fv': fv(r['ID'], a, b, gate),
             'Fm': comp['music_6_12'].get(r['ID'], 0.)} for r in rows[s]]
def evaluate(a, b, gate):
    f = files('val', a, b, gate); keep = [i for i, r in enumerate(rows['val']) if r['FILE_FAKE_mask'] == '1']
    ens, _ = fit_file([f[i] for i in keep], np.array([int(rows['val'][i]['FILE_FAKE']) for i in keep]))
    out = {}
    for s in ('val', 'test'):
        fs = f if s == 'val' else files(s, a, b, gate)
        preds = [{'ID': x['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, x), 'VOICE_FAKE_PROB': x['Fv'], 'MUSIC_FAKE_PROB': x['Fm'],
                  'VOICE_PRESENT_PROB': x['Pv'], 'MUSIC_PRESENT_PROB': x['Pm']} for x in fs]
        m = official_metrics(rows[s], preds)['metrics']
        out[s] = {k: round(m[k], 4) for k in ('file_eer', 'voice_eer', 'music_eer')} | {'ads': round(.5*(1-m['file_eer'])+.2*(1-m['voice_eer'])+.3*(1-m['music_eer']), 5)}
    return out
def val_voice(a, b, gate):   # selection criterion: val voice EER only (no FILE refit needed)
    fs = files('val', a, b, gate)
    preds = [{'ID': x['ID'], 'FILE_FAKE_PROB': 0., 'VOICE_FAKE_PROB': x['Fv'], 'MUSIC_FAKE_PROB': x['Fm'],
              'VOICE_PRESENT_PROB': x['Pv'], 'MUSIC_PRESENT_PROB': x['Pm']} for x in fs]
    return official_metrics(rows['val'], preds)['metrics']['voice_eer']
grid = [(a, b, g) for a, b in itertools.product([.5, 1], [.25, .5, 1, 2]) for g in ('soft', 'hard')]
sel = {k: val_voice(*k) for k in grid}; best = min(sel, key=sel.get)
print('val voice EER top-5:', sorted(((round(v, 4), k) for k, v in sel.items()))[:5], flush=True)
rep = {'S3 config (a=.5, no music member)': evaluate(.5, 0, 'soft'), f'song-aware VOICE val-best {best}': evaluate(*best)}
for k, v in rep.items(): print(k, v, flush=True)
json.dump({'selection_val_voice_eer': {str(k): v for k, v in sel.items()}, 'report': rep},
          open('research/top15/work/r3/cross_component_ads.json', 'w'), indent=1)

# Variant: song-aware Fv only for the VOICE_FAKE_PROB output; FILE rules keep the S3 Fv (decoupled), so FILE is unchanged.
def evaluate_decoupled(a, b, gate):
    base_v, base_t = files('val', .5, 0, 'soft'), files('test', .5, 0, 'soft')
    keep = [i for i, r in enumerate(rows['val']) if r['FILE_FAKE_mask'] == '1']
    ens, _ = fit_file([base_v[i] for i in keep], np.array([int(rows['val'][i]['FILE_FAKE']) for i in keep]))
    out = {}
    for s, base in (('val', base_v), ('test', base_t)):
        new = {x['ID']: x['Fv'] for x in files(s, a, b, gate)}
        preds = [{'ID': x['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, x), 'VOICE_FAKE_PROB': new[x['ID']], 'MUSIC_FAKE_PROB': x['Fm'],
                  'VOICE_PRESENT_PROB': x['Pv'], 'MUSIC_PRESENT_PROB': x['Pm']} for x in base]
        m = official_metrics(rows[s], preds)['metrics']
        out[s] = {k: round(m[k], 4) for k in ('file_eer', 'voice_eer', 'music_eer')} | {'ads': round(.5*(1-m['file_eer'])+.2*(1-m['voice_eer'])+.3*(1-m['music_eer']), 5)}
    return out
r2 = evaluate_decoupled(*best); print(f'song-aware VOICE output only, FILE decoupled {best}', r2, flush=True)
rep['song-aware VOICE output only (FILE keeps S3 Fv)'] = r2
json.dump({'selection_val_voice_eer': {str(k): v for k, v in sel.items()}, 'report': rep},
          open('research/top15/work/r3/cross_component_ads.json', 'w'), indent=1)
