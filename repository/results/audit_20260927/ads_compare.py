"""Official ADS on backbone-cache rows (DB protocol: heads trained on the train split only, 3 seeds, seed-mean probs):
baseline (DF-Arena voice + MERT-12 music) vs music MERT-6+12, plus voice DF-Arena+AD-head CDF fusion. FILE ensemble
refitted on val for every configuration (file_ensemble.fit); presence from the shared presence heads."""
import csv, json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'data_v4'); sys.path.insert(0, '.')
import train_backbone_heads as H
import full_bank_train as T
import decision_heads as dh
from component_fusion import fuse
from file_ensemble import fit as fit_file
from full_bank_evaluate import score_rows
from graph_scoring import official_metrics
from safetensors.torch import load_file
cache = Path('_workspace/backbone_cache')
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and (cache / f"{r['ID']}.npz").exists()]
        for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
print({s: len(v) for s, v in rows.items()}, flush=True)
pres = json.loads(H.PRESENCE.read_text())
P = {}   # presence per eval row, via the old cache (heads dir only needed for its own Fv/Fm, unused here)
for s in ('val', 'test'):
    for r, x in zip(rows[s], score_rows(rows[s], H.OLD[s], Path('_keep_best_20260926/train_duration_balanced'), 'cpu')):
        P[r['ID']] = dh.presence(pres, x['presence_features'], (x['Pv'], x['Pm']))
old_init = torch.load(H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
native = load_file(str(H.AD / 'model.safetensors'))
ad_init = {'weight': native['proj_fc.weight'].float(), 'bias': native['proj_fc.bias'].float()}
def seed_mean(route, layers, dfarena, init=None, fi=0):
    data = {s: H.items_for(rows[s], cache, layers, route, dfarena=dfarena) for s in rows}
    out = {}
    for seed in (41, 42, 43):
        model, _, _ = H.train(route, data['train'], data['val'], init, fi, seed=seed)
        for s in ('val', 'test'):
            p = T.predict(model, [x for x, _ in data[s]], route, fi, 256, 10 ** 6, 'cpu')
            for (x, _), v in zip(data[s], p):
                out.setdefault(x.row['ID'], []).append(float(v))
    return {k: float(np.mean(v)) for k, v in out.items()}
comp = {'voice_df': seed_mean('voice', [12], True, old_init['state_dict'], old_init['fake_label_index']),
        'voice_ad': seed_mean('voice', [12], False, ad_init, 0),
        'music_12': seed_mean('music', [12], False), 'music_6_12': seed_mean('music', [6, 12], False)}
json.dump(comp, open('_workspace/audit_20260927/ads_compare_scores.json', 'w'))
def evaluate(vkey, mkey, vspec=None):
    def files(s):
        out = []
        for r in rows[s]:
            vp, mp = P[r['ID']]
            fv = comp['voice_df'].get(r['ID'], 0.) if vspec is None else fuse(vspec, {'ours': comp['voice_df'].get(r['ID'], 0.), 'ad': comp['voice_ad'].get(r['ID'], 0.)})
            out.append({'ID': r['ID'], 'Pv': vp, 'Pm': mp, 'Fv': fv, 'Fm': comp[mkey].get(r['ID'], 0.)})
        return out
    fv = files('val'); keep = [i for i, r in enumerate(rows['val']) if r['FILE_FAKE_mask'] == '1']
    ens, _ = fit_file([fv[i] for i in keep], np.array([int(rows['val'][i]['FILE_FAKE']) for i in keep]))
    res = {}
    for s in ('val', 'test'):
        fs = files(s)
        preds = [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'],
                  'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in fs]
        m = official_metrics(rows[s], preds)['metrics']
        res[s] = {k: round(m[k], 4) for k in ('file_eer', 'voice_eer', 'music_eer')} | {'ads': round(.5*(1-m['file_eer'])+.2*(1-m['voice_eer'])+.3*(1-m['music_eer']), 5)}
    return res
vtab = lambda w: {'weights': {'ours': 1, 'ad': w}, 'tables': {'ours': dh.quantile_table([comp['voice_df'][r['ID']] for r in rows['val'] if r['ID'] in comp['voice_df']]),
                                                             'ad': dh.quantile_table([comp['voice_ad'][r['ID']] for r in rows['val'] if r['ID'] in comp['voice_ad']])}}
report = {'baseline (DF-Arena voice, MERT-12 music)': evaluate('voice_df', 'music_12'),
          'music MERT-6+12': evaluate('voice_df', 'music_6_12'),
          'music MERT-6+12 + voice fusion AD-head w=0.5': evaluate('voice_df', 'music_6_12', vtab(.5)),
          'music MERT-6+12 + voice fusion AD-head w=1': evaluate('voice_df', 'music_6_12', vtab(1.))}
for k, v in report.items(): print(k, v, flush=True)
json.dump(report, open('_workspace/audit_20260927/ads_compare.json', 'w'), indent=1)
