"""Does the S3-style gain hold on rows whose source/parents were NOT in AntiDeepfake/DF-Arena/SONICS training data?
Same scores as _workspace/audit_20260927/ads_compare_scores.json (DB protocol, 3 seeds); FILE refit on the same val subset."""
import csv, json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, 'data_v4'); sys.path.insert(0, '.')
import decision_heads as dh
from component_fusion import fuse
from file_ensemble import fit as fit_file
from full_bank_evaluate import score_rows
from graph_scoring import official_metrics
from fusion_fit import member_seen
B = Path('results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'); cache = Path('_workspace/backbone_cache')
comp = json.load(open('_workspace/audit_20260927/ads_compare_scores.json'))
pres = json.load(open('results/data_v4_20260923/data_v4_results_20260923T1639Z/train/presence_head.json'))
OLD = {'val': Path('_workspace/cache_v4/cache_tv'), 'test': Path('_workspace/cache_v4/cache_test')}
for subset in ('all', 'unseen', 'seen'):
    rows = {s: [r for r in csv.DictReader((B / f).open()) if r['split'] == s and (cache / f"{r['ID']}.npz").exists()
                and (subset == 'all' or (member_seen(r) == (subset == 'seen')))] for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
    P = {}
    for s in rows:
        for r, x in zip(rows[s], score_rows(rows[s], OLD[s], Path('_keep_best_20260926/train_duration_balanced'), 'cpu')):
            P[r['ID']] = dh.presence(pres, x['presence_features'], (x['Pv'], x['Pm']))
    vv = [r['ID'] for r in rows['val'] if comp['voice_df'].get(r['ID'], 0) > 0]
    spec = {'weights': {'ours': 1, 'ad': .5}, 'tables': {'ours': dh.quantile_table([comp['voice_df'][i] for i in vv]), 'ad': dh.quantile_table([comp['voice_ad'].get(i, 0.) for i in vv])}}
    def ev(mkey, fused):
        def files(s):
            out = []
            for r in rows[s]:
                fv = comp['voice_df'].get(r['ID'], 0.)
                if fused and fv > 0: fv = fuse(spec, {'ours': fv, 'ad': comp['voice_ad'].get(r['ID'], 0.)})
                vp, mp = P[r['ID']]; out.append({'ID': r['ID'], 'Pv': vp, 'Pm': mp, 'Fv': fv, 'Fm': comp[mkey].get(r['ID'], 0.)})
            return out
        fv = files('val'); k = [i for i, r in enumerate(rows['val']) if r['FILE_FAKE_mask'] == '1']
        ens, _ = fit_file([fv[i] for i in k], np.array([int(rows['val'][i]['FILE_FAKE']) for i in k]))
        res = {}
        for s in rows:
            fs = files(s)
            m = official_metrics(rows[s], [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'], 'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in fs])['metrics']
            res[s] = round(.5*(1-m['file_eer'])+.2*(1-m['voice_eer'])+.3*(1-m['music_eer']), 5)
        return res
    b, s3 = ev('music_12', False), ev('music_6_12', True)
    print(subset, {k: len(v) for k, v in rows.items()}, 'baseline', b, 'S3-style', s3, 'delta', {k: round(s3[k]-b[k], 5) for k in b}, flush=True)
