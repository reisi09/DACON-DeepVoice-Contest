"""Question: does S3-style voice fusion (DF-Arena heads + AntiDeepfake-embedding heads, val-CDF 1:0.5, DB protocol,
component scores from _workspace/audit_20260927/ads_compare_scores.json) still miss sung AI vocals (HAIM) as often as
the DB heads (40% below the pooled voice EER threshold)? Only the VOICE stem score matters for an AI cover. CPU, seconds."""
import csv, json, sys, numpy as np
from sklearn.metrics import roc_curve
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
sys.path.insert(0, R)
import decision_heads as dh
from component_fusion import fuse
comp = json.load(open(R + '_workspace/audit_20260927/ads_compare_scores.json'))
B = R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
rows = {s: [r for r in csv.DictReader(open(B + f)) if r['split'] == s and r['ID'] in comp['voice_ad']]
        for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
vv = [r['ID'] for r in rows['val'] if comp['voice_df'].get(r['ID'], 0) > 0]
spec = {'weights': {'ours': 1, 'ad': .5}, 'tables': {'ours': dh.quantile_table([comp['voice_df'][i] for i in vv]),
                                                   'ad': dh.quantile_table([comp['voice_ad'].get(i, 0.) for i in vv])}}
t = [r for r in rows['test'] if r['VOICE_PRESENT'] == '1' and r['VOICE_FAKE_mask'] == '1']
y = np.array([int(float(r['VOICE_FAKE'])) for r in t])
out = {}
for name, sc in (('db_df_only', [comp['voice_df'][r['ID']] for r in t]),
                 ('s3_fused', [fuse(spec, {'ours': comp['voice_df'][r['ID']], 'ad': comp['voice_ad'][r['ID']]}) if comp['voice_df'][r['ID']] > 0 else 0. for r in t])):
    s = np.array(sc); fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False); i = np.argmin(np.abs(fpr - 1 + tpr))
    hs = np.array([r['source'] == 'haim' and r['MUSIC_FAKE'] == '1' for r in t])
    rm = np.array([r['MUSIC_PRESENT'] == '1' and y[k] == 0 for k, r in enumerate(t)])
    out[name] = {'voice_eer': round(float((fpr[i] + 1 - tpr[i]) / 2), 5), 'n': len(t),
                 'haim_sung_fake_miss': [round(float((s[hs] < thr[i]).mean()), 4), int(hs.sum())],
                 'real_voice_in_music_FP': [round(float((s[rm] >= thr[i]).mean()), 4), int(rm.sum())]}
print(json.dumps(out, indent=1))
