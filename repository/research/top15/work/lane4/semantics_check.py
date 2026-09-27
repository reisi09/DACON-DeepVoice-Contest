"""Lane 4 check S (val, final S3 heads, CPU, read-only).
Q5: the held-out screen (_workspace/audit_20260927/ads_compare.py) scored Fv only for rows with a voice label and Fm only for
    rows with a music label (0 otherwise), while the runtime scores both stems of every file. How much does that change
    FILE EER when the FILE ensemble is refit (as the screen did)?  (in-sample on val; magnitude only)
Q6: max-over-windows voice pooling vs duration: Spearman(Fv, #windows | real, non-silent voice) for ours / AD / fused, and
    the same for music log-mean-exp.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

R = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'data_v4'))
sys.path.insert(0, str(R / 'research/top15/work/lane4'))
import decision_heads as dh  # noqa: E402
from file_ensemble import fit as fit_file, eer  # noqa: E402
from graph_scoring import official_metrics  # noqa: E402
exec(open(R / 'research/top15/work/lane4/cache_check.py').read().split('# ---------------- Q1 coverage')[0])  # imports + rows/have
exec(open(R / 'research/top15/work/lane4/cache_check.py').read().split('# ---------------- Q3 val reproduction with packaged heads')[1].split('base = [score(r) for r in val]')[0])
base = [score(r) for r in val]
out = {}
# runtime semantics (reproduced S3): already 0.054381 with the packaged ensemble; refit here for a like-for-like comparison
def files(sem):
    res = []
    for r, s in zip(val, base):
        fv = s['ours'] if s['ours'] == 0. else fuse(spec, {'ours': s['ours'], 'ad': s['ad']})
        fm = s['Fm']
        if sem == 'screen':
            fv = fv if (r['route'] == 'voice' or (r['VOICE_FAKE'] in ('0', '1') and r['VOICE_PRESENT'] == '1')) else 0.
            fm = fm if (r['route'] == 'music' or (r['MUSIC_FAKE'] in ('0', '1') and r['MUSIC_PRESENT'] == '1')) else 0.
        res.append(dict(ID=r['ID'], Pv=s['Pv'], Pm=s['Pm'], Fv=fv, Fm=fm))
    return res
keep = [i for i, r in enumerate(val) if r['FILE_FAKE_mask'] == '1']
y = np.array([int(val[i]['FILE_FAKE']) for i in keep])
for sem in ('runtime', 'screen'):
    fs = files(sem)
    e, rep = fit_file([fs[i] for i in keep], y)
    out[f'file_eer_refit_{sem}'] = round(eer(y, [dh.file_score(e, fs[i]) for i in keep]), 5)
    out[f'chosen_{sem}'] = rep['chosen_weights']
    out[f'candidates_alone_{sem}'] = rep['candidates_alone']
# Q6 duration dependence
nwin = {}
for r in val:
    with np.load(OLD['val'] / f"{r['ID']}.npz") as o:
        nwin[r['ID']] = (len(o['voice_embeddings']), len(o['mert_windows']), int(o['samples']))
def rho(sel, key):
    a = [s[key] if key != 'fused' else (s['ours'] if s['ours'] == 0 else fuse(spec, {'ours': s['ours'], 'ad': s['ad']})) for s in sel]
    return round(float(spearmanr(a, [nwin[s['ID']][2] for s in sel])[0]), 3)
rv = [s for r, s in zip(val, base) if r['VOICE_FAKE'] == '0' and r['VOICE_PRESENT'] == '1' and s['ours'] > 0]
fvv = [s for r, s in zip(val, base) if r['VOICE_FAKE'] == '1' and r['VOICE_PRESENT'] == '1' and s['ours'] > 0]
rm = [s for r, s in zip(val, base) if r['MUSIC_FAKE'] == '0' and r['MUSIC_PRESENT'] == '1']
out['spearman_score_vs_samples_real_voice'] = {'n': len(rv), 'ours_max': rho(rv, 'ours'), 'ad_max': rho(rv, 'ad'), 'fused': rho(rv, 'fused')}
out['spearman_score_vs_samples_fake_voice'] = {'n': len(fvv), 'ours_max': rho(fvv, 'ours'), 'ad_max': rho(fvv, 'ad'), 'fused': rho(fvv, 'fused')}
out['spearman_score_vs_samples_real_music_LME'] = {'n': len(rm), 'Fm': rho(rm, 'Fm')}
d = np.array([nwin[r['ID']][2] / 16000 for r in val])
out['val_duration_s'] = {'p50': round(float(np.median(d)), 2), 'p90': round(float(np.quantile(d, .9)), 2), 'max': round(float(d.max()), 2),
                         'frac_gt_30s': round(float((d > 30).mean()), 4)}
json.dump(out, sys.stdout, indent=1); print()
(R / 'research/top15/work/lane4/semantics_check.out.json').write_text(json.dumps(out, indent=1) + '\n')
