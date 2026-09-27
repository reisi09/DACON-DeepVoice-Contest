"""Question: can a plausible official-test composition (voice/music/mixed x real/fake, incl. discordant mixes)
turn the held-out ADS (~0.863, DB heads) into the LB ADS (~0.768-0.770)? If yes, the LB gap is largely a
composition effect and the weak cells are the levers; if no, the gap is domain shift (unseen generators/reals).
Inputs: held-out DB-head predictions + bank labels (read-only). CPU, seconds."""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
PRED = sys.argv[1] if len(sys.argv) > 1 else R + '_keep_best_20260926/train_duration_balanced/test_predictions_decision.csv'
b = pd.read_csv(R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/bank_test.csv')
p = pd.read_csv(PRED)
d = b.merge(p, on='ID', how='inner')
assert len(d) == len(p), (len(d), len(b), len(p))  # 6 bank rows have no prediction row


def eer(y, s, w=None):
    if w is not None and float(np.sum(w)) == 0:
        return float('nan'), float('nan')
    fpr, tpr, thr = roc_curve(y, s, pos_label=1, drop_intermediate=False, sample_weight=w)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fpr - fnr))
    return (fpr[i] + fnr[i]) / 2, thr[i]


def lab(x):
    return 'NA' if pd.isna(x) else str(int(x))


d['type'] = ['?' if vm == 0 else {(1, 0): 'voice', (0, 1): 'music', (1, 1): 'mixed', (0, 0): 'none'}[(v, m)]
             for v, m, vm in zip(d.VOICE_PRESENT, d.MUSIC_PRESENT, d.VOICE_PRESENT_mask)]
d['cell'] = d['type'] + '|v' + d.VOICE_FAKE.map(lab) + '|m' + d.MUSIC_FAKE.map(lab)


def ads(w):
    out = {}
    f = d.FILE_FAKE_mask == 1
    out['file'] = eer(d.FILE_FAKE[f], d.FILE_FAKE_PROB[f], w[f])[0]
    v = (d.VOICE_PRESENT == 1) & (d.VOICE_FAKE_mask == 1)
    out['voice'] = eer(d.VOICE_FAKE[v], d.VOICE_FAKE_PROB[v], w[v])[0]
    m = (d.MUSIC_PRESENT == 1) & (d.MUSIC_FAKE_mask == 1)
    out['music'] = eer(d.MUSIC_FAKE[m], d.MUSIC_FAKE_PROB[m], w[m])[0]
    out['ADS'] = .5 * (1 - out['file']) + .2 * (1 - out['voice']) + .3 * (1 - out['music'])
    return {k: round(float(x), 5) for k, x in out.items()}


cells = d.cell.value_counts()


def weights(spec):
    """spec: {cell: share}; share spread uniformly over that cell's rows; cells not named get 0."""
    w = np.zeros(len(d))
    for c, s in spec.items():
        idx = (d.cell == c).to_numpy()
        if idx.sum():
            w[idx] = s / idx.sum()
    return pd.Series(w * len(d), index=d.index)


V0, V1 = 'voice|v0|mNA', 'voice|v1|mNA'
M0, M1 = 'music|vNA|m0', 'music|vNA|m1'
X00, X01, X10, X11 = 'mixed|v0|m0', 'mixed|v0|m1', 'mixed|v1|m0', 'mixed|v1|m1'


def comp(tv, tm, tx, x):
    """type shares tv/tm/tx, real/fake 50/50 within voice & music; x = shares of X00,X01,X10,X11 within mixed."""
    return {V0: tv / 2, V1: tv / 2, M0: tm / 2, M1: tm / 2, X00: tx * x[0], X01: tx * x[1], X10: tx * x[2], X11: tx * x[3]}


res = {'n_cells': cells.to_dict(), 'natural': ads(pd.Series(np.ones(len(d)), index=d.index))}
scen = {
    'thirds_factorial_mixed': comp(1 / 3, 1 / 3, 1 / 3, (.25, .25, .25, .25)),
    'thirds_concordant_mixed': comp(1 / 3, 1 / 3, 1 / 3, (.5, 0, 0, .5)),
    'thirds_mixed_no_realvoice_aimusic': comp(1 / 3, 1 / 3, 1 / 3, (1 / 3, 0, 1 / 3, 1 / 3)),
    'voice_heavy_60_20_20_factorial': comp(.6, .2, .2, (.25, .25, .25, .25)),
    'music_heavy_20_40_40_factorial': comp(.2, .4, .4, (.25, .25, .25, .25)),
    'mixed_heavy_20_20_60_factorial': comp(.2, .2, .6, (.25, .25, .25, .25)),
    'voice_only': comp(1, 0, 0, (0, 0, 0, 0)),
}
for k, s in scen.items():
    res[k] = ads(weights(s))

# per-cell error at the pooled (natural) EER threshold of each metric
per = {}
for metric, ycol, scol, mask in (
        ('file', 'FILE_FAKE', 'FILE_FAKE_PROB', d.FILE_FAKE_mask == 1),
        ('voice', 'VOICE_FAKE', 'VOICE_FAKE_PROB', (d.VOICE_PRESENT == 1) & (d.VOICE_FAKE_mask == 1)),
        ('music', 'MUSIC_FAKE', 'MUSIC_FAKE_PROB', (d.MUSIC_PRESENT == 1) & (d.MUSIC_FAKE_mask == 1))):
    e, t = eer(d[ycol][mask], d[scol][mask])
    sub = d[mask]
    err = np.where(sub[ycol] == 1, sub[scol] < t, sub[scol] >= t)
    g = pd.DataFrame({'cell': sub.cell, 'y': sub[ycol], 'err': err}).groupby(['cell', 'y']).err.agg(['mean', 'size'])
    per[metric] = {'thr': float(t), 'eer': float(e),
                   'cells': {f'{c}|y{int(y)}': [round(float(r['mean']), 3), int(r['size'])] for (c, y), r in g.iterrows()}}
res['per_cell_error_at_pooled_threshold'] = per
# within-type EERs (pure calibration-free) for FILE
wt = {}
for t in ('voice', 'music', 'mixed'):
    s = (d.type == t) & (d.FILE_FAKE_mask == 1)
    if d.FILE_FAKE[s].nunique() == 2:
        wt[t] = round(float(eer(d.FILE_FAKE[s], d.FILE_FAKE_PROB[s])[0]), 5)
res['file_eer_within_type'] = wt
print(json.dumps(res, indent=1))
