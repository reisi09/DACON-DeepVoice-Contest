"""Question: data v4 has 0 rows of FAKE singing over REAL music ('AI covers': cloned vocal on a human backing track);
every fake-sung-voice row is a fully AI song (MUSIC_FAKE=1). If the official test contains AI covers, how would the
current FILE rule score them? Score-level simulation, assuming the runtime's stem routing (VOICE sees only the vocal
stem, MUSIC only the backing stem): pair (Pv, Fv) of held-out AI-song rows with (Pm, Fm) of held-out real music-only
rows, apply the shipped DB FILE ensemble (decision_heads.file_score), measure the miss rate at the natural held-out FILE
EER threshold and the pooled FILE EER when covers make up a share of the fakes. DB-head predictions; CPU, seconds."""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
sys.path.insert(0, R)
import decision_heads as dh

K = R + '_keep_best_20260926/train_duration_balanced/'
ens = json.load(open(K + 'file_ensemble.json'))
b = pd.read_csv(R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/bank_test.csv')
d = b.merge(pd.read_csv(K + 'test_predictions_decision.csv'), on='ID')
f = d[d.FILE_FAKE_mask == 1]


def eer(y, s):
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2, thr[i]


e0, thr = eer(f.FILE_FAKE, f.FILE_FAKE_PROB)
# check that file_score on the CSV components reproduces the CSV FILE score
rec = np.array([dh.file_score(ens, {'Pv': r.VOICE_PRESENT_PROB, 'Pm': r.MUSIC_PRESENT_PROB, 'Fv': r.VOICE_FAKE_PROB,
                                    'Fm': r.MUSIC_FAKE_PROB}) for r in f.itertuples()])
song = d[(d.VOICE_FAKE == 1) & (d.MUSIC_FAKE == 1) & (d.VOICE_PRESENT == 1) & (d.MUSIC_PRESENT == 1) & (d.source == 'haim')]  # sung AI vocals only
realm = d[(d.MUSIC_FAKE == 0) & (d.VOICE_PRESENT == 0) & (d.MUSIC_PRESENT == 1)]
rng = np.random.default_rng(0)
res = {'file_eer_natural': round(float(e0), 5), 'thr': float(thr), 'max_abs_reproduction_err': float(np.abs(rec - f.FILE_FAKE_PROB).max()),
       'n_song_voice_donors': len(song), 'song_donor_sources': song.source.value_counts().to_dict(),
       'n_real_music_donors': len(realm), 'real_music_sources': realm.source.value_counts().to_dict()}
miss, covers = [], []
for _ in range(20):
    m = realm.sample(len(song), replace=True, random_state=int(rng.integers(1 << 30)))
    s = np.array([dh.file_score(ens, {'Pv': a.VOICE_PRESENT_PROB, 'Fv': a.VOICE_FAKE_PROB, 'Pm': c.MUSIC_PRESENT_PROB, 'Fm': c.MUSIC_FAKE_PROB})
                  for a, c in zip(song.itertuples(), m.itertuples())])
    miss.append(float((s < thr).mean())); covers.append(s)
res['cover_miss_at_natural_thr'] = [round(float(np.mean(miss)), 4), round(float(np.std(miss)), 4)]
res['same_rows_as_ai_song_miss_at_natural_thr'] = round(float((song.FILE_FAKE_PROB < thr).mean()), 4)
res['voice_stem_alone_Fv_below_voice_thr_0.1913'] = round(float((song.VOICE_FAKE_PROB < 0.1912508).mean()), 4)
# pooled FILE EER when covers are a share of all fakes (added, fakes' total count held fixed by weighting)
y, s0 = f.FILE_FAKE.to_numpy(), f.FILE_FAKE_PROB.to_numpy()
nf = int(y.sum())
for share in (.05, .10, .20):
    k = int(round(share * nf / (1 - share)))
    es = [eer(np.r_[y, np.ones(k)], np.r_[s0, rng.choice(c, k)])[0] for c in covers[:10]]
    res[f'file_eer_if_covers_{int(share * 100)}pct_of_fakes'] = round(float(np.mean(es)), 5)
print(json.dumps(res, indent=1))
