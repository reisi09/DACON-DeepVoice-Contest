"""Questions: (1) do sequential real mixes (voice and music in turn, allowed by the official 'mixed' definition) get
more VOICE false positives than simultaneous ones, as window-max over bleed-only windows would predict?
(2) is channel count (stereo) label-correlated in the bank, and so a possible shortcut the runtime discards?
Read-only; DB held-out predictions; CPU seconds."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
D = R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/'
man = pd.read_json(D + 'manifest.jsonl', lines=True)[['ID', 'mix_mode', 'channels', 'snr_db']].rename(columns={'ID': 'orig_id'})
b = pd.concat([pd.read_csv(D + 'bank/bank_trainval.csv'), pd.read_csv(D + 'bank/bank_test.csv')]).merge(man, on='orig_id', how='left')
p = pd.read_csv(R + '_keep_best_20260926/train_duration_balanced/test_predictions_decision.csv')
t = b[b.split == 'test'].merge(p, on='ID')


def eer(y, s):
    fpr, tpr, thr = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2, thr[i]


out = {}
vm = (t.VOICE_PRESENT == 1) & (t.VOICE_FAKE_mask == 1)
_, thr = eer(t.VOICE_FAKE[vm], t.VOICE_FAKE_PROB[vm])
out['voice_pooled_thr'] = float(thr)
c = t[vm & t.mix_mode.notna()]
g = c.groupby(['mix_mode', 'VOICE_FAKE'])[['VOICE_FAKE', 'VOICE_FAKE_PROB']].apply(lambda x: pd.Series({
    'n': len(x), 'err_at_pooled_thr': float(np.mean(np.where(x.VOICE_FAKE == 1, x.VOICE_FAKE_PROB < thr, x.VOICE_FAKE_PROB >= thr))),
    'median_score': float(x.VOICE_FAKE_PROB.median())}))
out['voice_by_mix_mode_test'] = {f'{m}|vf{int(v)}': r.round(4).to_dict() for (m, v), r in g.iterrows()}
out['voice_eer_within_mode_test'] = {m: round(float(eer(x.VOICE_FAKE, x.VOICE_FAKE_PROB)[0]), 4)
                                     for m, x in c.groupby('mix_mode') if x.VOICE_FAKE.nunique() == 2}
# bleed check: real-music-only files -> how fake does the (bleed) voice stem look?
mo = t[(t.VOICE_PRESENT == 0) & (t.VOICE_PRESENT_mask == 1)]
out['voice_score_on_music_only_test'] = {'n': len(mo), 'frac_above_voice_thr': round(float((mo.VOICE_FAKE_PROB >= thr).mean()), 4),
                                         'median': round(float(mo.VOICE_FAKE_PROB.median()), 4)}
out['mix_mode_counts_by_split'] = b.groupby(['split', 'mix_mode']).size().unstack().fillna(0).astype(int).to_dict()
ch = b.groupby(['split', 'channels', 'FILE_FAKE']).size().unstack().fillna(0).astype(int)
out['channels_x_filefake'] = {f'{s}|ch{int(k)}': r.to_dict() for (s, k), r in ch.iterrows()}
st = b[b.channels == 2]
out['stereo_by_source_label_trainval'] = st[st.split != 'test'].groupby(['source', 'FILE_FAKE']).size().sort_values(ascending=False).head(15).items()
out['stereo_by_source_label_trainval'] = {f'{a}|{b}': int(n) for (a, b), n in out['stereo_by_source_label_trainval']}
print(json.dumps(out, indent=1, default=str))
