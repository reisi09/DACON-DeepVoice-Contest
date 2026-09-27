"""Source-adjusted duration effect on FP (reals) / FN (fakes) at the global held-out file-EER threshold (DB heads).
Logistic err ~ log2(duration) + source dummies (unpenalized); 95% CI by group-cluster bootstrap (200 reps)."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from lane3_checks import bank, PRED, eer_threshold
d = bank(); p = pd.read_csv(PRED / 'test_predictions_decision.csv')
x = d[(d.split == 'test') & (d.FILE_FAKE_mask == '1')].merge(p, on='ID')
x['y'] = x.FILE_FAKE.astype(int); thr, _ = eer_threshold(x.y, x.FILE_FAKE_PROB)
x['err'] = np.where(x.y == 0, x.FILE_FAKE_PROB >= thr, x.FILE_FAKE_PROB < thr).astype(int)
x['ld'] = np.log2(x.dur)
def fit(g):
    X = np.column_stack([g.ld.values, pd.get_dummies(g.source).values.astype(float)])
    return LogisticRegression(penalty=None, max_iter=5000, fit_intercept=False).fit(X, g.err.values).coef_[0][0]
rng = np.random.default_rng(0); out = {'threshold': thr}
for lab, g in (('real_FP', x[x.y == 0]), ('fake_FN', x[x.y == 1])):
    g = g[g.groupby('source').source.transform('size') >= 20]
    g = g[g.groupby('source').err.transform('nunique') > 1]      # sources with both outcomes carry the within-source signal
    b = fit(g); groups = g.group_id.unique(); idx = g.groupby('group_id').indices
    boot = [fit(g.iloc[np.concatenate([idx[k] for k in rng.choice(groups, len(groups))])]) for _ in range(200)]
    out[lab] = {'n': len(g), 'sources': g.source.nunique(), 'OR_per_duration_doubling': round(float(np.exp(b)), 3),
                'ci95': [round(float(np.exp(q)), 3) for q in np.percentile(boot, [2.5, 97.5])]}
json.dump(out, open('dur_adjust.json', 'w'), indent=1); print(json.dumps(out, indent=1))
