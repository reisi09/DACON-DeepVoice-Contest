"""Question: if official 'mixed' files were only the hardest held-out mixed sub-cells (simultaneous speech+music and
AI songs; no sequential mixes, no constructed real songs), and music-only files only the hardest music sources,
how low can ADS go at held-out per-row error rates? Upper-bounds how much of the ~0.095 LB gap composition explains."""
import contextlib, io, json, runpy
import pandas as pd
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(R + 'research/top15/work/review/v4/composition_check.py', run_name='lib')
d, ads, weights = g['d'], g['ads'], g['weights']
man = pd.read_json(R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/manifest.jsonl', lines=True)
mode = dict(zip(man.ID, man.mix_mode))
d['mode'] = d.orig_id.map(mode).fillna('')
hard = (d.type == 'mixed') & ((d['mode'] == 'simultaneous') | (d.source == 'haim'))
d['cell'] = d.cell.where(~(d.type == 'mixed'), d.cell + d['mode'].radd('|').where(hard, '|easy'))
print(d[d.type == 'mixed'].cell.value_counts().to_string())
V0, V1, M0, M1 = 'voice|v0|mNA', 'voice|v1|mNA', 'music|vNA|m0', 'music|vNA|m1'
hx = {c: n for c, n in d.cell.value_counts().items() if c.startswith('mixed') and not c.endswith('easy')}
out = {}
for name, (tv, tm, tx) in {'thirds': (1 / 3, 1 / 3, 1 / 3), 'mixed_heavy_20_20_60': (.2, .2, .6)}.items():
    # hard mixed cells, split equally over the (vf, mf) label combinations that exist among hard rows
    combos = {}
    for c in hx:
        combos.setdefault('|'.join(c.split('|')[1:3]), []).append(c)
    spec = {V0: tv / 2, V1: tv / 2, M0: tm / 2, M1: tm / 2}
    for k, cs in combos.items():
        for c in cs:
            spec[c] = tx / len(combos) / len(cs)
    out[name + '_hard_mixed_only'] = ads(weights(spec))
print(json.dumps(out, indent=1))
