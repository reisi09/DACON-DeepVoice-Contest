"""Read-only audit helper (project root, PYTHONDONTWRITEBYTECODE=1): rebuild the S7-recipe DB-protocol predictions exactly as
research/top15/work5/s8/s8_eval.py (S7_cpu: df_L 1 / ad_L 0.5 / wavlm8 1, music_sslam; work4/runs every-row preds, val/test only),
check the recorded E1 ADS 0.90921, then
  (1) scale check: replace Fv by a monotone per-component map (its own val CDF) -> VOICE EER must not move; FILE EER does
      if the FILE rule mixes differently calibrated scales (noisy_or / max_nopresence combine Fv with the raw Fm);
  (2) describe Fm (music head) on files without music (MUSIC_PRESENT=0), the population the music head never trains on."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path('.').resolve(); TOP = ROOT / 'research/top15'
sys.path[:0] = [str(TOP / 'work2' / 'backbones'), str(TOP / 'work3' / 'integration')]
import run as R  # noqa: F401  (sys.path setup, as s8_eval)
import fit_ft_fusion as F
from component_fusion import fuse
import decision_heads as dh

ST, W4 = TOP / 'work2' / 'stacking', TOP / 'work4' / 'runs'
frozen = set(json.loads((ST / 'frozen_rows.json').read_text())['ids'])
BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
rows = {s: F.bank_rows(BANK, s, frozen) for s in ('val', 'test')}
P = {**json.loads((ST / 'presence_val.json').read_text()), **json.loads((ST / 'presence_test.json').read_text())}
M, silent = {}, {}
for name, f, ks in (('df', 'voice_df__L.json', True), ('ad', 'voice_ad__L.json', False), ('wavlm', 'voice_wavlm8__every.json', False),
                    ('music', 'music_sslam__every.json', False)):
    r = json.loads((W4 / f).read_text()); M[name] = {}
    for s in ('val', 'test'):
        M[name].update(zip(r['every'][s]['ids'], np.mean(r['every'][s]['preds'], 0).tolist()))
        if ks:
            silent.update(zip(r['every'][s]['ids'], r['every'][s]['silent']))
ns = [r['ID'] for r in rows['val'] if not silent[r['ID']]]
tables = {n: dh.quantile_table([M[n][i] for i in ns]) for n in M}
spec = {'weights': {'ours': 1, 'ad': .5, 'wavlm': 1}, 'tables': {'ours': tables['df'], 'ad': tables['ad'], 'wavlm': tables['wavlm']}}
Fv = {i: 0. if silent[i] else fuse(spec, {'ours': M['df'][i], 'ad': M['ad'][i], 'wavlm': M['wavlm'][i]}) for s in rows for i in (r['ID'] for r in rows[s])}
vt = dh.quantile_table([Fv[i] for i in ns])   # val CDF of the fused VOICE (monotone map; silent rows stay 0)


def run(fv_map, weights=None):
    files = {s: [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fm': M['music'][r['ID']], 'Fv': fv_map(r['ID'])} for r in rows[s]] for s in rows}
    ens, rep, preds = F.file_preds(files, rows['val'])
    if weights:
        ens = {**ens, 'weights': weights}
        preds = {s: [{**p, 'FILE_FAKE_PROB': dh.file_score(ens, f)} for p, f in zip(preds[s], files[s])] for s in preds}
    return ens['weights'], {s: F.ads_of(rows[s], preds[s]) for s in rows}, files


w0, base, files = run(lambda i: Fv[i])
print('rows', {s: len(v) for s, v in rows.items()}, 'FILE weights (fit on val)', w0)
print('current      ', {s: base[s] for s in base})
assert base['test']['ads'] == .90921, base['test']
for label, w in (('fixed 0/9/1', {'max': 0., 'noisy_or': 9., 'max_nopresence': 1.}), ('refit', None)):
    _, alt, _ = run(lambda i: 0. if silent[i] else dh.to_cdf(vt, Fv[i]), w)
    _, cur, _ = run(lambda i: Fv[i], w)
    print(f'Fv -> val-CDF ({label:11s})', {s: alt[s] for s in alt}, '| dFILE_EER', {s: round(alt[s]['file_eer'] - cur[s]['file_eer'], 5) for s in alt},
          '| dVOICE_EER', {s: round(alt[s]['voice_eer'] - cur[s]['voice_eer'], 5) for s in alt})
lab = F.lab
for s in rows:
    real_v = [Fv[r['ID']] for r in rows[s] if lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') == 0 and not silent[r['ID']]]
    real_m = [M['music'][r['ID']] for r in rows[s] if lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') == 0]
    nomus = [(lab(r, 'FILE_FAKE'), M['music'][r['ID']], Fv[r['ID']]) for r in rows[s] if lab(r, 'MUSIC_PRESENT') == 0 and lab(r, 'FILE_FAKE') is not None]
    y = np.array([a for a, _, _ in nomus]); fm = np.array([b for _, b, _ in nomus]); fv = np.array([c for _, _, c in nomus])
    q = lambda x: np.round(np.quantile(x, [.5, .9]), 3).tolist()
    print(f'{s}: real-voice Fv q50/q90 {q(real_v)} (n {len(real_v)}) | real-music Fm q50/q90 {q(real_m)} (n {len(real_m)})')
    print(f'{s}: no-music files n {len(y)} (real {int((y == 0).sum())}) Fm q50/q90 real {q(fm[y == 0])} fake {q(fm[y == 1])}; '
          f'share with Fm > Fv (max_nopresence set by the music head): real {np.mean(fm[y == 0] > fv[y == 0]):.3f} fake {np.mean(fm[y == 1] > fv[y == 1]):.3f}')

# (3) oracle magnitude (labels used for gating, diagnostic only): Fm := 0 on files without music, val and test alike
nomusic = {r['ID'] for s in rows for r in rows[s] if lab(r, 'MUSIC_PRESENT') == 0}
for label, w in (('fixed 0/9/1', {'max': 0., 'noisy_or': 9., 'max_nopresence': 1.}), ('fit on val', None)):
    def run_m(fm_map):
        files = {s: [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fm': fm_map(r['ID']), 'Fv': Fv[r['ID']]} for r in rows[s]] for s in rows}
        ens, rep, preds = F.file_preds(files, rows['val'])
        if w:
            ens = {**ens, 'weights': w}
            preds = {s: [{**p, 'FILE_FAKE_PROB': dh.file_score(ens, f)} for p, f in zip(preds[s], files[s])] for s in preds}
        return {s: F.ads_of(rows[s], preds[s])['file_eer'] for s in rows}
    cur = run_m(lambda i: M['music'][i]); orc = run_m(lambda i: 0. if i in nomusic else M['music'][i])
    print(f'FILE EER ({label}): current {cur} | oracle Fm=0 on no-music files {orc}')

# (4) the same oracle for the voice route: Fv := 0 on files without voice (instrumental music)
novoice = {r['ID'] for s in rows for r in rows[s] if lab(r, 'VOICE_PRESENT') == 0}
for s in rows:
    nv = [(lab(r, 'FILE_FAKE'), Fv[r['ID']]) for r in rows[s] if r['ID'] in novoice and lab(r, 'FILE_FAKE') is not None]
    y = np.array([a for a, _ in nv]); v = np.array([b for _, b in nv])
    print(f'{s}: no-voice files n {len(y)} (real {int((y == 0).sum())}) Fv q50/q90 real {np.round(np.quantile(v[y == 0], [.5, .9]), 3).tolist()} fake {np.round(np.quantile(v[y == 1], [.5, .9]), 3).tolist()}')
w = {'max': 0., 'noisy_or': 9., 'max_nopresence': 1.}
def run_v(fv_map):
    files = {s: [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fm': M['music'][r['ID']], 'Fv': fv_map(r['ID'])} for r in rows[s]] for s in rows}
    ens, rep, preds = F.file_preds(files, rows['val'])
    ens = {**ens, 'weights': w}
    preds = {s: [{**p, 'FILE_FAKE_PROB': dh.file_score(ens, f)} for p, f in zip(preds[s], files[s])] for s in preds}
    return {s: F.ads_of(rows[s], preds[s])['file_eer'] for s in rows}
print('FILE EER (fixed 0/9/1): oracle Fv=0 on no-voice files', run_v(lambda i: 0. if i in novoice else Fv[i]))
