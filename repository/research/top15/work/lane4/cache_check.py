"""Lane 4 CPU checks on local caches + packaged S3 heads (read-only; hashes of heads == zip members, see pkg_check).

Q1 coverage : is the 74% backbone-cache coverage balanced over split/source/label/duration?
Q2 parity   : backbone_cache MERT slot for layer 12 (fp16) == cache_v4 mert_windows (fp32, the P1 feature, extracted with
              hidden_states[12]) up to fp16 rounding, and far from the layer-6 slot -> layer order; window counts, samples,
              voice_silent agreement; AntiDeepfake fake index 0 polarity (zero-shot ad_probs AUC on VOICE_FAKE).
Q3 val repro: runtime formulas (full_bank_submission.Engine.predict, lines 111-137) + packaged heads/tables reproduce
              validation_metrics.json on the 3,680 val rows -> fitting/runtime consistency of fusion + FILE ensemble.
Q4 variants : 48-min fallback (no AD member), fp16->fp32 dequantisation noise, equal-weight FILE ensemble, partial fallback.
"""
import csv, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

R = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(R))
import decision_heads as dh  # noqa: E402
from component_fusion import fuse  # noqa: E402
from graph_scoring import official_metrics, _roc_metrics  # noqa: E402
from localization_models import WindowMIL  # noqa: E402

BANK = R / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
BB = R / '_workspace/backbone_cache'
OLD = {'train': R / '_workspace/cache_v4/cache_tv', 'val': R / '_workspace/cache_v4/cache_tv', 'test': R / '_workspace/cache_v4/cache_test'}
F = R / '_workspace/final_backbone_20260927'
OUT = R / 'research/top15/work/lane4'
t0 = time.time()
rows = [r for r in csv.DictReader((BANK / 'bank_trainval.csv').open())] + [r for r in csv.DictReader((BANK / 'bank_test.csv').open())]
have = {r['ID'] for r in rows if (BB / f"{r['ID']}.npz").exists()}
report = {}

# ---------------- Q1 coverage
samples = {}
for s in ('tv', 'test'):
    samples.update({k: v['samples'] for k, v in json.loads((R / f'_workspace/cache_v4/cache_{s}/index.json').read_text())['items'].items()})
def cov(key):
    c = defaultdict(lambda: [0, 0])
    for r in rows:
        c[key(r)][0] += 1
        c[key(r)][1] += r['ID'] in have
    return {str(k): [n, m, round(m / n, 3)] for k, (n, m) in sorted(c.items(), key=lambda t: -t[1][0])}
report['coverage_split'] = cov(lambda r: r['split'])
report['coverage_route_label'] = cov(lambda r: (r['split'] == 'val', r['route'], r['MUSIC_FAKE' if r['route'] == 'music' else 'VOICE_FAKE']))
src = cov(lambda r: r['source'])
report['coverage_source_extremes'] = {k: v for k, v in src.items() if v[0] >= 100 and (v[2] < .6 or v[2] > .9)}
report['coverage_source_n'] = len(src)
dur = cov(lambda r: int(np.digitize(samples[r['ID']] / 16000, [2, 4, 8, 16, 30, 60])))
report['coverage_duration_bin(<2,2-4,4-8,8-16,16-30,30-60,>60 s)'] = dur

# ---------------- Q2 parity (random sample with both caches)
rng = np.random.default_rng(0)
both = [r for r in rows if r['ID'] in have]
sample = [both[i] for i in rng.choice(len(both), 400, replace=False)]
par = Counter(); d12, d6, q16, rel12 = [], [], [], []
for r in sample:
    with np.load(BB / f"{r['ID']}.npz") as b, np.load(OLD[r['split']] / f"{r['ID']}.npz") as o:
        L = list(b['layers'])
        par['samples_equal'] += int(b['samples']) == int(o['samples'])
        par['windows_equal'] += len(b['mert_layers']) == len(o['mert_windows'])
        par['silent_equal'] += bool(b['voice_silent']) == bool(o['voice_silent'])
        par['gain_capped_rows'] += float(o['normalization_gain']) != 1.
        if len(b['mert_layers']) != len(o['mert_windows']):
            continue
        ref = o['mert_windows'].astype(np.float32)
        l12 = b['mert_layers'][:, L.index(12)].astype(np.float32)
        l6 = b['mert_layers'][:, L.index(6)].astype(np.float32)
        d12.append(np.abs(l12 - ref).max()); d6.append(np.abs(l6 - ref).max())
        q16.append(np.abs(ref.astype(np.float16).astype(np.float32) - ref).max())
        rel12.append(np.linalg.norm(l12 - ref) / np.linalg.norm(ref))
par = dict(par, n=len(sample))
par.update(max_abs_l12_vs_v4=float(max(d12)), median_abs_l12_vs_v4=float(np.median(d12)),
           max_fp16_rounding_of_v4=float(max(q16)), max_rel_l2_l12_vs_v4=float(max(rel12)),
           min_abs_l6_vs_v4=float(min(d6)), median_abs_l6_vs_v4=float(np.median(d6)))
report['parity_sample'] = par
# window_truth usability for music rows (items_for silently drops truth when window counts differ)
mism = Counter()
for r in both:
    if r['route'] != 'music' and r.get('MUSIC_PRESENT') != '1':
        continue
    with np.load(BB / f"{r['ID']}.npz") as b, np.load(OLD[r['split']] / f"{r['ID']}.npz") as o:
        n = len(b['mert_layers'])
        mism['music_rows'] += 1
        mism['truth_len_mismatch'] += len(o['window_truth']) != n
        mism['rows_with_known_truth'] += bool(np.isfinite(o['window_truth']).any())
report['window_truth'] = dict(mism)
# AntiDeepfake polarity: zero-shot ad_probs (classifier index 0) vs VOICE_FAKE on voice-route val+test rows
ys, ps = [], []
for r in both:
    if r['route'] == 'voice' and r['split'] in ('val', 'test') and r['VOICE_FAKE'] in ('0', '1'):
        with np.load(BB / f"{r['ID']}.npz") as b:
            if bool(b['voice_silent']):
                continue
            ys.append(int(r['VOICE_FAKE'])); ps.append(float(b['ad_probs'].max()))
report['antideepfake_index0_zero_shot'] = {'n': len(ys), 'eer_max_window': _roc_metrics(ys, ps)[0], 'auc': _roc_metrics(ys, ps)[1]}

# ---------------- Q3 val reproduction with packaged heads
def load(name):
    return torch.load(F / name, map_location='cpu', weights_only=True)
music = []
for s in (41, 42, 43):
    st = load(f'music_{s}.pt')['model']; m = WindowMIL(st['mean'], st['scale']); m.load_state_dict(st); music.append(m.eval())
voice, ad = [], []
for s in (41, 42, 43):
    rec = load(f'voice_{s}.pt'); m = nn.Linear(1280, 2); m.load_state_dict(rec['model']); voice.append(m.eval()); fi = rec['fake_label_index']
    rec = load(f'voice_ad_{s}.pt'); m = nn.Linear(1280, 2); m.load_state_dict(rec['model']); ad.append(m.eval())
layers = json.loads((F / 'music_layers.json').read_text())
spec = json.loads((F / 'component_fusion.json').read_text())['voice']
ens = json.loads((F / 'file_ensemble.json').read_text())
pres = json.loads((R / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/train/presence_head.json').read_text())
val = [r for r in rows if r['split'] == 'val' and r['ID'] in have]

def half_noise(x):
    """fp32 value consistent with the stored fp16: uniform within +-half ulp of the fp16 grid."""
    x16 = x.astype(np.float16)
    ulp = np.abs(np.spacing(x16)).astype(np.float32)
    return x16.astype(np.float32) + rng.uniform(-.5, .5, x.shape).astype(np.float32) * ulp

def score(r, noise=False):
    with np.load(BB / f"{r['ID']}.npz") as b, np.load(OLD['val'] / f"{r['ID']}.npz") as o:
        L = list(b['layers'])
        f = b['mert_layers'][:, [L.index(k) for k in layers]].reshape(len(b['mert_layers']), -1)
        e = b['ad_embeddings']
        f = half_noise(f) if noise else f.astype(np.float32)
        e = half_noise(e) if noise else e.astype(np.float32)
        emb = torch.from_numpy(o['voice_embeddings']); silent = bool(o['voice_silent'])
        feats = json.loads(str(o['presence_features'])); legacy = tuple(float(x) for x in o['presence'])
    with torch.no_grad():
        f = torch.from_numpy(f)
        fm = float(np.mean([float((m.head((f - m.mean) / m.scale)[0].float().logsumexp(0) - np.log(len(f))).sigmoid()) for m in music]))
        vo = 0. if silent else float(np.mean([float(h(emb).softmax(-1)[:, fi].max()) for h in voice]))
        e = torch.from_numpy(e)
        fa = float(np.mean([float(h(e).softmax(-1)[:, 0].max()) for h in ad]))
    vp, mp = dh.presence(pres, feats, legacy)
    return dict(ID=r['ID'], ours=vo, ad=fa, Fm=fm, Pv=vp, Pm=mp)

def five(s, use_ad=True, weights=None):
    fv = s['ours'] if s['ours'] == 0. else fuse(spec, {'ours': s['ours'], **({'ad': s['ad']} if use_ad else {})})
    e = ens if weights is None else {**ens, 'weights': weights}
    d = dict(Pv=s['Pv'], Pm=s['Pm'], Fv=fv, Fm=s['Fm'])
    return {'ID': s['ID'], 'FILE_FAKE_PROB': dh.file_score(e, d), 'VOICE_FAKE_PROB': fv, 'MUSIC_FAKE_PROB': s['Fm'],
            'VOICE_PRESENT_PROB': s['Pv'], 'MUSIC_PRESENT_PROB': s['Pm']}

def metrics(preds):
    m = official_metrics(val, preds)['metrics']
    ads = .5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer'])
    return {**{k: round(v, 6) for k, v in m.items()}, 'ADS': round(ads, 6)}

base = [score(r) for r in val]
recorded = json.loads((F / 'validation_metrics.json').read_text())['metrics']
p0 = [five(s) for s in base]
report['val_repro'] = {'n': len(val), 'reproduced': metrics(p0), 'recorded': {k: round(v, 6) for k, v in recorded.items()}}
report['val_fallback_no_ad_all_files'] = metrics([five(s, use_ad=False) for s in base])
report['val_equal_weight_file_ensemble'] = metrics([five(s, weights={'max': 1, 'noisy_or': 1, 'max_nopresence': 1}) for s in base])
# partial fallback: files in random 30% scored without AD (mixed scoring functions in one ranking)
mask = rng.random(len(base)) < .3
report['val_partial_fallback_30pct'] = metrics([five(s, use_ad=not m) for s, m in zip(base, mask)])
noisy = [score(r, noise=True) for r in val]
p1 = [five(s) for s in noisy]
report['val_fp16_dequant_noise'] = {**metrics(p1),
    'max_abs_dFm': max(abs(a['MUSIC_FAKE_PROB'] - b['MUSIC_FAKE_PROB']) for a, b in zip(p0, p1)),
    'max_abs_dFv': max(abs(a['VOICE_FAKE_PROB'] - b['VOICE_FAKE_PROB']) for a, b in zip(p0, p1)),
    'max_abs_dFILE': max(abs(a['FILE_FAKE_PROB'] - b['FILE_FAKE_PROB']) for a, b in zip(p0, p1))}
# how often does the fused voice score move vs ours, and silent-count
report['val_silent_voice_rows'] = sum(s['ours'] == 0. for s in base)
report['seconds'] = round(time.time() - t0, 1)
(OUT / 'cache_check.out.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
print(json.dumps(report, indent=1, default=str))
