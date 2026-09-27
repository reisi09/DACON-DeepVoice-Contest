"""Audit A2: device parity of the frozen feature pipeline. Reference = H100 CUDA caches in the Existing9504 handoff
(parity/full_cache, torch 2.9.1). Candidates = local_features.py on MPS / CPU (torch 2.7.1) for the same audio bytes.
Pre-declared criteria: per-window cosine >= 0.999 for voice_embeddings and mert_windows, presence |d| <= 0.01,
same windowing (starts/samples), and P1-head probabilities |d| <= 0.01."""
import io, json, sys, zipfile, numpy as np, torch
from pathlib import Path
sys.path.insert(0, '.')
from full_bank_evaluate import load_selected_heads
A = Path('_workspace/audit_20260927'); h = 'results/existing9504_h100_20260920/completed/DACON_Existing9504_H100_Handoff.zip'
music, voice, fi, _ = load_selected_heads(Path('_workspace/p1_refit_cpu'), 'cpu')
def heads(d):
    f = torch.from_numpy(d['mert_windows']); e = torch.from_numpy(d['voice_embeddings'])
    with torch.no_grad():
        fm = np.mean([float((m.head((f - m.mean) / m.scale)[0].logsumexp(0) - np.log(len(f))).sigmoid()) for m in music])
        fv = 0. if bool(d['voice_silent']) else np.mean([float(v(e).softmax(-1)[:, fi].max()) for v in voice])
    return fv, fm
def cos(a, b): return float((np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)).min())
dev = sys.argv[1]
with zipfile.ZipFile(h) as z:
    samples = json.loads(z.read('parity/parity_receipt.json'))['samples']
    worst = {}
    for s in samples:
        ref = np.load(io.BytesIO(z.read(f"parity/full_cache/{s['file_stem']}.npz")))
        cand = np.load(Path('research/top15/work/orchestrator/fp16') / f'cache_{dev}' / f"{s['file_stem']}.npz")
        same = all(np.array_equal(ref[k], cand[k]) for k in ('starts', 'valid_samples', 'samples'))
        r = {'windows_equal': same, 'voice_cos_min': cos(ref['voice_embeddings'], cand['voice_embeddings']),
             'mert_cos_min': cos(ref['mert_windows'], cand['mert_windows']),
             'voice_max_abs': float(np.abs(ref['voice_embeddings'] - cand['voice_embeddings']).max()),
             'presence_max_abs': float(np.abs(ref['presence'] - cand['presence']).max()),
             'silent_equal': bool(ref['voice_silent']) == bool(cand['voice_silent'])}
        (a, b), (c, d) = heads(ref), heads(cand)
        r.update(dFv=abs(a - c), dFm=abs(b - d), Fv_ref=round(a, 4), Fm_ref=round(b, 4))
        print(f"{s['file_stem'][:38]:38s} win={r['windows_equal']} vcos={r['voice_cos_min']:.6f} mcos={r['mert_cos_min']:.6f} "
              f"pres={r['presence_max_abs']:.4f} dFv={r['dFv']:.4f} dFm={r['dFm']:.4f} silent_eq={r['silent_equal']}")
        for k, v in r.items():
            if isinstance(v, float): worst[k] = max(worst.get(k, 0), v) if not k.endswith('cos_min') else min(worst.get(k, 1), v)
    print(dev, 'WORST', {k: round(v, 6) for k, v in worst.items() if k not in ('Fv_ref', 'Fm_ref')})
