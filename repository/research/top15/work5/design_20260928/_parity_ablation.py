"""Read-only audit helper (run from the project root with PYTHONDONTWRITEBYTECODE=1).
Replays research/top15/work5/w/w_runtime_parity.py (the stricter 16:13 version: FILE parity included) for a heads dir and
its package's CPU verification predictions, then measures how much the SSLAM feature block and the WavLM fusion member
move the packaged outputs on the same 10 parity files (sensitivity only, not a retrained ablation).
  HEADS=... VERIFICATION=... python _parity_ablation.py"""
import io, json, os, sys, zipfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '.')
from full_bank_evaluate import load_selected_heads
from component_fusion import fuse
import decision_heads, voice_heads

F = Path(os.environ['HEADS']); A = Path('_workspace/audit_20260927')
h = 'results/existing9504_h100_20260920/completed/DACON_Existing9504_H100_Handoff.zip'
music, voice, fi, _ = load_selected_heads(F, 'cpu')
ad = [voice_heads.load(torch.load(F / f'voice_ad_{s}.pt', weights_only=True)).eval() for s in (41, 42, 43)]
wav = [voice_heads.load(torch.load(F / f'wavlm_{s}.pt', weights_only=True)).eval() for s in (41, 42, 43)]
spec = json.loads((F / 'component_fusion.json').read_text())['voice']
layers = json.loads((F / 'music_layers.json').read_text()); eat_layers = json.loads((F / 'music_eat_layers.json').read_text())
sslam_layers = json.loads((F / 'music_sslam_layers.json').read_text()); (wl,) = json.loads((F / 'voice_wavlm_layer.json').read_text())
ens = json.loads((F / 'file_ensemble.json').read_text())
ver = json.loads(Path(os.environ['VERIFICATION']).read_text()); runtime = {r['ID']: r for r in ver['predictions']}
no_wavlm = {**spec, 'weights': {k: v for k, v in spec['weights'].items() if k != 'wavlm'}}

# first-layer weight share per music feature block (inputs are standardized, so columns are comparable)
width = music[0].mean.numel(); blocks = {'mert6': (0, 1024), 'mert12': (1024, 2048), 'eat16': (2048, 3072), 'sslam5': (3072, width)}
share = {b: float(np.mean([m.head.network[0].weight[:, i:j].pow(2).sum().item() / m.head.network[0].weight.pow(2).sum().item() for m in music]))
         for b, (i, j) in blocks.items()}
print('music width', width, 'first-layer squared-norm share per block (seed mean):', {k: round(v, 4) for k, v in share.items()})

def music_prob(f):
    return float(np.mean([float((m.head((f - m.mean) / m.scale)[0].logsumexp(0) - np.log(len(f))).sigmoid()) for m in music]))

worst = {'voice': 0., 'music': 0., 'file': 0.}; dW, dS = [], []
with zipfile.ZipFile(h) as z, torch.no_grad():
    for s in json.loads(z.read('parity/parity_receipt.json'))['samples']:
        ref = np.load(io.BytesIO(z.read(f"parity/full_cache/{s['file_stem']}.npz"))); bb = np.load(A / 'backbone_parity' / f"{s['file_stem']}.npz")
        e = torch.from_numpy(ref['voice_embeddings']); L = list(bb['layers'])
        f = bb['mert_layers'][:, [L.index(k) for k in layers]].reshape(len(bb['mert_layers']), -1).astype(np.float32)
        with np.load(A / 'eat_parity' / f"{s['file_stem']}.npz") as d:
            E = list(d['eat_layers']); f = np.concatenate([f, d['eat_backing'][:, [E.index(k) for k in eat_layers]].reshape(len(f), -1).astype(np.float32)], 1)
        with np.load(A / 's6_parity' / f"{s['file_stem']}.npz") as d6:
            S_ = list(d6['sslam_layers']); f = torch.from_numpy(np.concatenate([f, d6['sslam_pt_backing'][:, [S_.index(k) for k in sslam_layers]].reshape(len(f), -1).astype(np.float32)], 1))
            W_ = list(d6['wavlm_layers']); ew = torch.from_numpy(d6['wavlm_voice'][:, W_.index(wl)].astype(np.float32))
        ours = 0. if bool(ref['voice_silent']) else float(np.mean([float(v(e).softmax(-1)[:, fi].max()) for v in voice]))
        fa = float(np.mean([float(m(torch.from_numpy(bb['ad_embeddings'].astype(np.float32))).softmax(-1)[:, 0].max()) for m in ad]))
        fw = float(np.mean([float(m(ew).softmax(-1)[:, 0].max()) for m in wav]))
        fv = 0. if ours == 0. else fuse(spec, {'ours': ours, 'ad': fa, 'wavlm': fw})
        fv_now = 0. if ours == 0. else fuse(no_wavlm, {'ours': ours, 'ad': fa})
        fm = music_prob(f)
        fm_nos = float(np.mean([float((m.head((torch.cat([f[:, :3072], m.mean[3072:].expand(len(f), -1)], 1) - m.mean) / m.scale)[0].logsumexp(0) - np.log(len(f))).sigmoid()) for m in music]))
        r = runtime[s['ID']]; pv, pm = float(r['VOICE_PRESENT_PROB']), float(r['MUSIC_PRESENT_PROB'])
        ff = decision_heads.file_score(ens, dict(Pv=pv, Pm=pm, Fv=fv, Fm=fm))
        ff_now = decision_heads.file_score(ens, dict(Pv=pv, Pm=pm, Fv=fv_now, Fm=fm))
        ff_nos = decision_heads.file_score(ens, dict(Pv=pv, Pm=pm, Fv=fv, Fm=fm_nos))
        dv, dm, df = abs(fv - float(r['VOICE_FAKE_PROB'])), abs(fm - float(r['MUSIC_FAKE_PROB'])), abs(ff - float(r['FILE_FAKE_PROB']))
        worst = {'voice': max(worst['voice'], dv), 'music': max(worst['music'], dm), 'file': max(worst['file'], df)}
        dW.append((abs(fv - fv_now), abs(ff - ff_now))); dS.append((abs(fm - fm_nos), abs(ff - ff_nos)))
        print(f"{s['ID'][:30]:30s} ours {ours:.4f} ad {fa:.4f} wavlm {fw:.4f} | VOICE {fv:.4f} (no-WavLM {fv_now:.4f}) | MUSIC {fm:.4f} (SSLAM@mean {fm_nos:.4f}) | FILE {ff:.4f} rt {float(r['FILE_FAKE_PROB']):.4f}")
print('PARITY WORST', {k: round(v, 5) for k, v in worst.items()}, 'n', len(runtime))
print('WavLM member removed: max |dVOICE| %.4f, max |dFILE| %.4f' % tuple(np.max(dW, 0)))
print('SSLAM block at mean:  max |dMUSIC| %.4f, max |dFILE| %.4f' % tuple(np.max(dS, 0)))
