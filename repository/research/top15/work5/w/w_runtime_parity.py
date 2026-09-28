"""S7 (S6 + SSLAM block-5 music features + WavLM hidden-8 voice member): training-feature path vs packaged runtime on the 10 parity files. EAT features:
eat_parity (research/top15/work2/backbones/extract_new_backbones.py on CPU, the training extractor). Offline: H100 parity caches (DF-Arena embeddings) +
backbone_parity (extract_backbones on CPU, fp32 AntiDeepfake) -> packaged heads -> component fusion. Runtime: the S3
package's own predictions from verify_datav4_submission (fp16-stored AntiDeepfake). Pre-declared tolerance: VOICE <= 0.02,
MUSIC <= 0.01 absolute."""
import io, json, os, sys, zipfile, numpy as np, torch
from pathlib import Path
sys.path.insert(0, '.')
from full_bank_evaluate import load_selected_heads
from component_fusion import fuse
F = Path(os.environ.get('HEADS', '_workspace/final_s7_20260928')); A = Path('_workspace/audit_20260927')
h = 'results/existing9504_h100_20260920/completed/DACON_Existing9504_H100_Handoff.zip'
music, voice, fi, _ = load_selected_heads(F, 'cpu')
ad = []
for s in (41, 42, 43):
    import voice_heads; ad.append(voice_heads.load(torch.load(F / f'voice_ad_{s}.pt', weights_only=True)).eval())
spec = json.loads((F / 'component_fusion.json').read_text())['voice']; layers = json.loads((F / 'music_layers.json').read_text()); eat_layers = json.loads((F / 'music_eat_layers.json').read_text()); sslam_layers = json.loads((F / 'music_sslam_layers.json').read_text()); (wl,) = json.loads((F / 'voice_wavlm_layer.json').read_text())
wav = [voice_heads.load(torch.load(F / f'wavlm_{s}.pt', weights_only=True)).eval() for s in (41, 42, 43)]
runtime = {r['ID']: r for r in json.loads(Path(os.environ.get('VERIFICATION', 'submissions/DataV4_S7_SSLAMWavLM/qa/verification.json')).read_text())['predictions']}
worst = {'voice': 0, 'music': 0}
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
        fv = 0. if bool(ref['voice_silent']) else float(np.mean([float(v(e).softmax(-1)[:, fi].max()) for v in voice]))
        fa = float(np.mean([float(m(torch.from_numpy(bb['ad_embeddings'].astype(np.float32))).softmax(-1)[:, 0].max()) for m in ad]))
        fw = float(np.mean([float(m(ew).softmax(-1)[:, 0].max()) for m in wav]))
        fv = 0. if fv == 0. else fuse(spec, {'ours': fv, 'ad': fa, 'wavlm': fw})
        fm = float(np.mean([float((m.head((f - m.mean) / m.scale)[0].logsumexp(0) - np.log(len(f))).sigmoid()) for m in music]))
        r = runtime[s['ID']]; dv, dm = abs(fv - float(r['VOICE_FAKE_PROB'])), abs(fm - float(r['MUSIC_FAKE_PROB']))
        worst = {'voice': max(worst['voice'], dv), 'music': max(worst['music'], dm)}
        print(f"{s['ID'][:34]:34s} voice offline {fv:.4f} runtime {float(r['VOICE_FAKE_PROB']):.4f} | music offline {fm:.4f} runtime {float(r['MUSIC_FAKE_PROB']):.4f}")
print('WORST', {k: round(v, 5) for k, v in worst.items()}, 'PASS' if worst['voice'] <= .02 and worst['music'] <= .01 else 'FAIL')
