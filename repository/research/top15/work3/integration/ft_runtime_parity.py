"""Packaged runtime vs the offline fit path on the 10 parity files, for any S4/R/A heads dir (s4_runtime_parity.py
generalised to the ft member). Runtime side: qa/verification.json of verify_datav4_submission.py on the package.
Offline side: H100 parity caches (DF-Arena embeddings) + --backbone-parity npz (frozen-AD embeddings, MERT layers) ->
packaged heads -> component fusion, with ft = max(ad_probs) of --ft-parity npz (data_v4/extract_backbones.py run with
the ft weights; for the frozen-AD stand-in this is --backbone-parity itself). Pre-declared tolerance (as S4):
VOICE <= 0.02, MUSIC <= 0.01 absolute.

  python ft_runtime_parity.py --heads FIT_DIR --verification PKG/qa/verification.json [--ft-parity DIR]
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
ap = argparse.ArgumentParser()
ap.add_argument('--heads', type=Path, required=True)
ap.add_argument('--verification', type=Path, required=True)
ap.add_argument('--backbone-parity', type=Path, default=ROOT / '_workspace/audit_20260927/backbone_parity')
ap.add_argument('--ft-parity', type=Path, help='npz dir with ad_probs from the ft weights (default: --backbone-parity)')
ap.add_argument('--handoff', type=Path, default=ROOT / 'results/existing9504_h100_20260920/completed/DACON_Existing9504_H100_Handoff.zip')
ap.add_argument('--code', type=Path, default=ROOT)
a = ap.parse_args()
sys.path.insert(0, str(a.code))
from component_fusion import fuse  # noqa: E402
from full_bank_evaluate import load_selected_heads  # noqa: E402
import voice_heads  # noqa: E402

music, voice, fi, _ = load_selected_heads(a.heads, 'cpu')
spec = json.loads((a.heads / 'component_fusion.json').read_text())['voice']
layers = json.loads((a.heads / 'music_layers.json').read_text())
ad = [voice_heads.load(torch.load(a.heads / f'voice_ad_{s}.pt', weights_only=True)).eval() for s in (41, 42, 43)] if spec['weights'].get('ad') else []
runtime = {r['ID']: r for r in json.loads(a.verification.read_text())['predictions']}
worst, rows = {'voice': 0., 'music': 0.}, []
with zipfile.ZipFile(a.handoff) as z, torch.no_grad():
    for s in json.loads(z.read('parity/parity_receipt.json'))['samples']:
        ref = np.load(io.BytesIO(z.read(f"parity/full_cache/{s['file_stem']}.npz")))
        bb = np.load(a.backbone_parity / f"{s['file_stem']}.npz")
        L = list(bb['layers'])
        f = torch.from_numpy(bb['mert_layers'][:, [L.index(k) for k in layers]].reshape(len(bb['mert_layers']), -1).astype(np.float32))
        e = torch.from_numpy(ref['voice_embeddings'])
        m = {'ours': 0. if bool(ref['voice_silent']) else float(np.mean([float(v(e).softmax(-1)[:, fi].max()) for v in voice]))}
        if ad:
            m['ad'] = float(np.mean([float(h(torch.from_numpy(bb['ad_embeddings'].astype(np.float32))).softmax(-1)[:, 0].max()) for h in ad]))
        if spec['weights'].get('ft'):
            m['ft'] = float(np.load((a.ft_parity or a.backbone_parity) / f"{s['file_stem']}.npz")['ad_probs'].max())
        fv = 0. if m['ours'] == 0. else fuse(spec, m)
        fm = float(np.mean([float((h.head((f - h.mean) / h.scale)[0].logsumexp(0) - np.log(len(f))).sigmoid()) for h in music]))
        r = runtime[s['ID']]
        dv, dm = abs(fv - float(r['VOICE_FAKE_PROB'])), abs(fm - float(r['MUSIC_FAKE_PROB']))
        worst = {'voice': max(worst['voice'], dv), 'music': max(worst['music'], dm)}
        rows.append({'ID': s['ID'], 'voice_offline': round(fv, 5), 'voice_runtime': round(float(r['VOICE_FAKE_PROB']), 5),
                     'music_offline': round(fm, 5), 'music_runtime': round(float(r['MUSIC_FAKE_PROB']), 5), 'members': {k: round(v, 5) for k, v in m.items()}})
ok = worst['voice'] <= .02 and worst['music'] <= .01
print(json.dumps({'fusion': spec['weights'], 'rows': rows, 'worst': {k: round(v, 5) for k, v in worst.items()}, 'pass': ok}, indent=1))
sys.exit(0 if ok else 1)
