"""S8 (S7 + SSLAM-pretrain block 5 on the RAW MIX appended to the music features): training-feature path vs the packaged
runtime (VESSL CPU run of verify_s8_vessl.py) on the 10 parity files. Offline: parity DF-Arena embeddings (handoff),
backbone_parity (AD, MERT), eat_parity (EAT), s6_parity (SSLAM backing, WavLM from fp16-rounded runtime stems), s8_parity
(mixsslam) -> packaged heads -> component fusion. Pre-declared tolerance (as S7): VOICE <= 0.02, MUSIC <= 0.01 absolute.
Heads/configs are read from TREE/model/trained (the receipt-verified package content); FILE is recomputed with the
packaged file_ensemble.json from offline Fv/Fm and the runtime presence (tolerance 0.02, declared before this run).
Exit 1 on any failure (tolerance, missing/duplicate IDs, nonfinite values, receipt/device mismatch).
  python s8_runtime_parity.py RECEIPT PARITY VERIFICATION_JSON TREE DEVICE"""
import json, sys
from pathlib import Path
import numpy as np, torch
RECEIPT, A, V, TREE = (Path(x) for x in sys.argv[1:5])
DEVICE = sys.argv[5]
F = TREE / 'model' / 'trained'
import math  # noqa: E402
sys.path[:0] = [str(TREE / 'model' / 'runtime'), '/root/proj']
from full_bank_evaluate import load_selected_heads  # noqa: E402
from component_fusion import fuse  # noqa: E402
import decision_heads  # noqa: E402
import voice_heads  # noqa: E402
music, voice, fi, _ = load_selected_heads(F, 'cpu')
ad = [voice_heads.load(torch.load(F / f'voice_ad_{s}.pt', weights_only=True)).eval() for s in (41, 42, 43)]
wav = [voice_heads.load(torch.load(F / f'wavlm_{s}.pt', weights_only=True)).eval() for s in (41, 42, 43)]
spec = json.loads((F / 'component_fusion.json').read_text())['voice']
layers, eat_layers = json.loads((F / 'music_layers.json').read_text()), json.loads((F / 'music_eat_layers.json').read_text())
sslam_layers, (wl,) = json.loads((F / 'music_sslam_layers.json').read_text()), json.loads((F / 'voice_wavlm_layer.json').read_text())
extras = json.loads((F / 'music_extra.json').read_text())
assert [e['source'] for e in extras] == ['mixsslam'], extras
ver, rec = json.loads(V.read_text()), json.loads(RECEIPT.read_text())
assert ver['submission_sha256'] == rec['sha256'] and ver['device'] == DEVICE, (ver['submission_sha256'], ver['device'])
ids = [r['ID'] for r in ver['predictions']]
samples = json.loads((A / 'parity' / 'parity_receipt.json').read_text())['samples']
assert len(ids) == len(set(ids)) == len(samples) == 10 and set(ids) == {s['ID'] for s in samples}, ids
runtime = {r['ID']: r for r in ver['predictions']}
ens = json.loads((F / 'file_ensemble.json').read_text())
worst_file = 0.
worst = {'voice': 0, 'music': 0}
with torch.no_grad():
    for s in samples:
        ref = np.load(A / 'parity' / 'full_cache' / f"{s['file_stem']}.npz"); bb = np.load(A / 'backbone_parity' / f"{s['file_stem']}.npz")
        e = torch.from_numpy(ref['voice_embeddings']); L = list(bb['layers'])
        f = bb['mert_layers'][:, [L.index(k) for k in layers]].reshape(len(bb['mert_layers']), -1).astype(np.float32)
        with np.load(A / 'eat_parity' / f"{s['file_stem']}.npz") as d:
            E = list(d['eat_layers']); f = np.concatenate([f, d['eat_backing'][:, [E.index(k) for k in eat_layers]].reshape(len(f), -1).astype(np.float32)], 1)
        with np.load(A / 's6_parity' / f"{s['file_stem']}.npz") as d6:
            S_ = list(d6['sslam_layers']); f = np.concatenate([f, d6['sslam_pt_backing'][:, [S_.index(k) for k in sslam_layers]].reshape(len(f), -1).astype(np.float32)], 1)
            W_ = list(d6['wavlm_layers']); ew = torch.from_numpy(d6['wavlm_voice'][:, W_.index(wl)].astype(np.float32))
        with np.load(A / 's8_parity' / f"{s['file_stem']}.npz") as d8:
            M_ = list(d8['sslam_layers']); x = d8['mixsslam'][:, [M_.index(k) for k in extras[0]['layers']]].reshape(len(f), -1).astype(np.float32)
            f = torch.from_numpy(np.concatenate([f, x], 1))
        fv = 0. if bool(ref['voice_silent']) else float(np.mean([float(v(e).softmax(-1)[:, fi].max()) for v in voice]))
        fa = float(np.mean([float(m(torch.from_numpy(bb['ad_embeddings'].astype(np.float32))).softmax(-1)[:, 0].max()) for m in ad]))
        fw = float(np.mean([float(m(ew).softmax(-1)[:, 0].max()) for m in wav]))
        fv = 0. if fv == 0. else fuse(spec, {'ours': fv, 'ad': fa, 'wavlm': fw})
        fm = float(np.mean([float((m.head((f - m.mean) / m.scale)[0].logsumexp(0) - np.log(len(f))).sigmoid()) for m in music]))
        r = runtime[s['ID']]; dv, dm = abs(fv - float(r['VOICE_FAKE_PROB'])), abs(fm - float(r['MUSIC_FAKE_PROB']))
        ff = decision_heads.file_score(ens, dict(Pv=float(r['VOICE_PRESENT_PROB']), Pm=float(r['MUSIC_PRESENT_PROB']), Fv=fv, Fm=fm))
        df = abs(ff - float(r['FILE_FAKE_PROB']))
        assert all(math.isfinite(v) for v in (fv, fm, ff, dv, dm, df)), s['ID']
        worst_file = max(worst_file, df)
        worst = {'voice': max(worst['voice'], dv), 'music': max(worst['music'], dm)}
        print(f"{s['ID'][:34]:34s} voice offline {fv:.4f} runtime {float(r['VOICE_FAKE_PROB']):.4f} | music offline {fm:.4f} runtime {float(r['MUSIC_FAKE_PROB']):.4f}")
ok = worst['voice'] <= .02 and worst['music'] <= .01 and worst_file <= .02
print('WORST', {k: round(v, 5) for k, v in worst.items()}, 'file', round(worst_file, 5), DEVICE, 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
