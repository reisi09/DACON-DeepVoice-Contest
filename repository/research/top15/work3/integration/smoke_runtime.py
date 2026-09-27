"""CPU plumbing check of the staged runtime voice fusion (Engine.load_fusion + Engine.fused_voice) on real audio, <1 GB RAM.
Question: given a fitted heads dir (S4 / R / A), does the runtime load exactly the right detectors and heads from
assets/antideepfake/<member>, compute ft = window max of the ft detector and ad = seed-mean ResMLP window max of the ad
detector, and fuse them with the fitted tables? Not loaded: DF-Arena, MERT, HTDemucs, PANNs (the full package run is
verify_datav4_submission.py, >6 GB). Detector weights: the real fp16 AntiDeepfake checkpoint cut to 2 (ad) / 3 (ft)
encoder layers, so the two detectors differ while the loader, key layout and windowing are the real ones.

  python smoke_runtime.py --work SCRATCH_DIR
"""
import argparse
import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT)]
import full_bank_submission as rt  # noqa: E402  (the staged copy: HERE is first on sys.path)
from antideepfake_runtime import AntiDeepfake  # noqa: E402
from component_fusion import fuse  # noqa: E402
import voice_heads  # noqa: E402

assert Path(rt.__file__).resolve().parent == HERE, rt.__file__


def cut(src: Path, dst: Path, n: int):
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / 'antideepfake_config.json').read_text()) | {'num_hidden_layers': n}
    (dst / 'antideepfake_config.json').write_text(json.dumps(cfg))
    with safe_open(str(src / 'model.safetensors'), 'pt') as f:
        keep = {k: f.get_tensor(k) for k in f.keys()
                if not (m := re.match(r'm_ssl\.model\.encoder\.layers\.(\d+)\.', k)) or int(m[1]) < n}
    save_file(keep, str(dst / 'model.safetensors'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--ad-weights', type=Path, default=ROOT / '_workspace/models/xls-r-1b-anti-deepfake-fp16')
    ap.add_argument('--audio', type=Path, default=ROOT / '_workspace/audit_20260927/parity_audio')
    ap.add_argument('--fits', type=Path, default=HERE / 'out', help='fit_ft_fusion.py p1 outputs p1_S4, p1_R, p1_A')
    a = ap.parse_args()
    torch.set_num_threads(4)
    assets = a.work / 'assets'
    for name, n in (('ad', 2), ('ft', 3)):
        if not (assets / 'antideepfake' / name / 'model.safetensors').exists():
            cut(a.ad_weights, assets / 'antideepfake' / name, n)
    ref = {n: AntiDeepfake(assets / 'antideepfake' / n) for n in ('ad', 'ft')}   # independent copies
    audio = []
    for p in sorted(a.audio.iterdir())[:4]:   # speech (flac) + music (wav); raw mix stands in for the voice stem
        x = librosa.load(p, sr=16000, mono=True)[0]
        audio.append((p.name, (x * min(1., .6 / float(np.abs(x).max()))).astype(np.float32)))
    out = {}
    for option, expect in (('S4', {'ad'}), ('R', {'ft'}), ('A', {'ad', 'ft'})):
        trained = a.fits / f'p1_{option}'
        e = rt.Engine.__new__(rt.Engine)
        e.device = torch.device('cpu')
        e.load_fusion(assets, trained)
        assert set(e.detectors) == expect and len(e.ad_heads) == 3 * ('ad' in expect), (option, e.detectors.keys())
        spec = json.loads((trained / 'component_fusion.json').read_text())['voice']
        heads = [voice_heads.load(torch.load(trained / f'voice_ad_{s}.pt', weights_only=True)).eval()
                 for s in (41, 42, 43)] if 'ad' in expect else []
        worst = 0.
        for name, x in audio:
            with torch.inference_mode():
                emb, prob = ref['ad'].windows(x)
                m = {'ours': .6}
                if 'ad' in expect:
                    m['ad'] = float(np.mean([float(h(torch.from_numpy(emb)).softmax(-1)[:, 0].max()) for h in heads]))
                if 'ft' in expect:
                    m['ft'] = float(ref['ft'].windows(x)[1].max())
                    assert abs(m['ft'] - float(prob.max())) > 1e-4, 'ft and ad detectors must differ for this check'
            got = e.fused_voice(x, .6)
            worst = max(worst, abs(got - fuse(spec, m)))
            out.setdefault(option, {})[name] = {'fused': round(got, 6), **{k: round(v, 6) for k, v in m.items()}}
        assert worst < 1e-6, (option, worst)
        out[option]['max_abs_vs_reference'] = worst
    bad = a.work / 'bad_trained'
    bad.mkdir(exist_ok=True)
    (bad / 'component_fusion.json').write_text(json.dumps({'voice': {'weights': {'ours': 1, 'ft_top30': 1}, 'tables': {}}}))
    e = rt.Engine.__new__(rt.Engine)
    e.device = torch.device('cpu')
    try:
        e.load_fusion(assets, bad)
        raise AssertionError('unsupported member accepted')
    except rt.SubmissionError:
        out['unsupported_member'] = 'SubmissionError raised'
    print(json.dumps(out, indent=1))
    print('smoke_runtime ok')


if __name__ == '__main__':
    main()
