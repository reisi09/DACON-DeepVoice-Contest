"""CPU smoke (answers: do MuQ / EAT-large load OFFLINE on the evaluation-server package set, does the ready-to-run
extractor work end to end on runtime preprocessing, are the features sane, and what do they cost relative to MERT-330M?).
Imports of packages absent from the server (timm, easydict, nnAudio, x_clip, pip muq) are blocked; HF is offline.
  python smoke.py            -> logs/smoke.json
"""
import csv
import gc
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

BLOCKED = {'timm', 'easydict', 'nnAudio', 'x_clip', 'muq'}


class Absent(importlib.abc.MetaPathFinder):
    """PathFinder that cannot see BLOCKED packages, as on the server (find_spec None -> ModuleNotFoundError on import)."""
    @staticmethod
    def find_spec(name, path=None, target=None):
        return None if name.split('.')[0] in BLOCKED else importlib.machinery.PathFinder.find_spec(name, path, target)

    @staticmethod
    def find_distributions(*args, **kwargs):   # importlib.metadata: hide the same distributions
        return (d for d in importlib.machinery.PathFinder.find_distributions(*args, **kwargs)
                if (d.metadata['Name'] or '').replace('-', '_').lower() not in {b.lower() for b in BLOCKED})


sys.meta_path = [Absent() if f is importlib.machinery.PathFinder else f for f in sys.meta_path]
assert importlib.util.find_spec('timm') is None and importlib.util.find_spec('easydict') is None
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import extract_new_backbones as X  # noqa: E402

AUDIO = ROOT / '_workspace/audit_20260927/parity_audio'
FILES = ['fake_16900_udio_1__81062a7b5a.mp3', 'librispeech_test-clean_1580-141083-0008__a832b3a4a5.flac']
MUQ, EAT = HERE / 'ckpt/MuQ-large-msd-iter', HERE / 'ckpt/EAT-large_epoch20_finetune_AS2M'
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)   # macOS: bytes
torch.set_num_threads(4)
report = {}

# 1. the ready-to-run extractor, unchanged, on CPU
manifest = HERE / 'logs/smoke_manifest.csv'
with manifest.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID', 'path'])
    for name in FILES:
        w.writerow([name.rsplit('.', 1)[0], name])
out = HERE / 'smoke_cache'
t = time.time()
sys.argv = ['x', '--manifest', str(manifest), '--input', str(AUDIO), '--assets', str(ROOT / '_workspace/assets_e9504'),
            '--muq', str(MUQ), '--eat', str(EAT), '--out', str(out), '--device', 'cpu']
X.main()
report['extract_seconds_2_files'] = round(time.time() - t, 1)
report['peak_rss_gb_after_extract'] = rss()
for name in FILES:
    with np.load(out / f"{name.rsplit('.', 1)[0]}.npz") as d:
        report[name] = {k: list(d[k].shape) for k in d.files} | {
            'samples': int(d['samples']), 'voice_silent': bool(d['voice_silent']),
            'finite': all(bool(np.isfinite(d[k].astype(np.float32)).all()) for k in ('muq_backing', 'muq_voice', 'eat_backing')),
            'muq_backing_layer_norms': [round(float(np.linalg.norm(d['muq_backing'][:, i].astype(np.float32), axis=-1).mean()), 2) for i in range(len(X.MUQ_LAYERS))],
            'cos_backing_vs_voice_L6': round(float(np.mean([np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b) for a, b in zip(
                d['muq_backing'][:, 3].astype(np.float32), d['muq_voice'][:, 3].astype(np.float32))])), 3)}
report['weights_sha256'] = json.loads((out / 'weights.json').read_text())

# 2. MuQ pipeline sanity: with correct preprocessing the model predicts its own Mel-RVQ targets on clean input far above chance
import librosa  # noqa: E402
wav = librosa.load(AUDIO / FILES[0], sr=24000, mono=True, duration=10.0)[0][None]
muq = X.load_muq(MUQ, 'cpu')
keys = list(load_file(str(MUQ / 'model.safetensors')).keys())
with torch.inference_mode():
    x = torch.from_numpy(wav)
    targets, _ = muq.model.get_targets(x)
    logits, _ = muq.model.get_predictions(x)
    lg = logits['melspec_2048']                              # (1, T, 8 * 1024): codebook-major like get_loss()
    codes = targets['melspec_2048'].view(1, lg.shape[-1] // 1024, -1)   # (1, 8, T'): codebook-major like tokenize()
    T = min(lg.shape[1], codes.shape[-1])
    lg, codes = lg[:, :T], codes[..., :T]
    acc = (lg.view(1, T, -1, 1024).argmax(-1).transpose(1, 2) == codes).float().mean(-1)[0]
report['muq_rvq_in_checkpoint'] = any(k.startswith('model.rvq.') for k in keys)
report['muq_token_accuracy_per_codebook_10s_udio'] = [round(float(a), 3) for a in acc]
report['muq_chance_accuracy'] = round(1 / 1024, 5)

# 3. fp16-stored weights (packaged) vs HF fp32 weights, same window
from mert12_submission import window_layout, windows  # noqa: E402
audio16 = librosa.load(AUDIO / FILES[0], sr=16000, mono=True)[0].astype(np.float32)
chunks = windows(audio16, window_layout(len(audio16)))[:8]
f16 = X.muq_features(muq, chunks, 'cpu').astype(np.float32)
from muq_core import MuQ, MuQConfig  # noqa: E402
m32 = MuQ(MuQConfig(**json.loads((MUQ / 'config.json').read_text())))
m32.load_state_dict(load_file(str(MUQ / 'model.safetensors')), strict=True)
m32.eval()
f32 = X.muq_features(m32, chunks, 'cpu').astype(np.float32)
report['muq_fp16_weights_rel_diff_per_layer'] = [round(float(np.linalg.norm(f16[:, i] - f32[:, i]) / np.linalg.norm(f32[:, i])), 5) for i in range(f16.shape[1])]
del m32
gc.collect()

# 4. CPU cost per batch of 8 runtime windows (4 threads): MuQ vs EAT-large vs MERT-330M (the current music encoder)
def timed(fn, reps=2):
    fn()
    t = time.time()
    for _ in range(reps):
        fn()
    return round((time.time() - t) / reps, 2)


report['cpu_seconds_per_8_windows'] = {'muq': timed(lambda: X.muq_features(muq, chunks, 'cpu'))}
del muq
gc.collect()
eat = X.load_eat(EAT, 'cpu')
report['cpu_seconds_per_8_windows']['eat_large'] = timed(lambda: X.eat_features(eat, chunks, 'cpu'))
del eat
gc.collect()
from transformers import AutoFeatureExtractor, AutoModel  # noqa: E402
processor = AutoFeatureExtractor.from_pretrained(ROOT / '_workspace/assets_e9504/mert', local_files_only=True)
mert = AutoModel.from_pretrained(ROOT / '_workspace/assets_e9504/mert', trust_remote_code=True, local_files_only=True).eval()


def run_mert():   # full_bank_submission.Engine.predict music path
    with torch.inference_mode():
        audio = [librosa.resample(c, orig_sr=16000, target_sr=24000, res_type='soxr_hq') for c in chunks]
        values = processor(audio, sampling_rate=24000, return_tensors='pt', padding=True).input_values
        mert(values, output_hidden_states=True)


report['cpu_seconds_per_8_windows']['mert_330m'] = timed(run_mert)
report['peak_rss_gb'] = rss()
report['params_millions'] = {'muq': 333.4, 'eat_large': 309.4, 'mert_330m': round(sum(p.numel() for p in mert.parameters()) / 1e6, 1)}
(HERE / 'logs/smoke.json').write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
