"""Does the runtime load an fp16-stored MERT pytorch_model.bin into the same fp32 model and outputs as the fp32 bin?
Loads both with the runtime's exact call (full_bank_submission.Engine: AutoModel.from_pretrained(trust_remote_code,
local_files_only, output_loading_info)) and checks: empty loading info, every parameter/buffer fp32 and bitwise equal,
and bitwise-equal hidden layers 6/12 on real 24 kHz windows (the runtime music features). CPU, ~3.5 GB RAM.

  python check_mert_fp16.py --fp16-bin BIN [--mert FP32_DIR] [--work DIR] [--audio FILE ...]
BIN: e.g. model/assets/mert/pytorch_model.bin extracted from a --mert-fp16 package.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ap = argparse.ArgumentParser()
ap.add_argument('--fp16-bin', type=Path, required=True)
ap.add_argument('--mert', type=Path, default=ROOT / '_workspace/assets_e9504/mert')
ap.add_argument('--work', type=Path, required=True)
ap.add_argument('--audio', type=Path, nargs='+', default=[ROOT / '_workspace/audit_20260927/parity_audio/haim_musicgen_musicgen_small_00029__f7f6acf7a5.mp3',
                                                             ROOT / '_workspace/audit_20260927/parity_audio/core_63ab76f979000832863b_base__2b1a70e367.wav'])
a = ap.parse_args()
os.environ.update(HF_HOME=str(a.work / 'hf_home'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
sys.path.insert(0, str(ROOT))
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mert12_submission import AutoFeatureExtractor, AutoModel  # noqa: E402

torch.set_num_threads(4)
half = a.work / 'mert_fp16'
half.mkdir(parents=True, exist_ok=True)
for f in a.mert.iterdir():
    if f.name != 'pytorch_model.bin' and f.is_file():
        shutil.copy(f, half / f.name)
shutil.copy(a.fp16_bin, half / 'pytorch_model.bin')
stored = torch.load(half / 'pytorch_model.bin', map_location='cpu', weights_only=True)
dtypes = sorted({str(v.dtype) for v in stored.values()})
del stored
models = {}
for tag, d in (('fp32', a.mert), ('fp16_bin', half)):
    model, info = AutoModel.from_pretrained(d, trust_remote_code=True, local_files_only=True, output_loading_info=True)
    assert not any(info.values()), (tag, info)
    models[tag] = model.eval().requires_grad_(False)
s32, s16 = (models[t].state_dict() for t in ('fp32', 'fp16_bin'))
assert s32.keys() == s16.keys()
not_fp32 = sorted(k for k, v in s16.items() if v.is_floating_point() and v.dtype != torch.float32)
unequal = sorted(k for k in s32 if not torch.equal(s32[k], s16[k]))
processor = AutoFeatureExtractor.from_pretrained(a.mert, local_files_only=True)
worst = 0.
with torch.inference_mode():
    for p in a.audio:
        x = librosa.load(p, sr=24000, mono=True, duration=8.)[0]
        wins = [x[i:i + 96960] for i in range(0, max(1, len(x) - 96960 + 1), 48480)][:2]   # 4.04 s windows
        v = processor(wins, sampling_rate=24000, return_tensors='pt', padding=True).input_values
        h = {t: m(v, output_hidden_states=True).hidden_states for t, m in models.items()}
        for k in (6, 12):
            worst = max(worst, float((h['fp32'][k].mean(1) - h['fp16_bin'][k].mean(1)).abs().max()))
report = {'stored_dtypes': dtypes, 'loaded_params_not_fp32': not_fp32, 'state_tensors_unequal': unequal,
          'n_state_tensors': len(s32), 'hidden_6_12_max_abs_diff': worst, 'transformers': __import__('transformers').__version__,
          'torch': torch.__version__}
print(json.dumps(report, indent=1))
assert 'torch.float16' in dtypes, 'bin is not fp16-stored'
assert not not_fp32 and not unequal and worst == 0., 'fp16 MERT bin does not reproduce the fp32 model'
print('check_mert_fp16 ok')
