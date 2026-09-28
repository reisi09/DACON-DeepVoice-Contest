# /// script
# requires-python = ">=3.11"
# ///
# Run with the existing job venv, after suspending its feature worker.
# Compares identical train-only excerpts; this is not a full-pipeline benchmark.
import csv
import gc
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

JOB = Path('/root/dacon_existing9504_a100_20260919')
config = json.loads((JOB / 'config/run_config.json').read_text())
sys.path[:0] = [config['code_dir'], config['source_dir']]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import soundfile as sf
import torch
from dacon9504a100.gpu import configure_numerics
from mert12_submission import pin_torchaudio_loader, window_layout, windows
from v6_runtime import baseline_module
configure_numerics()
base = baseline_module(Path(config['assets_dir']))
pin_torchaudio_loader(base)
rows = list(csv.DictReader(Path(config['manifest']).open()))
report = {'scope': '4 distinct train excerpts per case; equal duration within case; model inference timing, excludes decoding, cache writes, other model stages', 'cases': {}}
separator = base.load_htdemucs_model().to('cuda').eval().requires_grad_(False)
backings = {}

def measured(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start

def parity(left, right):
    a, b = left.detach().float().cpu().numpy(), right.detach().float().cpu().numpy()
    return {'max_abs': float(np.max(np.abs(a-b))), 'rmse': float(np.sqrt(np.mean((a-b)**2))),
            'atol': 1e-4, 'rtol': 1e-3, 'passed': bool(np.allclose(a,b,atol=1e-4,rtol=1e-3))}

with tempfile.TemporaryDirectory(prefix='dacon-batch-probe-', dir='/tmp') as temporary:
    for name, prefix, seconds in [('speech4s','asvspoof2019_la:',4), ('music30s','bank_fma_',30)]:
        prepared, means, stds, selected = [], [], [], []
        for row in rows:
            if row['split'] != 'train' or not row['ID'].startswith(prefix):
                continue
            raw = base.load_audio(Path(config['data_dir']) / row['path'])
            if len(raw) < 16000 * seconds:
                continue
            peak = float(np.abs(raw).max())
            gain = 1. if row.get('input_kind','').startswith(('legacy','constructed')) or peak <= .6 else .6 / peak
            raw = (raw * gain)[:16000 * seconds]
            path = Path(temporary) / (name + str(len(selected)) + '.wav')
            sf.write(path, raw, 16000, subtype='FLOAT')
            wave = base.load_track(path, separator.audio_channels, separator.samplerate).float()
            mono = wave.mean(0); mean, std = mono.mean(), mono.std()
            if float(std) < 1e-8:
                continue
            prepared.append((wave-mean)/std); means.append(mean); stds.append(std); selected.append(row['ID'])
            if len(selected) == 4:
                break
        assert len(selected) == 4
        batch = torch.stack(prepared)
        kwargs = dict(device='cuda',shifts=0,split=True,overlap=.25,progress=False)
        def sequential():
            return torch.cat([base.apply_model(separator,x[None],**kwargs) for x in prepared])
        def grouped():
            return base.apply_model(separator,batch,**kwargs)
        sequential(); grouped()  # warm both execution paths before measuring
        serial_times, batch_times = [], []
        for _ in range(3):
            serial, elapsed = measured(sequential); serial_times.append(elapsed)
            grouped_output, elapsed = measured(grouped); batch_times.append(elapsed)
        comparison = parity(serial,grouped_output)
        report['cases'][name] = {'IDs': selected, 'excerpt_seconds':seconds,
            'demucs': {'sequential_seconds':serial_times,'batch4_seconds':batch_times,
                'median_speedup':statistics.median(serial_times)/statistics.median(batch_times),'parity':comparison}}
        originals = []
        for i, sources in enumerate(grouped_output):
            sources = sources * stds[i] + means[i]
            music = torch.stack([sources[j] for j,s in enumerate(separator.sources) if s!='vocals']).sum(0).mean(0,keepdim=True)
            music = base.torchaudio.functional.resample(music,separator.samplerate,16000)[0]
            originals.append(music[:seconds*16000].cpu().numpy().astype(np.float32))
        backings[name] = originals
        print('DEMUCs',name,json.dumps(report['cases'][name]),flush=True)
        del serial,grouped_output,batch,prepared

del separator
gc.collect();torch.cuda.empty_cache()
from transformers import AutoFeatureExtractor,AutoModel
import librosa
processor=AutoFeatureExtractor.from_pretrained(Path(config['assets_dir'])/'mert',local_files_only=True)
model=AutoModel.from_pretrained(Path(config['assets_dir'])/'mert',trust_remote_code=True,local_files_only=True).to('cuda').eval().requires_grad_(False)
for name, originals in backings.items():
    inputs=[]
    for raw in originals:
        pieces=windows(raw,window_layout(len(raw)))
        resampled=[librosa.resample(x,orig_sr=16000,target_sr=24000,res_type='soxr_hq') for x in pieces]
        inputs.append(processor(resampled,sampling_rate=24000,return_tensors='pt',padding=True).input_values)
    joined=torch.cat(inputs)
    def forward(value):
        with torch.inference_mode():
            return model(value.to('cuda'),output_hidden_states=True).hidden_states[12].mean(1).float().cpu()
    def sequential_mert():
        return torch.cat([forward(x) for x in inputs])
    def batch_mert():
        return torch.cat([forward(x) for x in joined.split(64)])
    sequential_mert();batch_mert()
    serial_times,batch_times=[],[]
    for _ in range(3):
        a,elapsed=measured(sequential_mert);serial_times.append(elapsed)
        b,elapsed=measured(batch_mert);batch_times.append(elapsed)
    result={'windows_per_file':[len(x) for x in inputs],'pooled_batch_cap':64,
        'sequential_seconds':serial_times,'pooled_seconds':batch_times,
        'median_speedup':statistics.median(serial_times)/statistics.median(batch_times),
        'parity':parity(a,b)}
    report['cases'][name]['mert']=result
    print('MERT',name,json.dumps(result),flush=True)
report['all_parity_passed']=all(c[stage]['parity']['passed'] for c in report['cases'].values() for stage in ['demucs','mert'])
Path('/tmp/dacon_batch_probe_result.json').write_text(json.dumps(report,indent=2)+'\n')
print('BENCHMARK_COMPLETE',report['all_parity_passed'],flush=True)
