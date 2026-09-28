"""Rebuild the byte-identical ZIP and test its offline H100 inference on real audio.

Run in the existing workspace: python full_bank_submission_smoke.py STAGING_DIR
The staging directory contains package_small.zip, submit.json, and zip_order.json.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
import torch


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main(stage):
    job = Path('/root/balanced7144_h100_20260919')
    work = job/'submission_qa'
    work.mkdir(exist_ok=True)
    package = work/'package'
    package.mkdir(exist_ok=True)
    report = json.loads((stage/'submit.json').read_text())
    with zipfile.ZipFile(stage/'package_small.zip') as archive:
        archive.extractall(package)
    import imageio_ffmpeg
    assets = Path('/root/redesign_h100_20260917/assets')
    for name, expected in report['member_sha256'].items():
        target = package/name
        if not target.exists():
            source = (assets/name.removeprefix('model/assets/') if name.startswith('model/assets/')
                      else Path(imageio_ffmpeg.get_ffmpeg_exe()))
            if digest(source) != expected:
                raise ValueError('Staging identity mismatch: '+name)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        if digest(target) != expected:
            raise ValueError('Packaged bytes changed: '+name)
    archive_path = work/'submit.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in json.loads((stage/'zip_order.json').read_text()):
            if name in ('script.py', 'requirements.txt', 'model/manifest.json'):
                archive.writestr(zipfile.ZipInfo(name), (package/name).read_bytes())
            else:
                with (package/name).open('rb') as incoming, archive.open(name, 'w', force_zip64=True) as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 8*1024*1024)
    if digest(archive_path) != report['sha256']:
        raise ValueError('Reconstructed ZIP differs from the local submission ZIP')
    # Extract the verified ZIP itself, so every executed file comes from the deliverable.
    runtime = work/'extracted'
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC failure')
        archive.extractall(runtime)
    sys.path.insert(0, str(runtime/'model/runtime'))
    from full_bank_submission import Engine, load_heads, PREDICTION_COLUMNS, SUBMISSION_COLUMNS
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    rows = read_rows(job/'output/dataset_manifest.csv')
    index = json.loads((job/'cache/index.json').read_text())
    complete = json.loads((job/'output/features/features_complete.json').read_text())
    if set(index['items']) != {row['ID'] for row in rows} or digest(job/'cache/index.json') != complete['index_sha256']:
        raise ValueError('Incomplete or changed 7,144-row feature index')
    regular = []
    for route in ('voice', 'music'):
        for label in ('0', '1'):
            eligible = [r for r in rows if r['split'] == 'val' and r['route'] == route
                        and r['FILE_FAKE'] == label and not r['input_kind'].startswith('legacy')]
            eligible.sort(key=lambda r: float(r['duration_seconds']))
            regular.append(eligible[len(eligible)//2])
    regular.append(next(r for r in rows if r['ID'] == 'bank_fma_97601'))
    def cached_peak(row):
        with np.load(job/'cache'/index['items'][row['ID']]['path'], allow_pickle=False) as data:
            return float(data['source_peak'])
    legacy = [max((r for r in rows if r['split'] == 'val' and r['input_kind'].startswith('legacy')
                   and r['FILE_FAKE'] == label), key=cached_peak)
              for label in ('0', '1')]
    chosen = regular+legacy
    music, voice, fake = load_heads(runtime/'model/trained', 'cuda')
    reference = {}
    provenance = []
    stored = {r['ID']:r for r in read_rows(job/'output/train/validation_predictions.csv')}
    for row in chosen:
        entry = index['items'][row['ID']]
        cache = job/'cache'/entry['path']
        audio = job/'input/dataset'/row['path']
        if digest(cache) != entry['sha256'] or digest(audio) != row['audio_sha256']:
            raise ValueError('QA input/cache bytes changed')
        with np.load(cache, allow_pickle=False) as data:
            features = torch.from_numpy(data['mert_windows']).cuda()
            embeddings = torch.from_numpy(data['voice_embeddings']).cuda()
            vp, mp = map(float, data['presence'])
            silent = bool(data['voice_silent'])
            provenance.append({'ID':row['ID'], 'split':row['split'], 'route':row['route'],
                'label':row['FILE_FAKE'], 'input_kind':row['input_kind'], 'audio_sha256':row['audio_sha256'],
                'cache_sha256':entry['sha256'], 'seconds':int(data['samples'])/16000,
                'source_peak':float(data['source_peak']), 'normalization_gain':float(data['normalization_gain'])})
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            ms = []
            for head in music:
                logits, _ = head.head((features-head.mean)/head.scale)
                ms.append(float((logits.float().logsumexp(0)-np.log(len(features))).sigmoid()))
            vs = [0. if silent else float(head(embeddings).float().softmax(-1)[:,fake].max()) for head in voice]
        mf, vf = float(np.mean(ms)), float(np.mean(vs))
        reference[row['ID']] = [max(vp*vf, mp*mf), vf, mf, vp, mp]
        if row['ID'] in stored:
            np.testing.assert_allclose(reference[row['ID']], [float(stored[row['ID']][k]) for k in PREDICTION_COLUMNS], atol=1e-7, rtol=0)
    del music, voice, head, features, embeddings, logits
    gc.collect()
    torch.cuda.empty_cache()
    (work/'input_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (work/'reference.json').write_text(json.dumps(reference, indent=2)+'\n')
    guard = work/'guard'
    guard.mkdir(exist_ok=True)
    (guard/'sitecustomize.py').write_text("import sys\ndef deny(event,args):\n    if event in {'socket.connect','socket.getaddrinfo'}: raise RuntimeError('OFFLINE_NETWORK_BLOCKED')\nsys.addaudithook(deny)\nprint('OFFLINE_GUARD_ACTIVE',flush=True)\n")
    env = os.environ | {'PYTHONPATH':str(guard)+os.pathsep+os.environ.get('PYTHONPATH',''),
        'HF_HOME':str(work/'empty_hf_cache'), 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1'}
    tolerance = 2e-3  # CUDA bf16 head rounding and different full-float MERT batch sizes.
    def run_cli(selected, name):
        folder = work/name
        inputs = folder/'inputs'
        inputs.mkdir(parents=True, exist_ok=True)
        for row in selected:
            source = job/'input/dataset'/row['path']
            target = inputs/(row['ID']+source.suffix)
            if not target.exists():
                os.link(source, target)
        sample = folder/'sample.csv'
        with sample.open('w',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
            writer.writeheader()
            writer.writerows(dict(ID=r['ID'], **dict.fromkeys(PREDICTION_COLUMNS, 0)) for r in selected)
        with (folder/'runtime.log').open('w') as log:
            subprocess.run([sys.executable, str(runtime/'script.py'), '--test-dir',str(inputs),
                '--sample-submission',str(sample),'--output',str(folder/'predictions.csv'),'--device','cuda'],
                cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if 'OFFLINE_GUARD_ACTIVE' not in (folder/'runtime.log').read_text():
            raise ValueError('Network guard was not loaded')
        actual = read_rows(folder/'predictions.csv')
        if [r['ID'] for r in actual] != [r['ID'] for r in selected]:
            raise ValueError('Submission ID/order contract changed')
        error = max(float(np.max(np.abs(np.array([float(r[k]) for k in PREDICTION_COLUMNS])-reference[r['ID']]))) for r in actual)
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(f'Packaged inference parity failed: {name} {error}')
        return error
    began = time.time()
    regular_error = run_cli(regular, 'regular')
    subset_error = run_cli(regular[1::-1], 'reordered_subset')
    # Legacy inputs were deliberately not peak-normalized during feature extraction.
    # Test that exact preprocessing separately; the public CLI always caps new inputs.
    os.environ.update(env)
    decoder = runtime/'model/runtime/bin/ffmpeg'
    os.environ['PATH'] = str(decoder.parent)+os.pathsep+os.environ['PATH']
    def deny(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo'}:
            raise RuntimeError('OFFLINE_NETWORK_BLOCKED')
    sys.addaudithook(deny)
    engine = Engine(runtime/'model/assets', runtime/'model/trained', 'cuda')
    legacy_predictions = {r['ID']:engine.predict(job/'input/dataset'/r['path'], preserve_gain=True) for r in legacy}
    legacy_error = max(float(np.max(np.abs(np.array([legacy_predictions[r['ID']][k] for k in PREDICTION_COLUMNS])-reference[r['ID']]))) for r in legacy)
    if not np.isfinite(legacy_error) or legacy_error > tolerance:
        raise ValueError(f'Legacy gain-preserving parity failed: {legacy_error}')
    return {'passed':True, 'submission_sha256':report['sha256'], 'identical_zip_verified':True,
        'feature_index_rows':len(index['items']), 'regular_files':len(regular), 'legacy_files':len(legacy),
        'regular_max_abs_error':regular_error, 'reordered_subset_max_abs_error':subset_error,
        'legacy_preserve_gain_max_abs_error':legacy_error, 'absolute_tolerance':tolerance, 'relative_tolerance':0,
        'network_connections_forbidden':True, 'gpu':torch.cuda.get_device_name(0), 'seconds':time.time()-began,
        'official_l4_validated':False, 'independent_test':False, 'official_submission':False,
        'legacy_scope':'Packaged Engine with preserve_gain=True, separate from the default public CLI'}


if __name__ == '__main__':
    stage = Path(sys.argv[1])
    started = time.time()
    try:
        result = main(stage)
    except Exception:
        result = {'passed':False, 'traceback':traceback.format_exc()}
    work = Path('/root/balanced7144_h100_20260919/submission_qa')
    with zipfile.ZipFile(stage/'evidence.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('input_provenance.json','reference.json','regular/runtime.log','regular/predictions.csv',
                     'regular/runtime_receipt.json','reordered_subset/runtime.log',
                     'reordered_subset/predictions.csv','reordered_subset/runtime_receipt.json'):
            if (work/name).exists():
                archive.write(work/name, name)
    result['evidence_sha256'] = digest(stage/'evidence.zip')
    if result['passed']:
        # Only remove reproducible QA copies; trained checkpoints, caches and audio stay intact.
        for name in ('package', 'extracted'):
            shutil.rmtree(work/name)
        (work/'submit.zip').unlink()
        result['temporary_cleanup'] = {name:not (work/name).exists() for name in ('package','extracted','submit.zip')}
    result['started_at'], result['finished_at'] = started, time.time()
    temporary = stage/'smoke_result.tmp'
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(stage/'smoke_result.json')
    print(json.dumps(result),flush=True)
    sys.exit(0 if result['passed'] else 1)
