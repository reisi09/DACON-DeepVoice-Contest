"""Reopen the actual ZIP and verify CPU execution and A100 cached-feature parity separately."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from c08_submission import score_ensemble
from music_temporal import load_temporal_head
from temporal_resolution import load_pair


def main():
    output=Path(__file__).resolve().parent
    archive=output/'M330_MLP_full1206_submit.zip'
    receipt=json.loads(archive.with_suffix('.json').read_text())
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    stage=ROOT/f'_workspace/mlp_full_package_check_{int(time.time())}'
    if stage.exists():raise FileExistsError('Use a fresh extraction for package verification')
    with zipfile.ZipFile(archive) as z:
        assert {name.split('/')[0] for name in z.namelist()}=={'script.py','requirements.txt','model'}
        assert all((stage/name).resolve().is_relative_to(stage.resolve()) for name in z.namelist())
        z.extractall(stage)
    inputs=stage/'data/test';inputs.mkdir(parents=True)
    old=ROOT/'_workspace/c08_package_check'
    real_id='025e3d128394bcf37fdd0947'
    shutil.copy2(old/'data/test/smoke.wav',inputs/'smoke.wav')
    shutil.copy2(old/f'real_data/test/{real_id}.wav',inputs/f'{real_id}.wav')
    sample=old/'real_data/sample_submission.csv'
    with sample.open() as f:fields=csv.DictReader(f).fieldnames
    with (stage/'data/sample_submission.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows([dict(ID='smoke'),dict(ID=real_id)])
    (stage/'output').mkdir()
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    env.pop('PYTHONPATH',None)
    started=time.time()
    with (output/'offline_inference.log').open('w') as log:
        subprocess.run([str(ROOT/'.venv/bin/python'),'script.py','--device','cpu'],cwd=stage,
                       env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300,check=True)
    seconds=time.time()-started
    with (stage/'output/submission.csv').open() as f:actual=list(csv.DictReader(f))
    assert [r['ID'] for r in actual]==['smoke',real_id]
    real=actual[1]
    run=ROOT/'results/mlp_full_a100_20260916'
    manifest=json.loads((run/'unpacked/reference330_features_manifest.json').read_text())
    item=manifest['records'][real_id]
    p=run/'smoke_reference.npz'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    record,_=load_pair(p,item['identity'],1024)
    heads=[load_temporal_head(run/f'unpacked/final_run/fold{i}/temporal_music.pt') for i in range(5)]
    baseline=json.loads((ROOT/'submissions/C08/cpu_real_audio.json').read_text())['h100_reference']
    expected=score_ensemble(heads,record,baseline['VOICE_FAKE_PROB'],
                           (baseline['VOICE_PRESENT_PROB'],baseline['MUSIC_PRESENT_PROB']),'cpu')
    columns=['FILE_FAKE_PROB','VOICE_FAKE_PROB','MUSIC_FAKE_PROB','VOICE_PRESENT_PROB','MUSIC_PRESENT_PROB']
    measured=[float(real[k]) for k in columns]
    cross_environment_error=float(np.max(np.abs(np.array(measured)-expected)))
    with np.load(output/'cpu_record.npz') as z:cpu_record={k:z[k] for k in z.files}
    cpu_expected=score_ensemble(heads,cpu_record,measured[1],(measured[3],measured[4]),'cpu')
    np.testing.assert_allclose(measured,cpu_expected,atol=1e-6,rtol=0)
    with (output/'gpu_submission.csv').open() as f:gpu=list(csv.DictReader(f))[0]
    assert gpu['ID']==real_id
    with np.load(output/'gpu_record.npz') as z:gpu_record={k:z[k] for k in z.files}
    np.testing.assert_allclose(gpu_record['music'],record['music'],atol=1e-6,rtol=0)
    gpu_measured=[float(gpu[k]) for k in columns]
    gpu_expected=score_ensemble(heads,record,gpu_measured[1],(gpu_measured[3],gpu_measured[4]),'cpu')
    np.testing.assert_allclose(gpu_measured,gpu_expected,atol=1e-4,rtol=0)
    result=dict(status='verified',zip_sha256=receipt['sha256'],fresh_zip_extraction=True,
                offline_cpu_inference=True,files=2,seconds=seconds,predictions=actual,
                cpu_captured_feature_parity_max_abs_error=float(np.max(np.abs(np.array(measured)-cpu_expected))),
                gpu_cached_reference_max_abs_error=float(np.max(np.abs(np.array(gpu_measured)-gpu_expected))),
                cross_environment_max_abs_error=cross_environment_error,
                cross_environment_parity=False,
                limitation='CPU and training A100 use different device and torch/librosa/numpy versions. Music difference0.00694 remains cross-environment drift; no claim of device-independent parity.',
                gpu_real_audio_files=1,
                tolerance=1e-4,L4_full_runtime_verified=False,official_score=None,utc=time.time())
    (output/'package_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
