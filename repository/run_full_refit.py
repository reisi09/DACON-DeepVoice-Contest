"""Refit the selected AST recipe, then package and replay its final checkpoint."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
from collections import Counter

import numpy as np
import soundfile as sf
import torch

from ast_multitask import ASTMultiTask, export_checkpoint, load_processor
from full_refit import RefitJob, read_full_data, train_stage
from graph_features import atomic_json, digest
from graph_scoring import PREDICTION_COLUMNS
from package_submission import verify_packaged
from package_v2_submission import build_ast_submission
from run_experiment import write_csv
from train_v2 import seed_everything
from v2_inference import generate_submission


def main() -> None:
    """Only emit READY after all-data coverage and actual offline replay succeed."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'mps', 'cpu'), required=True)
    parser.add_argument('--seconds', type=int, default=3600)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + args.seconds
    device = torch.device(args.device)
    seed_everything()
    torch.set_num_threads(4)
    data = read_full_data(args.data.resolve())
    counts = dict(Counter(r['split'] for r in data.rows))
    assert counts == {'dev_train': 633, 'dev_select': 122, 'test_generator': 176, 'test_source': 98}
    assert data.manifest_hashes == {
        'development': '4fd72e3731de71c5ae4fccd4ca8a1e4d05f9e88a5f8143a60209e7cf9c2de01a',
        'locked_test': 'a11ae794de70f83c0a325f884d8ba920b4a1b3653a2936777a08a0be7c0c735d'}
    assert digest(args.assets / 'model.safetensors') == 'ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d'
    inventory = [{**r, 'audio_sha256': digest(data.root / r['path'])} for r in data.rows]
    atomic_json(root / 'full_training_inventory.json', inventory)
    protocol = {'purpose': 'final_refit', 'input_files': len(data.rows), 'original_roles': counts,
                'warmup_epochs': 8, 'finetune_epochs': 17,
                'initialization': 'fresh pinned AudioSet AST; fresh heads, full-data head warmup',
                'checkpoint_selection': 'fixed final epoch; no validation selection',
                'independent_validation_score': None, 'deadline_unix': deadline,
                'device': str(device), 'python': sys.version, 'torch': torch.__version__,
                'training_inventory_sha256': digest(root / 'full_training_inventory.json')}
    atomic_json(root / 'protocol.json', protocol)
    processor = load_processor(args.assets)
    model = ASTMultiTask.from_pretrained(args.assets, 'frozen')
    warmup = train_stage(model, processor, RefitJob(data, root / 'warmup', device, 8, deadline-120))
    model.configure_trainable('last_block')
    finetune = train_stage(model, processor, RefitJob(data, root / 'finetune', device, 17, deadline-120))
    assert all(v == 8 for v in warmup['exposures'].values()) and len(warmup['exposures']) == 1029
    assert all(v == 17 for v in finetune['exposures'].values()) and len(finetune['exposures']) == 1029
    checkpoint = root / 'final_checkpoint'
    export_checkpoint(model, processor, checkpoint)
    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    archive = build_ast_submission(checkpoint, root / 'DACON_236749_AST_full1029_epoch17.zip')
    unpacked = root / 'offline_replay'
    with zipfile.ZipFile(archive) as packed:
        assert packed.testzip() is None
        assert digest(checkpoint / 'model.safetensors') == __import__('hashlib').sha256(packed.read('model/ast/model.safetensors')).hexdigest()
        packed.extractall(unpacked)
    reference = root / 'reference'
    (reference / 'model').mkdir(parents=True)
    shutil.copytree(checkpoint, reference / 'model/ast')
    audio_dir = reference / 'data/test'
    audio_dir.mkdir(parents=True)
    # These transformed public clips verify deployment, not independent accuracy.
    waveform, rate = sf.read(data.root / data.rows[0]['path'], dtype='float32')
    sf.write(audio_dir / 'mono.wav', waveform, rate)
    sf.write(audio_dir / 'stereo.wav', np.column_stack((waveform, waveform*.7)), rate)
    sf.write(audio_dir / 'stress_60s.wav', np.tile(waveform, int(np.ceil(60*rate/len(waveform))))[:60*rate], rate)
    ids = ['mono', 'stereo', 'stress_60s']
    write_csv(reference / 'data/sample_submission.csv', [{'ID': i, **{c: 0 for c in PREDICTION_COLUMNS}} for i in ids])
    generate_submission(reference)
    shutil.copytree(reference / 'data', unpacked / 'data')
    guard = root / 'network_guard'
    guard.mkdir()
    (guard / 'sitecustomize.py').write_text("import socket\ndef deny(*a,**kw):\n    raise RuntimeError('offline replay forbids network')\nsocket.socket.connect=deny\nsocket.create_connection=deny\n")
    previous = os.environ.get('PYTHONPATH')
    os.environ['PYTHONPATH'] = str(guard)
    try:
        predicted = verify_packaged(unpacked, deadline)
    finally:
        if previous is None:
            os.environ.pop('PYTHONPATH', None)
        else:
            os.environ['PYTHONPATH'] = previous
    import csv
    with (reference / 'output/submission.csv').open() as stream:
        expected = list(csv.DictReader(stream))
    assert [r['ID'] for r in predicted] == ids
    delta = max(abs(float(a[c])-float(b[c])) for a,b in zip(expected,predicted,strict=True) for c in PREDICTION_COLUMNS)
    assert delta <= 1e-5, delta
    atomic_json(root / 'verification.json', {'network_blocked': True, 'max_delta': delta,
                'files': ids, 'inference_device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'L4_full_runtime': 'unverified', 'zip_sha256': digest(archive),
                'zip_bytes': archive.stat().st_size,
                'checkpoint_sha256': digest(checkpoint / 'model.safetensors')})
    atomic_json(root / 'finished.json', {'status': 'ready_not_submitted', 'full_training_files': 1029,
                'warmup_epochs': 8, 'finetune_epochs': 17, 'independent_validation_score': None,
                'elapsed_seconds': time.time()-started})
    print('FULL_REFIT_READY', archive, flush=True)


if __name__ == '__main__':
    main()
