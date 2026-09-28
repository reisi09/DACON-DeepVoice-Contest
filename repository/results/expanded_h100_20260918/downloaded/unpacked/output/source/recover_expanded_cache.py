"""Seal fully computed, interrupted feature files before resuming the frozen notebook."""
from pathlib import Path
import csv
import gc
import json
import time
import importlib.metadata

import numpy as np
import nbformat
from nbclient import NotebookClient
from eight_contract import sha256_file
from localization_features import atomic_json, frame_truth, window_layout, WINDOW
from localization_train import read_items


def recover(job: Path, deadline: float) -> None:
    began = time.time()
    source = job / 'input/source'
    cache, output = job / 'cache', job / 'output'
    manifest = job / 'input/dataset/manifest.csv'
    rows = list(csv.DictReader(manifest.open()))
    previous = json.loads((output / 'feature_progress.json').read_text())
    evaluation = sum(row['split'] != 'train' for row in rows)
    if previous.get('stage') != 'fixed_voice' or previous.get('done') != evaluation or previous.get('total') != evaluation:
        raise ValueError('Only the fully completed fixed-branch stage may be sealed')
    if (cache / 'index.json').exists() or (output / 'train/config.json').exists():
        raise ValueError('An existing index or training run must not be replaced')
    identity = json.loads((output / 'feature_identity.json').read_text())
    assert identity['manifest_sha256'] == sha256_file(manifest)
    assert identity['source_sha256'] == sha256_file(source / 'localization_features.py')
    for name, digest in identity['implementation_sha256'].items():
        assert sha256_file(source / name) == digest, name
    for name, version in identity['versions'].items():
        assert importlib.metadata.version(name) == version, name
    assets = Path('/root/redesign_h100_20260917/assets')
    for name, digest in identity['all_frozen_assets_sha256'].items():
        assert sha256_file(assets / name) == digest, name
    index = {'identity': identity, 'items': {}}
    windows = 0
    for number, row in enumerate(rows, 1):
        if time.time() >= deadline:
            raise TimeoutError('Original experiment deadline reached during recovery')
        audio = job / 'input/dataset' / row['path']
        assert sha256_file(audio) == row['audio_sha256'], row['ID']
        features, prefix = cache / 'features' / (row['ID'] + '.npz'), cache / 'prefix' / (row['ID'] + '.npy')
        with np.load(features, allow_pickle=False) as saved:
            starts, valid = saved['starts'], saved['valid_samples']
            length = int(starts[-1] + valid[-1])
            assert np.array_equal(starts, window_layout(length)), row['ID']
            assert np.array_equal(valid, np.minimum(WINDOW, length - starts)), row['ID']
            targets, mask = frame_truth(row, starts, valid)
            np.testing.assert_array_equal(saved['frame_targets'], targets)
            np.testing.assert_array_equal(saved['frame_mask'], mask)
            assert len(saved['mert_windows']) == len(starts) and np.isfinite(saved['mert_windows']).all(), row['ID']
            if row['split'] != 'train':
                for key in ('VOICE_FAKE_PROB', 'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB'):
                    assert np.isfinite(saved[key]).all() and 0 <= float(saved[key]) <= 1, (row['ID'], key)
        encoded = np.load(prefix, mmap_mode='r', allow_pickle=False)
        assert encoded.shape == (len(starts), 1214, 768) and encoded.dtype == np.float16, row['ID']
        assert np.isfinite(encoded).all(), row['ID']
        # Unwritten memmap windows are zero; a learned AST prefix cannot be all-zero.
        assert np.any(encoded != 0, axis=(1, 2)).all(), row['ID']
        del encoded
        index['items'][row['ID']] = {'features': str(features.relative_to(cache)), 'ast_prefix': str(prefix.relative_to(cache)),
                                    'features_sha256': sha256_file(features), 'ast_prefix_sha256': sha256_file(prefix)}
        windows += len(starts)
        if number % 200 == 0:
            atomic_json(output / 'recovery_progress.json', {'verified_files': number, 'total': len(rows)})
    assert windows == json.loads((job / 'input/data_preparation.json').read_text())['windows']
    recovery = {'reason': 'completed arrays lacked final index after workspace interruption', 'previous_progress': previous,
                'files_verified': len(rows), 'windows_verified': windows, 'verification_seconds': time.time() - began,
                'original_stage_timers_unavailable': True, 'deadline_epoch': deadline}
    index.update(windows=windows, fixed_branch_rows=evaluation, stage_seconds={},
                 elapsed_seconds=previous['elapsed_seconds'] + recovery['verification_seconds'], recovery=recovery)
    atomic_json(cache / 'index.json', index)
    items = read_items(manifest, cache, job / 'input/ast_assets')
    assert len(items) == len(rows)
    del items
    gc.collect()
    atomic_json(output / 'feature_cache_receipt.json', index)
    atomic_json(output / 'cache_recovery.json', recovery)
    atomic_json(output / 'feature_progress.json', {'stage': 'completed', 'files': len(rows), 'windows': windows, 'recovered_finalization': True})


def main() -> None:
    import os
    job = Path(os.environ['EXPANDED_JOB_ROOT'])
    deadline = float(os.environ['DEADLINE_EPOCH']) - 600
    recover(job, deadline)
    notebook = nbformat.read(job / 'input/DACON_Expanded_H100_Experiments.ipynb', as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        if 'capacity = canary(' in cell.source:
            cell.source = cell.source.replace("capacity = canary(INPUT, JOB/'cache', OUTPUT, ASSETS, DEADLINE)", "capacity = json.loads((OUTPUT/'capacity_gate.json').read_text())")
        if "command(feature_arguments(INPUT/'dataset'" in cell.source:
            cell.source = cell.source.replace("command(feature_arguments(INPUT/'dataset', JOB/'cache', OUTPUT, ASSETS,\n                              INPUT/'ast_assets', DEADLINE), OUTPUT/'features.log', DEADLINE)", "assert (OUTPUT/'cache_recovery.json').is_file()")
            assert "command(feature_arguments" not in cell.source, 'Feature cell recovery substitution did not match'
    notebook.cells.insert(0, nbformat.v4.new_markdown_cell('Recovery: completed feature arrays were verified against frozen identities, labels and numeric integrity before training. Original deadline preserved; no held-out predictions existed before recovery.'))
    target = job / 'output/DACON_Expanded_H100_Executed.ipynb'
    client = NotebookClient(notebook, timeout=None, kernel_name='expanded-h100', resources={'metadata': {'path': str(job)}})
    client.on_cell_executed = lambda **kwargs: nbformat.write(notebook, target)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, target)


if __name__ == '__main__':
    main()
