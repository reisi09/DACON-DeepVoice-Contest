"""Protect effective-batch math, benchmark isolation, and holdout selection order."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch
from transformers import ASTConfig, ASTModel

import localization_train as runner
from localization_models import ASTLocalizer, WindowMIL, frame_targets_to_patches


def item(identifier: str, windows: int, label: int, prefix: Path, *, split: str = 'train') -> runner.Item:
    rng = np.random.default_rng(int(identifier))
    frames = np.full((windows, 1024), label, dtype=np.float32)
    mask = np.arange(1024)[None] < np.full((windows, 1), 48)
    frames[0, :16] = np.nan
    truth = np.full(windows, label, dtype=np.float32)
    truth[0] = np.nan
    row = {'ID': identifier, 'split': split, 'source': 'example', 'generator': str(label),
           'parent_id': identifier, 'MUSIC_FAKE': str(label), 'MUSIC_PRESENT': '1',
           'probe_family': 'test', 'duration_seconds': '8'}
    patches = frame_targets_to_patches(torch.from_numpy(frames), torch.from_numpy(mask)).numpy()
    return runner.Item(row, rng.normal(size=(windows, 4)).astype(np.float32), frames, mask,
                       truth, patches, prefix, {})


def test_microbatch_loss_is_invariant_for_variable_bags_and_unknown_truth(tmp_path: Path) -> None:
    # Unequal local-label counts used to bias accumulated gradients by microbatch size.
    torch.manual_seed(10)
    examples = [item(str(i), i + 1, i % 2, tmp_path / 'unused') for i in range(4)]
    full = WindowMIL(torch.zeros(4), torch.ones(4))
    split = copy.deepcopy(full)
    optimizer_full = torch.optim.SGD(full.parameters(), lr=.05)
    optimizer_split = torch.optim.SGD(split.parameters(), lr=.05)
    deadline = time.time() + 60
    loss_full = runner.train_step(full, optimizer_full, examples, 'mil', 4, 20, 'cpu', deadline)
    loss_split = runner.train_step(split, optimizer_split, examples, 'mil', 1, 20, 'cpu', deadline)
    assert loss_full == pytest.approx(loss_split, abs=1e-7)
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, split.state_dict()[name], rtol=1e-6, atol=1e-7)
    before = runner.parameters_to_save(full)
    with pytest.raises(TimeoutError):
        runner.train_step(full, optimizer_full, examples, 'mil', 4, 20, 'cpu', time.time() - 1)
    for name, value in before.items():
        assert torch.equal(value, full.state_dict()[name])


def test_benchmark_restores_weights_and_rng_after_real_optimizer_steps(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = WindowMIL(torch.zeros(4), torch.ones(4))
    before = runner.parameters_to_save(model)
    rng = torch.get_rng_state().clone()
    examples = [item('1', 2, 1, tmp_path / 'unused')]
    selected, records = runner.benchmark(model, examples, 'mil', 'cpu', time.time() + 60)
    assert selected == 128 and records[0]['files_per_second'] > 0
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name])


def test_real_three_arm_runner_locks_dev_selection_before_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise cached tiny native AST + actual samplers/optimizers/serialization end to end.
    torch.set_num_threads(1)
    torch.manual_seed(12)
    config = ASTConfig(hidden_size=8, num_hidden_layers=2, num_attention_heads=2, intermediate_size=16,
                       max_length=1024, num_mel_bins=16, patch_size=16, time_stride=10, frequency_stride=10)
    backbone = ASTModel(config)
    encoder = ASTLocalizer(copy.deepcopy(backbone)).eval()
    cache = tmp_path / 'cache'
    cache.mkdir()
    examples = []
    entries = {}
    for i in range(8):
        current = item(str(i), 1 + i % 2, i % 2, cache / f'{i}.npy', split='train' if i < 4 else 'dev' if i < 6 else 'test')
        examples.append(current)
        np.save(current.prefix, encoder.encode_prefix(torch.randn(len(current.mert), 1024, 16)).numpy().astype(np.float16))
        np.savez(cache / f'{i}.npz', mert_windows=current.mert, frame_targets=current.frames, frame_mask=current.frame_mask)
        entries[str(i)] = {'features': f'{i}.npz', 'ast_prefix': f'{i}.npy',
                           'features_sha256': runner.sha256_file(cache / f'{i}.npz'),
                           'ast_prefix_sha256': runner.sha256_file(current.prefix)}
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(examples[0].row))
        writer.writeheader()
        writer.writerows(example.row for example in examples)
    config.to_json_file(tmp_path / 'config.json')
    (tmp_path / 'preprocessor_config.json').write_text('{}')
    (tmp_path / 'model.safetensors').write_bytes(b'bound tiny fixture')
    identity = {'manifest_sha256': runner.sha256_file(manifest), 'implementation_sha256':
                {name: runner.sha256_file(Path(runner.__file__).parent / name) for name in
                 ('localization_models.py', 'ast_multitask.py', 'mert12_submission.py')}, 'ast_assets_sha256':
                {name: runner.sha256_file(tmp_path / name) for name in ('config.json', 'preprocessor_config.json', 'model.safetensors')}}
    (cache / 'index.json').write_text(json.dumps({'items': entries, 'identity': identity}))
    monkeypatch.setattr(runner, 'original_training_items', lambda train: list(train))
    monkeypatch.setattr(ASTLocalizer, 'from_pretrained', classmethod(lambda cls, path: cls(copy.deepcopy(backbone))))
    original_predict = runner.predict
    output = tmp_path / 'output'
    heldout_calls = []

    def guarded_predict(model, items, microbatch, window_limit, device, deadline):
        if any(example.row['split'] == 'test' for example in items):
            lock = json.loads((output / 'candidate_lock.json').read_text())
            assert len(lock['checkpoints']) == 3
            assert lock['winner_by_dev_only'] in ('mean', 'mil', 'ast')
            heldout_calls.append(True)
        return original_predict(model, items, microbatch, window_limit, device, deadline)

    monkeypatch.setattr(runner, 'predict', guarded_predict)
    args = argparse.Namespace(manifest=manifest, cache_dir=cache, ast_assets=tmp_path, output_dir=output,
                              deadline_epoch=time.time() + 120, device='cpu', steps=1, eval_every=1, seeds=[41],
                              effective_batch=256, evaluate_training=False)
    runner.run(args)
    result = json.loads((output / 'results.json').read_text())
    assert result['status'] == 'completed' and len(heldout_calls) == 4
    exposure = json.loads((output / 'expanded_exposure.json').read_text())['41']['counts']
    assert set(exposure) == {'0', '1', '2', '3'} and sum(exposure.values()) == 256
    ast_saved = torch.load(output / 'ast_41.pt', weights_only=True)
    assert all(name.startswith(('backbone.encoder.layer.1.', 'backbone.layernorm.', 'head.')) for name in ast_saved['model'])
    for arm in ('mean', 'mil', 'ast'):
        predictions = np.load(output / f'{arm}_41_predictions.npz')
        assert predictions['ids'].tolist() == [str(i) for i in range(4, 8)]
        assert np.isfinite(predictions['music_probability']).all()
        assert result['metrics'][arm]['test']['known_eligible'] == 2
        assert result['metrics'][arm]['test']['official_score'] is None
    control = np.load(output / 'original831_mean_41_predictions.npz')['music_probability']
    primary = np.load(output / 'mean_41_predictions.npz')['music_probability']
    np.testing.assert_array_equal(control, primary)
    with (cache / '0.npy').open('ab') as stream:
        stream.write(b'changed cached bytes')
    with pytest.raises(runner.TrainingContractError, match='Feature cache bytes changed'):
        runner.read_items(manifest, cache, tmp_path)
