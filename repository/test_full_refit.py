"""Protect full-data coverage and actual last-block optimization."""
import csv
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import ASTConfig, ASTFeatureExtractor, ASTModel

from ast_multitask import ASTMultiTask
from graph_scoring import LABEL_COLUMNS
import full_refit


def test_all_roles_train_each_epoch_without_changing_frozen_blocks(tmp_path, monkeypatch):
    rows = []
    for index, role in enumerate(('dev_train', 'dev_select', 'test_generator', 'test_source')):
        path = f'{index}.wav'
        sf.write(tmp_path / path, np.sin(np.arange(4000) * (.01 + index * .002)), 16000)
        rows.append({'ID': str(index), 'path': path, 'split': role,
                     **{c: str(index % 2) for c in LABEL_COLUMNS}})
    for folder, subset in [('development', rows[:2]), ('locked_test', rows[2:])]:
        (tmp_path / folder).mkdir()
        with (tmp_path / folder / 'manifest.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    data = full_refit.read_full_data(tmp_path)
    config = ASTConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=32, max_length=1024, num_mel_bins=128)
    model = ASTMultiTask(ASTModel(config), 'last_block')
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    monkeypatch.setattr(full_refit, 'save_training_checkpoint', lambda *a: None)
    job = full_refit.RefitJob(data, tmp_path / 'result', torch.device('cpu'), 2, time.time()+120)
    report = full_refit.train_stage(model, ASTFeatureExtractor(), job)
    assert report['exposures'] == {'0': 2, '1': 2, '2': 2, '3': 2}
    assert all(r['unique_files'] == 4 and r['head_gradient'] > 0 and r['backbone_gradient'] > 0
               for r in report['epochs'])
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters() if not p.requires_grad)
    assert 'proxy' not in report and 'best_epoch' not in report
    with (tmp_path / 'locked_test/manifest.csv').open('a') as f:
        f.write('0,0.wav,test_source,0,0,0,0,0\n')
    with pytest.raises(ValueError, match='duplicate'):
        full_refit.read_full_data(tmp_path)
