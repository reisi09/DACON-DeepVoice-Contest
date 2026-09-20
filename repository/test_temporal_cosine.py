"""Protect cosine geometry, training loss, scheduler wiring and saved inference.

Run: OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q test_temporal_cosine.py
"""

from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pytest
import torch
from torch import nn

from music_temporal import TemporalMusic, _batch, fit_temporal_head, load_temporal_head
from temporal_backbone import COSINE_C08
from test_temporal_augmentation import samples
from v7_ablation import source_weights


def test_cosine_geometry_and_trunk_gradients() -> None:
    # Given: wider features, including invalid windows that must not affect inference.
    torch.manual_seed(41)
    model = TemporalMusic(.1, simplified=True, compact=True, input_dim=1024,
                          softmax_output=True, cosine_output=True).eval()
    x = torch.randn(4, 7, 1024)
    mask = torch.zeros(4, 7, dtype=torch.bool)
    mask[:, -2:] = True
    h = torch.randn(4, 64)
    # When: compare the actual head with independently computed cosine scores.
    w = model.output.weight.detach().numpy()
    v = h.numpy()
    expected = 10 * (v / np.linalg.norm(v, axis=1, keepdims=True)) @ (
        w / np.linalg.norm(w, axis=1, keepdims=True)).T
    np.testing.assert_allclose(model.output(h).detach().numpy(), expected, atol=1e-6)
    torch.testing.assert_close(model.output(h * 7), model.output(h), atol=1e-6, rtol=1e-6)
    assert not any(isinstance(layer, nn.Linear) for layer in model.modules())
    altered = x.clone()
    altered[mask] = float('nan')
    torch.testing.assert_close(model(x, mask), model(altered, mask), atol=0, rtol=0)
    logits = model(x, mask)
    nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1]), label_smoothing=.05).backward()
    # Then: both representatives and every temporal convolution participate in learning.
    for parameter in [model.output.weight, model.projection.weight,
                      *(layer.weight for layer in model.blocks[0].convs)]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.all(parameter.grad.flatten(1).abs().sum(1) > 0)
    assert model.projection.weight.grad[:, 768:].abs().sum() > 0
    assert model(x, torch.ones_like(mask)).softmax(-1)[:, 1].max() < 1e-6


def test_fit_uses_smoothed_targets_but_validation_and_reload_do_not(tmp_path: Path) -> None:
    # Given: one deterministic batch, so its expected loss can be calculated independently.
    rows, records = samples()
    records = {key: dict(record, music=np.pad(record['music'], ((0, 0), (0, 256))))
               for key, record in records.items()}
    recipe = replace(COSINE_C08, epochs=1, batch_size=8, dropout=0.,
                     crop_probability=0., feature_mask_probability=0., input_dim=1024)
    torch.manual_seed(recipe.seed)
    initial = TemporalMusic(0., simplified=True, compact=True, softmax_output=True,
                            cosine_output=True, input_dim=1024)
    x, mask = _batch(rows[:8], records, 'cpu', input_dim=1024)
    log_p = initial(x, mask).log_softmax(-1).detach().numpy()
    targets = np.array([[.975, .025] if r['MUSIC_FAKE'] == '0' else [.025, .975]
                        for r in rows[:8]])
    expected = (-targets * log_p).sum(1) @ source_weights(rows[:8], 0.)
    # When: run the real fit/save/load path.
    model, report = fit_temporal_head(rows[:8], records, (rows[8:], records), tmp_path,
                                      time.time() + 60, recipe=recipe)
    loaded = load_temporal_head(tmp_path / 'temporal_music.pt')
    x_val, mask_val = _batch(rows[8:], records, 'cpu', input_dim=1024)
    logits = loaded(x_val, mask_val)
    labels = np.array([int(r['MUSIC_FAKE']) for r in rows[8:]])
    raw_loss = -logits.log_softmax(-1).detach().numpy()[np.arange(4), labels]
    # Then: smoothing affects training only; the persisted classifier reproduces inference.
    np.testing.assert_allclose(report['train_losses'][0], expected, atol=1e-6)
    np.testing.assert_allclose(report['best_validation_loss'],
                               raw_loss @ source_weights(rows[8:], None), atol=1e-6)
    torch.testing.assert_close(model(x_val, mask_val), logits, atol=0, rtol=0)
    assert not torch.equal(initial.output.weight, loaded.output.weight)


def test_plateau_reduces_actual_fit_lr_after_five_stale_epochs_and_respects_floor(tmp_path: Path) -> None:
    # Given: an unreachable improvement threshold forces a known plateau without mocking losses.
    rows, records = samples()
    recipe = replace(COSINE_C08, epochs=12, min_delta=100., learning_rate=2e-6,
                     dropout=0., crop_probability=0., feature_mask_probability=0.)
    # When: the actual training loop observes five, then ten, stale validation epochs.
    _, scheduled = fit_temporal_head(rows[:8], records, (rows[8:], records), tmp_path / 'scheduled',
                                     time.time() + 60, recipe=recipe)
    _, fixed = fit_temporal_head(rows[:8], records, (rows[8:], records), tmp_path / 'fixed',
                                 time.time() + 60, recipe=replace(recipe, lr_plateau=False))
    # Then: epoch 7 uses half the LR; further reductions stop at the floor and affect optimization.
    assert scheduled['learning_rates'] == [2e-6] * 6 + [1e-6] * 6
    assert fixed['learning_rates'] == [2e-6] * 12
    assert abs(scheduled['train_losses'][-1] - fixed['train_losses'][-1]) > 1e-6


def test_invalid_head_and_smoothing_combinations_fail_before_training(tmp_path: Path) -> None:
    # Given/When: incompatible heads or smoothing that would be silently ignored by BCE.
    with pytest.raises(ValueError, match='Cosine output requires'):
        TemporalMusic(simplified=True, compact=True, softmax_output=True,
                      mlp_output=True, cosine_output=True)
    rows, records = samples()
    for recipe in (replace(COSINE_C08, cosine_output=False, softmax_output=False),
                   replace(COSINE_C08, label_smoothing=float('nan')),
                   replace(COSINE_C08, learning_rate=1e-7)):
        with pytest.raises(ValueError):
            fit_temporal_head(rows[:8], records, (rows[8:], records), tmp_path,
                              time.time() + 60, recipe=recipe)
    # Then: invalid configurations cannot produce a misleading trained checkpoint.
    assert not (tmp_path / 'temporal_music.pt').exists()
