from pathlib import Path
import time

import numpy as np
import torch

from ads_cached_experiments import Cohort, Heads, bce, file_fusion, probabilities, train_arm
from graph_scoring import LABEL_COLUMNS
from v5_fusion import joint_probabilities


def cohort() -> tuple[Cohort, dict[str, torch.Tensor]]:
    labels, rows = [], []
    for i in range(288):
        voice, music = i % 9 // 3, i % 3
        truth = [int(voice == 2 or music == 2), int(voice == 2) if voice else np.nan,
                 int(music == 2) if music else np.nan, int(voice > 0), int(music > 0)]
        role = 'train' if i < 270 else 'selection' if i < 279 else 'calibration'
        rows.append(dict(ID=f'row{i}', role=role, **{name: str(int(value)) if np.isfinite(value) else ''
                                                  for name, value in zip(LABEL_COLUMNS, truth, strict=True)}))
        labels.append(truth)
    x = torch.randn((288, 12), generator=torch.Generator().manual_seed(19))
    data = Cohort(rows, x, torch.tensor(labels), ())
    indices = {role: torch.tensor([i for i, row in enumerate(rows) if row['role'] == role])
               for role in ('train', 'selection', 'calibration')}
    return data, indices


def test_unknown_targets_cannot_backpropagate_fake_evidence() -> None:
    target = torch.tensor([[1., float('nan'), 1., 0., 1.], [0., 0., float('nan'), 1., 0.]])
    logits = torch.zeros_like(target, requires_grad=True)
    bce(logits, target, torch.ones((5, 2))).backward()
    assert torch.equal(logits.grad[torch.isnan(target)], torch.zeros(2))
    assert torch.all(logits.grad[torch.isfinite(target)].abs() > 0)
    assert torch.all(torch.sign(logits.grad[torch.isfinite(target)]) ==
                     1 - 2 * target[torch.isfinite(target)])


def test_all_heads_update_full_tail_pass_and_saved_checkpoint_replays(tmp_path: Path) -> None:
    data, indices = cohort()
    first_batch = torch.randperm(270, generator=torch.Generator().manual_seed(41))[:128]
    data.labels[first_batch, 2] = torch.nan
    samplers = []
    torch.set_num_threads(2)
    for arm in ('independent5', 'shared5', 'joint9'):
        output = tmp_path / arm
        output.mkdir()
        record = train_arm(data, indices, dict(arm=arm, seed=41, epochs=2), output, time.monotonic() + 60)
        assert record['completed'] and record['updates'] == 6
        assert all(epoch['train_rows'] == 270 for epoch in record['history'])
        assert all(p['gradient_norm'] > 0 and p['max_update'] > 0
                   for p in record['gradient_and_update_audit'].values())
        if arm == 'independent5':
            assert all(p['update'] > 1 for name, p in record['gradient_and_update_audit'].items()
                       if name.startswith('heads.2.'))
        saved = torch.load(output / 'best.pt', weights_only=True)
        assert saved['best_epoch'] == min(record['history'], key=lambda row: row['selection_bce'])['epoch']
        restored = Heads(12, arm)
        restored.load_state_dict(saved['model'])
        with torch.inference_mode(), np.load(output / 'predictions.npz') as result:
            np.testing.assert_allclose(probabilities(restored, data.features).numpy(), result['probabilities'], rtol=0, atol=1e-7)
        samplers.append(record['sampler_sha256'])
    assert len(set(samplers)) == 1


def test_joint_roundoff_preserves_valid_probabilities() -> None:
    logits = torch.tensor([2.69995546, 18.44358444, 1.30518007, 8.88653660, -19.21269989,
                           15.71553326, 3.19232607, 25.96337891, 14.08759022])
    model = Heads(12, 'joint9')
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.heads[0][-1].bias.copy_(logits)
        predicted = probabilities(model, torch.zeros((1, 12)))
    assert float(joint_probabilities(logits[None])[0, 4]) > 1
    assert torch.all((predicted >= 0) & (predicted <= 1))
    torch.testing.assert_close(predicted.double(), joint_probabilities(logits.double()[None]), rtol=0, atol=2e-7)


def test_file_fit_preserves_components_and_never_fits_heldout_rows(tmp_path: Path) -> None:
    data, indices = cohort()
    data.labels[0, 0] = torch.nan
    random = np.random.default_rng(42)
    baseline = random.uniform(.05, .95, (288, 5))
    baseline[:, 1:] = .1 + .7 * data.labels[:, 1:].nan_to_num().numpy() + .1 * random.random((288, 4))
    first = tmp_path / 'first'
    first.mkdir()
    report = file_fusion(data, baseline, first, time.monotonic() + 60)
    assert report['final_loss'] < report['initial_loss'] and report['max_weight'] > 0
    with np.load(first / 'predictions.npz') as result:
        assert np.array_equal(result['probabilities'][:, 1:], baseline[:, 1:])
    altered = baseline.copy()
    altered[0, 1:] = 1 - altered[0, 1:]
    altered[270:, 1:] = 1 - altered[270:, 1:]
    labels = data.labels.clone()
    labels[270:, 0] = 1 - labels[270:, 0]
    second = tmp_path / 'second'
    second.mkdir()
    file_fusion(Cohort(data.rows, data.features, labels, ()), altered, second, time.monotonic() + 60)
    before = torch.load(first / 'model.pt', weights_only=True)
    after = torch.load(second / 'model.pt', weights_only=True)
    assert torch.equal(before['mean'], after['mean']) and torch.equal(before['scale'], after['scale'])
    assert all(torch.equal(value, after['model'][name]) for name, value in before['model'].items())
