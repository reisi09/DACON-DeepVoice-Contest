"""Controlled frozen-MERT data ablations; reuse V6 scoring and file-level MIL."""

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from graph_scoring import official_metrics
from v6_core import COLUMNS, Heads, fusion_features
from v6_data import assert_disjoint, write_json
from v6_train import mil_logits


@dataclass(frozen=True, slots=True)
class Recipe:
    name: str
    nsynth_mass: float | None
    equal_windows: bool = False
    seed: int = 41
    epochs: int = 300
    patience: int = 20
    min_delta: float = 1e-4


class EarlyStopping:
    """Track validation improvements and retain a restorable best checkpoint."""

    def __init__(self, patience: int, min_delta: float) -> None:
        if patience < 1 or not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError('Invalid early stopping configuration')
        self.patience, self.min_delta = patience, min_delta
        self.best_loss = self.reference = float('inf')
        self.selected_epoch = self.stale = 0
        self.history: list[float] = []
        self.state: dict[str, torch.Tensor] = {}

    def step(self, model: torch.nn.Module, loss: float, epoch: int) -> bool:
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite validation loss')
        self.history.append(loss)
        if loss < self.best_loss:
            self.best_loss, self.selected_epoch = loss, epoch
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if loss < self.reference - self.min_delta:
            self.reference, self.stale = loss, 0
        else:
            self.stale += 1
        return self.stale >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if not self.state:
            raise ValueError('No evaluated checkpoint to restore')
        model.load_state_dict(self.state)


def source_weights(rows: list[dict[str, str]], mass: float | None) -> np.ndarray:
    labels = np.array([int(r['MUSIC_FAKE']) for r in rows])
    real, fake = labels == 0, labels == 1
    if not real.any() or not fake.any():
        raise ValueError('Both music classes are required')
    if mass is None:
        return np.where(real, .5 / real.sum(), .5 / fake.sum())
    synth = np.array(['nsynth' in r['source'].lower() for r in rows]) & real
    ordinary = real & ~synth
    if not ordinary.any() or not 0 <= mass < .5 or (mass > 0 and not synth.any()):
        raise ValueError('Invalid real-source composition or NSynth mass')
    providers = [r['source'].split('/')[1] for r in rows if r['MUSIC_FAKE'] == '1']
    counts = Counter(providers)
    weights = np.zeros(len(rows))
    weights[ordinary] = (.5 - mass) / ordinary.sum()
    if synth.any():
        weights[synth] = mass / synth.sum()
    for i, row in enumerate(rows):
        if fake[i]:
            provider = row['source'].split('/')[1]
            weights[i] = .5 / len(counts) / counts[provider]
    np.testing.assert_allclose([weights[real].sum(), weights[fake].sum()], [.5, .5])
    return weights


def fit_head(rows, records, recipe: Recipe, deadline: float, *, validation=None):
    if recipe.epochs < 1 or recipe.patience < 1 or not np.isfinite(recipe.min_delta) or recipe.min_delta < 0:
        raise ValueError('Invalid epoch limit, patience or minimum improvement')
    if validation is not None:
        assert_disjoint(rows, validation[0])
    weights = source_weights(rows, recipe.nsynth_mass)
    selected = [(row, weight) for row, weight in zip(rows, weights, strict=True) if weight > 0]
    bags = [records[r['ID']]['music'][records[r['ID']]['music_audible'] > 0]
            for r, _ in selected]
    if any(len(bag) == 0 for bag in bags):
        raise ValueError('A training file has no audible music; review instead of silently dropping it')
    windows = np.concatenate(bags).astype('float32')
    if recipe.equal_windows:
        # Equal total contribution per file, including files with only one unique crop.
        mean = np.stack([b.mean(0) for b in bags]).mean(0)
        second = np.stack([(b.astype('float64') ** 2).mean(0) for b in bags]).mean(0)
        scale = np.sqrt(np.maximum(second - mean.astype('float64') ** 2, 0)).clip(1e-4).astype('float32')
    else:
        mean, scale = windows.mean(0), windows.std(0).clip(1e-4)
    x = torch.from_numpy((windows - mean) / scale)
    sizes = np.array([len(bag) for bag in bags])
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    indices = torch.repeat_interleave(torch.arange(len(bags)), torch.from_numpy(sizes))
    y = torch.tensor([int(r['MUSIC_FAKE']) for r, _ in selected], dtype=torch.float32)
    weight = torch.tensor([w for _, w in selected], dtype=torch.float32)
    torch.manual_seed(recipe.seed)
    model = torch.nn.Linear(768, 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    if validation is not None:
        val_rows, val_records = validation
        val_bags = [val_records[r['ID']]['music'] for r in val_rows]
        val_x = torch.from_numpy(((np.concatenate(val_bags) - mean) / scale).astype('float32'))
        val_audible = torch.from_numpy(np.concatenate([
            val_records[r['ID']]['music_audible'] for r in val_rows]).astype('float32'))
        val_indices = torch.repeat_interleave(torch.arange(len(val_rows)),
                                              torch.tensor([len(b) for b in val_bags]))
        val_y = torch.tensor([int(r['MUSIC_FAKE']) for r in val_rows], dtype=torch.float32)
        val_weight = torch.tensor(source_weights(val_rows, None), dtype=torch.float32)
    rng = np.random.default_rng(recipe.seed)
    losses = []
    stopping = EarlyStopping(recipe.patience, recipe.min_delta)
    for epoch in range(1, recipe.epochs + 1):
        if time.time() >= deadline:
            raise TimeoutError('Training deadline; completed candidates remain saved')
        optimizer.zero_grad(set_to_none=True)
        if recipe.equal_windows:
            chosen = offsets + (rng.random(len(bags)) * sizes).astype(int)
            logits = model(x[chosen]).flatten()
        else:
            logits = mil_logits(model(x).flatten(), indices, len(bags))
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none') * weight).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite weighted training loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if validation is not None:
            with torch.inference_mode():
                probabilities = torch.sigmoid(model(val_x).flatten()) * val_audible
                file_probabilities = mil_logits(probabilities, val_indices, len(val_rows))
                val_loss = float((torch.nn.functional.binary_cross_entropy(
                    file_probabilities, val_y, reduction='none') * val_weight).sum())
            if stopping.step(model, val_loss, epoch):
                break
    if validation is not None:
        stopping.restore(model)
    head = Heads(mean, scale, model.weight.detach().numpy().ravel().copy(),
                 np.array(model.bias.detach().item()), np.zeros(5), np.array(0.))
    return head, losses, {'files': len(bags), 'windows': int(sizes.sum()),
                          'real_loss_mass': float(weight[y == 0].sum()),
                          'fake_loss_mass': float(weight[y == 1].sum()),
                          'epochs_completed': len(losses),
                          'selected_epoch': stopping.selected_epoch if validation is not None else len(losses),
                          'early_stopping': validation is not None,
                          'validation_losses': stopping.history,
                          'best_validation_loss': stopping.best_loss if validation is not None else None}


def evaluate(head: Heads, rows, records):
    predictions = [dict(ID=r['ID'], **dict(zip(COLUMNS, head.predict(records[r['ID']]).tolist()))) for r in rows]
    return predictions, official_metrics(rows, predictions)


def run_candidate(recipe: Recipe, context, deadline: float):
    split, records, output = context
    rows = [r for r in split['expert'] if r['MUSIC_PRESENT'] == '1' and r['MUSIC_FAKE'] in ('0', '1')]
    head, losses, composition = fit_head(rows, records, recipe, deadline)
    fusion_rows = [r for r in split['fusion'] if r['FILE_FAKE'] in ('0', '1')]
    matrix = []
    for row in fusion_rows:
        record = records[row['ID']]
        music = head.music(record['music'], record['music_audible'])
        matrix.append(fusion_features(record['presence'], record['voice'], music, record['raw']))
    fusion = LogisticRegression(C=1., class_weight='balanced', max_iter=2000, random_state=41)
    fusion.fit(matrix, [int(r['FILE_FAKE']) for r in fusion_rows])
    if fusion.n_iter_.max() >= fusion.max_iter:
        raise RuntimeError('Fusion fitting did not converge')
    head = Heads(head.mean, head.scale, head.music_weight, head.music_bias,
                 fusion.coef_[0], np.array(fusion.intercept_[0]))
    destination = output / recipe.name
    destination.mkdir(parents=True, exist_ok=True)
    head.save(destination / 'heads.npz')
    predictions, full = evaluate(head, split['dev'], records)
    unique = list({r['pcm_sha256']: r for r in reversed(split['dev'])}.values())
    _, dedup = evaluate(head, unique, records)
    panels = {}
    for name, allowed in {'seen': {'music_seen', 'music_real'}, 'unseen': {'music_ood', 'music_real'},
                          'mixture': {'mixture'}, 'splice': {'splice'}}.items():
        _, metrics = evaluate(head, [r for r in unique if r['panel'] in allowed], records)
        panels[name] = metrics['metrics']['music_eer']
    report = dict(recipe=asdict(recipe), composition=composition,
        full=full, deduplicated=dedup, panels=panels, initial_loss=losses[0], final_loss=losses[-1])
    write_json(destination / 'report.json', report)
    write_json(destination / 'predictions.json', predictions)
    write_json(destination / 'losses.json', losses)
    # Reference save/reload parity protects the artifact actually returned to the user.
    example = records[unique[0]['ID']]
    np.testing.assert_array_equal(head.predict(example), Heads.load(destination / 'heads.npz').predict(example))
    return report
