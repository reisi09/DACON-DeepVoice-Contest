"""Experimental residual temporal CNN over ordered, cached MERT music windows."""

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import NotRequired, TypedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from v6_data import assert_disjoint, sha256, write_json
from v7_ablation import EarlyStopping, source_weights
from temporal_augmentation import augment_records, empty_counts


class MusicRecord(TypedDict):
    music: np.ndarray
    music_audible: np.ndarray
    starts: np.ndarray
    bin_times: NotRequired[np.ndarray]


Rows = list[dict[str, str]]
Records = dict[str, MusicRecord]


@dataclass(frozen=True, slots=True)
class TemporalRecipe:
    epochs: int = 300
    patience: int = 20
    min_delta: float = 1e-4
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = .01
    nsynth_mass: float = .05
    seed: int = 41
    dropout: float = .2
    crop_probability: float = 0.
    feature_mask_probability: float = 0.
    simplified: bool = False
    compact: bool = False
    temporal_bins: int = 1
    softmax_output: bool = False
    input_dim: int = 768
    mlp_output: bool = False
    cosine_output: bool = False
    label_smoothing: float = 0.
    lr_plateau: bool = False
    alternative_output: str = ''


class CosineClassifier(nn.Module):
    """Two learned class directions with a checkpointed, fixed logit scale of 10."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, channels))
        self.register_buffer('scale', torch.tensor(10.))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(F.normalize(features, dim=-1),
                                     F.normalize(self.weight, dim=-1))


class ResidualTemporalBlock(nn.Module):
    """Non-causal temporal convolutions; mask every layer to isolate padding."""

    def __init__(self, dilation: int, dropout: float = .2, *, channels: int = 128,
                 layers: int = 2) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(layers)])
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = hidden
        for norm, conv in zip(self.norms, self.convs, strict=True):
            hidden = norm(hidden.transpose(1, 2)).transpose(1, 2).masked_fill(mask, 0)
            hidden = self.dropout(F.gelu(conv(hidden))).masked_fill(mask, 0)
        return (residual + hidden).masked_fill(mask, 0)


class TemporalMusic(nn.Module):
    """Original gated head, two-layer max head, or compact three-layer max head."""

    def __init__(self, dropout: float = .2, *, simplified: bool = False,
                 compact: bool = False, temporal_bins: int = 1,
                 softmax_output: bool = False, input_dim: int = 768,
                 mlp_output: bool = False, cosine_output: bool = False,
                 alternative_output: str = '') -> None:
        super().__init__()
        if compact and not simplified:
            message = 'Compact head requires max-only simplified pooling'
            raise ValueError(message)
        self.simplified = simplified
        self.compact = compact
        if temporal_bins not in (1, 8) or ((temporal_bins != 1 or softmax_output) and not compact):
            message = 'Resolution and softmax experiments require the compact head and 1 or 8 bins'
            raise ValueError(message)
        if mlp_output and (not compact or not softmax_output or temporal_bins != 1):
            raise ValueError('MLP output requires compact one-bin two-class classification')
        if cosine_output and (mlp_output or not compact or not softmax_output or temporal_bins != 1):
            message = 'Cosine output requires compact one-bin two-class classification without MLP'
            raise ValueError(message)
        if input_dim not in (768, 1024, 1280) or (input_dim != 768 and (not compact or temporal_bins != 1 or (softmax_output and not (mlp_output or cosine_output)))):
            raise ValueError('1024-dimensional inputs require compact one-bin sigmoid, MLP or cosine output')
        if input_dim == 1280 and (not mlp_output or alternative_output):
            raise ValueError('AF-Whisper inputs require the original compact MLP head')
        self.input_dim = input_dim
        if alternative_output not in ('', 'prototype', 'attention', 'conv_mean', 'conv_lme'):
            raise ValueError('Unknown alternative classifier')
        if alternative_output and not (mlp_output and input_dim == 1024):
            raise ValueError('Alternative classifiers require the compact MERT330 MLP recipe')
        self.alternative_output = alternative_output
        self.temporal_bins = temporal_bins
        self.softmax_output = softmax_output
        self.architecture = ('mert-residual-tcn1-max-v1' if simplified
                             else 'mert-residual-tcn3-gated-max-v1')
        if compact:
            self.architecture = 'mert-residual-64x3-max-v1'
            if temporal_bins != 1 or softmax_output:
                self.architecture = f'mert-residual-64x3-max-b{temporal_bins}-softmax{int(softmax_output)}-v1'
        if input_dim == 1024:
            self.architecture = "mert-residual-64x3-max-d1024-v1"
        if mlp_output:
            self.architecture = f'mert-residual-64x3-max-d{input_dim}-mlp32-softmax-v1'
            if input_dim == 1280:
                self.architecture = 'afwhisper-residual-64x3-max-d1280-mlp32-softmax-v1'
        if cosine_output:
            self.architecture = f'mert-residual-64x3-max-d{input_dim}-cosine-softmax-v1'
        if alternative_output:
            self.architecture = f'mert-residual-64x3-d1024-{alternative_output}-v1'
        channels = 64 if compact else 128
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Conv1d(input_dim, channels, 1)
        self.blocks = nn.ModuleList([ResidualTemporalBlock(d, dropout, channels=channels,
                                                          layers=3 if compact else 2)
                                     for d in ((1,) if simplified else (1, 2, 4))])
        if not simplified:
            self.gate_tanh = nn.Conv1d(128, 64, 1)
            self.gate_sigmoid = nn.Conv1d(128, 64, 1)
            self.gate_score = nn.Conv1d(64, 1, 1, bias=False)
        if cosine_output:
            self.output = CosineClassifier(channels)
        else:
            self.output = (nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,2))
                           if mlp_output else nn.Linear(channels if simplified else 256, 2 if softmax_output else 1))
        if alternative_output == 'prototype':
            self.output = nn.Identity()
            self.prototypes = nn.Parameter(torch.randn(2, 3, channels))
        elif alternative_output == 'attention':
            self.attention_score = nn.Conv1d(channels, 1, 1)
        elif alternative_output.startswith('conv_'):
            self.output = nn.Conv1d(channels, 2, 1)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim or features.shape[1] < 1:
            raise ValueError(f'Expected nonempty [files, ordered windows, {self.input_dim}] MERT features')
        if padding_mask.shape != features.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError('Expected boolean [files, windows] invalid-window mask')
        features = features.masked_fill(padding_mask[..., None], 0)
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite valid MERT features')
        mask = padding_mask[:, None, :]
        hidden = self.projection(self.input_norm(features).transpose(1, 2))
        hidden = F.gelu(hidden).masked_fill(mask, 0)
        for block in self.blocks:
            hidden = block(hidden, mask)
        empty = padding_mask.all(dim=1)
        if self.alternative_output:
            safe_mask = mask.clone()
            safe_mask[empty, :, 0] = False
            if self.alternative_output.startswith('conv_'):
                scores = self.output(hidden)
                count = (~safe_mask).sum(2).to(scores.dtype)
                if self.alternative_output == 'conv_mean':
                    logits = scores.masked_fill(safe_mask, 0).sum(2) / count
                else:
                    logits = scores.masked_fill(safe_mask, -torch.inf).logsumexp(2) - count.log()
            elif self.alternative_output == 'attention':
                attention = self.attention_score(hidden).masked_fill(safe_mask, -torch.inf).softmax(2)
                logits = self.output((hidden * attention).sum(2))
            else:
                pooled = hidden.masked_fill(mask, -torch.inf).amax(2).masked_fill(empty[:, None], 0)
                similarities = 10 * torch.einsum('bc,kpc->bkp', F.normalize(pooled, dim=-1),
                                                F.normalize(self.prototypes, dim=-1))
                logits = similarities.logsumexp(2) - np.log(3.)
            return torch.where(empty[:, None], logits.new_tensor([0., -100.]), logits)
        if self.simplified:
            maxima = hidden.masked_fill(mask, -torch.inf).amax(dim=2).masked_fill(empty[:, None], 0)
            if self.softmax_output:
                logits = self.output(maxima)
                return torch.where(empty[:, None], logits.new_tensor([0., -100.]), logits)
            return self.output(maxima).squeeze(1).masked_fill(empty, -100.)
        gates = torch.tanh(self.gate_tanh(hidden)) * torch.sigmoid(self.gate_sigmoid(hidden))
        scores = self.gate_score(gates).squeeze(1)
        safe_mask = padding_mask.clone()
        safe_mask[empty, 0] = False
        attention = scores.masked_fill(safe_mask, -torch.inf).softmax(dim=1)
        attended = (hidden * attention[:, None, :]).sum(dim=2)
        maxima = hidden.masked_fill(mask, -torch.inf).amax(dim=2).masked_fill(empty[:, None], 0)
        logits = self.output(torch.cat([attended, maxima], dim=1)).squeeze(1)
        return logits.masked_fill(empty, -100.)


def _batch(rows: Rows, records: Records, device: str, *,
           temporal_bins: int = 1, input_dim: int = 768) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep masked slots in place, so filtering cannot collapse the time sequence."""
    length = max(len(records[r['ID']]['music']) for r in rows) * temporal_bins
    features = torch.zeros(len(rows), length, input_dim)
    mask = torch.ones(len(rows), length, dtype=torch.bool)
    for index, row in enumerate(rows):
        record = records[row['ID']]
        count = len(record['music'])
        starts = np.asarray(record['starts'])
        expected_shape = (count, input_dim) if temporal_bins == 1 else (count, temporal_bins, input_dim)
        if (record['music'].shape != expected_shape or not count
                or record['music_audible'].shape != (count,) or starts.shape != (count,)
                or not np.isfinite(starts).all() or np.any(starts < 0)
                or np.any(np.diff(starts) <= 0)
                or not np.isin(record['music_audible'], [0, 1]).all()):
            raise ValueError(f'Invalid or unordered cached music sequence: {row["ID"]}')
        if temporal_bins == 1:
            features[index, :count] = torch.from_numpy(record['music']).float()
            mask[index, :count] = torch.from_numpy(record['music_audible'] == 0)
        else:
            # The final original window can overlap; order bins by their audio positions.
            positions = np.asarray(record.get("bin_times", []))
            if positions.shape != (count, temporal_bins) or not np.isfinite(positions).all():
                raise ValueError(f"Missing or invalid MERT bin timestamps: {row['ID']}")
            order = np.argsort(positions.ravel(), kind='stable')
            size = count * temporal_bins
            features[index, :size] = torch.from_numpy(record['music'].reshape(size, input_dim)[order]).float()
            mask[index, :size] = torch.from_numpy(np.repeat(record['music_audible'] == 0, temporal_bins)[order])
    return features.to(device), mask.to(device)


def predict_temporal_music(model: TemporalMusic, rows: Rows, records: Records,
                           device: str = 'cpu', batch_size: int = TemporalRecipe().batch_size) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            logits = model(*_batch(rows[start:start + batch_size], records, device,
                                   temporal_bins=model.temporal_bins, input_dim=model.input_dim))
            p = logits.softmax(-1)[:, 1] if model.softmax_output else logits.sigmoid()
            probabilities.extend(p.cpu().tolist())
    return np.asarray(probabilities)


def fit_temporal_head(rows: Rows, records: Records, validation: tuple[Rows, Records],
                      output: Path, deadline: float, *, recipe: TemporalRecipe = TemporalRecipe(),
                      device: str = 'cpu') -> tuple[TemporalMusic, dict]:
    """Fit an experimental music-only candidate; never replace the submitted C3 head."""
    val_rows, val_records = validation
    assert_disjoint(rows, val_rows)
    if (not rows or not val_rows or recipe.epochs < 1 or recipe.batch_size < 1
            or not np.isfinite(recipe.learning_rate) or recipe.learning_rate <= 0
            or not np.isfinite(recipe.weight_decay) or recipe.weight_decay < 0):
        raise ValueError('Invalid temporal training inputs or optimizer settings')
    if not all(0 <= p <= 1 for p in (recipe.crop_probability, recipe.feature_mask_probability)):
        message = 'Augmentation probabilities must be finite and between zero and one'
        raise ValueError(message)
    if not 0 <= recipe.label_smoothing < 1 or (recipe.label_smoothing and not recipe.softmax_output):
        message = 'Label smoothing requires two-class logits and a finite value in [0, 1)'
        raise ValueError(message)
    if recipe.lr_plateau and recipe.learning_rate < 1e-6:
        message = 'Plateau initial learning rate must be at least the 1e-6 floor'
        raise ValueError(message)
    if len({r['ID'] for r in rows + val_rows}) != len(rows + val_rows):
        raise ValueError('Duplicate temporal training/validation IDs')
    weights = torch.tensor(source_weights(rows, recipe.nsynth_mass), dtype=torch.float32, device=device)
    val_weights = torch.tensor(source_weights(val_rows, None), dtype=torch.float32, device=device)
    labels = torch.tensor([int(r['MUSIC_FAKE']) for r in rows], dtype=torch.float32, device=device)
    val_labels = torch.tensor([int(r['MUSIC_FAKE']) for r in val_rows], dtype=torch.float32, device=device)
    for row in rows:
        if not np.any(records[row['ID']]['music_audible']):
            raise ValueError(f'No usable training windows: {row["ID"]}')
    torch.manual_seed(recipe.seed)
    rng = np.random.default_rng(recipe.seed)
    crop_rng = np.random.default_rng(recipe.seed + 10000)
    mask_rng = np.random.default_rng(recipe.seed + 20000)
    augmentation_history = []
    model = TemporalMusic(recipe.dropout, simplified=recipe.simplified, compact=recipe.compact,
                          temporal_bins=recipe.temporal_bins, softmax_output=recipe.softmax_output,
                          input_dim=recipe.input_dim, mlp_output=recipe.mlp_output,
                          cosine_output=recipe.cosine_output,
                          alternative_output=recipe.alternative_output).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                                  weight_decay=recipe.weight_decay)
    # PyTorch reduces when bad epochs exceed patience: 4 means the fifth stale epoch.
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.5, patience=4, threshold=recipe.min_delta,
        threshold_mode='abs', min_lr=1e-6) if recipe.lr_plateau else None)
    stopping = EarlyStopping(recipe.patience, recipe.min_delta)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'temporal_music.pt'
    losses = []
    learning_rates = []
    optimizer_steps = 0
    for epoch in range(1, recipe.epochs + 1):
        learning_rates.append(float(optimizer.param_groups[0]['lr']))
        model.train()
        order = rng.permutation(len(rows))
        train_loss = 0.
        augmentation_counts = empty_counts()
        for start in range(0, len(rows), recipe.batch_size):
            if time.time() >= deadline:
                raise TimeoutError('Temporal training deadline; no complete model claimed')
            indices = order[start:start + recipe.batch_size]
            selected = [rows[i] for i in indices]
            optimizer.zero_grad(set_to_none=True)
            training_records, counts = augment_records(selected, records, recipe, crop_rng, mask_rng)
            for key in augmentation_counts:
                augmentation_counts[key] += counts[key]
            logits = model(*_batch(selected, training_records, device, temporal_bins=recipe.temporal_bins, input_dim=recipe.input_dim))
            index = torch.tensor(indices, device=device)
            losses_per_file = (F.cross_entropy(logits, labels[index].long(), reduction='none',
                                               label_smoothing=recipe.label_smoothing)
                               if model.softmax_output else
                               F.binary_cross_entropy_with_logits(logits, labels[index], reduction='none'))
            weighted = losses_per_file * weights[index]
            loss = weighted.sum() * len(rows) / len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite temporal training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
            train_loss += float(weighted.detach().sum())
        losses.append(train_loss)
        augmentation_history.append(augmentation_counts)
        model.eval()
        val_loss = 0.
        with torch.inference_mode():
            for start in range(0, len(val_rows), recipe.batch_size):
                if time.time() >= deadline:
                    raise TimeoutError('Temporal validation deadline')
                selected = val_rows[start:start + recipe.batch_size]
                logits = model(*_batch(selected, val_records, device, temporal_bins=recipe.temporal_bins, input_dim=recipe.input_dim))
                end = start + len(selected)
                losses_per_file = (F.cross_entropy(logits, val_labels[start:end].long(), reduction='none')
                                   if model.softmax_output else F.binary_cross_entropy_with_logits(
                                       logits, val_labels[start:end], reduction='none'))
                val_loss += float((losses_per_file * val_weights[start:end]).sum())
        stop = stopping.step(model, val_loss, epoch)
        if scheduler is not None:
            scheduler.step(val_loss)
        if stopping.selected_epoch == epoch:
            temporary = target.with_suffix('.partial')
            torch.save(dict(architecture=model.architecture,
                            dropout=recipe.dropout, state_dict=stopping.state), temporary)
            temporary.replace(target)
        write_json(output / 'progress.json', dict(epoch=epoch, train_loss=train_loss,
            optimizer_steps=optimizer_steps,
            learning_rate=learning_rates[-1], next_learning_rate=float(optimizer.param_groups[0]['lr']),
            augmentation=augmentation_counts,
            validation_loss=val_loss, best_epoch=stopping.selected_epoch,
            best_validation_loss=stopping.best_loss, utc=time.time()))
        if stop:
            break
    stopping.restore(model)
    model.eval()
    report = dict(status='complete', scope='experimental music head; no full-data OOF or submission integration',
        recipe=asdict(recipe), loss='cross_entropy' if recipe.softmax_output else 'bce_with_logits', parameters=sum(p.numel() for p in model.parameters()),
        epochs_completed=len(losses), selected_epoch=stopping.selected_epoch,
        optimizer_steps=optimizer_steps,
        learning_rates=learning_rates,
        scheduler=(dict(name='ReduceLROnPlateau', mode='min', factor=.5, patience=4,
                        stale_epochs_before_reduction=5, min_lr=1e-6,
                        threshold=recipe.min_delta, threshold_mode='abs',
                        monitor='unsmoothed class-balanced validation cross entropy'
                        if recipe.softmax_output else 'class-balanced validation BCE')
                   if recipe.lr_plateau else None),
        augmentation_history=augmentation_history,
        train_losses=losses, validation_losses=stopping.history, best_validation_loss=stopping.best_loss,
        train_ids=[r['ID'] for r in rows], validation_ids=[r['ID'] for r in val_rows],
        checkpoint_sha256=sha256(target), input=f'{recipe.temporal_bins} vectors per 4.0375s cached MERT window; masked slots retained',
        source_sha256={name: sha256(Path(__file__).with_name(name))
                       for name in ('music_temporal.py', 'v7_ablation.py', 'temporal_augmentation.py')})
    write_json(output / 'training.json', report)
    return model, report


def load_temporal_head(path: Path, device: str = 'cpu') -> TemporalMusic:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['architecture'] == 'afwhisper-residual-64x3-max-d1280-mlp32-softmax-v1':
        model = TemporalMusic(payload.get('dropout', .2), simplified=True, compact=True,
                              input_dim=1280, softmax_output=True, mlp_output=True).to(device)
        model.load_state_dict(payload['state_dict'], strict=True)
        return model.eval()
    alternatives = {f'mert-residual-64x3-d1024-{name}-v1': name
                    for name in ('prototype', 'attention', 'conv_mean', 'conv_lme')}
    if payload['architecture'] in alternatives:
        model = TemporalMusic(payload.get('dropout', .2), simplified=True, compact=True,
            input_dim=1024, softmax_output=True, mlp_output=True,
            alternative_output=alternatives[payload['architecture']]).to(device)
        model.load_state_dict(payload['state_dict'], strict=True)
        return model.eval()
    architectures = {
        'mert-residual-tcn3-gated-max-v1': (False, False, 1, False),
        'mert-residual-tcn1-max-v1': (True, False, 1, False),
        'mert-residual-64x3-max-v1': (True, True, 1, False),
        'mert-residual-64x3-max-d1024-v1': (True, True, 1, False),
        'mert-residual-64x3-max-b8-softmax0-v1': (True, True, 8, False),
        'mert-residual-64x3-max-b1-softmax1-v1': (True, True, 1, True),
        'mert-residual-64x3-max-b8-softmax1-v1': (True, True, 8, True),
        'mert-residual-64x3-max-d768-mlp32-softmax-v1': (True, True, 1, True),
        'mert-residual-64x3-max-d1024-mlp32-softmax-v1': (True, True, 1, True),
        'mert-residual-64x3-max-d768-cosine-softmax-v1': (True, True, 1, True),
        'mert-residual-64x3-max-d1024-cosine-softmax-v1': (True, True, 1, True),
    }
    if payload['architecture'] not in architectures:
        raise ValueError('Unexpected temporal music architecture')
    simplified, compact, bins, softmax = architectures[payload['architecture']]
    model = TemporalMusic(payload.get('dropout', .2), simplified=simplified, compact=compact,
                          temporal_bins=bins, softmax_output=softmax,
                          input_dim=1024 if payload['architecture'] in ('mert-residual-64x3-max-d1024-v1',
                              'mert-residual-64x3-max-d1024-mlp32-softmax-v1',
                              'mert-residual-64x3-max-d1024-cosine-softmax-v1') else 768,
                          mlp_output=payload['architecture'].endswith('-mlp32-softmax-v1'),
                          cosine_output=payload['architecture'].endswith('-cosine-softmax-v1')).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.eval()
