"""Experimental residual temporal CNN over ordered, cached MERT music windows."""

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import TypedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from v6_data import assert_disjoint, sha256, write_json
from v7_ablation import EarlyStopping, source_weights


class MusicRecord(TypedDict):
    music: np.ndarray
    music_audible: np.ndarray
    starts: np.ndarray


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


class ResidualTemporalBlock(nn.Module):
    """Two non-causal temporal convolutions; mask every layer to isolate padding."""

    def __init__(self, dilation: int, dropout: float = .2) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(2)])
        self.convs = nn.ModuleList([
            nn.Conv1d(128, 128, 3, padding=dilation, dilation=dilation) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = hidden
        for norm, conv in zip(self.norms, self.convs, strict=True):
            hidden = norm(hidden.transpose(1, 2)).transpose(1, 2).masked_fill(mask, 0)
            hidden = self.dropout(F.gelu(conv(hidden))).masked_fill(mask, 0)
        return (residual + hidden).masked_fill(mask, 0)


class TemporalMusic(nn.Module):
    """Six temporal Conv1D layers plus gated-attention and max feature pooling."""

    def __init__(self, dropout: float = .2) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(768)
        self.projection = nn.Conv1d(768, 128, 1)
        self.blocks = nn.ModuleList([ResidualTemporalBlock(d, dropout) for d in (1, 2, 4)])
        self.gate_tanh = nn.Conv1d(128, 64, 1)
        self.gate_sigmoid = nn.Conv1d(128, 64, 1)
        self.gate_score = nn.Conv1d(64, 1, 1, bias=False)
        self.output = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != 768 or features.shape[1] < 1:
            raise ValueError('Expected nonempty [files, ordered windows, 768] MERT features')
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
        gates = torch.tanh(self.gate_tanh(hidden)) * torch.sigmoid(self.gate_sigmoid(hidden))
        scores = self.gate_score(gates).squeeze(1)
        empty = padding_mask.all(dim=1)
        safe_mask = padding_mask.clone()
        safe_mask[empty, 0] = False
        attention = scores.masked_fill(safe_mask, -torch.inf).softmax(dim=1)
        attended = (hidden * attention[:, None, :]).sum(dim=2)
        maxima = hidden.masked_fill(mask, -torch.inf).amax(dim=2).masked_fill(empty[:, None], 0)
        logits = self.output(torch.cat([attended, maxima], dim=1)).squeeze(1)
        return logits.masked_fill(empty, -100.)


def _batch(rows: Rows, records: Records, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep masked slots in place, so filtering cannot collapse the time sequence."""
    length = max(len(records[r['ID']]['music']) for r in rows)
    features = torch.zeros(len(rows), length, 768)
    mask = torch.ones(len(rows), length, dtype=torch.bool)
    for index, row in enumerate(rows):
        record = records[row['ID']]
        count = len(record['music'])
        starts = np.asarray(record['starts'])
        if (record['music'].shape != (count, 768) or not count
                or record['music_audible'].shape != (count,) or starts.shape != (count,)
                or not np.isfinite(starts).all() or np.any(starts < 0)
                or np.any(np.diff(starts) <= 0)
                or not np.isin(record['music_audible'], [0, 1]).all()):
            raise ValueError(f'Invalid or unordered cached music sequence: {row["ID"]}')
        features[index, :count] = torch.from_numpy(record['music']).float()
        mask[index, :count] = torch.from_numpy(record['music_audible'] == 0)
    return features.to(device), mask.to(device)


def predict_temporal_music(model: TemporalMusic, rows: Rows, records: Records,
                           device: str = 'cpu', batch_size: int = TemporalRecipe().batch_size) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            probabilities.extend(model(*_batch(rows[start:start + batch_size], records,
                                               device)).sigmoid().cpu().tolist())
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
    model = TemporalMusic(recipe.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                                  weight_decay=recipe.weight_decay)
    stopping = EarlyStopping(recipe.patience, recipe.min_delta)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'temporal_music.pt'
    losses = []
    optimizer_steps = 0
    for epoch in range(1, recipe.epochs + 1):
        model.train()
        order = rng.permutation(len(rows))
        train_loss = 0.
        for start in range(0, len(rows), recipe.batch_size):
            if time.time() >= deadline:
                raise TimeoutError('Temporal training deadline; no complete model claimed')
            indices = order[start:start + recipe.batch_size]
            selected = [rows[i] for i in indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(*_batch(selected, records, device))
            index = torch.tensor(indices, device=device)
            weighted = F.binary_cross_entropy_with_logits(logits, labels[index], reduction='none') * weights[index]
            loss = weighted.sum() * len(rows) / len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite temporal training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
            train_loss += float(weighted.detach().sum())
        losses.append(train_loss)
        model.eval()
        val_loss = 0.
        with torch.inference_mode():
            for start in range(0, len(val_rows), recipe.batch_size):
                if time.time() >= deadline:
                    raise TimeoutError('Temporal validation deadline')
                selected = val_rows[start:start + recipe.batch_size]
                logits = model(*_batch(selected, val_records, device))
                end = start + len(selected)
                val_loss += float((F.binary_cross_entropy_with_logits(
                    logits, val_labels[start:end], reduction='none') * val_weights[start:end]).sum())
        stop = stopping.step(model, val_loss, epoch)
        if stopping.selected_epoch == epoch:
            temporary = target.with_suffix('.partial')
            torch.save(dict(architecture='mert-residual-tcn3-gated-max-v1',
                            dropout=recipe.dropout, state_dict=stopping.state), temporary)
            temporary.replace(target)
        write_json(output / 'progress.json', dict(epoch=epoch, train_loss=train_loss,
            optimizer_steps=optimizer_steps,
            validation_loss=val_loss, best_epoch=stopping.selected_epoch,
            best_validation_loss=stopping.best_loss, utc=time.time()))
        if stop:
            break
    stopping.restore(model)
    model.eval()
    report = dict(status='complete', scope='experimental music head; no full-data OOF or submission integration',
        recipe=asdict(recipe), parameters=sum(p.numel() for p in model.parameters()),
        epochs_completed=len(losses), selected_epoch=stopping.selected_epoch,
        optimizer_steps=optimizer_steps,
        train_losses=losses, validation_losses=stopping.history, best_validation_loss=stopping.best_loss,
        train_ids=[r['ID'] for r in rows], validation_ids=[r['ID'] for r in val_rows],
        checkpoint_sha256=sha256(target), input='Ordered 4.0375s cached MERT windows; masked slots retained',
        source_sha256={name: sha256(Path(__file__).with_name(name))
                       for name in ('music_temporal.py', 'v7_ablation.py')})
    write_json(output / 'training.json', report)
    return model, report


def load_temporal_head(path: Path, device: str = 'cpu') -> TemporalMusic:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['architecture'] != 'mert-residual-tcn3-gated-max-v1':
        raise ValueError('Unexpected temporal music architecture')
    model = TemporalMusic(payload.get('dropout', .2)).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.eval()
