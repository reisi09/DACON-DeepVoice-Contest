"""Training and checkpoint helpers for redesign residual heads."""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from redesign_checkpoint import (
    Arm, Records, Rows, identity_hash, load_checkpoint, save_checkpoint, sha256,
    training_identity,
)
from redesign_models import MusicOutput, MusicResidual, StemMusicWrapper, make_control


@dataclass(frozen=True, slots=True)
class FitResult:
    model: StemMusicWrapper | MusicResidual
    checkpoint: Path
    report: dict[str, str | int | float | list[str] | list[float]]


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """Average only known targets; unknown entries have no gradient path."""
    if logits.shape != targets.shape or known.shape != logits.shape:
        raise ValueError("Masked BCE tensors must share a shape")
    if not known.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[known], targets[known])


def _pad(
    rows: Rows, records: Records, device: str, *, raw_equal_stem: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(len(records[row["ID"]]["stem"]) for row in rows)
    stem = torch.zeros(len(rows), length, 1024, device=device)
    raw = torch.zeros_like(stem)
    aux = torch.zeros(len(rows), length, 6, device=device)
    valid = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        record = records[row["ID"]]
        count = len(record["stem"])
        record_valid = np.asarray(record["valid"], dtype=bool) & np.asarray(
            record["stem_audible"], dtype=bool
        )
        if (record["stem"].shape != (count, 1024) or record["raw"].shape != (count, 1024)
                or record["aux"].shape != (count, 6) or record_valid.shape != (count,)):
            raise ValueError(f"Invalid aligned feature record: {row['ID']}")
        stem[index, :count] = torch.as_tensor(record["stem"], device=device)
        raw[index, :count] = torch.as_tensor(
            record["stem"] if raw_equal_stem else record["raw"], device=device
        )
        item_aux = np.asarray(record["aux"], dtype=np.float32).copy()
        if raw_equal_stem:
            item_aux[:, 0] = item_aux[:, 1]
            item_aux[:, 3] = item_aux[:, 4]
        aux[index, :count] = torch.as_tensor(item_aux, device=device)
        valid[index, :count] = torch.as_tensor(record_valid, device=device)
    return stem, raw, aux, valid


def _targets(rows: Rows, records: Records, windows: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bag = torch.zeros(len(rows), dtype=torch.long, device=device)
    bag_known = torch.zeros(len(rows), dtype=torch.bool, device=device)
    event = torch.zeros(len(rows), windows, device=device)
    event_known = torch.zeros(len(rows), windows, dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        if row.get("MUSIC_PRESENT") == "1" and row.get("MUSIC_FAKE") in ("0", "1"):
            bag[index] = int(row["MUSIC_FAKE"])
            bag_known[index] = True
        intervals = json.loads(row.get("event_intervals") or "[]")
        starts = np.asarray(records[row["ID"]]["starts"], dtype=np.int64)
        valid_ratio = np.asarray(records[row["ID"]]["aux"], dtype=np.float32)[:, 5]
        for window, start in enumerate(starts):
            end = start + int(round(64600 * float(valid_ratio[window])))
            for interval in intervals:
                component = str(interval.get("component", ""))
                label = interval.get("label")
                first = int(round(float(interval.get("start", -1)) * 16000))
                last = int(round(float(interval.get("end", -1)) * 16000))
                if component in ("music", "accompaniment") and label in (0, 1, "0", "1") and first <= start and end <= last:
                    event[index, window] = int(label)
                    event_known[index, window] = True
                    break
    return bag, bag_known, event, event_known


def _balanced_weights(rows: Rows, labels: torch.Tensor, known: torch.Tensor, device: str) -> torch.Tensor:
    weights = torch.zeros(len(rows), device=device)
    classes = sorted(set(int(labels[index]) for index in range(len(rows)) if bool(known[index])))
    for label in classes:
        indices = [index for index in range(len(rows)) if bool(known[index]) and int(labels[index]) == label]
        groups = sorted({rows[index]["group_id"] for index in indices})
        for group in groups:
            members = [index for index in indices if rows[index]["group_id"] == group]
            weights[members] = 1.0 / (len(classes) * len(groups) * len(members))
    return weights


def _weighted_ce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    eligible = weights > 0
    if not eligible.any():
        return logits.sum() * 0.0
    values = F.cross_entropy(logits[eligible], targets[eligible], reduction="none")
    return (values * weights[eligible]).sum() / weights[eligible].sum()


def _weighted_event_bce(
    logits: torch.Tensor, targets: torch.Tensor, known: torch.Tensor, row_weights: torch.Tensor
) -> torch.Tensor:
    weights = row_weights[:, None] * known
    if not (weights > 0).any():
        return logits.sum() * 0.0
    values = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (values * weights).sum() / weights.sum()


def _forward(
    model: StemMusicWrapper | MusicResidual,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> MusicOutput:
    stem, raw, aux, valid = batch
    if isinstance(model, MusicResidual):
        return model(stem, raw, aux, valid)
    return model(stem, valid)


def fit_music(
    rows: Rows,
    records: Records,
    validation: Rows,
    output: Path,
    *,
    arm: Arm,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: str,
    deadline: float | None,
    raw_equal_stem: bool = False,
) -> FitResult:
    """Fit with grouped internal validation already removed by the caller."""
    if not rows or not validation or epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("Training and grouped early-stop rows are required")
    config: dict[str, str | int | float | list[str]] = {
        "arm": arm, "seed": seed, "epochs": epochs, "patience": patience,
        "batch_size": batch_size, "raw_equal_stem": str(raw_equal_stem),
        "train_ids": sorted(row["ID"] for row in rows),
        "earlystop_ids": sorted(row["ID"] for row in validation),
    }
    identity = training_identity(rows, validation, records, config)
    checkpoint = output / "best.pt"
    if checkpoint.exists():
        model, report = load_checkpoint(checkpoint, device=device, expected_identity=identity)
        return FitResult(model, checkpoint, report)
    control = make_control(seed)
    model: StemMusicWrapper | MusicResidual = (
        MusicResidual(control) if arm in ("B", "D") else StemMusicWrapper(control)
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    train_batch = _pad(rows, records, device, raw_equal_stem=raw_equal_stem)
    valid_batch = _pad(validation, records, device, raw_equal_stem=raw_equal_stem)
    train_targets = _targets(rows, records, train_batch[0].shape[1], device)
    valid_targets = _targets(validation, records, valid_batch[0].shape[1], device)
    train_targets = (train_targets[0], train_targets[1] & train_batch[3].any(dim=1),
                     train_targets[2], train_targets[3])
    valid_targets = (valid_targets[0], valid_targets[1] & valid_batch[3].any(dim=1),
                     valid_targets[2], valid_targets[3])
    if not train_targets[1].any() or not valid_targets[1].any():
        raise ValueError("Training and early-stop sets need eligible music labels")
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    updates = 0
    history: list[float] = []
    train_weights = _balanced_weights(rows, train_targets[0], train_targets[1], device)
    valid_weights = _balanced_weights(validation, valid_targets[0], valid_targets[1], device)
    torch.manual_seed(seed + 100000)
    rng = np.random.default_rng(seed + 100000)
    for epoch in range(1, epochs + 1):
        if deadline is not None and time.time() >= deadline:
            report = {**config, "seed": seed, "status": "stopped_deadline_restart_from_seed",
                      "epoch": epoch - 1, "optimizer_updates": updates, "identity": identity,
                      "validation_loss": history, "exact_resume": "false"}
            save_checkpoint(output / "last.pt", model, arm, identity, report)
            raise TimeoutError("Absolute training deadline reached; restart-bound last.pt saved")
        model.train()
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), batch_size):
            if deadline is not None and time.time() >= deadline:
                report = {**config, "seed": seed, "status": "stopped_deadline_restart_from_seed",
                          "epoch": epoch - 1, "optimizer_updates": updates, "identity": identity,
                          "validation_loss": history, "exact_resume": "false"}
                save_checkpoint(output / "last.pt", model, arm, identity, report)
                raise TimeoutError("Absolute training deadline reached inside epoch; restart from seed")
            chosen = torch.as_tensor(order[start:start + batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            output_value = _forward(model, tuple(tensor[chosen] for tensor in train_batch))
            bag, bag_known, event, event_known = train_targets
            loss = 0.27 * _weighted_ce(output_value.bag_logits, bag[chosen], train_weights[chosen])
            loss = loss + 0.10 * _weighted_event_bce(
                output_value.window_logits[..., 1] - output_value.window_logits[..., 0],
                event[chosen], event_known[chosen] & train_batch[3][chosen], train_weights[chosen],
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        model.eval()
        with torch.inference_mode():
            output_value = _forward(model, valid_batch)
            bag, known, event, event_known = valid_targets
            loss = 0.27 * _weighted_ce(output_value.bag_logits, bag, valid_weights)
            loss = loss + 0.10 * _weighted_event_bce(
                output_value.window_logits[..., 1] - output_value.window_logits[..., 0],
                event, event_known & valid_batch[3], valid_weights,
            )
            value = float(loss)
        history.append(value)
        if value < best_loss - 1e-4:
            best_loss, best_epoch, stale = value, epoch, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No finite early-stop checkpoint was selected")
    model.load_state_dict(best_state)
    report = {**config, "seed": seed, "status": "complete", "selected_epoch": best_epoch,
              "epochs_completed": len(history), "optimizer_updates": updates, "identity": identity,
              "validation_loss": history}
    save_checkpoint(checkpoint, model, arm, identity, report)
    loaded, _ = load_checkpoint(checkpoint, device=device, expected_identity=identity)
    model.eval()
    with torch.inference_mode():
        before = _forward(model, valid_batch).bag_logits
        after = _forward(loaded, valid_batch).bag_logits
    if not torch.allclose(before, after, atol=1e-7, rtol=0):
        raise RuntimeError("Saved checkpoint changed validation predictions")
    return FitResult(loaded, checkpoint, report)
