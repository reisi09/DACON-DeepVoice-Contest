"""Bounded training over cached V5 per-file temporal features."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Final, Literal, TypedDict

import numpy as np
import torch

from v5_features import EncodedFile
from v5_fusion import (
    FusionContractError,
    TemporalFusion,
    balanced_known_label_bce,
    joint_partial_label_loss,
)

BATCH_SIZE: Final = 8
LEARNING_RATE: Final = 1e-4


class EpochUpdate(TypedDict):
    epoch: int
    train_loss: float
    dev_known_bce: float
    seconds: float
    learning_rate: float


class TrainingMetadata(TypedDict):
    status: Literal["complete", "censored_deadline"]
    joint: bool
    seed: int
    epochs_requested: int
    epochs_completed: int
    updates: int
    train_examples: int
    dev_examples: int
    exposures: dict[str, int]
    all_model_gradients_connected: bool
    disconnected_parameters: list[str]
    best_epoch: int
    best_dev_known_bce: float
    elapsed_seconds: float
    batch_size: int
    learning_rate: float
    normalization: str
    loss: str
    epoch_updates: list[EpochUpdate]


def _atomic_json(path: Path, value: TrainingMetadata) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _snapshot(model: TemporalFusion) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _atomic_state(path: Path, state: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _validate(train: list[EncodedFile], dev: list[EncodedFile], epochs: int) -> int:
    if not train or not dev or epochs < 1:
        raise FusionContractError("Training, dev, and at least one epoch are required")
    if len({item.identifier for item in train}) != len(train):
        raise FusionContractError("Training identifiers must be unique")
    dimensions = {item.features.shape[1] for item in (*train, *dev) if item.features.ndim == 2}
    if len(dimensions) != 1:
        raise FusionContractError("Every feature matrix must share one nonempty width")
    input_dim = dimensions.pop()
    if input_dim < 1:
        raise FusionContractError("Feature width must be positive")
    for item in (*train, *dev):
        labels = np.asarray(item.labels, dtype=np.float32)
        if (
            item.features.ndim != 2
            or item.features.shape[0] < 1
            or item.times.shape != (item.features.shape[0],)
            or labels.shape != (5,)
            or not np.isfinite(item.features).all()
            or not np.isfinite(item.times).all()
            or np.any(~np.isnan(labels) & (labels != 0) & (labels != 1))
        ):
            raise FusionContractError(f"Invalid encoded feature contract: {item.identifier}")
    if not any(np.isfinite(item.labels).any() for item in dev):
        raise FusionContractError("Dev requires at least one known label")
    return input_dim


def _batch(
    examples: list[EncodedFile], indices: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = [examples[index] for index in indices]
    windows = max(len(item.times) for item in selected)
    width = selected[0].features.shape[1]
    features = torch.zeros(len(selected), windows, width, dtype=torch.float32)
    times = torch.zeros(len(selected), windows, dtype=torch.float32)
    padding = torch.ones(len(selected), windows, dtype=torch.bool)
    labels = torch.full((len(selected), 5), torch.nan, dtype=torch.float32)
    for row, item in enumerate(selected):
        count = len(item.times)
        features[row, :count] = torch.from_numpy(item.features).float()
        times[row, :count] = torch.from_numpy(item.times).float()
        padding[row, :count] = False
        labels[row] = torch.tensor(item.labels, dtype=torch.float32)
    return features.to(device), padding.to(device), times.to(device), labels.to(device)


def _evaluate(
    model: TemporalFusion, examples: list[EncodedFile], device: torch.device
) -> tuple[np.ndarray, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    weighted_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(examples), BATCH_SIZE):
            indices = list(range(start, min(start + BATCH_SIZE, len(examples))))
            features, padding, times, labels = _batch(examples, indices, device)
            result = model(features, padding, times)
            loss = balanced_known_label_bce(result.probabilities, labels)
            weighted_loss += float(loss) * len(indices)
            predictions.append(result.probabilities.cpu().numpy())
    return np.concatenate(predictions), weighted_loss / len(examples)


def train_fusion(
    train: list[EncodedFile],
    dev: list[EncodedFile],
    joint: bool,
    output: Path,
    deadline: float,
    epochs: int = 20,
    seed: int = 20_260_913,
) -> tuple[TemporalFusion, np.ndarray, TrainingMetadata]:
    """Train every cached file once per completed epoch and return the best dev model."""
    started = time.time()
    input_dim = _validate(train, dev, epochs)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalFusion(input_dim, joint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    steps_per_epoch = math.ceil(len(train) / BATCH_SIZE)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    eval_started = time.time()
    _evaluate(model, dev, device)
    best_score = float("inf")
    eval_reserve = max(0.05, (time.time() - eval_started) * 1.25)
    best_epoch = completed = updates = 0
    best_state = _snapshot(model)
    _atomic_state(output / "best.pt", best_state)
    exposures = {item.identifier: 0 for item in train}
    disconnected = [name for name, _ in model.named_parameters()]
    connected = False
    history: list[EpochUpdate] = []
    batch_budget = 0.05
    status: Literal["complete", "censored_deadline"] = "complete"
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        epoch_losses: list[float] = []
        order = torch.randperm(len(train), generator=generator).tolist()
        for start in range(0, len(order), BATCH_SIZE):
            if time.time() + batch_budget + eval_reserve >= deadline:
                status = "censored_deadline"
                break
            batch_started = time.time()
            indices = order[start : start + BATCH_SIZE]
            features, padding, times, labels = _batch(train, indices, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result = model(features, padding, times)
            loss = (
                joint_partial_label_loss(result.logits, labels)
                if joint
                else balanced_known_label_bce(result.probabilities, labels)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite temporal fusion loss")
            loss.backward()
            disconnected = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is None
                or not torch.isfinite(parameter.grad).all()
                or torch.count_nonzero(parameter.grad) == 0
            ]
            if disconnected:
                raise FusionContractError(f"Disconnected temporal parameters: {disconnected}")
            connected = True
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            updates += 1
            epoch_losses.append(float(loss.detach()))
            for index in indices:
                exposures[train[index].identifier] += 1
            batch_budget = max(batch_budget, (time.time() - batch_started) * 1.25)
        if status == "censored_deadline":
            break
        completed = epoch
        _, dev_score = _evaluate(model, dev, device)
        if dev_score < best_score:
            best_score, best_epoch, best_state = dev_score, epoch, _snapshot(model)
            _atomic_state(output / "best.pt", best_state)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "dev_known_bce": dev_score,
                "seconds": time.time() - epoch_started,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
    _atomic_state(output / "final.pt", _snapshot(model))
    model.load_state_dict(best_state, strict=True)
    probabilities, restored_score = _evaluate(model, dev, device)
    if not math.isfinite(best_score):
        best_score = restored_score
    metadata: TrainingMetadata = {
        "status": status, "joint": joint, "seed": seed,
        "epochs_requested": epochs, "epochs_completed": completed, "updates": updates,
        "train_examples": len(train), "dev_examples": len(dev), "exposures": exposures,
        "all_model_gradients_connected": connected,
        "disconnected_parameters": disconnected,
        "best_epoch": best_epoch, "best_dev_known_bce": best_score,
        "elapsed_seconds": time.time() - started, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "normalization": "per-window LayerNorm; no dataset statistics",
        "loss": "joint partial-label marginal NLL" if joint else "balanced known-label BCE",
        "epoch_updates": history,
    }
    _atomic_json(output / "training.json", metadata)
    return model, probabilities, metadata
