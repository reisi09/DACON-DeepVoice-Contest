"""Resumable optimizer primitives shared by the V4 training trials."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np
import torch

CHECKPOINT_SECONDS: Final = 60.0
RunStatus = Literal["complete", "interrupted", "stopped_deadline"]
NamedParameters = Sequence[tuple[str, torch.nn.Parameter]]


@dataclass(frozen=True, slots=True)
class Batch:
    ids: tuple[str, ...]
    inputs: torch.Tensor
    targets: torch.Tensor
    sample_weights: torch.Tensor | None = None
    file_indices: torch.Tensor | None = None
    window_targets: torch.Tensor | None = None
    starts: tuple[tuple[int, ...], ...] = ()
    exposure_records: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunState:
    epoch: int = 0
    batch_index: int = 0
    optimizer_steps: int = 0
    exposures: dict[str, int] = field(default_factory=dict)
    window_exposures: int = 0
    exposure_records: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    state: RunState


def build_optimizer(
    encoder: NamedParameters,
    head: NamedParameters,
    *,
    encoder_lr: float,
    head_lr: float,
) -> torch.optim.AdamW:
    """Apply declared LR and bias/norm decay without omitting a trainable layer."""
    named = (*encoder, *head)
    if not named or len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError("optimizer parameters must be nonempty and unique")
    if not all(parameter.requires_grad for _, parameter in named):
        raise ValueError("full fine-tuning requires every learned parameter")
    groups: list[dict[str, float | list[torch.nn.Parameter]]] = []
    for parameters, learning_rate in ((encoder, encoder_lr), (head, head_lr)):
        for decay in (0.01, 0.0):
            selected = [
                parameter
                for _, parameter in parameters
                if (parameter.ndim >= 2) == (decay > 0)
            ]
            if selected:
                groups.append({"params": selected, "lr": learning_rate, "weight_decay": decay})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
    planned_epochs: int = 30,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm one full epoch, then decay over the fixed 30-epoch horizon."""
    if steps_per_epoch < 1 or planned_epochs < 2:
        raise ValueError("scheduler requires positive steps and at least two epochs")
    total_steps = steps_per_epoch * planned_epochs

    def multiplier(step: int) -> float:
        if step < steps_per_epoch:
            return 0.1 + 0.9 * step / steps_per_epoch
        progress = (step - steps_per_epoch) / (total_steps - steps_per_epoch)
        return 0.1 + 0.45 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _rng_state() -> dict[str, torch.Tensor | tuple | list[torch.Tensor] | None]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, torch.Tensor | tuple | list[torch.Tensor] | None]) -> None:
    python_state = state["python"]
    numpy_state = state["numpy"]
    torch_state = state["torch"]
    cuda_state = state["cuda"]
    assert isinstance(python_state, tuple)
    assert isinstance(numpy_state, tuple)
    assert isinstance(torch_state, torch.Tensor)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None and torch.cuda.is_available():
        assert isinstance(cuda_state, list)
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: RunState,
    contract: str,
) -> None:
    """Atomically bind weights, optimizer, scheduler, RNG, and exposure progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "contract": contract,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": asdict(state),
            "rng": _rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract: str,
) -> RunState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["contract"] != contract:
        raise ValueError("checkpoint contract mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng"])
    raw = payload["state"]
    return RunState(
        epoch=int(raw["epoch"]),
        batch_index=int(raw["batch_index"]),
        optimizer_steps=int(raw["optimizer_steps"]),
        exposures={str(key): int(value) for key, value in raw["exposures"].items()},
        window_exposures=int(raw.get("window_exposures", 0)),
        exposure_records=tuple(map(str, raw.get("exposure_records", ()))),
    )


def _default_loss(model: torch.nn.Module, batch: Batch) -> torch.Tensor:
    logits = model(batch.inputs).reshape(-1)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch.targets, reduction="none"
    )
    return losses.mean() if batch.sample_weights is None else (
        losses * batch.sample_weights
    ).sum() / batch.sample_weights.sum()


def run_batches(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batches: Sequence[Batch],
    state: RunState,
    checkpoint: Path,
    deadline: float,
    *,
    now: Callable[[], float] = time.time,
    max_updates: int | None = None,
    contract: str = "test-contract",
    loss_fn: Callable[[torch.nn.Module, Batch], torch.Tensor] = _default_loss,
    accumulation: int = 1,
) -> RunResult:
    """Run deterministic batches and durably stop at a budget or requested boundary."""
    model.train()
    exposures = dict(state.exposures)
    current = state
    saved_at = now()
    if accumulation < 1:
        raise ValueError("accumulation must be positive")
    for start in range(state.batch_index, len(batches), accumulation):
        if now() >= deadline:
            save_checkpoint(checkpoint, model, optimizer, scheduler, current, contract)
            return RunResult("stopped_deadline", current)
        optimizer.zero_grad(set_to_none=True)
        try:
            group = [batches[index] for index in range(start, min(start + accumulation, len(batches)))]
        except TimeoutError:
            save_checkpoint(checkpoint, model, optimizer, scheduler, current, contract)
            return RunResult("stopped_deadline", current)
        if now() >= deadline:
            save_checkpoint(checkpoint, model, optimizer, scheduler, current, contract)
            return RunResult("stopped_deadline", current)
        for batch in group:
            loss = loss_fn(model, batch) / len(group)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite training gradient")
        optimizer.step()
        scheduler.step()
        for batch in group:
            for identifier in batch.ids:
                exposures[identifier] = exposures.get(identifier, 0) + 1
        current = RunState(
            epoch=state.epoch,
            batch_index=start + len(group),
            optimizer_steps=current.optimizer_steps + 1,
            exposures=exposures.copy(),
            window_exposures=current.window_exposures + sum(batch.inputs.shape[0] for batch in group),
            exposure_records=current.exposure_records + tuple(
                record for batch in group for record in batch.exposure_records),
        )
        if now() - saved_at >= CHECKPOINT_SECONDS:
            save_checkpoint(checkpoint, model, optimizer, scheduler, current, contract)
            saved_at = now()
        if (max_updates is not None and current.optimizer_steps - state.optimizer_steps >= max_updates
                and current.batch_index < len(batches)):
            save_checkpoint(checkpoint, model, optimizer, scheduler, current, contract)
            return RunResult("interrupted", current)
    save_checkpoint(checkpoint, model, optimizer, scheduler, current, contract)
    return RunResult("complete", current)
