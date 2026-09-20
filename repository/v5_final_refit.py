"""Resumable fixed-epoch V5 refits over all permissible public examples."""

# allow: SIZE_OK — requested single module for expert and temporal final-refit state machines.

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import torch
from torch import nn

from v5_dataset import SOURCE_PANELS, Example, training_crop
from v5_features import EncodedFile
from v5_fusion import (
    FusionContractError,
    TemporalFusion,
    balanced_known_label_bce,
    joint_partial_label_loss,
)
from v5_fusion_training import BATCH_SIZE, LEARNING_RATE, _batch
from v5_training import (
    ExpertKind,
    _audit_before_step,
    _crop_targets,
    _disable_native_augmentation,
    _forward,
    _gradient_context,
    _optimizer,
    _training_mode,
    masked_balanced_bce,
)

Arm = Literal["B1", "B2"]
FusionName = Literal["B3", "B4"]
Status = Literal["complete", "incomplete_deadline"]
EXPERT_EPOCHS: Final[dict[tuple[Arm, ExpertKind], int]] = {
    ("B1", "voice"): 4, ("B1", "music"): 5, ("B1", "presence"): 5,
    ("B2", "voice"): 4, ("B2", "music"): 1, ("B2", "presence"): 1,
}
COSINE_HORIZON_EPOCHS: Final = 60
FUSION_EPOCHS: Final = 20


RefitMetadata = dict[str, bool | float | int | str | dict[str, int | float | str]]


@dataclass(frozen=True, slots=True)
class FinalExpertConfig:
    arm: Arm
    kind: ExpertKind
    output: Path
    deadline: float
    seed: int = 20_260_913
    device: str = "cuda"
    microbatch: int | None = None
    accumulation: int | None = None
    clock: Callable[[], float] = time.time


@dataclass(frozen=True, slots=True)
class FinalFusionConfig:
    name: FusionName
    joint: bool
    output: Path
    deadline: float
    seed: int = 20_260_913
    device: str | None = None
    clock: Callable[[], float] = time.time


class FinalRefitError(RuntimeError):
    """A final-refit artifact would violate its completion contract."""


def _atomic_json(path: Path, value: RefitMetadata) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_state(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _device(name: str | None) -> torch.device:
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def _metadata(status: Status, epochs: int, completed: int, exposures: dict[str, int], resumed: int, elapsed: float) -> RefitMetadata:
    return {"status": status, "full_data_refit": {"status": status},
            "epochs_requested": epochs, "epochs_completed": completed,
            "exposures": exposures, "resumed_from_epoch": resumed,
            "elapsed_seconds": elapsed, "dev_selection": False}


def _crop_labels(row: Example) -> tuple[float, ...]:
    return row.labels if row.panel in SOURCE_PANELS else (math.nan, math.nan, math.nan, *row.labels[3:])


def _eligible_final(row: Example, kind: ExpertKind) -> bool:
    labels = _crop_labels(row)
    if kind == "voice":
        return labels[3] == 1.0 and math.isfinite(labels[1])
    if kind == "music":
        return labels[4] == 1.0 and math.isfinite(labels[2])
    return math.isfinite(labels[3]) or math.isfinite(labels[4])


def _existing_complete(job: Path, epochs: int, identifiers: set[str]) -> RefitMetadata | None:
    metadata_path, final = job / "training.json", job / "final.pt"
    if not metadata_path.is_file():
        return None
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        return None
    valid = (final.is_file() and value.get("epochs_requested") == epochs
             and value.get("epochs_completed") == epochs
             and value.get("exposures") == {identifier: epochs for identifier in identifiers}
             and value.get("full_data_refit") == {"status": "complete"}
             and value.get("state_sha256") == _digest(final))
    if not valid:
        raise FinalRefitError(f"Invalid completed refit artifacts: {job}")
    return value


def _restore(job: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler) -> tuple[int, int, dict[str, int], dict[str, float], dict[str, float]]:
    path = job / "checkpoint.pt"
    if not path.is_file():
        return 0, 0, {}, {}, {}
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    return saved["epoch"], saved["updates"], saved["exposures"], saved["gradient_audit"], saved["update_audit"]


def _checkpoint(job: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, updates: int, exposures: dict[str, int], gradients: dict[str, float], deltas: dict[str, float]) -> None:
    _atomic_state(job / "checkpoint.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict(), "epoch": epoch, "updates": updates,
                  "exposures": exposures, "gradient_audit": gradients, "update_audit": deltas})


def train_final_expert(model: nn.Module, rows: list[Example], config: FinalExpertConfig) -> RefitMetadata:
    """Refit one pretrained expert on every eligible example without model selection."""
    epochs = EXPERT_EPOCHS[(config.arm, config.kind)]
    eligible = [row for row in rows if _eligible_final(row, config.kind)]
    excluded = {row.id: "unknown_crop_target" for row in rows if row not in eligible}
    identifiers = [row.id for row in eligible]
    if not eligible or len(set(identifiers)) != len(identifiers) or epochs < 1:
        raise FinalRefitError("Expert refit requires unique eligible IDs and positive epochs")
    job = config.output / "jobs" / f"{config.arm}_{config.kind}"
    job.mkdir(parents=True, exist_ok=True)
    existing = _existing_complete(job, epochs, set(identifiers))
    if existing is not None:
        return existing
    if (job / "final.pt").exists():
        raise FinalRefitError(f"Untrusted final checkpoint exists: {job / 'final.pt'}")
    device = _device(config.device)
    model.to(device)
    _disable_native_augmentation(model)
    microbatch = config.microbatch or (1 if config.kind == "voice" else 16)
    accumulation = config.accumulation or (8 if config.kind == "voice" else 1)
    steps = math.ceil(math.ceil(len(eligible) / microbatch) / accumulation)
    optimizer = _optimizer(model, config.kind)
    horizon = max(1, steps * COSINE_HORIZON_EPOCHS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.1 + 0.45 * (1 + math.cos(math.pi * min(step, horizon) / horizon)))
    completed, updates, exposures, gradients, deltas = _restore(job, model, optimizer, scheduler)
    resumed = completed
    if exposures and (set(exposures) != set(identifiers)
                      or any(value != completed for value in exposures.values())):
        raise FinalRefitError("Expert resume checkpoint does not match the full-data cohort")
    exposures = {identifier: exposures.get(identifier, 0) for identifier in identifiers}
    population = torch.tensor([_crop_targets(_crop_labels(row), config.kind) for row in eligible],
                              dtype=torch.float32, device=device)
    started = time.perf_counter()
    for epoch in range(completed, epochs):
        if config.clock() >= config.deadline:
            break
        torch.manual_seed(config.seed + epoch)
        _training_mode(model, microbatch)
        order = eligible.copy()
        np.random.default_rng(config.seed + epoch).shuffle(order)
        consumed: list[str] = []
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        batches = [order[start:start + microbatch] for start in range(0, len(order), microbatch)]
        expired = False
        for batch_index, batch in enumerate(batches):
            if config.clock() >= config.deadline:
                expired = True
                break
            prepared = [training_crop(row, epoch, int(model.window_samples), config.arm == "B2", config.seed)
                        for row in batch]
            audio = torch.from_numpy(np.stack([item[0] for item in prepared])).to(device)
            targets = torch.tensor([_crop_targets(item[1], config.kind) for item in prepared],
                                   dtype=torch.float32, device=device)
            with _gradient_context(model, not gradients), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                loss = masked_balanced_bce(_forward(model, audio), targets, population) / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite final expert loss")
            loss.backward()
            consumed.extend(row.id for row in batch)
            pending += 1
            if pending == accumulation or batch_index + 1 == len(batches):
                probes = None
                if not gradients:
                    gradients, probes = _audit_before_step(model, ())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite final expert gradient")
                optimizer.step()
                scheduler.step()
                updates += 1
                if probes is not None:
                    deltas = {group: abs(float(parameter.reshape(-1)[index]) - before)
                              for group, (parameter, index, before) in probes.items()}
                    if any(value == 0 for value in deltas.values()):
                        raise FinalRefitError("Expert parameter update audit failed")
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        if expired or set(consumed) != set(identifiers) or len(consumed) != len(identifiers):
            break
        completed = epoch + 1
        for identifier in consumed:
            exposures[identifier] += 1
        _checkpoint(job, model, optimizer, scheduler, completed, updates, exposures, gradients, deltas)
        print(json.dumps({"event": "final_refit_epoch_complete", "job": f"{config.arm}_{config.kind}",
                          "epoch": completed, "epochs": epochs, "updates": updates}), flush=True)
    status: Status = "complete" if completed == epochs and all(value == epochs for value in exposures.values()) else "incomplete_deadline"
    metadata = _metadata(status, epochs, completed, exposures, resumed, time.perf_counter() - started)
    metadata.update({"arm": config.arm, "kind": config.kind, "augmented": config.arm == "B2",
                     "updates": updates, "gradient_audit": gradients, "update_audit": deltas,
                     "cosine_horizon_epochs": COSINE_HORIZON_EPOCHS,
                     "eligible_examples": len(eligible), "excluded_ids": excluded})
    if status == "complete":
        final = job / "final.pt"
        _atomic_state(final, {name: value.detach().cpu() for name, value in model.state_dict().items()})
        metadata["state_sha256"] = _digest(final)
    _atomic_json(job / "training.json", metadata)
    if status == "complete":
        (job / "checkpoint.pt").unlink(missing_ok=True)
    return metadata


def _validate_features(examples: list[EncodedFile]) -> int:
    widths = {item.features.shape[1] for item in examples if item.features.ndim == 2}
    if not examples or len({item.identifier for item in examples}) != len(examples) or len(widths) != 1:
        raise FusionContractError("Fusion refit requires unique examples of one feature width")
    for item in examples:
        labels = np.asarray(item.labels)
        if (item.features.shape[0] < 1 or item.times.shape != (item.features.shape[0],)
                or labels.shape != (5,) or not np.isfinite(item.features).all()
                or not np.isfinite(item.times).all() or not np.isfinite(labels).any()
                or np.any(~np.isnan(labels) & (labels != 0) & (labels != 1))):
            raise FusionContractError(f"Invalid final fusion example: {item.identifier}")
    return widths.pop()


def train_final_fusion(examples: list[EncodedFile], config: FinalFusionConfig) -> RefitMetadata:
    """Refit original B3/B4 temporal fusion on all cached B2 features."""
    epochs = FUSION_EPOCHS
    if config.joint != (config.name == "B4"):
        raise FinalRefitError("B3 must be non-joint and B4 must be joint")
    width = _validate_features(examples)
    identifiers = [item.identifier for item in examples]
    job = config.output / config.name
    job.mkdir(parents=True, exist_ok=True)
    existing = _existing_complete(job, epochs, set(identifiers))
    if existing is not None:
        return existing
    if (job / "final.pt").exists():
        raise FinalRefitError(f"Untrusted final checkpoint exists: {job / 'final.pt'}")
    device = _device(config.device)
    torch.manual_seed(config.seed)
    model = TemporalFusion(width, config.joint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    steps = math.ceil(len(examples) / BATCH_SIZE)
    total_steps, warmup = steps * epochs, steps

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    completed, updates, exposures, gradients, deltas = _restore(job, model, optimizer, scheduler)
    resumed = completed
    if exposures and (set(exposures) != set(identifiers)
                      or any(value != completed for value in exposures.values())):
        raise FinalRefitError("Temporal resume checkpoint does not match the full-data cohort")
    exposures = {identifier: exposures.get(identifier, 0) for identifier in identifiers}
    started = time.perf_counter()
    connected = bool(gradients)
    for epoch in range(completed, epochs):
        if config.clock() >= config.deadline:
            break
        torch.manual_seed(config.seed + epoch)
        generator = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(len(examples), generator=generator).tolist()
        consumed: list[str] = []
        expired = False
        for start in range(0, len(order), BATCH_SIZE):
            if config.clock() >= config.deadline:
                expired = True
                break
            indices = order[start:start + BATCH_SIZE]
            features, padding, times, labels = _batch(examples, indices, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result = model(features, padding, times)
            loss = (joint_partial_label_loss(result.logits, labels) if config.joint
                    else balanced_known_label_bce(result.probabilities, labels))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite final temporal loss")
            loss.backward()
            disconnected = [name for name, parameter in model.named_parameters()
                            if parameter.grad is None or not torch.isfinite(parameter.grad).all()
                            or torch.count_nonzero(parameter.grad) == 0]
            if disconnected:
                raise FusionContractError(f"Disconnected temporal parameters: {disconnected}")
            gradients = {"model": float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))}
            optimizer.step()
            scheduler.step()
            updates += 1
            connected = True
            consumed.extend(examples[index].identifier for index in indices)
        if expired or set(consumed) != set(identifiers) or len(consumed) != len(identifiers):
            break
        completed = epoch + 1
        for identifier in consumed:
            exposures[identifier] += 1
        _checkpoint(job, model, optimizer, scheduler, completed, updates, exposures, gradients, deltas)
        print(json.dumps({"event": "final_refit_epoch_complete", "job": config.name,
                          "epoch": completed, "epochs": epochs, "updates": updates}), flush=True)
    status: Status = "complete" if completed == epochs and all(value == epochs for value in exposures.values()) else "incomplete_deadline"
    metadata = _metadata(status, epochs, completed, exposures, resumed, time.perf_counter() - started)
    metadata.update({"name": config.name, "joint": config.joint, "input_dim": width,
                     "updates": updates, "all_model_gradients_connected": connected,
                     "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
                     "schedule": "one-epoch warmup then cosine", "encoder_adaptation": False})
    if status == "complete":
        final = job / "final.pt"
        _atomic_state(final, {name: value.detach().cpu() for name, value in model.state_dict().items()})
        metadata["state_sha256"] = _digest(final)
    _atomic_json(job / "training.json", metadata)
    if status == "complete":
        (job / "checkpoint.pt").unlink(missing_ok=True)
    return metadata
