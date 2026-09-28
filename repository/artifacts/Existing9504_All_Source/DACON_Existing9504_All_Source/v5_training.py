"""Budget-bounded full fine-tuning for the three V5 audio experts."""

# allow: SIZE_OK — indivisible budgeted training state machine and its public result schema.

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, assert_never

import numpy as np
import torch
from torch import nn

from graph_scoring import _roc_metrics
from v5_dataset import Example, read_audio, training_crop
from v5_experts import (
    MusicExpert,
    PresenceExpert,
    VoiceExpert,
    deterministic_gradient_audit,
    parameter_groups,
)

ExpertKind = Literal["voice", "music", "presence"]
Precision = Literal["bf16", "fp16", "fp32"]
TrainStatus = Literal["completed", "budget_stop"]
MilestoneStatus = Literal["completed", "censored"]
BASE_LR: Final = {"voice": 1e-6, "music": 3e-5, "presence": 1e-5}
LABEL_INDEX: Final = {"voice": (1,), "music": (2,), "presence": (3, 4)}
HEAD_NAMES: Final = {"voice": ("voice_fake",), "music": ("music_fake",),
                     "presence": ("voice_presence", "music_presence")}


@dataclass(frozen=True, slots=True)
class TrainConfig:
    kind: ExpertKind
    output: Path
    deadline: float
    stage_deadline: float | None = None
    epochs: int = 60
    max_epochs: int = 90
    seed: int = 20_260_913
    augmented: bool = False
    evaluation_interval: int | None = None
    microbatch: int | None = None
    accumulation: int | None = None
    precision: Precision = "bf16"
    device: str = "cuda"
    known_unused_parameters: tuple[str, ...] = ()
    profiled_step_seconds: float = 0.0
    clock: Callable[[], float] = time.time


@dataclass(frozen=True, slots=True)
class HeadMetric:
    name: str
    count: int
    loss: float | None
    eer: float | None
    auc: float | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    predictions: tuple[tuple[str, tuple[float, ...]], ...]
    window_starts: tuple[tuple[str, tuple[int, ...]], ...]
    heads: tuple[HeadMetric, ...]
    loss: float
    selection_metric: float


@dataclass(frozen=True, slots=True)
class FirstStepAudit:
    gradient_norms: dict[str, float]
    parameter_deltas: dict[str, float]


@dataclass(frozen=True, slots=True)
class EpochMetric:
    epoch: int
    updates: int
    train_loss: float
    seconds: float
    source_exposures: dict[str, int]
    dev: EvaluationResult | None


@dataclass(frozen=True, slots=True)
class Milestone:
    epoch: int
    status: MilestoneStatus


@dataclass(frozen=True, slots=True)
class TrainResult:
    status: TrainStatus
    epochs_completed: int
    updates: int
    seconds: float
    best_path: Path | None
    final_path: Path | None
    epoch_metrics: tuple[EpochMetric, ...]
    first_step_audit: FirstStepAudit | None
    milestones: tuple[Milestone, ...]


class TrainingError(RuntimeError):
    """The requested expert training contract cannot be completed safely."""


class BudgetExpired(TrainingError):
    """The immutable run or stage deadline was reached."""


def masked_balanced_bce(
    logits: torch.Tensor, targets: torch.Tensor, population: torch.Tensor
) -> torch.Tensor:
    """Give each population class equal epoch mass, then average across heads."""
    logits_2d = logits.reshape(-1, 1) if logits.ndim == 1 else logits
    targets_2d = targets.reshape(-1, 1) if targets.ndim == 1 else targets
    population_2d = population.reshape(-1, 1) if population.ndim == 1 else population
    if logits_2d.shape != targets_2d.shape or logits_2d.shape[1] != population_2d.shape[1]:
        raise TrainingError("logits, targets, and population heads must match")
    head_losses: list[torch.Tensor] = []
    for head in range(logits_2d.shape[1]):
        known = torch.isfinite(targets_2d[:, head])
        if not torch.any(known):
            continue
        population_known = torch.isfinite(population_2d[:, head])
        classes = [label for label in (0.0, 1.0)
                   if torch.any(population_known & (population_2d[:, head] == label))]
        total = int(population_known.sum())
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_2d[known, head], targets_2d[known, head], reduction="none")
        weights = torch.ones_like(losses)
        for label in classes:
            count = int((population_known & (population_2d[:, head] == label)).sum())
            weights[targets_2d[known, head] == label] = total / (len(classes) * count)
        head_losses.append((losses * weights).mean())
    if not head_losses:
        raise TrainingError("batch has no known binary supervision")
    return torch.stack(head_losses).mean()


def _targets(row: Example, kind: ExpertKind) -> tuple[float, ...]:
    return tuple(row.labels[index] for index in LABEL_INDEX[kind])


def _crop_targets(labels: tuple[float, ...], kind: ExpertKind) -> tuple[float, ...]:
    return tuple(labels[index] for index in LABEL_INDEX[kind])


def _eligible(row: Example, kind: ExpertKind) -> bool:
    values = _targets(row, kind)
    match kind:
        case "voice":
            return row.labels[3] == 1.0 and math.isfinite(values[0])
        case "music":
            return row.labels[4] == 1.0 and math.isfinite(values[0])
        case "presence":
            return any(math.isfinite(value) for value in values)
        case unreachable:
            assert_never(unreachable)


def _starts(samples: int, window: int) -> tuple[int, ...]:
    if samples <= window:
        return (0,)
    last = samples - window
    starts = list(range(0, last + 1, window))
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def _limit(config: TrainConfig) -> float:
    return min(config.deadline, config.stage_deadline or config.deadline)


def _check_budget(deadline: float, clock: Callable[[], float], reserve: float = 0.0) -> None:
    if clock() + reserve >= deadline:
        raise BudgetExpired


def _forward(model: nn.Module, audio: torch.Tensor) -> torch.Tensor:
    logits, _embeddings = model(audio)
    return logits.float()


def evaluate_expert(
    model: nn.Module,
    rows: Sequence[Example],
    kind: ExpertKind,
    *,
    device: str | torch.device,
    deadline: float = float("inf"),
    clock: Callable[[], float] = time.time,
) -> EvaluationResult:
    """Max-pool probabilities over a tail-anchored grid covering every sample."""
    model.eval()
    predictions: list[tuple[str, tuple[float, ...]]] = []
    grids: list[tuple[str, tuple[int, ...]]] = []
    pooled_logits: list[torch.Tensor] = []
    target_rows: list[tuple[float, ...]] = []
    window = int(model.window_samples)
    with torch.inference_mode():
        for row in rows:
            if not _eligible(row, kind):
                continue
            _check_budget(deadline, clock)
            audio = np.asarray(read_audio(row.path), dtype=np.float32)
            starts = _starts(len(audio), window)
            windows = np.zeros((len(starts), window), dtype=np.float32)
            for index, start in enumerate(starts):
                clip = audio[start:start + window]
                windows[index, :len(clip)] = clip
            _check_budget(deadline, clock)
            batch_size = 1 if kind == "voice" else 16
            window_logits: list[torch.Tensor] = []
            for offset in range(0, len(windows), batch_size):
                _check_budget(deadline, clock)
                window_logits.append(_forward(
                    model, torch.from_numpy(windows[offset:offset + batch_size]).to(device),
                ))
                _check_budget(deadline, clock)
            pooled = torch.cat(window_logits).max(dim=0).values.reshape(-1)
            probabilities = tuple(torch.sigmoid(pooled).cpu().tolist())
            predictions.append((row.id, probabilities))
            grids.append((row.id, starts))
            pooled_logits.append(pooled.cpu())
            target_rows.append(_targets(row, kind))
    if not pooled_logits:
        raise TrainingError("development rows contain no eligible supervision")
    logits = torch.stack(pooled_logits)
    targets = torch.tensor(target_rows, dtype=torch.float32)
    loss = float(masked_balanced_bce(logits, targets, targets))
    heads: list[HeadMetric] = []
    for index, name in enumerate(HEAD_NAMES[kind]):
        known = torch.isfinite(targets[:, index])
        truth = targets[known, index].int().tolist()
        scores = torch.sigmoid(logits[known, index]).tolist()
        if len(set(truth)) == 2:
            eer, auc = _roc_metrics(truth, scores)
        else:
            eer, auc = None, None
        head_loss = (float(masked_balanced_bce(logits[:, index], targets[:, index], targets[:, index]))
                     if truth else None)
        heads.append(HeadMetric(name, len(truth), head_loss, eer, auc))
    if kind == "presence":
        valid = [head.auc for head in heads if head.auc is not None]
    else:
        valid = [1.0 - head.eer for head in heads if head.eer is not None]
    selection = sum(valid) / len(valid) if valid else -loss
    return EvaluationResult(tuple(predictions), tuple(grids), tuple(heads), loss, selection)


def _save(path: Path, model: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({name: value.detach().cpu().float() for name, value in model.state_dict().items()}, temporary)
    temporary.replace(path)


def _parameter_groups(
    model: nn.Module,
) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    match model:
        case VoiceExpert() | MusicExpert() | PresenceExpert():
            resolved = parameter_groups(model)
            return list(resolved.encoder), list(resolved.head)
        case _:
            named = [(name, parameter) for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
            return ([item for item in named if not item[0].startswith("head.")],
                    [item for item in named if item[0].startswith("head.")])


def _optimizer(model: nn.Module, kind: ExpertKind) -> torch.optim.AdamW:
    encoder_named, head_named = _parameter_groups(model)
    encoder = [parameter for _name, parameter in encoder_named]
    head = [parameter for _name, parameter in head_named]
    groups = [{"params": encoder, "lr": BASE_LR[kind]},
              {"params": head, "lr": BASE_LR[kind] * 3}]
    return torch.optim.AdamW([group for group in groups if group["params"]], weight_decay=0.01)


def _gradient_context(model: nn.Module, enabled: bool) -> AbstractContextManager[None]:
    if not enabled:
        return nullcontext()
    match model:
        case VoiceExpert() | MusicExpert() | PresenceExpert():
            return deterministic_gradient_audit(model)
        case _:
            return nullcontext()


def _disable_native_augmentation(model: nn.Module) -> None:
    """Keep B1/B2 different only through the dataset's declared augmentation."""
    match model:
        case VoiceExpert():
            ssl_model = model.wrapper.get_submodule("backbone.ssl_model")
            ssl_model.config.apply_spec_augment = False
        case PresenceExpert():
            model.cnn.spec_augmenter = nn.Identity()
        case MusicExpert():
            return
        case _:
            return


def _training_mode(model: nn.Module, microbatch: int) -> int:
    """Restore stochastic training while freezing unstable singleton BN statistics."""
    model.train()
    if microbatch != 1:
        return 0
    frozen = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            frozen += 1
    return frozen


def _audit_before_step(
    model: nn.Module, known_unused: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, tuple[nn.Parameter, int, float]]]:
    norms: dict[str, float] = {}
    probes: dict[str, tuple[nn.Parameter, int, float]] = {}
    for group, named in zip(("encoder", "head"), _parameter_groups(model), strict=True):
        for name, parameter in named:
            if parameter.grad is None:
                if name not in known_unused:
                    raise TrainingError(f"active parameter has no first-step gradient: {name}")
                continue
            norm = float(parameter.grad.detach().float().norm())
            norms[group] = math.hypot(norms.get(group, 0.0), norm)
            peak = int(parameter.grad.detach().abs().argmax())
            if group not in probes or abs(float(parameter.grad.reshape(-1)[peak])) > abs(probes[group][2]):
                probes[group] = (parameter, peak, float(parameter.reshape(-1)[peak]))
    if not norms or any(not math.isfinite(value) or value == 0.0 for value in norms.values()):
        raise TrainingError("first-step encoder gradient audit failed")
    return norms, probes


def _finish(
    status: TrainStatus, started: float, updates: int, reports: list[EpochMetric],
    best: Path | None, final: Path | None, audit: FirstStepAudit | None,
) -> TrainResult:
    completed = len(reports)
    milestones = tuple(Milestone(epoch, "completed" if completed >= epoch else "censored")
                       for epoch in (10, 30, 60))
    return TrainResult(status, completed, updates, time.perf_counter() - started, best, final,
                       tuple(reports), audit, milestones)


def train_expert(
    model: nn.Module, train_rows: Sequence[Example], dev_rows: Sequence[Example], config: TrainConfig
) -> TrainResult:
    """Train all active parameters without crossing either immutable deadline."""
    if not 1 <= config.epochs <= config.max_epochs <= 90:
        raise TrainingError("epochs must be within the configured maximum of 90")
    rows = [row for row in train_rows if _eligible(row, config.kind)]
    if not rows:
        raise TrainingError("training rows contain no eligible supervision")
    if len({row.id for row in rows}) != len(rows):
        raise TrainingError("eligible training IDs must be unique")
    device = torch.device(config.device)
    model.to(device)
    _disable_native_augmentation(model)
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    microbatch = config.microbatch or (1 if config.kind == "voice" else 16)
    accumulation = config.accumulation or (8 if config.kind == "voice" else 1)
    interval = config.evaluation_interval or (2 if config.kind == "voice" else 5)
    steps_per_epoch = math.ceil(math.ceil(len(rows) / microbatch) / accumulation)
    optimizer = _optimizer(model, config.kind)
    total_steps = max(1, steps_per_epoch * 60)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.1 + 0.45 *
        (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.precision == "fp16")
    population = torch.tensor([_targets(row, config.kind) for row in rows], dtype=torch.float32,
                              device=device)
    reports: list[EpochMetric] = []
    updates = 0
    best_score = float("-inf")
    best_path: Path | None = None
    audit: FirstStepAudit | None = None
    started = time.perf_counter()
    estimated_training = config.profiled_step_seconds * steps_per_epoch
    estimated_evaluation = 0.0
    deadline = _limit(config)
    try:
        _check_budget(deadline, config.clock, config.profiled_step_seconds)
        for epoch in range(config.epochs):
            if estimated_training and estimated_evaluation:
                _check_budget(deadline, config.clock,
                              (estimated_training + estimated_evaluation) * 1.15)
            frozen_batch_norms = _training_mode(model, microbatch)
            epoch_started = time.perf_counter()
            training_started = config.clock()
            ordered = rows.copy()
            random.Random(config.seed + epoch).shuffle(ordered)
            exposures: dict[str, int] = {}
            losses: list[float] = []
            optimizer.zero_grad(set_to_none=True)
            groups_since_step = 0
            batches = [ordered[index:index + microbatch] for index in range(0, len(ordered), microbatch)]
            for batch_index, batch in enumerate(batches):
                _check_budget(deadline, config.clock)
                prepared = [training_crop(row, epoch, int(model.window_samples), config.augmented,
                                          config.seed) for row in batch]
                _check_budget(deadline, config.clock)
                audio = torch.from_numpy(np.stack([item[0] for item in prepared])).to(device)
                targets = torch.tensor([_crop_targets(item[1], config.kind) for item in prepared],
                                       dtype=torch.float32, device=device)
                use_amp = device.type == "cuda" and config.precision != "fp32"
                amp_dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
                with _gradient_context(model, audit is None), torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp,
                ):
                    loss = masked_balanced_bce(_forward(model, audio), targets, population) / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite expert loss")
                scaler.scale(loss).backward()
                losses.append(float(loss.detach()) * accumulation)
                for row in batch:
                    for group in row.groups:
                        exposures[group] = exposures.get(group, 0) + 1
                groups_since_step += 1
                final_batch = batch_index + 1 == len(batches)
                if groups_since_step == accumulation or final_batch:
                    scaler.unscale_(optimizer)
                    probes = None
                    norms: dict[str, float] = {}
                    if audit is None:
                        norms, probes = _audit_before_step(model, config.known_unused_parameters)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("non-finite expert gradient")
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    updates += 1
                    if probes is not None:
                        deltas = {group: abs(float(parameter.reshape(-1)[index]) - before)
                                  for group, (parameter, index, before) in probes.items()}
                        if any(value == 0.0 for value in deltas.values()):
                            raise TrainingError("first-step parameter delta audit failed")
                        audit = FirstStepAudit(norms, deltas)
                        print(json.dumps({"event": "first_step_audit", "updates": updates,
                                          "gradient_norms": norms, "parameter_deltas": deltas},
                                         sort_keys=True), flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    groups_since_step = 0
                _check_budget(deadline, config.clock)
            if any(sum(group in row.groups for row in ordered) != count
                   for group, count in exposures.items()):
                raise TrainingError("complete epoch exposure ledger mismatch")
            estimated_training = max(estimated_training, config.clock() - training_started)
            next_epoch_fits = (not estimated_evaluation or
                               config.clock() + (estimated_training + estimated_evaluation) * 1.15
                               < deadline)
            should_evaluate = (epoch == 0 or (epoch + 1) % interval == 0 or
                               epoch + 1 == config.epochs or not next_epoch_fits)
            evaluation_started = config.clock()
            dev = evaluate_expert(model, dev_rows, config.kind, device=device, deadline=deadline,
                                  clock=config.clock) if should_evaluate else None
            if dev is not None:
                estimated_evaluation = max(estimated_evaluation,
                                           config.clock() - evaluation_started)
            elapsed = time.perf_counter() - epoch_started
            reports.append(EpochMetric(epoch + 1, updates, sum(losses) / len(losses), elapsed,
                                       exposures, dev))
            print(json.dumps({"event": "epoch_complete", "epoch": epoch + 1,
                              "updates": updates, "seconds": elapsed,
                              "batch_norm_stats_frozen": frozen_batch_norms,
                              "dev_selection_metric": (dev.selection_metric if dev else None)},
                             sort_keys=True), flush=True)
            if dev is not None and dev.selection_metric > best_score:
                best_score = dev.selection_metric
                best_path = config.output / "best.pt"
                _save(best_path, model)
    except BudgetExpired:
        return _finish("budget_stop", started, updates, reports, best_path, None, audit)
    final_path = config.output / "final.pt"
    _save(final_path, model)
    return _finish("completed", started, updates, reports, best_path, final_path, audit)
