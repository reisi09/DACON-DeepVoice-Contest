"""Fixed frozen-feature heads and exposure-controlled samplers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


class HeadContractError(ValueError):
    """Frozen-feature head input violates the controlled comparison."""


@dataclass(frozen=True, slots=True)
class HeadResult:
    model: nn.Module
    checkpoint: Path
    updates: int
    draw_counts: dict[str, int]
    draw_sha256: str


def balanced_draws(
    rows: Sequence[Mapping[str, str | int]], *, steps: int, batch_size: int, seed: int, mode: str,
) -> list[list[int]]:
    """Draw class, then source/generator, then parent, then view uniformly."""
    if steps < 1 or batch_size < 2 or batch_size % 2 or not rows:
        raise HeadContractError("positive steps and a positive even batch size are required")
    labels = sorted({int(row["label"]) for row in rows})
    if labels != [0, 1]:
        raise HeadContractError("both classes are required")
    if mode not in ("parent", "source_generator_parent", "variant_parent"):
        raise HeadContractError(f"unknown sampler mode: {mode}")
    by_label = {label: [index for index, row in enumerate(rows) if int(row["label"]) == label] for label in labels}
    rng = np.random.default_rng(seed)
    parent_cursors: dict[tuple[int | str, ...], int] = {}
    schedule: list[list[int]] = []
    for _ in range(steps):
        batch: list[int] = []
        for label in labels:
            for position in range(batch_size // 2):
                ordinal = len(schedule) * (batch_size // 2) + position
                eligible = by_label[label]
                cell: tuple[int | str, ...] = (label,)
                if mode == "source_generator_parent":
                    sources = sorted({str(rows[index]["source"]) for index in eligible})
                    source = sources[ordinal % len(sources)]
                    eligible = [index for index in eligible if str(rows[index]["source"]) == source]
                    generators = sorted({str(rows[index].get("generator", "")) or "REAL" for index in eligible})
                    generator = generators[(ordinal // len(sources)) % len(generators)]
                    eligible = [index for index in eligible if (str(rows[index].get("generator", "")) or "REAL") == generator]
                    cell = (label, source, generator)
                elif mode == "variant_parent":
                    variants = sorted({str(rows[index]["variant"]) for index in eligible})
                    variant = variants[ordinal % len(variants)]
                    eligible = [index for index in eligible if str(rows[index]["variant"]) == variant]
                    cell = (label, variant)
                parents = sorted({str(rows[index]["parent_id"]) for index in eligible})
                cursor = parent_cursors.get(cell, 0)
                parent = parents[cursor % len(parents)]
                parent_cursors[cell] = cursor + 1
                members = [index for index in eligible if str(rows[index]["parent_id"]) == parent]
                batch.append(members[int(rng.integers(len(members)))])
        rng.shuffle(batch)
        schedule.append(batch)
    return schedule


def consistency_draws(
    rows: Sequence[Mapping[str, str | int]], paired_indices: Sequence[tuple[int, int]], *, steps: int, seed: int,
) -> list[tuple[int, int]]:
    """Spend one pair/update, balancing classes then cycling seed-shuffled parents/views.

    Parent coverage takes priority over exact aggregate codec counts. A finite
    budget may cover only a subset; no parent repeats until its class is exhausted.
    """
    groups: dict[int, dict[str, list[tuple[int, int]]]] = {0: {}, 1: {}}
    for first, second in paired_indices:
        if not (0 <= first < len(rows) and 0 <= second < len(rows)):
            raise HeadContractError("consistency pair index is outside the feature rows")
        label = int(rows[first]["label"])
        parent = str(rows[first]["parent_id"])
        if label not in groups or label != int(rows[second]["label"]) or parent != str(rows[second]["parent_id"]):
            raise HeadContractError("consistency pairs must share a binary label and parent")
        groups[label].setdefault(parent, []).append((first, second))
    if steps < 1 or not all(groups.values()):
        raise HeadContractError("consistency training requires matched pairs from both classes")
    rng = np.random.default_rng(seed)
    parents_by_label = {label: sorted(parents) for label, parents in groups.items()}
    for label, parents in parents_by_label.items():
        rng.shuffle(parents)
        for parent in parents:
            rng.shuffle(groups[label][parent])
    schedule: list[tuple[int, int]] = []
    for step in range(steps):
        label, ordinal = step % 2, step // 2
        parents = parents_by_label[label]
        pairs = groups[label][parents[ordinal % len(parents)]]
        schedule.append(pairs[(ordinal // len(parents)) % len(pairs)])
    return schedule


def consistency_objective(
    clean_logits: torch.Tensor,
    transformed_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight: float,
    clean_features: torch.Tensor | None = None,
    transformed_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Equal-view CE plus probability and optional normalized-feature consistency."""
    ce = 0.5 * (
        functional.binary_cross_entropy_with_logits(clean_logits, labels)
        + functional.binary_cross_entropy_with_logits(transformed_logits, labels)
    )
    probability = functional.mse_loss(clean_logits.sigmoid(), transformed_logits.sigmoid())
    feature = clean_logits.sum() * 0.0
    if clean_features is not None and transformed_features is not None:
        feature = functional.mse_loss(
            functional.normalize(clean_features, dim=-1),
            functional.normalize(transformed_features, dim=-1),
        )
    return ce + weight * (probability + feature)


def consistency_penalty(
    clean_logits: torch.Tensor,
    transformed_logits: torch.Tensor,
    clean_features: torch.Tensor,
    transformed_features: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    """Consistency only; CE exposure stays identical between E3 arms."""
    return weight * (
        functional.mse_loss(clean_logits.sigmoid(), transformed_logits.sigmoid())
        + functional.mse_loss(functional.normalize(clean_features, dim=-1),
                              functional.normalize(transformed_features, dim=-1))
    )


class FrozenHead(nn.Module):
    """Small matched MLP used by every frozen-feature comparison."""

    def __init__(self, width: int, architecture: str = "mlp") -> None:
        super().__init__()
        hidden = min(128, max(16, width // 4))
        self.network = (nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1))
                        if architecture == "mlp" else nn.Sequential(nn.Identity(), nn.Identity(), nn.Linear(width, 1)))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network[:2](features)
        return self.network[2](hidden).squeeze(-1), hidden


def fit_head(
    name: str,
    rows: Sequence[Mapping[str, str | int]],
    features: np.ndarray,
    output: Path,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    mode: str,
    device: str = "cpu",
    paired_indices: Sequence[tuple[int, int]] | None = None,
    consistency_weight: float = 0.0,
    architecture: str = "mlp",
) -> HeadResult:
    """Fit exactly ``steps`` AdamW updates and persist a reproducible checkpoint."""
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(rows) or not np.isfinite(matrix).all():
        raise HeadContractError("finite two-dimensional aligned features are required")
    draws = balanced_draws(rows, steps=steps, batch_size=batch_size, seed=seed, mode=mode)
    torch.manual_seed(seed)
    if architecture not in ("linear", "mlp"):
        raise HeadContractError("architecture must be linear or mlp")
    model = FrozenHead(matrix.shape[1], architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    x = torch.as_tensor(matrix, device=device)
    y = torch.as_tensor([int(row["label"]) for row in rows], dtype=torch.float32, device=device)
    pairs = (consistency_draws(rows, paired_indices or [], steps=steps, seed=seed)
             if consistency_weight else [])
    for step, indices in enumerate(draws):
        chosen = torch.as_tensor(indices, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x[chosen])
        loss = functional.binary_cross_entropy_with_logits(logits, y[chosen])
        if consistency_weight:
            pair = pairs[step]
            first, first_hidden = model(x[[pair[0]]])
            second, second_hidden = model(x[[pair[1]]])
            loss = loss + consistency_penalty(
                first, second, first_hidden, second_hidden, weight=consistency_weight,
            )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{name}.pt"
    flat = [index for batch in draws for index in batch]
    counts = {"class_0": sum(int(rows[index]["label"]) == 0 for index in flat),
              "class_1": sum(int(rows[index]["label"]) == 1 for index in flat)}
    for field in ("variant", "source", "generator"):
        for value in sorted({str(rows[index].get(field, "")) for index in flat}):
            counts[f"{field}:{value or 'REAL'}"] = sum(str(rows[index].get(field, "")) == value for index in flat)
            for label in (0, 1):
                counts[f"class_{label}|{field}:{value or 'REAL'}"] = sum(
                    int(rows[index]["label"]) == label and str(rows[index].get(field, "")) == value for index in flat)
    identity = "\n".join(str(rows[index]["ID"]) for index in flat).encode()
    draw_sha256 = hashlib.sha256(identity).hexdigest()
    torch.save({"model": model.state_dict(), "input_width": matrix.shape[1], "steps": steps,
                "batch_size": batch_size, "seed": seed, "mode": mode,
                "consistency_weight": consistency_weight, "architecture": architecture,
                "draw_counts": counts, "draw_sha256": draw_sha256}, checkpoint)
    return HeadResult(model.eval(), checkpoint, steps, counts, draw_sha256)


def predict(model: nn.Module, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Predict fake probabilities from frozen vectors."""
    with torch.inference_mode():
        logits, _ = model(torch.as_tensor(features, dtype=torch.float32, device=device))
    return logits.sigmoid().cpu().numpy()
