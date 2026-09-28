"""Data, optimization, and prediction helpers for AntiDeepfake E7."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from eight_audio import load
from eight_lora import (
    LoRALinear,
    adapter_state_dict,
    frozen_parameter_probe,
    load_adapter_state_dict,
    tensor_checksum,
)
from eight_metrics import binary_summary


WINDOW: Final = 64_600
HOP: Final = 32_300
TRAIN_VARIANTS: Final = frozenset({"base", "telephone", "mp3_32k"})
EVAL_VARIANTS: Final = frozenset({"base", "opus_telephone"})


class E7Error(RuntimeError):
    """Raised when E7 data or numerical invariants fail."""


@dataclass(frozen=True, slots=True)
class View:
    row: dict[str, str]
    start: int
    waveform: np.ndarray


def make_windows(
    audio: np.ndarray,
    window: int = WINDOW,
    hop: int = HOP,
) -> list[tuple[int, np.ndarray]]:
    """Return common fixed windows, zero-padding short clips and deduplicating tail."""
    if audio.ndim != 1 or len(audio) < 400 or not np.isfinite(audio).all():
        raise E7Error("Audio must be a finite mono waveform with at least 400 samples")
    starts = list(range(0, max(1, len(audio) - window + 1), hop))
    if len(audio) > window and starts[-1] != len(audio) - window:
        starts.append(len(audio) - window)
    result: list[tuple[int, np.ndarray]] = []
    for start in starts:
        piece = np.zeros(window, dtype=np.float32)
        source = audio[start : start + window]
        piece[: len(source)] = source
        result.append((start, piece))
    return result


def balanced_schedule(
    labels: list[int],
    steps: int,
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    """Build the same deterministic 1:1 class exposure for both trained arms."""
    if batch_size != 2 or steps < 1:
        raise E7Error("E7 requires positive steps and batch_size=2")
    classes = {label: np.flatnonzero(np.asarray(labels) == label) for label in (0, 1)}
    if any(not len(indices) for indices in classes.values()):
        raise E7Error("E7 training requires trusted REAL and FAKE labels")
    rng = np.random.default_rng(seed)
    return [[int(rng.choice(classes[0])), int(rng.choice(classes[1]))] for _ in range(steps)]


def load_views(config: dict) -> tuple[dict[str, list[View]], dict]:
    """Load only E7 core speech without using holdout labels for training."""
    from eight_contract import dataset_manifest, read_manifest, validate_manifest

    manifest = dataset_manifest(Path(config["root"]))
    receipt = validate_manifest(manifest.parent)
    rows = read_manifest(manifest)
    selected = [
        row for row in rows
        if row["kind"] == "speech"
        and row["probe_family"] == "core"
        and row["VOICE_FAKE"] in {"0", "1"}
        and ((row["split"] == "train" and row["variant"] in TRAIN_VARIANTS)
             or (row["split"] in {"calibration", "dev", "test"} and row["variant"] in EVAL_VARIANTS))
    ]
    by_split: dict[str, list[View]] = defaultdict(list)
    for row in selected:
        audio = load(manifest.parent / row["path"])
        for start, waveform in make_windows(audio):
            by_split[row["split"]].append(View(row, start, waveform))
    missing = {"train", "calibration", "dev", "test"} - set(by_split)
    if missing:
        raise E7Error(f"E7 is missing required splits: {sorted(missing)}")
    audit = {
        "manifest": str(manifest), "manifest_validation": receipt, "files": len(selected),
        "windows": {split: len(views) for split, views in by_split.items()},
        "train_class_counts": dict(Counter(row["VOICE_FAKE"] for row in selected if row["split"] == "train")),
        "train_generator_counts": dict(Counter(row["generator"] for row in selected if row["split"] == "train")),
        "train_variants": sorted({row["variant"] for row in selected if row["split"] == "train"}),
        "eval_variants": sorted({row["variant"] for row in selected if row["split"] != "train"}),
        "speakers": {split: len({atom for view in views for atom in view.row["group_id"].split("|")
                                  if atom.startswith("speaker:")}) for split, views in by_split.items()},
    }
    return dict(by_split), audit


def batch(views: list[View], indices: list[int], device: str) -> torch.Tensor:
    return torch.from_numpy(np.stack([views[index].waveform for index in indices])).to(device)


def logits(model: nn.Module, views: list[View], device: str, batch_size: int = 2) -> torch.Tensor:
    """Run deterministic eval-mode inference on already-normalized shared views."""
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(views), batch_size):
            audio = batch(views, list(range(start, min(start + batch_size, len(views)))), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunks.append(model(audio).float().cpu())
    return torch.cat(chunks)


def teacher_cache(model: nn.Module, views: list[View], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache teacher logits and the exact pooled encoder inputs to its classifier."""
    outputs: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(views), 2):
            audio = batch(views, list(range(start, min(start + 2, len(views)))), device)
            normalized = F.layer_norm(audio, audio.shape[-1:])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pooled = model.get_submodule("encoder")(normalized).last_hidden_state.mean(1)
                outputs.append(model.get_submodule("classifier")(pooled).float().cpu())
                embeddings.append(pooled.float().cpu())
    return torch.cat(outputs), torch.cat(embeddings)


def predictions(views: list[View], values: torch.Tensor) -> list[dict]:
    """Average window fake probabilities into per-file evidence rows."""
    grouped: dict[str, list[tuple[View, float]]] = defaultdict(list)
    for view, score in zip(views, values.softmax(-1)[:, 0].tolist(), strict=True):
        grouped[view.row["ID"]].append((view, float(score)))
    result = []
    for identifier, pairs in sorted(grouped.items()):
        row = pairs[0][0].row
        result.append({
            "ID": identifier, "split": row["split"], "label": int(row["VOICE_FAKE"]),
            "score": float(np.mean([score for _, score in pairs])), "component_id": row["group_id"],
            "source": row["source"], "generator": row["generator"], "codec": row["variant"],
            "parent_id": row.get("parent_id", row.get("parent_ids", "")), "windows": len(pairs),
        })
    return result


def summary(rows: list[dict], threshold: float | None = None) -> Mapping[str, float | int | None]:
    return binary_summary([row["label"] for row in rows], [row["score"] for row in rows], threshold)


def smoke(
    model: nn.Module,
    train: list[View],
    initial: dict[str, torch.Tensor],
    device: str,
) -> Mapping[str, float | int | bool]:
    """Prove real adapter gradients, one update, frozen base, and exact restore."""
    indices = balanced_schedule([int(view.row["VOICE_FAKE"]) for view in train], 1, 2, 41)[0]
    audio = batch(train, indices, device)
    targets = torch.tensor([1 - int(train[index].row["VOICE_FAKE"]) for index in indices], device=device)
    before_logits, base_before = logits(model, [train[index] for index in indices], device), frozen_parameter_probe(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(model(audio), targets)
    loss.backward()
    b_gradients = [p.grad for name, p in model.named_parameters() if name.endswith("lora_b.weight")]
    nonzero = sum(g is not None and float(g.abs().sum()) > 0 for g in b_gradients)
    if nonzero == 0 or any(g is None or not torch.isfinite(g).all() for g in b_gradients):
        raise E7Error("GPU smoke failed: LoRA B gradients are absent, zero, or non-finite")
    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
    optimizer.step()
    changed = tensor_checksum(adapter_state_dict(model)) != tensor_checksum(initial)
    base_unchanged = frozen_parameter_probe(model) == base_before
    load_adapter_state_dict(model, initial)
    restored = torch.equal(logits(model, [train[index] for index in indices], device), before_logits)
    if not changed or not base_unchanged or not restored:
        raise E7Error("GPU smoke failed: update, frozen-base, or restore invariant")
    return {"loss": float(loss), "nonzero_lora_b_gradients": nonzero,
            "lora_b_tensors": len(b_gradients), "adapter_changed": changed,
            "frozen_probe_unchanged": base_unchanged, "restored_logits_exact": restored}


def set_adapter_training(model: nn.Module, enabled: bool) -> None:
    """Select head-only or adapter-plus-head optimization without changing forwards."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad_(enabled)
            module.lora_b.requires_grad_(enabled)
    model.get_submodule("classifier").requires_grad_(True)


def train(
    model: nn.Module, train_views: list[View], teacher: torch.Tensor,
    initial: dict[str, torch.Tensor], schedule: list[list[int]], kl_weight: float,
    device: str, deadline: float, cached_embeddings: torch.Tensor | None = None,
) -> tuple[list[dict], dict]:
    """Complete one fixed-budget arm with finite gradients and clipped updates."""
    load_adapter_state_dict(model, initial)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=0.01)
    curve: list[dict] = []
    base_before = frozen_parameter_probe(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for step, indices in enumerate(schedule, 1):
        if time.time() >= deadline:
            raise TimeoutError("E7 deadline reached before completing fixed updates")
        audio = None if cached_embeddings is not None else batch(train_views, indices, device)
        targets = torch.tensor([1 - int(train_views[index].row["VOICE_FAKE"]) for index in indices], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = (model.get_submodule("classifier")(cached_embeddings[indices].to(device))
                      if cached_embeddings is not None else model(audio))
            ce = F.cross_entropy(output, targets)
            kl = F.kl_div(F.log_softmax(output.float(), -1), F.softmax(teacher[indices].to(device), -1), reduction="batchmean")
            loss = ce + kl_weight * kl
        if not torch.isfinite(loss):
            raise E7Error(f"Non-finite E7 loss at step {step}")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise E7Error(f"Missing or non-finite E7 gradient at step {step}")
        norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        curve.append({"step": step, "loss": float(loss), "ce": float(ce), "kl": float(kl),
                      "grad_norm_before_clip": float(norm)})
    state = adapter_state_dict(model)
    report = {"optimizer_updates": len(curve), "wall_seconds": time.time() - started,
              "peak_vram_bytes": torch.cuda.max_memory_allocated(), "adapter_checksum": tensor_checksum(state),
              "adapter_changed": tensor_checksum(state) != tensor_checksum(initial),
              "frozen_probe_unchanged": frozen_parameter_probe(model) == base_before}
    if not report["adapter_changed"] or not report["frozen_probe_unchanged"]:
        raise E7Error("E7 optimization did not change adapters or changed frozen parameters")
    return curve, report
