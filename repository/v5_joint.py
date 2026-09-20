"""Bounded end-to-end adaptation after cached temporal fusion training."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from v5_dataset import Example, read_audio
from v5_features import AudioViews, window_starts
from v5_fusion import joint_partial_label_loss
from v5_reporting import write_json


def live_features(models: dict[str, torch.nn.Module], view: AudioViews, *, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Regenerate evidence from current encoders; never reuse stale embeddings."""
    raw = read_audio(view.raw)
    starts = window_starts(len(raw), 64_600)
    if training and len(starts) > 2:
        starts = (starts[0], starts[-1])
    blocks = []
    for kind in ("voice", "music", "presence"):
        model = models[kind]
        device = next(model.parameters()).device
        width = model.window_samples
        waveform = read_audio(view.voice if kind == "voice" else view.music) if kind != "presence" else raw
        rows = []
        for start in starts:
            tensors = []
            for source in (raw, waveform):
                clip = np.pad(source[start:start + width], (0, max(0, width - len(source[start:]))))
                tensor = torch.from_numpy(clip).unsqueeze(0).to(device)
                logits, embedding = checkpoint(model, tensor, use_reentrant=False) if training else model(tensor)
                tensors.append((logits.float(), embedding.float()))
            (raw_logits, raw_embedding), (stem_logits, stem_embedding) = tensors
            rows.append(torch.cat((raw_embedding, stem_embedding, raw_logits, stem_logits,
                                   raw_logits - stem_logits), dim=1))
        blocks.append(torch.cat(rows, dim=0))
    features = torch.cat(blocks, dim=1).unsqueeze(0)
    times = torch.tensor(starts, device=features.device, dtype=torch.float32).unsqueeze(0) / 16_000 + view.offset_seconds
    return features, times


def adapt_joint(
    models: dict[str, torch.nn.Module], fusion: torch.nn.Module, rows: list[Example],
    views: dict[str, AudioViews], output: Path, deadline: float,
) -> dict[str, int | float | str | dict]:
    """Run up to five full passes, explicitly recording any budget-censored pass."""
    output.mkdir(parents=True, exist_ok=True)
    all_models = {**models, "fusion": fusion}
    for model in all_models.values():
        model.train()
        # Small-context adaptation keeps pretrained normalization statistics fixed.
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.eval()
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": 1e-7 if kind == "voice" else (1e-5 if kind == "fusion" else 1e-6)}
        for kind, model in all_models.items()], weight_decay=.01)
    started = time.time()
    epochs, updates, seen = 0, 0, set()
    audit = {}
    status = "completed"
    step_seconds = 30.0
    for epoch in range(5):
        ordered = rows.copy()
        random.Random(20260913 + epoch).shuffle(ordered)
        for row in ordered:
            if time.time() + step_seconds * 1.5 >= deadline:
                status = "budget_stop"
                break
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features, times = live_features(models, views[row.id], training=True)
                mask = torch.zeros(times.shape, dtype=torch.bool, device=features.device)
                output_value = fusion(features, mask, times)
                targets = torch.tensor([row.labels], device=features.device, dtype=torch.float32)
                loss = joint_partial_label_loss(output_value.logits.float(), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Joint adaptation produced nonfinite loss")
            loss.backward()
            if not audit:
                for kind, model in all_models.items():
                    norm = math.sqrt(sum(float(p.grad.float().square().sum()) for p in model.parameters() if p.grad is not None))
                    if not math.isfinite(norm) or norm <= 0:
                        raise RuntimeError(f"Joint loss is disconnected from {kind}")
                    audit[kind] = {"gradient_norm": norm}
            torch.nn.utils.clip_grad_norm_([p for model in all_models.values() for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            seen.add(row.id)
            step_seconds = max(step_seconds, time.time() - step_start)
            if updates % 10 == 0:
                print(f"JOINT updates={updates} epoch={epoch + 1} loss={float(loss):.5f}", flush=True)
        if status == "budget_stop":
            break
        epochs += 1
    result = {"status": status, "epochs_completed": epochs, "updates": updates,
              "distinct_sources": len(seen), "seconds": time.time() - started,
              "training_windows_per_context_max": 2, "gradient_audit": audit,
              "cache_policy": "All predictions regenerated from current encoder weights"}
    write_json(output / "joint_training.json", result)
    return result


def predict_joint(
    models: dict[str, torch.nn.Module], fusion: torch.nn.Module, rows: list[Example],
    views: dict[str, AudioViews], deadline: float,
) -> np.ndarray:
    for model in (*models.values(), fusion):
        model.eval()
    predictions = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            if time.time() >= deadline:
                raise TimeoutError("Joint full-file validation exceeded its reserved budget")
            features, times = live_features(models, views[row.id], training=False)
            mask = torch.zeros(times.shape, dtype=torch.bool, device=features.device)
            predictions.append(fusion(features, mask, times).probabilities.float().cpu().numpy()[0])
    return np.stack(predictions)
