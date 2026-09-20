"""One bounded full-fine-tuning trial and full-file prediction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import numpy as np
from safetensors.torch import save_file

from v4_batches import music_supervision_rows
from v4_data import load_audio
from v4_evaluation import evaluate_development
from v4_model import LoadedModel, PreparedWindows, pool_bag_logits, predict_files, window_logits
from v4_training import Batch, RunState, build_optimizer, build_scheduler, load_checkpoint, run_batches, save_checkpoint
from v4_training_data import EpochBatches

Row = Mapping[str, str]


@dataclass(frozen=True, slots=True)
class TrialSpec:
    name: str
    kind: Literal["sonics", "ast"]
    encoder_lr: float
    head_lr: float
    epochs: int
    max_updates: int | None = None
    p3: bool = False
    p4: bool = False
    p4_control: bool = False
    patience: int = 5
    seed: int = 20_260_913
    contract_hash: str = ""


def _hash(value: Mapping[str, str | int | float | bool | None]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reset_training_seed(seed: int) -> str:
    """Reset every stochastic backend and return the exact initial RNG receipt."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    digest = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()), protocol=5))
    digest.update(torch.get_rng_state().numpy().tobytes())
    for state in torch.cuda.get_rng_state_all() if torch.cuda.is_available() else ():
        digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _save_weights(path: Path, model: torch.nn.Module) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary)
    temporary.replace(path)


def _atomic_payload(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _finish(output: Path, contract: str, result: dict) -> dict:
    receipt = output / "trial_receipt.json"
    if receipt.is_file():
        result = {**result, "trial_receipt": str(receipt),
                  "trial_receipt_sha256": _file_digest(receipt)}
    artifacts = {}
    for key in ("weights", "final_weights"):
        path = Path(str(result[key])) if result.get(key) else None
        if path is not None and path.is_file():
            artifacts[key] = _file_digest(path)
    _atomic_payload(output / "terminal_result.json",
                    {"contract": contract, "artifacts": artifacts, "result": result})
    return result


def _load_terminal(output: Path, contract: str) -> dict | None:
    path = output / "terminal_result.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != contract:
        raise ValueError("terminal result contract mismatch")
    result = payload["result"]
    for key, expected in payload.get("artifacts", {}).items():
        artifact = Path(str(result[key]))
        if not artifact.is_file() or _file_digest(artifact) != expected:
            raise ValueError(f"terminal result {key} hash mismatch")
    return result


def _write_exposure_records(output: Path, state: RunState) -> Path:
    path = output / "exposure_records.json"
    _atomic_payload(path, {"records": [json.loads(value) for value in state.exposure_records]})
    return path


def _weighted_music_loss(logits: torch.Tensor, metadata: PreparedWindows, targets: torch.Tensor,
                         weights: torch.Tensor, *, p4: bool, p4_control: bool,
                         file_count: int) -> torch.Tensor:
    file_indices = metadata.file_indices.to(logits.device)
    if not p4 or p4_control:
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets[file_indices], reduction="none")
        if p4_control:
            per_bag = [losses[file_indices == index].mean() * weights[index]
                       for index in range(file_count)]
            return torch.stack(per_bag).sum()
        return (losses * weights[file_indices]).sum()
    pooled = pool_bag_logits(logits, file_indices, file_count)
    result = (torch.nn.functional.binary_cross_entropy_with_logits(
        pooled, targets, reduction="none") * weights).sum()
    window_targets = metadata.window_targets.to(logits.device)
    known = torch.isfinite(window_targets)
    if torch.any(known):
        auxiliary = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[known], window_targets[known], reduction="none")
        known_files = file_indices[known]
        per_bag = []
        for index in range(file_count):
            selected = auxiliary[known_files == index]
            if selected.numel():
                per_bag.append(weights[index] * selected.mean())
        if per_bag:
            result = result + 0.5 * torch.stack(per_bag).sum()
    return result


def _early_stop(epoch: int, patience_epoch: int, patience: int) -> bool:
    return patience > 0 and epoch >= 5 and epoch - patience_epoch > patience


def predict_rows(
    loaded: LoadedModel,
    rows: Sequence[Row],
    root: Path,
    baseline: Sequence[Mapping[str, str | float]],
    *,
    p4: bool,
    deadline: float,
) -> list[dict[str, str | float]]:
    baseline_by_id = {str(row["ID"]): row for row in baseline}
    if len(baseline_by_id) != len(rows):
        raise ValueError("baseline prediction coverage mismatch")
    result: list[dict[str, str | float]] = []
    loaded.model.eval()
    for offset in range(0, len(rows), 8):
        if time.time() >= deadline:
            raise TimeoutError("deadline reached during full-file prediction")
        selected = rows[offset : offset + 8]
        values = predict_files(loaded, [load_audio(root / row["path"]) for row in selected], p4=p4).probabilities
        for row, probability in zip(selected, values.tolist(), strict=True):
            base = baseline_by_id[str(row["ID"])]
            result.append({"ID": row["ID"],
                "FILE_FAKE_PROB": max(float(base["VOICE_PRESENT_PROB"]) * float(base["VOICE_FAKE_PROB"]),
                                      float(base["MUSIC_PRESENT_PROB"]) * probability),
                "VOICE_FAKE_PROB": base["VOICE_FAKE_PROB"], "MUSIC_FAKE_PROB": probability,
                "VOICE_PRESENT_PROB": base["VOICE_PRESENT_PROB"],
                "MUSIC_PRESENT_PROB": base["MUSIC_PRESENT_PROB"]})
    return result


def write_predictions(path: Path, rows: Sequence[Mapping[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB",
            "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"))
        writer.writeheader()
        writer.writerows(rows)


def profile_update(
    loaded: LoadedModel,
    groups: tuple[Sequence[tuple[str, torch.nn.Parameter]], Sequence[tuple[str, torch.nn.Parameter]]],
    rows: Sequence[Row],
    data_root: Path,
    inference_row: Row,
    codec_cache: Path,
    deadline: float,
) -> dict[str, str | int | float]:
    """Measure real preparation, one full-parameter update, and one full-file inference."""
    eligible = music_supervision_rows(rows)
    microbatch = 2
    batches = EpochBatches(loaded, eligible, data_root, 0, microbatch=microbatch, p3=False,
                           p4=False, p4_control=False, codec_cache=data_root / ".unused", deadline=deadline)
    prepared_at = time.perf_counter()
    batch = batches[0]
    preparation_seconds = time.perf_counter() - prepared_at
    p3_preparation_seconds = 0.0
    if loaded.kind == "sonics":
        p3_batches = EpochBatches(loaded, eligible, data_root, 0, microbatch=microbatch, p3=True,
                                  p4=False, p4_control=False, codec_cache=codec_cache,
                                  deadline=deadline)
        p3_at = time.perf_counter()
        p3_batches[0]
        p3_preparation_seconds = time.perf_counter() - p3_at
    before = {name: value.detach().cpu().clone() for name, value in loaded.model.named_parameters()}
    optimizer = build_optimizer(groups[0], groups[1], encoder_lr=1e-5 if loaded.kind == "sonics" else 1e-6,
                                head_lr=3e-5 if loaded.kind == "sonics" else 1e-4)
    optimizer.zero_grad(set_to_none=True)
    if next(loaded.model.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    update_at = time.perf_counter()
    logits = loaded.model(batch.inputs.to(next(loaded.model.parameters()).device)).reshape(-1).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch.targets.to(logits.device))
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(loaded.model.parameters(), 1.0))
    optimizer.step()
    if next(loaded.model.parameters()).is_cuda:
        torch.cuda.synchronize()
    update_seconds = time.perf_counter() - update_at
    absolute_update = sum(float((parameter.detach().cpu() - before[name]).abs().sum())
                          for name, parameter in loaded.model.named_parameters())
    inference_at = time.perf_counter()
    predict_files(loaded, [load_audio(data_root / inference_row["path"])], p4=False)
    if next(loaded.model.parameters()).is_cuda:
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_at
    return {"kind": loaded.kind, "sample_file": inference_row["ID"],
            "preparation_seconds": preparation_seconds, "update_seconds": update_seconds,
            "p3_preparation_seconds": p3_preparation_seconds,
            "full_file_inference_seconds": inference_seconds, "gradient_norm": gradient_norm,
            "absolute_parameter_update": absolute_update,
            "trainable_parameters": sum(parameter.numel() for parameter in loaded.model.parameters()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0}


def train_trial(
    loaded: LoadedModel,
    groups: tuple[Sequence[tuple[str, torch.nn.Parameter]], Sequence[tuple[str, torch.nn.Parameter]]],
    spec: TrialSpec,
    train_rows: Sequence[Row],
    dev_rows: Sequence[Row],
    baseline: Sequence[Mapping[str, str | float]],
    data_root: Path,
    output: Path,
    deadline: float,
) -> dict[str, str | int | float | bool | list | dict]:
    seed_rng_sha256 = reset_training_seed(spec.seed)
    initial_weights_sha256 = model_fingerprint(loaded.model)
    eligible = music_supervision_rows(train_rows)
    if not eligible:
        raise ValueError("training manifest has no eligible music supervision")
    microbatch, accumulation = ((2, 4) if spec.p4 else (8, 4))
    probe_batches = EpochBatches(loaded, eligible, data_root, 0, microbatch=microbatch,
        p3=spec.p3, p4=spec.p4, p4_control=spec.p4_control,
        codec_cache=output / "codec_cache", deadline=deadline)
    steps_per_epoch = math.ceil(len(probe_batches) / accumulation)
    optimizer = build_optimizer(groups[0], groups[1], encoder_lr=spec.encoder_lr, head_lr=spec.head_lr)
    scheduler = build_scheduler(optimizer, steps_per_epoch=steps_per_epoch)
    contract = _hash({**asdict(spec), "seed_rng_sha256": seed_rng_sha256,
        "initial_weights_sha256": initial_weights_sha256,
        "train_ids": hashlib.sha256("\n".join(row["ID"] for row in eligible).encode()).hexdigest()})
    latest = output / "latest.pt"
    terminal = _load_terminal(output, contract)
    if terminal is not None:
        return terminal
    _atomic_payload(output / "trial_receipt.json", {"name": spec.name, "seed": spec.seed,
        "seed_rng_sha256": seed_rng_sha256, "initial_weights_sha256": initial_weights_sha256,
        "source_contract_hash": spec.contract_hash, "trial_contract": contract,
        "resume_checkpoint_sha256": _file_digest(latest) if latest.is_file() else ""})
    best = output / "best.safetensors"
    state = load_checkpoint(latest, loaded.model, optimizer, scheduler, contract) if latest.is_file() else RunState()
    reports_path = output / "epochs.json"
    reports = json.loads(reports_path.read_text()) if reports_path.is_file() else []
    best_metadata = output / "best.json"
    if best_metadata.is_file():
        saved_best = json.loads(best_metadata.read_text(encoding="utf-8"))
        if saved_best["contract"] != contract or not best.is_file():
            raise ValueError("best checkpoint contract mismatch")
        if _file_digest(best) != saved_best["checkpoint_sha256"]:
            raise ValueError("best checkpoint hash mismatch")
        best_score, best_epoch = float(saved_best["score"]), int(saved_best["epoch"])
        best_updates = int(saved_best["optimizer_steps"])
        best_windows = int(saved_best["window_exposures"])
    else:
        best_score, best_epoch, best_updates, best_windows = float("-inf"), 0, 0, 0
    patience_score, patience_epoch = float("-inf"), 0
    for report in reports:
        score = float(report["rank_score"])
        if score >= patience_score + 0.002:
            patience_score, patience_epoch = score, int(report["epoch"])
    started = time.perf_counter()

    def complete_fixed_update() -> dict:
        predictions = predict_rows(loaded, dev_rows, data_root, baseline,
                                   p4=spec.p4 and not spec.p4_control,
                                   deadline=deadline)
        write_predictions(output / "fixed_update_dev_predictions.csv", predictions)
        evaluation = evaluate_development(dev_rows, baseline, predictions)
        final = output / "final.safetensors"
        _save_weights(final, loaded.model)
        value = {"name": spec.name, "status": "completed", "fixed_updates": True,
            "updates": state.optimizer_steps, "epochs_completed": len(reports),
            "window_exposures": state.window_exposures, "run_total_updates": state.optimizer_steps,
            "run_total_window_exposures": state.window_exposures, "best_epoch": 0,
            "best_score": float(evaluation["rank_score"]), "weights": str(final),
            "reports": reports, "evaluation": evaluation,
            "exposure_records_path": str(_write_exposure_records(output, state)),
            "seconds": time.perf_counter() - started,
            "trainable_parameters": sum(p.numel() for p in loaded.model.parameters())}
        return _finish(output, contract, value)

    while state.epoch < spec.epochs:
        batches = EpochBatches(loaded, eligible, data_root, state.epoch, microbatch=microbatch,
            p3=spec.p3, p4=spec.p4, p4_control=spec.p4_control,
            codec_cache=output / "codec_cache", deadline=deadline)

        def loss_fn(_model: torch.nn.Module, batch: Batch) -> torch.Tensor:
            if batch.file_indices is None or batch.window_targets is None:
                raise RuntimeError("prepared batch metadata is missing")
            metadata = PreparedWindows(batch.inputs, batch.file_indices, batch.starts, batch.window_targets)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=next(loaded.model.parameters()).is_cuda):
                logits = window_logits(loaded, metadata)
            logits = logits.float()
            targets = batch.targets.to(logits.device)
            weights = batch.sample_weights.to(logits.device)
            return _weighted_music_loss(logits, metadata, targets, weights, p4=spec.p4,
                                        p4_control=spec.p4_control, file_count=len(batch.ids))

        remaining = None if spec.max_updates is None else spec.max_updates - state.optimizer_steps
        if remaining is not None and remaining <= 0:
            return complete_fixed_update()
        result = run_batches(loaded.model, optimizer, scheduler, batches, state, latest, deadline,
                             max_updates=remaining, contract=contract, loss_fn=loss_fn, accumulation=accumulation)
        state = result.state
        exposure_path = _write_exposure_records(output, state)
        fixed_complete = (result.status == "interrupted" and spec.max_updates is not None
                          and state.optimizer_steps >= spec.max_updates)
        if result.status != "complete" and not fixed_complete:
            if result.status == "stopped_deadline" and len(reports) >= 5 and best.is_file():
                return _finish(output, contract, {"name": spec.name, "status": "completed", "resource_censored": True,
                        "updates": best_updates, "epochs_completed": len(reports),
                        "completed_epoch_updates": int(reports[-1]["optimizer_steps"]),
                        "completed_epoch_window_exposures": int(reports[-1]["window_exposures"]),
                        "run_total_updates": state.optimizer_steps,
                        "run_total_window_exposures": state.window_exposures,
                        "window_exposures": best_windows, "best_epoch": best_epoch,
                        "best_score": best_score, "weights": str(best), "reports": reports,
                        "exposure_records_path": str(exposure_path),
                        "seconds": time.perf_counter() - started,
                        "trainable_parameters": sum(p.numel() for p in loaded.model.parameters())})
            return {"name": spec.name, "status": result.status, "updates": state.optimizer_steps,
                    "epochs_completed": len(reports), "window_exposures": state.window_exposures,
                    "exposure_records_path": str(exposure_path),
                    "seconds": time.perf_counter() - started}
        if fixed_complete:
            return complete_fixed_update()
        expected = {row["ID"] for row in eligible}
        previous = {key: int(value) for key, value in (reports[-1]["cumulative_exposures"].items() if reports else ())}
        if any(state.exposures.get(identifier, 0) - previous.get(identifier, 0) != 1 for identifier in expected):
            raise RuntimeError("complete epoch did not expose every eligible file exactly once")
        predictions = predict_rows(loaded, dev_rows, data_root, baseline,
                                   p4=spec.p4 and not spec.p4_control, deadline=deadline)
        write_predictions(output / f"epoch_{state.epoch + 1:03d}_dev_predictions.csv", predictions)
        evaluation = evaluate_development(dev_rows, baseline, predictions)
        score = float(evaluation["rank_score"])
        report = {"epoch": state.epoch + 1, "optimizer_steps": state.optimizer_steps,
                  "window_exposures": state.window_exposures,
                  "rank_score": score, "evaluation": evaluation,
                  "cumulative_exposures": state.exposures}
        reports.append(report)
        reports_path.parent.mkdir(parents=True, exist_ok=True)
        reports_path.write_text(json.dumps(reports, allow_nan=False), encoding="utf-8")
        if score > best_score or not best.is_file():
            temporary = best.with_suffix(".tmp")
            save_file({name: value.detach().cpu() for name, value in loaded.model.state_dict().items()}, temporary)
            temporary.replace(best)
            best_score, best_epoch = score, state.epoch + 1
            best_updates, best_windows = state.optimizer_steps, state.window_exposures
            _atomic_payload(best_metadata, {"contract": contract, "epoch": best_epoch,
                "optimizer_steps": state.optimizer_steps, "score": best_score,
                "window_exposures": state.window_exposures,
                "checkpoint_sha256": _file_digest(best)})
        if score >= patience_score + 0.002:
            patience_score, patience_epoch = score, state.epoch + 1
        state = RunState(state.epoch + 1, 0, state.optimizer_steps, state.exposures,
                         state.window_exposures, state.exposure_records)
        save_checkpoint(latest, loaded.model, optimizer, scheduler, state, contract)
        if spec.max_updates is not None and state.optimizer_steps >= spec.max_updates:
            break
        if _early_stop(state.epoch, patience_epoch, spec.patience):
            break
    final = output / "final.safetensors"
    _save_weights(final, loaded.model)
    return _finish(output, contract, {"name": spec.name, "status": "completed", "updates": best_updates,
            "run_total_updates": state.optimizer_steps,
            "run_total_window_exposures": state.window_exposures,
            "completed_epoch_updates": int(reports[-1]["optimizer_steps"]),
            "completed_epoch_window_exposures": int(reports[-1]["window_exposures"]),
            "epochs_completed": len(reports), "best_epoch": best_epoch,
            "best_score": best_score, "weights": str(best), "final_weights": str(final), "reports": reports,
            "exposure_records_path": str(_write_exposure_records(output, state)),
            "seconds": time.perf_counter() - started, "window_exposures": best_windows,
            "trainable_parameters": sum(p.numel() for p in loaded.model.parameters())})
