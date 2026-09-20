"""Bounded AST training on development data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from graph_features import atomic_json, digest
from graph_scoring import LABEL_COLUMNS, PREDICTION_COLUMNS, official_metrics
from v2_inference import load_audio as _load_audio

SEED = 20260913
RESERVE_SECONDS = 20 * 60
CONFIGS = {
    "AST_FROZEN": {"mode": "frozen", "epochs": 8, "batch_size": 16,
                   "accumulation": 1, "head_lr": 3e-4, "backbone_lr": 0.0},
    "AST_LAST_BLOCK": {"mode": "last_block", "epochs": 6, "batch_size": 8,
                       "accumulation": 2, "head_lr": 1e-4, "backbone_lr": 1e-5},
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _json_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def read_development_manifest(path: Path, data_root: Path) -> list[dict[str, str]]:
    """Reject locked-test inputs before any audio file is opened."""
    resolved = path.resolve()
    if path.name != "manifest.csv" or path.parent.name != "development" or "locked_test" in resolved.parts:
        raise ValueError("training accepts development manifest only")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["ID"] for row in rows}) != len(rows):
        raise ValueError("development manifest must contain unique rows")
    for row in rows:
        role = row.get("role", row.get("split", ""))
        if role not in {"dev_train", "dev_select"}:
            raise ValueError("training accepts development roles only")
        audio = (data_root / row["path"]).resolve()
        if not audio.is_relative_to(data_root.resolve()):
            raise ValueError("manifest audio path escapes data root")
    return rows


def _train_window(audio: np.ndarray, identifier: str, epoch: int) -> np.ndarray:
    rng = random.Random(f"{SEED}:{epoch}:{identifier}")
    if len(audio) > 160000:
        start = rng.randrange(len(audio) - 160000 + 1)
        audio = audio[start:start + 160000]
    gain = 10 ** (rng.uniform(-6.0, 6.0) / 20)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


class FrozenEmbeddingCache:
    """Content-address frozen pooler values; provenance drift creates a new key."""

    def __init__(self, directory: Path, model_sha: str, processor_signature: str):
        self.directory = directory
        self.signature = _json_hash({"model_sha256": model_sha,
                                     "processor": processor_signature,
                                     "augmentation": "seed20260913-fixed-crop-gain-v1"})
        self.misses = 0

    def path_for(self, audio_path: Path, identifier: str) -> Path:
        key = hashlib.sha256(f"{digest(audio_path)}:{identifier}:{self.signature}".encode()).hexdigest()
        return self.directory / f"{key}.pt"

    def get(self, audio_path: Path, identifier: str) -> torch.Tensor | None:
        path = self.path_for(audio_path, identifier)
        if not path.exists():
            self.misses += 1
            if self.misses % 25 == 0:
                print(f"FROZEN_CACHE {self.misses} embeddings", flush=True)
            return None
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["signature"] != self.signature or record["audio_sha256"] != digest(audio_path):
            raise ValueError("frozen embedding cache provenance mismatch")
        return record["embedding"]

    def put(self, audio_path: Path, identifier: str, embedding: torch.Tensor) -> None:
        path = self.path_for(audio_path, identifier)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({"signature": self.signature, "audio_sha256": digest(audio_path),
                    "embedding": embedding.detach().cpu()}, temporary)
        temporary.replace(path)


def _targets(rows: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    values = [[float(row[column]) if row.get(column, "") != "" else math.nan
               for column in LABEL_COLUMNS] for row in rows]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _proxy(report: dict) -> float:
    weights = {"file_eer": .45, "voice_eer": .18, "music_eer": .27,
               "voice_presence_auc": .05, "music_presence_auc": .05}
    available = {name: value for name, value in report["metrics"].items() if value is not None}
    total = sum(weights[name] for name in available)
    if not total:
        return float("-inf")
    return sum(weights[name] * ((1 - value) if name.endswith("eer") else value)
               for name, value in available.items()) / total


def _trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                             epoch: int, config_hash: str, reports: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"epoch": epoch, "config_hash": config_hash, "model": _trainable_state(model),
                "optimizer": optimizer.state_dict(), "rng": _rng_state(), "reports": reports}, temporary)
    temporary.replace(path)


def load_training_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                             config_hash: str) -> tuple[int, list[dict]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["config_hash"] != config_hash:
        raise ValueError("checkpoint configuration mismatch")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if unexpected or trainable.intersection(missing):
        raise ValueError("checkpoint trainable parameters do not match model")
    optimizer.load_state_dict(state["optimizer"])
    _restore_rng(state["rng"])
    return int(state["epoch"]), list(state["reports"])


def _optimizer(model: torch.nn.Module, config: dict) -> torch.optim.AdamW:
    head_ids = {id(parameter) for parameter in model.heads.parameters()}
    heads = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in head_ids]
    backbone = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in head_ids]
    groups = [{"params": heads, "lr": config["head_lr"]}]
    if backbone:
        groups.append({"params": backbone, "lr": config["backbone_lr"]})
    return torch.optim.AdamW(groups, weight_decay=.01)


def extend_checkpoint(source: Path, destination: Path, old_config: dict, new_config: dict) -> None:
    """Preserve optimizer/RNG state while explicitly extending only the training budget."""
    allowed = {"epochs", "patience"}
    if {k: v for k, v in old_config.items() if k not in allowed} != {
        k: v for k, v in new_config.items() if k not in allowed
    }:
        raise ValueError("extension may change only epochs and patience")
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError("extension requires a new checkpoint destination")
    state = torch.load(source, map_location="cpu", weights_only=False)
    if state["config_hash"] != _json_hash(old_config):
        raise ValueError("source checkpoint configuration mismatch")
    if new_config["epochs"] <= max(old_config["epochs"], state["epoch"]):
        raise ValueError("extension must increase the epoch budget")
    state["config_hash"] = _json_hash(new_config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)


def _batch_inputs(rows: list[dict[str, str]], root: Path, processor, epoch: int, device: torch.device):
    from ast_multitask import prepare_inputs
    windows = [_train_window(_load_audio(root / row["path"]), row["ID"], epoch) for row in rows]
    return prepare_inputs(processor, tuple(windows)).to(device), _targets(rows, device)


def _frozen_logits(model, processor, rows: list[dict[str, str]], root: Path,
                   cache: FrozenEmbeddingCache, device: torch.device) -> torch.Tensor:
    from ast_multitask import prepare_inputs
    embeddings = []
    for row in rows:
        path = root / row["path"]
        embedding = cache.get(path, row["ID"])
        if embedding is None:
            window = _train_window(_load_audio(path), row["ID"], 0)
            with torch.no_grad():
                embedding = model.encode(prepare_inputs(processor, (window,)).to(device))[0].cpu()
            cache.put(path, row["ID"], embedding)
        embeddings.append(embedding)
    pooled = torch.stack(embeddings).to(device)
    return torch.cat([head(pooled) for head in model.heads], dim=-1)


@torch.no_grad()
def predict_rows(model, processor, rows: list[dict[str, str]], root: Path,
                 device: torch.device,
                 deadline: float | None = None,
                 cache: FrozenEmbeddingCache | None = None) -> list[dict[str, str | float]]:
    from ast_multitask import aggregate_window_logits, prepare_inputs, split_audio
    model.eval()
    predictions = []
    for row in rows:
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("prediction deadline reached before completing a file")
        path = root / row["path"]
        embeddings = cache.get(path, row["ID"] + "#eval") if cache else None
        if embeddings is None:
            windows = split_audio(_load_audio(path))
            if cache:
                embeddings = model.encode(prepare_inputs(processor, windows).to(device)).cpu()
                cache.put(path, row["ID"] + "#eval", embeddings)
        logits = (torch.cat([head(embeddings.to(device)) for head in model.heads], dim=-1)
                  if cache else model(prepare_inputs(processor, windows).to(device)))
        probabilities = aggregate_window_logits(logits).float().cpu().tolist()
        predictions.append({"ID": row["ID"], **dict(zip(PREDICTION_COLUMNS, probabilities, strict=True))})
    return predictions


def train_candidate(name: str, model, processor, rows: list[dict[str, str]], data_root: Path,
                    output: Path, deadline: float, manifest_hash: str,
                    asset_model_sha: str) -> tuple[object, dict]:
    from ast_multitask import compute_pos_weights, export_checkpoint, masked_multitask_loss
    config = {**CONFIGS[name], "seed": SEED, "manifest_sha256": manifest_hash,
              "asset_model_sha256": asset_model_sha,
              "weight_decay": .01, "gradient_clip": 1.0}
    config_hash = _json_hash(config)
    if time.time() >= deadline - RESERVE_SECONDS:
        raise TimeoutError(f"not enough time to start {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = _optimizer(model, config)
    train = [row for row in rows if row.get("role", row.get("split")) == "dev_train"]
    select = [row for row in rows if row.get("role", row.get("split")) == "dev_select"]
    pos_weights = compute_pos_weights(_targets(train, device))
    processor_signature = _json_hash({name: getattr(processor, name) for name in
                                      ("sampling_rate", "num_mel_bins", "max_length", "mean", "std")})
    frozen_cache = FrozenEmbeddingCache(output / name / "pooler_cache", asset_model_sha,
                                        processor_signature) if name == "AST_FROZEN" else None
    checkpoint = output / name / "last.pt"
    start_epoch, reports = (0, [])
    if checkpoint.exists():
        start_epoch, reports = load_training_checkpoint(checkpoint, model, optimizer, config_hash)
    best_proxy = max((report["proxy"] for report in reports), default=float("-inf"))
    best_epoch = next((r["epoch"] for r in reports if r["proxy"] == best_proxy), 0)
    best_state = None
    if start_epoch and (output / name / "best.pt").exists():
        best_state = torch.load(output / name / "best.pt", map_location="cpu", weights_only=True)
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        if config.get("patience", 0) and epoch - 1 - best_epoch >= config["patience"]:
            break
        if time.time() >= deadline - RESERVE_SECONDS:
            break
        started = time.perf_counter()
        model.train()
        before_epoch = {key: value.clone() for key, value in _trainable_state(model).items()}
        optimizer.zero_grad(set_to_none=True)
        shuffled = list(train)
        random.Random(SEED + epoch).shuffle(shuffled)
        losses = []
        gradient_norms = []
        head_gradient_norms = []
        backbone_gradient_norms = []
        for batch_index in range(0, len(shuffled), config["batch_size"]):
            if time.time() >= deadline - RESERVE_SECONDS:
                raise TimeoutError(f"deadline reached during {name}; incomplete epoch was discarded")
            batch = shuffled[batch_index:batch_index + config["batch_size"]]
            targets = _targets(batch, device)
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
            with amp:
                logits = (_frozen_logits(model, processor, batch, data_root, frozen_cache, device)
                          if frozen_cache else model(_batch_inputs(batch, data_root, processor, 0, device)[0]))
                loss = masked_multitask_loss(logits, targets, pos_weights) / config["accumulation"]
            loss.backward()
            losses.append(float(loss.detach()) * config["accumulation"])
            step = (batch_index // config["batch_size"] + 1) % config["accumulation"] == 0
            last = batch_index + config["batch_size"] >= len(shuffled)
            if step or last:
                heads = [p for p in model.heads.parameters() if p.requires_grad]
                backbone = [p for parameter_name, p in model.named_parameters()
                            if p.requires_grad and not parameter_name.startswith("heads.")]
                head_gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(heads, float("inf"))))
                if backbone:
                    backbone_gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(backbone, float("inf"))))
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                gradient_norms.append(float(norm))
                if not math.isfinite(gradient_norms[-1]):
                    raise FloatingPointError(f"non-finite gradient in {name}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if max(head_gradient_norms, default=0.0) <= 0 or (
            name == "AST_LAST_BLOCK" and max(backbone_gradient_norms, default=0.0) <= 0
        ):
            raise FloatingPointError(f"intended trainable path received no gradient in {name}")
        predictions = predict_rows(model, processor, select, data_root, device,
                                   deadline=deadline - RESERVE_SECONDS, cache=frozen_cache)
        metric_report = official_metrics(select, predictions)
        proxy = _proxy(metric_report)
        report = {"epoch": epoch, "mean_loss": float(np.mean(losses)), "proxy": proxy,
                  "metric_report": metric_report, "seconds": time.perf_counter() - started,
                  "max_gradient_norm": max(gradient_norms, default=0.0),
                  "max_head_gradient_norm": max(head_gradient_norms, default=0.0),
                  "max_backbone_gradient_norm": max(backbone_gradient_norms, default=0.0),
                  "trainable_absolute_update": sum(
                      float((value - before_epoch[key]).abs().sum())
                      for key, value in _trainable_state(model).items()),
                  "head_absolute_update": sum(
                      float((value - before_epoch[key]).abs().sum())
                      for key, value in _trainable_state(model).items() if key.startswith("heads.")),
                  "backbone_absolute_update": sum(
                      float((value - before_epoch[key]).abs().sum())
                      for key, value in _trainable_state(model).items() if not key.startswith("heads."))}
        reports.append(report)
        if proxy > best_proxy:
            best_proxy, best_state = proxy, _trainable_state(model)
            best_epoch = epoch
            best_path = output / name / "best.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = best_path.with_suffix(".tmp")
            torch.save(best_state, temporary)
            temporary.replace(best_path)
        save_training_checkpoint(checkpoint, model, optimizer, epoch, config_hash, reports)
        atomic_json(output / name / f"epoch_{epoch:03d}.json", report)
        print(f"TRAIN_EPOCH {name} {epoch}/{config['epochs']} "
              f"loss={report['mean_loss']:.6f} proxy={proxy:.6f} "
              f"head_grad={report['max_head_gradient_norm']:.6f} "
              f"backbone_grad={report['max_backbone_gradient_norm']:.6f}", flush=True)
    if best_state is None:
        raise TimeoutError(f"{name} completed no epoch")
    model.load_state_dict(best_state, strict=False)
    exported = output / name / "best"
    export_checkpoint(model, processor, exported)
    predictions = predict_rows(model, processor, select, data_root, device,
                               deadline=deadline - RESERVE_SECONDS, cache=frozen_cache)
    changed = max(report.get("trainable_absolute_update", 0.0) for report in reports)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    result = {"candidate": name, "checkpoint": str(exported), "config": config,
              "config_hash": config_hash, "best_proxy": best_proxy, "epochs_completed": len(reports),
              "best_epoch": best_epoch,
              "predictions": predictions, "reports": reports, "trainable_names": trainable_names,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "trainable_absolute_update": changed,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0}
    prediction_path = output / name / "dev_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ID", *PREDICTION_COLUMNS])
        writer.writeheader()
        writer.writerows(predictions)
    atomic_json(output / name / "report.json", result)
    return model, result


def train_candidates(ast_assets: Path, manifest_path: Path, data_root: Path,
                     output: Path, deadline: float) -> dict:
    from ast_multitask import ASTMultiTask, load_processor
    seed_everything()
    rows = read_development_manifest(manifest_path, data_root)
    processor = load_processor(ast_assets)
    asset_model_sha = digest(ast_assets / "model.safetensors")
    results = {}
    frozen, results["AST_FROZEN"] = train_candidate(
        "AST_FROZEN", ASTMultiTask.from_pretrained(ast_assets, "frozen"), processor,
        rows, data_root, output, deadline, digest(manifest_path), asset_model_sha)
    if time.time() < deadline - RESERVE_SECONDS:
        last = ASTMultiTask.from_pretrained(ast_assets, "last_block")
        last.heads.load_state_dict(frozen.heads.state_dict())
        _, results["AST_LAST_BLOCK"] = train_candidate(
            "AST_LAST_BLOCK", last, processor, rows, data_root, output, deadline,
            digest(manifest_path), asset_model_sha)
    else:
        results["AST_LAST_BLOCK"] = {"candidate": "AST_LAST_BLOCK", "status": "skipped_deadline"}
    report = {"candidates": results, "manifest_sha256": digest(manifest_path),
              "finished_unix": time.time()}
    atomic_json(output / "training_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    train_candidates(args.assets, args.manifest, args.data_root, args.output, args.deadline)


if __name__ == "__main__":
    main()
