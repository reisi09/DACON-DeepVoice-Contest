"""File-level MIL music fitting and disjoint logistic fusion; frozen encoders."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from graph_scoring import official_metrics
from v6_core import COLUMNS, VERSION, Heads, fusion_features
from v6_data import assert_disjoint, sha256, write_json
from v6_runtime import load_cache


def mil_logits(logits: torch.Tensor, bags: torch.Tensor, count: int) -> torch.Tensor:
    return torch.full((count,), -torch.inf, dtype=logits.dtype, device=logits.device).scatter_reduce(
        0, bags, logits, reduce="amax", include_self=True)


def fit_music(features: list[np.ndarray], labels: np.ndarray, epochs: int,
              seed: int = 41, deadline: float = float("inf")) -> tuple[Heads, list[float]]:
    if epochs < 1 or len(features) != len(labels) or set(labels) != {0, 1}:
        raise ValueError("Music training requires both classes, aligned bags and positive epochs")
    torch.manual_seed(seed)
    windows = np.concatenate(features).astype("float32")
    mean, scale = windows.mean(axis=0), windows.std(axis=0).clip(1e-4)
    x = torch.from_numpy((windows - mean) / scale)
    indices = torch.repeat_interleave(torch.arange(len(features)),
                                      torch.tensor([len(f) for f in features]))
    y = torch.from_numpy(labels.astype("float32"))
    model = torch.nn.Linear(x.shape[1], 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    class_weights = torch.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (1 - y).sum()))
    trace = []
    for epoch in range(epochs):
        if time.monotonic() >= deadline:
            raise TimeoutError("Music fitting reached the run deadline; no final artifact exported")
        optimizer.zero_grad(set_to_none=True)
        logits = mil_logits(model(x).flatten(), indices, len(features))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, weight=class_weights)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite music training loss")
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach()))
    return Heads(mean, scale, model.weight.detach().numpy().ravel().copy(),
                 np.array(model.bias.detach().item()), np.zeros(5), np.array(0.)), trace


def feature_record(row: dict, cache: Path, cache_id: str) -> dict[str, np.ndarray]:
    if sha256(Path(row["path"])) != row["audio_sha256"]:
        raise ValueError(f"Audio changed since split audit: {row['ID']}")
    identity = row["audio_sha256"] + ":" + cache_id
    path = cache / (hashlib.sha256(identity.encode()).hexdigest() + ".npz")
    result = load_cache(path, identity)
    if result is None:
        raise ValueError(f"Missing or stale feature cache: {row['ID']}")
    return result


def train(split: dict, cache: Path, cache_id: str, asset_id: str, output: Path,
          epochs: int, deadline: float) -> dict:
    assert_disjoint(split["expert"], split["fusion"])
    assert_disjoint(split["expert"] + split["fusion"], split["dev"])
    records = {r["ID"]: feature_record(r, cache, cache_id)
               for partition in ("expert", "fusion", "dev") for r in split[partition]}
    eligible = [r for r in split["expert"] if r["MUSIC_PRESENT"] == "1" and r["MUSIC_FAKE"] in ("0", "1")]
    features = [records[r["ID"]]["music"][records[r["ID"]]["music_audible"] > 0] for r in eligible]
    if any(not len(x) for x in features):
        raise ValueError("A labeled music file has no audible music windows")
    head, losses = fit_music(features, np.array([int(r["MUSIC_FAKE"]) for r in eligible]),
                             epochs, split["seed"], deadline)
    fusion_rows = [r for r in split["fusion"] if r["FILE_FAKE"] in ("0", "1")]
    matrix = []
    for row in fusion_rows:
        record = records[row["ID"]]
        music = head.music(record["music"], record["music_audible"])
        matrix.append(fusion_features(record["presence"], record["voice"], music, record["raw"]))
    if time.monotonic() >= deadline:
        raise TimeoutError("Run deadline reached before fusion fitting")
    fusion = LogisticRegression(C=1., class_weight="balanced", max_iter=2000, random_state=split["seed"])
    fusion.fit(matrix, [int(r["FILE_FAKE"]) for r in fusion_rows])
    if fusion.n_iter_.max() >= fusion.max_iter:
        raise ValueError("Fusion did not converge")
    final = Heads(head.mean, head.scale, head.music_weight, head.music_bias,
                  fusion.coef_[0], np.array(fusion.intercept_[0]))
    predictions = [dict(ID=r["ID"], **dict(zip(COLUMNS, final.predict(records[r["ID"]]).tolist())))
                   for r in split["dev"]]
    report = official_metrics(split["dev"], predictions)
    output.mkdir(parents=True, exist_ok=True)
    final.save(output / "heads.npz")
    write_json(output / "dev_predictions.json", predictions)
    write_json(output / "dev_metrics.json", report)
    write_json(output / "losses.json", losses)
    metadata = dict(status="complete", purpose=split["purpose"], version=VERSION,
                    epochs_requested=epochs, epochs_completed=epochs,
                    music_files=len(eligible), fusion_files=len(fusion_rows), dev_files=len(split["dev"]),
                    asset_id=asset_id, cache_id=cache_id, heads_sha256=sha256(output / "heads.npz"),
                    split_sha256=hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
                    seed=split["seed"], voice="frozen pretrained XLS-R+AASIST LA; no new voice fine-tuning",
                    dev_status=split["dev_status"], initial_loss=losses[0], final_loss=losses[-1])
    write_json(output / "training.json", metadata)
    return metadata


def trained_heads(directory: Path, asset_id: str) -> Heads:
    metadata = json.loads((directory / "training.json").read_text())
    if (metadata.get("status") != "complete" or metadata.get("purpose") != "final"
            or metadata.get("version") != VERSION
            or metadata.get("asset_id") != asset_id
            or metadata.get("epochs_completed", 0) < 1
            or metadata.get("epochs_completed") != metadata.get("epochs_requested")
            or metadata.get("music_files", 0) < 2 or metadata.get("fusion_files", 0) < 2
            or metadata.get("heads_sha256") != sha256(directory / "heads.npz")):
        raise ValueError("Incomplete, mismatched or modified training artifact")
    return Heads.load(directory / "heads.npz")
