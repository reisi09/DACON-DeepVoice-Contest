"""Conditional voice-residual eligibility and fitting."""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from redesign_fit import Records, Rows, masked_bce
from redesign_models import VoiceResidual
from redesign_oof import connected_groups, fold_groups, internal_earlystop, row_groups


@dataclass(frozen=True, slots=True)
class VoiceFit:
    status: str
    reason: str
    checkpoint: Path | None
    eligible_ids: tuple[str, ...]


def eligible_voice_rows(rows: Rows) -> Rows:
    eligible = []
    for row in rows:
        intervals = json.loads(row.get("event_intervals") or "[]")
        has_truth = any(str(item.get("component", "")) in ("voice", "vocal")
                        and item.get("label") in (0, 1, "0", "1") for item in intervals)
        if (row.get("VOICE_PRESENT") == "1" and row.get("VOICE_FAKE") in ("0", "1")
                and row.get("kind") in ("speech", "singing") and has_truth):
            eligible.append(row)
    return eligible


def voice_gate(rows: Rows) -> tuple[bool, str]:
    eligible = eligible_voice_rows(rows)
    singing = {row["VOICE_FAKE"] for row in eligible if row.get("kind") == "singing"}
    telephone = {row["VOICE_FAKE"] for row in eligible if row.get("domain") == "telephone"}
    if singing != {"0", "1"} or telephone != {"0", "1"}:
        return False, "known real/fake singing and telephone labels are both required"
    return True, "eligible"


def fit_voice(
    rows: Rows,
    records: Records,
    output: Path,
    *,
    seed: int,
    epochs: int,
    patience: int,
    device: str,
    deadline: float | None,
    require_gate: bool = True,
) -> VoiceFit:
    """Fit only when the predeclared singing and telephone label gate is complete."""
    allowed, reason = voice_gate(rows)
    eligible = eligible_voice_rows(rows)
    if require_gate and not allowed:
        return VoiceFit("provisional_skip", reason, None, tuple(row["ID"] for row in eligible))
    fitting, validation = internal_earlystop(eligible, seed)
    torch.manual_seed(seed)
    model = VoiceResidual().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    best_loss, stale, best_state = float("inf"), 0, None
    for _epoch in range(epochs):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Absolute voice training deadline reached")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        bag, targets, known = _voice_batch(model, fitting, records, device)
        loss = 0.18 * masked_bce(bag, targets, known)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            bag, targets, known = _voice_batch(model, validation, records, device)
            value = float(0.18 * masked_bce(bag, targets, known))
        if value < best_loss - 1e-4:
            best_loss, stale = value, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        return VoiceFit("provisional_skip", "internal validation had no eligible voice labels", None,
                        tuple(row["ID"] for row in eligible))
    model.load_state_dict(best_state)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "voice.pt"
    temporary = checkpoint.with_suffix(".partial")
    source_sha = _sha256(Path(__file__))
    torch.save({"state_dict": model.state_dict(), "seed": seed, "epochs": epochs,
                "patience": patience, "source_sha256": source_sha}, temporary)
    temporary.replace(checkpoint)
    loaded = load_voice(checkpoint, device)
    with torch.inference_mode():
        before = _voice_batch(model.eval(), validation, records, device)[0]
        after = _voice_batch(loaded, validation, records, device)[0]
    if not torch.allclose(before, after, atol=1e-7, rtol=0):
        raise RuntimeError("Saved voice checkpoint changed validation predictions")
    return VoiceFit("complete", "eligible", checkpoint, tuple(row["ID"] for row in eligible))


def load_voice(path: Path, device: str = "cpu") -> VoiceResidual:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = VoiceResidual().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def predict_voice(
    checkpoint: Path, rows: Rows, records: Records, *, device: str = "cpu"
) -> dict[str, dict[str, float | list[float]]]:
    model = load_voice(checkpoint, device)
    result: dict[str, dict[str, float | list[float]]] = {}
    with torch.inference_mode():
        for row in rows:
            record = records[row["ID"]]
            count = len(record["voice_logits"])
            embeddings = torch.as_tensor(record["voice_embedding"], device=device)[None]
            logits = torch.as_tensor(record["voice_logits"], device=device)[None]
            valid = torch.as_tensor(np.asarray(record["valid"], dtype=bool), device=device)[None]
            bag, windows = model(embeddings, logits, valid)
            probabilities = windows.sigmoid()[0]
            probabilities = probabilities.masked_fill(~valid[0], 0)
            result[row["ID"]] = {"voice": float(bag.sigmoid()[0]),
                                  "windows": probabilities.cpu().tolist(), "count": count}
    return result


def build_voice_oof(
    rows: Rows, records: Records, output: Path, *, seed: int, epochs: int, patience: int,
    device: str, deadline: float | None, folds: int, training_rows: Rows | None = None,
) -> tuple[dict[str, dict[str, float | list[float]]], list[dict[str, str | int | list[str]]], dict[str, list[str]], dict[str, list[str]]]:
    partitions = connected_groups(rows)
    predictions: dict[str, dict[str, float | list[float]]] = {}
    receipts: list[dict[str, str | int | list[str]]] = []
    held_map: dict[str, list[str]] = {}
    trained_map: dict[str, list[str]] = {}
    for fold, held_components in enumerate(fold_groups(rows, folds, 41)):
        held = set(held_components)
        outer_valid = [row for row in rows if partitions[row["ID"]] in held]
        held_atoms = {group for row in outer_valid for group in row_groups(row)}
        pool = rows if training_rows is None else training_rows
        outer_train = [row for row in pool if held_atoms.isdisjoint(row_groups(row))]
        fit = fit_voice(outer_train, records, output / f"fold{fold}", seed=seed, epochs=epochs,
                        patience=patience, device=device, deadline=deadline, require_gate=False)
        if fit.checkpoint is None:
            raise RuntimeError("Voice OOF fold has no eligible checkpoint")
        values = predict_voice(fit.checkpoint, outer_valid, records, device=device)
        fitting, early = internal_earlystop(eligible_voice_rows(outer_train), seed)
        train_groups = sorted({group for row in fitting for group in row_groups(row)})
        early_groups = sorted({group for row in early for group in row_groups(row)})
        held_groups = sorted({group for row in outer_valid for group in row_groups(row)})
        artifact_ids = {row["ID"] for row in rows}
        for row in outer_valid:
            predictions[row["ID"]] = values[row["ID"]]
            held_map[row["ID"]] = list(row_groups(row))
            trained_map[row["ID"]] = train_groups
        receipts.append({"fold": fold, "held_out_groups": held_groups,
                         "trained_groups": train_groups, "earlystop_groups": early_groups,
                         "held_out_ids": sorted(row["ID"] for row in outer_valid),
                         "train_ids": sorted(row["ID"] for row in fitting if row["ID"] in artifact_ids),
                         "earlystop_ids": sorted(row["ID"] for row in early if row["ID"] in artifact_ids),
                         "auxiliary_train_ids": sorted(row["ID"] for row in fitting if row["ID"] not in artifact_ids),
                         "auxiliary_earlystop_ids": sorted(row["ID"] for row in early if row["ID"] not in artifact_ids),
                         "checkpoint": str(fit.checkpoint),
                         "checkpoint_sha256": _sha256(fit.checkpoint), "selected_epoch": -1})
    return predictions, receipts, held_map, trained_map


def _sha256(path: Path) -> str:
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _voice_batch(
    model: VoiceResidual, rows: Rows, records: Records, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(len(records[row["ID"]]["voice_logits"]) for row in rows)
    embeddings = torch.zeros(len(rows), length, 1280, device=device)
    logits = torch.zeros(len(rows), length, device=device)
    valid = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
    targets = torch.zeros(len(rows), device=device)
    known = torch.zeros(len(rows), dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        record = records[row["ID"]]
        count = len(record["voice_logits"])
        embeddings[index, :count] = torch.as_tensor(record["voice_embedding"], device=device)
        logits[index, :count] = torch.as_tensor(record["voice_logits"], device=device)
        valid[index, :count] = torch.as_tensor(np.asarray(record["valid"], dtype=bool), device=device)
        if row.get("VOICE_PRESENT") == "1" and row.get("VOICE_FAKE") in ("0", "1"):
            targets[index] = int(row["VOICE_FAKE"])
            known[index] = True
    bag, _ = model(embeddings, logits, valid)
    return bag, targets, known
