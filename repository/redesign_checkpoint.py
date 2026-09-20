"""Content-bound checkpoint helpers shared by training and inference."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from redesign_models import MusicResidual, StemMusicWrapper, make_control

Arm = Literal["A", "B", "C", "D"]
Rows = list[dict[str, str]]
Records = dict[str, dict[str, np.ndarray]]
Report = dict[str, str | int | float | list[str] | list[float]]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity_hash(value: dict[str, str | int | float | list[str]]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def training_identity(
    rows: Rows, validation: Rows, records: Records, config: dict[str, str | int | float | list[str]]
) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    for row in sorted(rows + validation, key=lambda item: item["ID"]):
        fields = {name: row.get(name, "") for name in (
            "ID", "group_id", "pair_id", "view", "audio_sha256", "MUSIC_FAKE",
            "MUSIC_PRESENT", "event_intervals")}
        digest.update(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode())
        for name in ("stem", "raw", "aux", "starts", "valid", "stem_audible"):
            values = np.ascontiguousarray(records[row["ID"]][name])
            digest.update(name.encode())
            digest.update(str(values.shape).encode())
            digest.update(values.dtype.str.encode())
            digest.update(values.tobytes())
    for name in ("redesign_models.py", "redesign_fit.py", "redesign_checkpoint.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def save_checkpoint(
    path: Path, model: StemMusicWrapper | MusicResidual, arm: Arm, identity: str, report: Report,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save({"arm": arm, "identity": identity, "state_dict": model.state_dict(), "report": report}, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path, *, device: str = "cpu", expected_identity: str | None = None,
) -> tuple[StemMusicWrapper | MusicResidual, Report]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if expected_identity is not None and payload["identity"] != expected_identity:
        raise ValueError("Checkpoint resume identity mismatch")
    arm: Arm = payload["arm"]
    seed = int(payload["report"]["seed"])
    control = make_control(seed)
    model: StemMusicWrapper | MusicResidual = (
        MusicResidual(control) if arm in ("B", "D") else StemMusicWrapper(control))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload["report"]
