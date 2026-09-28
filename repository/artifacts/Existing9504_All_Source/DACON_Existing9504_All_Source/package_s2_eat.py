"""Package the sealed S2 EAT epoch-four checkpoint for offline DACON inference."""
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "safetensors"]
# ///
# Run: .venv/bin/python package_s2_eat.py

import hashlib
import json
from pathlib import Path
import shutil
from typing import Final
import zipfile

import torch
from safetensors.torch import save_file

from package_submission import _model_members

CHECKPOINT_SHA: Final = "81bf4daac2eb11857d9562a0ff01a0ddbfadd92237210a2c7cbcf053880fc6b6"
WORKER_SHA: Final = "9c076487890c6e438a9deac80a83be0e04ce4b96d512e6219612efd0cbfbd727"
IDENTITY: Final = "0b27a1d2ef741e6a42505cc49ddd34d1cd1b8da5e029c37909ef3e5250733565"
ENTRYPOINT: Final = '''import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(ROOT/'model/runtime'))
from s2_eat_submission import main
if __name__ == '__main__':
    main(ROOT)
'''
REQUIREMENTS: Final = '''--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.7.1+cu128
torchaudio==2.7.1+cu128
torchvision==0.22.1+cu128
timm==1.0.15
'''


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package(root: Path) -> Path:
    """Export weights losslessly; preserve exact worker forward code and baseline assets."""
    run = root / "results/fullft8_h100x1_20260916/downloaded"
    checkpoint = run / "h100x1_fullft8_run/07_S2_EAT_fullft/best.pt"
    worker = run / "batch_worker.py"
    if digest(checkpoint) != CHECKPOINT_SHA or digest(worker) != WORKER_SHA:
        message = "Source checkpoint or training worker differs from the sealed S2 run"
        raise ValueError(message)
    output = root / "submissions/S2_EAT"
    stage = output / "staged"
    eat = stage / "model/assets/eat"
    trained = stage / "model/trained"
    runtime = stage / "model/runtime"
    for folder in [eat, trained, runtime]:
        folder.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["epoch"] != 3 or state["identity"] != IDENTITY:
        message = "Expected the S2 tune-selected epoch-four checkpoint"
        raise ValueError(message)
    encoder = {key.removeprefix("encoder."): value for key, value in state["model"].items()
               if key.startswith("encoder.")}
    head = {key.removeprefix("head."): value for key, value in state["model"].items()
            if key.startswith("head.")}
    if len(encoder) + len(head) != len(state["model"]):
        message = "Checkpoint contains unexported model parameters"
        raise ValueError(message)
    save_file(encoder, eat / "model.safetensors", metadata={"format": "pt"})
    torch.save(head, trained / "head.pt")
    source_assets = root / "_workspace/fullft7_upload/models/eat"
    sealed_assets = json.loads((source_assets.parent.parent / "model_manifest.json").read_text())["sha256"]
    for source in source_assets.iterdir():
        if source.suffix in (".py", ".json", ".md"):
            if digest(source) != sealed_assets["eat/" + source.name]:
                message = f"EAT source differs from training: {source.name}"
                raise ValueError(message)
            shutil.copyfile(source, eat / source.name)
    source = worker.read_text()
    model_code = source[source.index("class ResidualTemporalBlock"):source.index("def cache_frozen_features")]
    imports = "from pathlib import Path\nfrom typing import Literal\nimport torch, torchaudio\nfrom torch import nn\nfrom torch.nn import functional as F\nfrom transformers import AutoModel\n\n"
    (runtime / "fullft_model.py").write_text(imports + model_code)
    for name in ["s2_eat_submission.py", "graph_scoring.py"]:
        shutil.copyfile(root / name, runtime / name)
    (stage / "script.py").write_text(ENTRYPOINT)
    (stage / "requirements.txt").write_text(REQUIREMENTS)
    with zipfile.ZipFile(root / "baseline_original/baseline_submit.zip") as original:
        for member in _model_members(original):
            if member.is_dir():
                continue
            target = stage / "model/assets/baseline" / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with original.open(member) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
        (stage / "model/assets/baseline/script.py").write_bytes(original.read("script.py"))
    for name, expected in sealed_assets.items():
        if name.startswith("htdemucs/") and digest(stage / "model/assets/baseline/model" / name) != expected:
            message = "Packaged HTDemucs differs from the S2 training separator"
            raise ValueError(message)
    choice = {"trial": "07_S2_EAT_fullft", "seed": 41, "selected_epoch": 4,
              "checkpoint_sha256": CHECKPOINT_SHA, "worker_sha256": WORKER_SHA,
              "identity": IDENTITY, "separator": "original frozen HTDemucs",
              "seconds": 5, "evaluation_windows": 8, "official_score": None,
              "historical_predictions_included": False, "retrained": False}
    files = sorted(p for p in stage.rglob("*") if p.is_file() and p.name != "manifest.json")
    hashes = {str(p.relative_to(stage)): digest(p) for p in files}
    (stage / "model/manifest.json").write_text(json.dumps({"choice": choice, "sha256": hashes}, indent=2))
    target = output / "submit.zip"
    temporary = output / "submit.partial"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in [*files, stage / "model/manifest.json"]:
            archive.write(path, str(path.relative_to(stage)))
    with zipfile.ZipFile(temporary) as archive:
        expanded = sum(member.file_size for member in archive.infolist())
        if temporary.stat().st_size > 10_000_000_000 or expanded > 32_000_000_000:
            message = "Submission exceeds official size limits"
            raise ValueError(message)
        if archive.testzip() is not None:
            message = "Submission ZIP CRC check failed"
            raise ValueError(message)
    temporary.replace(target)
    report = {"archive": str(target.relative_to(root)), "sha256": digest(target),
              "bytes": target.stat().st_size, "expanded_bytes": expanded,
              "choice": choice, "crc_verified": True, "gpu_end_to_end_verified": False}
    (output / "package_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return target


if __name__ == "__main__":
    package(Path(__file__).resolve().parent)
