from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import ASTConfig, ASTFeatureExtractor, ASTModel

import ast_multitask as ast
import package_hybrid_submission as hybrid
import package_submission
import v2_inference


def test_fresh_hybrid_package_routes_mock_g0_and_real_ast_per_file(tmp_path: Path, monkeypatch) -> None:
    # Given: a real tiny AST checkpoint and a G0 builder replacement with known per-ID values.
    torch.manual_seed(236749)
    model = ast.ASTMultiTask(
        ASTModel(ASTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
                            intermediate_size=32, max_length=1024, num_mel_bins=128)),
        "frozen",
    ).eval()
    checkpoint = ast.export_checkpoint(model, ASTFeatureExtractor(), tmp_path / "checkpoint")

    def fake_g0_builder(_original: Path, destination: Path, _selection: object) -> Path:
        script = '''from pathlib import Path
ROOT = Path(__file__).resolve().parent
def main():
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    (output / "submission.csv").write_text(
        "ID,FILE_FAKE_PROB,VOICE_FAKE_PROB,MUSIC_FAKE_PROB,VOICE_PRESENT_PROB,MUSIC_PRESENT_PROB\\n"
        "b,0.11,0.22,0.33,0.44,0.55\\n"
        "a,0.66,0.77,0.88,0.99,0.12\\n", encoding="utf-8")
'''
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("script.py", script)
        return destination

    monkeypatch.setattr(package_submission, "build_submission", fake_g0_builder)
    archive = hybrid.build_hybrid_submission(Path("unused.zip"), checkpoint, tmp_path / "hybrid.zip")
    fresh = tmp_path / "fresh"
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read("model/ast/LICENSE.txt") == Path("AST_LICENSE.txt").read_bytes()
        bundle.extractall(fresh)
    test_dir = fresh / "data" / "test"
    test_dir.mkdir(parents=True)
    time = np.arange(8_000, dtype=np.float32) / 16_000
    sf.write(test_dir / "a.wav", 0.2 * np.sin(2 * np.pi * 220 * time), 16_000)
    sf.write(test_dir / "b.wav", 0.2 * np.sin(2 * np.pi * 440 * time), 16_000)
    (fresh / "data" / "sample_submission.csv").write_text(
        "ID,FILE_FAKE_PROB,VOICE_FAKE_PROB,MUSIC_FAKE_PROB,VOICE_PRESENT_PROB,MUSIC_PRESENT_PROB\n"
        "b,0,0,0,0,0\na,0,0,0,0,0\n", encoding="utf-8")
    blocker = tmp_path / "offline"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(
        "import socket\ndef blocked(*args, **kwargs): raise RuntimeError('network forbidden')\n"
        "socket.create_connection = blocked\nsocket.socket.connect = blocked\n", encoding="utf-8")
    reference = tmp_path / "reference"
    shutil.copytree(fresh / "model", reference / "model")
    shutil.copytree(fresh / "data", reference / "data")
    v2_inference.generate_submission(reference)
    with (reference / "output" / "submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        ast_rows = {row["ID"]: row for row in csv.DictReader(stream)}

    # When: a fresh, network-blocked archive invokes both runtimes and routes their outputs.
    subprocess.run([sys.executable, str(fresh / "script.py")], cwd=fresh,
                   env=os.environ | {"PYTHONPATH": str(blocker)}, check=True, timeout=120)
    with (fresh / "output" / "submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        first = list(csv.DictReader(stream))

    # Then: sample order holds, G0 owns file/voice fake, AST owns the other three, per-file isolation holds.
    assert [row["ID"] for row in first] == ["b", "a"]
    assert (first[0]["FILE_FAKE_PROB"], first[0]["VOICE_FAKE_PROB"]) == ("0.11", "0.22")
    assert (first[1]["FILE_FAKE_PROB"], first[1]["VOICE_FAKE_PROB"]) == ("0.66", "0.77")
    for row in first:
        expected = ast_rows[row["ID"]]
        assert [row[column] for column in hybrid.AST_COLUMNS] == [expected[column] for column in hybrid.AST_COLUMNS]
    before_a = next(row for row in first if row["ID"] == "a")
    sf.write(test_dir / "b.wav", np.zeros(8_000, dtype=np.float32), 16_000)
    subprocess.run([sys.executable, str(fresh / "script.py")], cwd=fresh,
                   env=os.environ | {"PYTHONPATH": str(blocker)}, check=True, timeout=120)
    with (fresh / "output" / "submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert next(row for row in csv.DictReader(stream) if row["ID"] == "a") == before_a
