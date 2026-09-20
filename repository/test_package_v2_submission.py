from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import ASTConfig, ASTFeatureExtractor, ASTModel

import ast_multitask as ast
import package_v2_submission as package
import v2_inference


def _checkpoint(directory: Path) -> Path:
    """Create a genuinely executable small AST checkpoint for package integration."""
    torch.manual_seed(236749)
    model = ast.ASTMultiTask(
        ASTModel(
            ASTConfig(
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=32,
                max_length=1024,
                num_mel_bins=128,
            )
        ),
        "frozen",
    ).eval()
    return ast.export_checkpoint(model, ASTFeatureExtractor(), directory)


def _write_input(root: Path) -> tuple[Path, Path]:
    """Create two different evaluator-format WAVs and the exact requested ID order."""
    test_dir = root / "data" / "test"
    test_dir.mkdir(parents=True)
    sample = root / "data" / "sample_submission.csv"
    sample.write_text(
        "ID,FILE_FAKE_PROB,VOICE_FAKE_PROB,MUSIC_FAKE_PROB,VOICE_PRESENT_PROB,MUSIC_PRESENT_PROB\n"
        "b,0,0,0,0,0\n"
        "a,0,0,0,0,0\n",
        encoding="utf-8",
    )
    time = np.arange(8_000, dtype=np.float32) / 16_000
    sf.write(test_dir / "a.wav", 0.2 * np.sin(2 * np.pi * 220 * time), 16_000)
    sf.write(test_dir / "b.wav", 0.2 * np.sin(2 * np.pi * 440 * time), 16_000)
    return test_dir / "a.wav", test_dir / "b.wav"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_fresh_offline_package_replays_file_local_predictions(tmp_path: Path) -> None:
    # Given: a real exported AST checkpoint and two evaluator files in non-filesystem ID order.
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    source_root = tmp_path / "reference"
    (source_root / "model").mkdir(parents=True)
    shutil.copytree(checkpoint, source_root / "model" / "ast")
    audio_a, audio_b = _write_input(source_root)
    v2_inference.generate_submission(source_root)
    expected = _rows(source_root / "output" / "submission.csv")
    archive = package.build_ast_submission(checkpoint, tmp_path / "submission.zip")
    unpacked = tmp_path / "fresh"
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert names == {
            "model/ast/config.json",
            "model/ast/preprocessor_config.json",
            "model/ast/multitask_config.json",
            "model/ast/model.safetensors",
            "model/ast/MODEL_INFO.json",
            "model/ast/LICENSE.txt",
            "model/runtime/ast_multitask.py",
            "model/runtime/v2_inference.py",
            "script.py",
            "requirements.txt",
        }
        bundle.extractall(unpacked)
    copied_a, copied_b = _write_input(unpacked)
    assert copied_a.read_bytes() == audio_a.read_bytes()
    assert copied_b.read_bytes() == audio_b.read_bytes()
    blocker = tmp_path / "offline"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(
        "import socket\ndef blocked(*args, **kwargs): raise RuntimeError('network forbidden')\n"
        "socket.create_connection = blocked\nsocket.socket.connect = blocked\n",
        encoding="utf-8",
    )

    # When: the freshly extracted archive runs with process-level network access blocked.
    environment = os.environ | {"PYTHONPATH": str(blocker)}
    subprocess.run(
        [sys.executable, str(unpacked / "script.py")],
        cwd=unpacked,
        env=environment,
        check=True,
        timeout=120,
    )
    actual = _rows(unpacked / "output" / "submission.csv")

    # Then: CSV order and every probability replay, and changing b cannot alter a.
    assert [row["ID"] for row in actual] == ["b", "a"]
    assert np.max(
        np.abs(
            np.asarray([[float(value) for key, value in row.items() if key != "ID"] for row in actual])
            - np.asarray([[float(value) for key, value in row.items() if key != "ID"] for row in expected])
        )
    ) <= 1e-5
    before_a = next(row for row in actual if row["ID"] == "a")
    sf.write(unpacked / "data" / "test" / "b.wav", np.zeros(8_000, dtype=np.float32), 16_000)
    subprocess.run(
        [sys.executable, str(unpacked / "script.py")],
        cwd=unpacked,
        env=environment,
        check=True,
        timeout=120,
    )
    assert next(row for row in _rows(unpacked / "output" / "submission.csv") if row["ID"] == "a") == before_a


def test_missing_local_weights_are_rejected_before_packaging(tmp_path: Path) -> None:
    # Given: a checkpoint directory whose model tensor was removed.
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / "model.safetensors").unlink()

    # When / Then: packaging fails locally; no alternate download route exists.
    with pytest.raises(ValueError, match="model.safetensors"):
        package.build_ast_submission(checkpoint, tmp_path / "submission.zip")
