"""High-signal checks for full coverage and serialized inference parity."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import zipfile

import numpy as np
import torch

from eight_heads import FrozenHead
from mert12_full_refit import coverage_draws
from mert12_submission import (compare, load_heads, load_track_torchaudio,
                               pin_torchaudio_loader, predict_music, window_layout, windows)
from v6_runtime import baseline_module


def test_runtime_pins_training_decoder_with_real_smoke_audio(tmp_path: Path, monkeypatch) -> None:
    archive = Path("_workspace/mert12_a100_20260918/upload/smoke_inputs.zip")
    member = "original/test/core_000209f68a0dc6367dcb_base.wav"
    path = tmp_path / "audio.wav"
    with zipfile.ZipFile(archive) as saved:
        path.write_bytes(saved.read(member))
    monkeypatch.setenv("PATH", "")
    base = baseline_module(Path("_workspace/redesign_full_20260917/assets"))
    pin_torchaudio_loader(base)
    decoded = base.load_track(path, 2, 44_100).numpy()
    assert base.load_track is load_track_torchaudio
    assert hashlib.sha256(decoded.tobytes()).hexdigest() == "30d9f785cf8dbef2264f5aeaa689083ec3bfdf5bef2a7dbcd70e2c3cb45835ee"


def test_coverage_schedule_exposes_every_row_without_changing_class_balance() -> None:
    rows = [{"ID": f"real-{index}", "label": 0, "source": "real", "generator": "",
             "parent_id": f"r{index}"} for index in range(37)]
    rows += [{"ID": f"fake-{index}", "label": 1, "source": "fake", "generator": "g",
              "parent_id": f"f{index}"} for index in range(173)]
    schedule, report = coverage_draws(rows, 41)
    assert report["after_unique"] == len(rows)
    assert not report["after_zero_exposure_ids"]
    assert all(sum(rows[index]["label"] == label for index in batch) == 16
               for batch in schedule for label in (0, 1))


def test_exact_tail_zero_padding_and_deduplication() -> None:
    audio = np.arange(70_000, dtype=np.float32)
    starts = window_layout(len(audio))
    extracted = windows(audio, starts)
    assert starts.tolist() == [0, 5_400]
    np.testing.assert_array_equal(extracted[-1], audio[5_400:70_000])
    short = windows(np.ones(100, dtype=np.float32), window_layout(100))
    assert float(short[0, :100].sum()) == 100.0 and float(short[0, 100:].sum()) == 0.0


def test_serialized_heads_match_direct_file_mean_inference(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(5, 1024)).astype(np.float32)
    mean = rng.normal(size=1024).astype(np.float32)
    scale = rng.uniform(0.2, 2.0, size=1024).astype(np.float32)
    paths = []
    direct = []
    pooled = features.mean(0)
    for seed in (41, 42, 43):
        torch.manual_seed(seed)
        model = FrozenHead(1024, "mlp").eval()
        with torch.inference_mode():
            logits, _ = model(torch.from_numpy((pooled - mean) / scale)[None])
            direct.append(float(logits.sigmoid()[0]))
        path = tmp_path / f"head_seed{seed}.pt"
        torch.save({"model": model.state_dict(), "input_width": 1024, "seed": seed, "steps": 500,
                    "batch_size": 32, "mode": "source_generator_parent+coverage_prefix",
                    "architecture": "mlp", "mean": torch.from_numpy(mean),
                    "scale": torch.from_numpy(scale)}, path)
        paths.append(path)
    loaded = predict_music(load_heads(paths, "cpu"), features, "cpu")
    permuted = predict_music(load_heads(paths, "cpu"), features[::-1].copy(), "cpu")
    assert abs(loaded - float(np.mean(direct))) < 1e-7
    assert abs(loaded - permuted) < 1e-7


def test_smoke_comparison_binds_cached_music_and_file_independence(tmp_path: Path) -> None:
    header = "ID,FILE_FAKE_PROB,VOICE_FAKE_PROB,MUSIC_FAKE_PROB,VOICE_PRESENT_PROB,MUSIC_PRESENT_PROB\n"
    rows = {"a": "a,0.3,0.2,0.25,0.8,0.9\n", "b": "b,0.4,0.1,0.44,0.7,0.8\n"}
    reference, permuted, subset = (tmp_path / name for name in ("reference.csv", "permuted.csv", "subset.csv"))
    reference.write_text(header + rows["a"] + rows["b"])
    permuted.write_text(header + rows["b"] + rows["a"])
    subset.write_text(header + rows["a"])
    cached = tmp_path / "cached.json"
    cached.write_text(json.dumps({"music_probability_by_id": {"a": 0.25, "b": 0.44}}))
    output = tmp_path / "smoke.json"
    compare(reference, permuted, subset, cached, output)
    report = json.loads(output.read_text())
    assert report["status"] == "passed" and report["cached_raw_music_max_abs_error"] == 0.0


def test_notebook_cli_options_are_accepted_by_actual_parsers() -> None:
    notebook = json.loads(Path("DACON_MERT12_Full_A100.ipynb").read_text())
    lines = [line for cell in notebook["cells"] for line in cell.get("source", [])
             if "subprocess.run" in line]
    checked = 0
    for line in lines:
        options = set(re.findall(r"'(\-\-[a-z-]+)'", line))
        if "mert12_full_refit.py" in line:
            command = [sys.executable, "mert12_full_refit.py", "--help"]
        elif "package_mert12_submission.py" in line:
            command = [sys.executable, "package_mert12_submission.py", "--help"]
        elif "mert12_submission.py" in line:
            subcommand = "cached-reference" if "'cached-reference'" in line else "compare"
            command = [sys.executable, "mert12_submission.py", subcommand, "--help"]
        elif "SMOKE_PACKAGE/'script.py'" in line:
            command = [sys.executable, "_workspace/redesign_full_20260917/assets/baseline/script.py", "--help"]
        else:
            continue
        help_text = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        assert options <= set(re.findall(r"\-\-[a-z-]+", help_text))
        checked += 1
    assert checked == 5
