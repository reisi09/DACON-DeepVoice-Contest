from __future__ import annotations

import math
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from recovery_sonics import SonicsContractError, SonicsPredictor


class MeanLogit(torch.nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform.mean(dim=1, keepdim=True) * 20.0


def _predictor() -> SonicsPredictor:
    predictor = SonicsPredictor.__new__(SonicsPredictor)
    predictor.model = MeanLogit().eval()
    predictor.device = torch.device("cpu")
    return predictor


def test_predict_covers_late_tail_and_aggregates_file_locally() -> None:
    # Given: two silent windows followed by a short, high-logit tail.
    window = 80_000
    audio = np.concatenate((np.zeros(window * 2, dtype=np.float32), np.ones(8_000, dtype=np.float32)))

    # When: the whole file is predicted.
    result = _predictor().predict(audio)

    # Then: the final full window is anchored to the tail and changes the file mean.
    expected_tail = 1.0 / (1.0 + math.exp(-2.0))
    assert result["starts"] == [0, window, 88_000]
    assert result["covered_samples"] == audio.size
    covered_end = 0
    for start in result["starts"]:
        assert start <= covered_end
        covered_end = max(covered_end, min(audio.size, start + window))
    assert covered_end == audio.size
    assert result["window_probabilities"][-1] == pytest.approx(expected_tail, abs=1e-6)
    assert result["mean_probability"] == pytest.approx((0.5 + 0.5 + expected_tail) / 3.0, abs=1e-6)


def test_predict_rejects_empty_or_nonfinite_audio() -> None:
    # Given: malformed waveform inputs at the predictor boundary.
    predictor = _predictor()

    # When / Then: no invalid file can reach model inference.
    with pytest.raises(SonicsContractError, match="nonempty"):
        predictor.predict(np.array([], dtype=np.float32))
    with pytest.raises(SonicsContractError, match="finite"):
        predictor.predict(np.array([0.0, np.nan], dtype=np.float32))


def test_strict_asset_hash_rejects_tampered_weights_before_model_load(tmp_path) -> None:
    # Given: a local asset directory whose checkpoint differs from the pinned receipt.
    assets = Path(__file__).resolve().parent / "_workspace" / "recovery_v3" / "sonics_assets"
    shutil.copy2(assets / "config.json", tmp_path / "config.json")
    (tmp_path / "pytorch_model.bin").write_bytes(b"tampered")

    # When / Then: loading fails before an unverified pickle can be deserialized.
    with pytest.raises(SonicsContractError, match="SHA-256 mismatch"):
        SonicsPredictor(tmp_path, "cpu")


def test_offline_portable_directory_matches_source_bundle(tmp_path) -> None:
    # Given: a fresh bundle directory containing only the assets and sibling SONICS source.
    root = Path(__file__).resolve().parent
    source_assets = root / "_workspace" / "recovery_v3" / "sonics_assets"
    source_vendor = root / "_workspace" / "recovery_v3" / "sonics_vendor"
    portable_root = tmp_path / "work"
    portable_assets = portable_root / "sonics_assets"
    shutil.copytree(source_assets, portable_assets)
    shutil.copytree(source_vendor, portable_root / "sonics_vendor")
    source_files = sorted(path.relative_to(source_vendor) for path in source_vendor.rglob("*") if path.is_file())
    portable_files = sorted(path.relative_to(portable_root / "sonics_vendor") for path in (portable_root / "sonics_vendor").rglob("*") if path.is_file())
    assert portable_files == source_files
    for relative_path in source_files:
        assert (portable_root / "sonics_vendor" / relative_path).read_bytes() == (source_vendor / relative_path).read_bytes()
    code = """
import json
import sys
from pathlib import Path
import numpy as np
from recovery_sonics import SonicsPredictor
predictor = SonicsPredictor(Path(sys.argv[1]), 'cpu')
result = predictor.predict(np.zeros(80000, dtype=np.float32))
print(json.dumps(result, sort_keys=True))
"""
    environment = os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    # When: independent Python processes load the source and copied offline layouts.
    source = subprocess.run([sys.executable, "-c", code, str(source_assets)], cwd=root, env=environment, text=True, capture_output=True, check=True)
    portable = subprocess.run([sys.executable, "-c", code, str(portable_assets)], cwd=root, env=environment, text=True, capture_output=True, check=True)

    # Then: the portable location resolves its sibling source and predicts identically.
    source_result = json.loads(source.stdout)
    portable_result = json.loads(portable.stdout)
    assert portable_result["mean_probability"] == pytest.approx(source_result["mean_probability"], abs=1e-7)
    assert portable_result["window_probabilities"] == pytest.approx(source_result["window_probabilities"], abs=1e-7)
