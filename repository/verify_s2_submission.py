"""Verify exact weight export, crop parity and real offline CPU submission inference."""
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchaudio", "transformers", "timm", "demucs", "soundfile"]
# ///
# Run: .venv/bin/python verify_s2_submission.py

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

sys.dont_write_bytecode = True


def verify(root: Path) -> None:
    """Protect against wrong weights, altered crop ordering, missing assets and invalid CSV output."""
    output = root / "submissions/S2_EAT"
    stage = output / "staged"
    run = root / "results/fullft8_h100x1_20260916/downloaded"
    source = run / "h100x1_fullft8_run/07_S2_EAT_fullft/best.pt"
    state = torch.load(source, map_location="cpu", weights_only=True)["model"]
    exported = {"encoder." + key: value for key, value in load_file(stage / "model/assets/eat/model.safetensors").items()}
    exported.update({"head." + key: value for key, value in torch.load(
        stage / "model/trained/head.pt", weights_only=True).items()})
    assert set(state) == set(exported)
    assert all(torch.equal(value, exported[key]) for key, value in state.items())
    tensors = len(state)
    del state, exported
    with zipfile.ZipFile(output / "submit.zip") as archive:
        names = archive.namelist()
        assert len(set(names)) == len(names)
        assert {name.split("/")[0] for name in names} == {"model", "script.py", "requirements.txt"}
        manifest = json.loads(archive.read("model/manifest.json"))
        assert set(names) == set(manifest["sha256"]) | {"model/manifest.json"}
        for name, digest in manifest["sha256"].items():
            with archive.open(name) as incoming:
                assert hashlib.file_digest(incoming, "sha256").hexdigest() == digest
            with (stage / name).open("rb") as incoming:
                assert hashlib.file_digest(incoming, "sha256").hexdigest() == digest
    sys.path.insert(0, str(stage / "model/runtime"))
    from s2_eat_submission import evaluation_bag
    worker = (run / "batch_worker.py").read_text()
    start = worker.index("        width = int(config['seconds'] * 16000)")
    end = worker.index("        temporary = target.with_suffix('.partial.npz')", start)
    import math
    import textwrap
    reference = compile(textwrap.dedent(worker[start:end]), "original_worker_crops", "exec")
    rng = np.random.default_rng(41)
    for length in [64_000, 80_000, 176_003, 960_000]:
        music = rng.normal(size=length).astype("float32")
        scope = {"music": music, "np": np, "math": math, "config": {"seconds": 5, "pool_windows": 8}}
        exec(reference, scope)
        np.testing.assert_array_equal(evaluation_bag(music)[0].numpy(), scope["pools"][1])
    print("EXACT_TENSOR_EXPORT_AND_WORKER_CROP_PARITY", tensors, flush=True)
    with tempfile.TemporaryDirectory(prefix="s2-offline-") as folder:
        work = Path(folder)
        audio_dir = work / "test"
        audio_dir.mkdir()
        sources = sorted((root / "v4_validation/audio/eval/dev").glob("*.wav"))[:2]
        for index, path in enumerate(sources):
            audio, rate = sf.read(path, dtype="float32")
            sf.write(audio_dir / f"probe{index}.{'wav' if index == 0 else 'flac'}", audio[:rate * (4 + index)], rate,
                     subtype="PCM_16")
        columns = ["ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"]
        sample = work / "sample_submission.csv"
        with sample.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows([["probe1", 0, 0, 0, 0, 0], ["probe0", 0, 0, 0, 0, 0]])
        (work / "sitecustomize.py").write_text(
            "import socket\ndef blocked(*args, **kwargs):\n    raise RuntimeError('Network forbidden in package check')\n"
            "socket.socket.connect = blocked\nsocket.create_connection = blocked\n")
        environment = dict(os.environ, HF_HOME=str(work / "hf"), HF_MODULES_CACHE=str(work / "hf/modules"),
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(work), PYTHONDONTWRITEBYTECODE="1")
        prediction = work / "output/submission.csv"
        begun = time.monotonic()
        with (output / "offline_cpu.log").open("w") as log:
            subprocess.run([sys.executable, str(stage / "script.py"), "--device", "cpu",
                            "--test-dir", str(audio_dir), "--sample-submission", str(sample),
                            "--output", str(prediction)], cwd=work, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=1200)
        from graph_scoring import validate_predictions
        with prediction.open() as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == columns
            rows = list(reader)
        validate_predictions(rows, ["probe1", "probe0"])
        assert [row["ID"] for row in rows] == ["probe1", "probe0"]
        for row in rows:
            expected = max(float(row["VOICE_PRESENT_PROB"]) * float(row["VOICE_FAKE_PROB"]),
                           float(row["MUSIC_PRESENT_PROB"]) * float(row["MUSIC_FAKE_PROB"]))
            assert abs(float(row["FILE_FAKE_PROB"]) - expected) < 2e-10
        (output / "offline_cpu_predictions.csv").write_bytes(prediction.read_bytes())
        report = {"exact_exported_tensors": tensors, "crop_parity_lengths": [64000, 80000, 176003, 960000],
                  "zip_member_sha256_verified": True, "network_connections_blocked": True,
                  "empty_hf_cache": True, "real_audio_formats": ["wav", "flac"],
                  "all_five_outputs_valid": True, "sample_order_preserved": True,
                  "seconds_cpu": time.monotonic() - begun, "gpu_end_to_end_verified": False,
                  "l4_1200_file_time_limit_verified": False}
        (output / "runtime_verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
