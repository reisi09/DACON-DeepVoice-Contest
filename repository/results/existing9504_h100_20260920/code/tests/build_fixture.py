"""Build a small synthetic deployment for CPU integration tests (never used on the A100).

Real code, random weights: the actual baseline/script.py, DF-Arena and MERT
sources are packed with shrunken configs (DF-Arena XLS-R: 2 layers; MERT: 13
layers so hidden_states[12] exists) and random checkpoints, a small random
HTDemucs, and a random Cnn14.  Audio is synthetic.  The deployment mirrors the
production contract: deployment.json files -> {key, bytes, sha256}, keys
relative to the object-volume root, audio in shard ZIPs, assets.zip with the
27 asset paths + asset_hashes.json, source.zip with source/*.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_assets(baseline_src: Path, assets: Path) -> dict[str, str]:
    if assets.exists():
        shutil.rmtree(assets)
    shutil.copytree(baseline_src / "baseline", assets / "baseline")
    shutil.copytree(baseline_src / "mert", assets / "mert")
    xls = assets / "baseline/model/df_arena_1b/facebook/wav2vec2-xls-r-1b/config.json"
    config = json.loads(xls.read_text())
    config["num_hidden_layers"] = 2
    xls.write_text(json.dumps(config, indent=2))
    mert_config = json.loads((assets / "mert/config.json").read_text())
    mert_config["num_hidden_layers"] = 13
    mert_config["conv_dim"] = [32] * 7          # CPU-speed fixture only; hidden size stays 1024
    mert_config["intermediate_size"] = 512
    (assets / "mert/config.json").write_text(json.dumps(mert_config, indent=2))
    torch.manual_seed(0)
    # DF-Arena (real code) random weights.
    model_dir = assets / "baseline/model"
    sys.path.insert(0, str(model_dir))
    previous = Path.cwd()
    os.chdir(model_dir / "df_arena_1b")
    try:
        from df_arena_1b.configuration_antispoofing import DF_Arena_1B_Config
        from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing
        df = DF_Arena_1B_Antispoofing(DF_Arena_1B_Config())
    finally:
        os.chdir(previous)
    torch.save(df.state_dict(), model_dir / "df_arena_1b/pytorch_model.bin")
    del df
    # MERT (real remote code) random weights with 13 layers.
    sys.path.insert(0, str(assets))
    import importlib
    configuration = importlib.import_module("mert.configuration_MERT")
    modeling_source = (assets / "mert/modeling_MERT.py").read_text().replace(
        "from .configuration_MERT import MERTConfig", "from mert.configuration_MERT import MERTConfig")
    namespace: dict = {"__name__": "fixture_mert"}
    exec(compile(modeling_source, "modeling_MERT.py", "exec"), namespace)
    mert = namespace["MERTModel"](configuration.MERTConfig(**mert_config))
    torch.save(mert.state_dict(), assets / "mert/pytorch_model.bin")
    del mert
    # Small random HTDemucs saved in demucs' package format; filename checksum = sha256 prefix.
    from demucs.htdemucs import HTDemucs
    demucs = HTDemucs(sources=["drums", "bass", "other", "vocals"], audio_channels=2, samplerate=44100,
                      channels=8, t_layers=1, t_heads=2, segment=4, dconv_depth=1)
    args, kwargs = demucs._init_args_kwargs
    buffer = io.BytesIO()
    torch.save({"klass": HTDemucs, "args": args, "kwargs": kwargs,
                "state": {k: v.clone() for k, v in demucs.state_dict().items()}, "training_args": {}}, buffer)
    data = buffer.getvalue()
    for old in (model_dir / "htdemucs").glob("*.th"):
        old.unlink()
    # demucs LocalRepo verifies the file-name checksum (sha256 prefix), so the
    # fixture file is named by its own checksum under the production signature.
    (model_dir / f"htdemucs/955717e8-{hashlib.sha256(data).hexdigest()[:8]}.th").write_bytes(data)
    # Random Cnn14 (same architecture the baseline loads through panns_inference).
    from panns_inference.models import Cnn14
    cnn = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    torch.save({"model": cnn.state_dict()}, model_dir / "panns/Cnn14_mAP=0.431.pth")
    hashes = {}
    for path in sorted(assets.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            hashes[path.relative_to(assets).as_posix()] = sha(path)
    (assets / "asset_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return hashes


def synth(rng: np.random.Generator, seconds: float, kind: str) -> np.ndarray:
    t = np.arange(int(seconds * 16000)) / 16000
    if kind == "voice":
        f0 = rng.uniform(110, 220)
        signal = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    else:
        chord = rng.uniform(100, 400, size=3)
        signal = sum(np.sin(2 * np.pi * f * t) for f in chord) + 0.3 * np.sign(np.sin(2 * np.pi * 2 * t))
    signal = signal + 0.05 * rng.standard_normal(len(t))
    return (0.3 * signal / np.abs(signal).max()).astype(np.float32)


def build(out: Path, baseline_src: Path, source_dir: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    objvol = out / "objvol"
    deploy = objvol / "deploy"
    work = out / "work"
    for path in (deploy / "dataset", objvol / "shards", work):
        path.mkdir(parents=True, exist_ok=True)
    hashes = build_assets(baseline_src, work / "assets")
    with zipfile.ZipFile(deploy / "assets.zip", "w", zipfile.ZIP_STORED) as archive:
        for name in [*hashes, "asset_hashes.json"]:
            archive.write(work / "assets" / name, name)
    with zipfile.ZipFile(deploy / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.glob("*.py")):
            archive.write(path, f"source/{path.name}")
    rng = np.random.default_rng(7)
    plan = []  # (split, route, subtype, label, seconds, extension, exposure)
    for split, reps in (("train", 1), ("val", 0), ("test", 0)):
        for label in (0, 1):
            plan += [(split, "voice", "speech", label, 3.0 + label, ".flac", True)]
            plan += [(split, "music", "orig", label, 14.0 + 3 * label, ".wav" if label else ".mp3", label == 0)]
            plan += [(split, "music", "legacy", label, 8.0, ".wav", True)]
            if reps:
                plan += [(split, "voice", "speech", label, 2.5, ".flac", False)]
                plan += [(split, "voice", "singing", label, 3.0, ".flac", True),
                         (split, "voice", "singing", label, 3.5, ".flac", False)]
                plan += [(split, "music", "orig", label, 20.0, ".wav", False)]
    rows, audio = [], {}
    for number, (split, route, subtype, label, seconds, extension, exposure) in enumerate(plan):
        signal = synth(rng, seconds, "voice" if route == "voice" else "music")
        buffer = io.BytesIO()
        if extension == ".mp3":
            import lameenc
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(16000)
            encoder.set_channels(1)
            encoder.set_quality(2)
            pcm = (signal * 32767).astype("<i2").tobytes()
            data = encoder.encode(pcm) + encoder.flush()
        else:
            sf.write(buffer, signal, 16000, format="FLAC" if extension == ".flac" else "WAV",
                     subtype="PCM_16" if extension == ".flac" else "FLOAT")
            data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        identifier = f"fx:{split}_{route}_{subtype}_{label}_{number:02d}"
        path = f"audio/{digest}{extension}"
        audio[path] = data
        group = f"speaker:{split}:{number // 4}" if route == "voice" else f"artist:{split}:{number}"
        source = {"speech": "asvspoof2019_la", "singing": "CtrSVDD", "orig": "FMA" if not label else "SONICS",
                  "legacy": "Echoes/fixture/TTA" if label else "FMA/FMA"}[subtype]
        voice = route == "voice"
        row = {"ID": identifier, "split": split, "route": route, "path": path, "audio_sha256": digest,
               "source_sha256": digest, "source": source, "generator": ("A01" if voice else "fixturegen") if label else "REAL",
               "group_id": group, "component_id": hashlib.sha256(group.encode()).hexdigest(),
               "parent_ids": "", "parent_id": "", "pair_id": "",
               "historical_development_exposure": str(exposure),
               "kind": "a_cappella_singing" if subtype == "singing" else ("mixed" if subtype == "legacy" else ""),
               "input_kind": "legacy_crop" if subtype == "legacy" else "original_full_duration",
               "variant": "base" if subtype == "legacy" else "",
               "event_intervals": json.dumps([{"component": "music", "start": 0, "end": seconds, "label": label}])
               if subtype == "legacy" else "",
               "duration_seconds": str(seconds),
               "FILE_FAKE": str(label), "FILE_FAKE_mask": "1",
               "VOICE_FAKE": str(label) if voice else "", "VOICE_FAKE_mask": "1" if voice else "0",
               "MUSIC_FAKE": "" if voice else str(label), "MUSIC_FAKE_mask": "0" if voice else "1",
               "VOICE_PRESENT": "1" if voice else "", "VOICE_PRESENT_mask": "1" if voice else "0",
               "MUSIC_PRESENT": "0" if voice else "1", "MUSIC_PRESENT_mask": "1"}
        rows.append(row)
    with (deploy / "dataset" / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = sorted(audio)
    shards = []
    for index, chunk in enumerate((names[: len(names) // 2], names[len(names) // 2:])):
        shard = objvol / "shards" / f"audio_{index:03d}.zip"
        with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as archive:
            for name in chunk:
                archive.writestr(name, audio[name])
            archive.writestr("audio/unused_member.wav", b"not selected")
        shards.append({"key": f"shards/{shard.name}", "bytes": shard.stat().st_size, "sha256": sha(shard), "etag": "x"})
    upload = {"all_sources_uploaded": True, "shards": shards,
              "items": [{"ID": r["ID"], "path": r["path"], "audio_sha256": r["audio_sha256"]} for r in rows]}
    (deploy / "audio_upload.json").write_text(json.dumps(upload, indent=2))
    files = {}
    for name in ("dataset/manifest.csv", "audio_upload.json", "assets.zip", "source.zip"):
        path = deploy / name
        files[name] = {"key": f"deploy/{name}", "bytes": path.stat().st_size, "sha256": sha(path), "etag": "x"}
    splits = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    deployment = {"workspace": "fixture", "rows": len(rows), "runtime_manifest_sha256": files["dataset/manifest.csv"]["sha256"],
                  "split_counts": splits, "files": files}
    (deploy / "deployment.json").write_text(json.dumps(deployment, indent=2))
    expectations = {
        "rows": len(rows), "splits": splits,
        "per_class": {s: splits[s] // 2 for s in splits},
        "routes": {s: {route: sum(r["split"] == s and r["route"] == route for r in rows) for route in ("voice", "music")}
                   for s in splits},
        "historical_exposure": {s: sum(r["split"] == s and r["historical_development_exposure"] == "True" for r in rows)
                                for s in splits},
        "singing": {s: sum(r["split"] == s and r["source"] == "CtrSVDD" for r in rows) for s in splits},
        "singing_source": "CtrSVDD"}
    summary = {"objvol": str(objvol), "deploy": str(deploy), "asset_hashes": hashes, "expectations": expectations,
               "rows": len(rows)}
    (out / "fixture.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-src", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.baseline_src, args.source_dir)["expectations"], indent=2))
