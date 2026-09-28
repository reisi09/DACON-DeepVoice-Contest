"""Build a portable training bundle or a completed, offline submission archive."""

import argparse
import json
import zipfile
from pathlib import Path

from v6_core import VERSION
from v6_data import sha256, write_json
from v6_runtime import asset_identity
from v6_train import trained_heads

RUNTIME = ("v6_core.py", "v6_data.py", "v6_runtime.py", "v6_train.py", "run_v6_final.py", "graph_scoring.py")
REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.7.1+cu128
torchaudio==2.7.1+cu128
torchvision==0.22.1+cu128
transformers==4.57.6
accelerate==1.12.0
onnxruntime-gpu==1.23.2
demucs==4.0.1
panns-inference==0.1.1
librosa==0.10.2.post1
soundfile==0.13.1
numpy==2.2.6
pandas==2.2.3
scipy==1.15.3
scikit-learn==1.8.0
einops==0.8.1
matplotlib==3.9.2
"""


def asset_files(assets: Path) -> list[Path]:
    files = [assets / "baseline" / "script.py"]
    files.extend(p for p in (assets / "baseline" / "model").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("."))
    files.extend(assets / "mert" / name for name in (
        "config.json", "configuration_MERT.py", "modeling_MERT.py", "preprocessor_config.json",
        "pytorch_model.bin", "README.md"))
    files.extend(assets / "voice" / name for name in ("w2v2-aasist.onnx", "README.md", "meta.yaml"))
    return sorted(files)


def seal_assets(assets: Path) -> None:
    manifest = dict(version=VERSION, sources={
        "baseline": "Organizer-supplied baseline_submit.zip; copied loaders and model files",
        "voice": "SpeechAntiSpoofingBenchmarks/W2V2-AASIST@196128e5a5101d5cb6ac7701597891bc7de7e7b5",
        "voice_variant": "ASVspoof2019 LA, XLS-R300M+AASIST; frozen published ONNX",
        "voice_upstream": "TakHemlata/SSL_Anti-spoofing@4acaa61dcef5f7610f43aa4d0b29c4559b970cd2",
        "mert": "m-a-p/MERT-v1-95M@12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
        "mert_license": "CC-BY-NC-4.0; model card bundled"},
        sha256={str(p.relative_to(assets)): sha256(p) for p in asset_files(assets)})
    write_json(assets / "assets.json", manifest)


def package(assets: Path, work: Path, output: Path, *, training: bool = False) -> dict:
    root = Path(__file__).resolve().parent
    identity = asset_identity(assets)
    if not training:
        trained_heads(work, identity)
    members = {f"model/runtime/{name}": root / name for name in RUNTIME}
    members.update({f"model/assets/{p.relative_to(assets)}": p for p in asset_files(assets)})
    members["model/assets/assets.json"] = assets / "assets.json"
    if training:
        members = {name.removeprefix("model/runtime/"): p for name, p in members.items()}
        members.update({name: root / name for name in ("package_v6_final.py", "V6_H100_Experiment.ipynb", "V6_FINAL_README.md")})
        for split in ("train", "dev"):
            manifest = root / "v4_validation" / f"{split}.csv"
            members[f"v4_validation/{split}.csv"] = manifest
            import csv
            with manifest.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    members[f"v4_validation/{row['path']}"] = root / "v4_validation" / row["path"]
    else:
        members.update({f"model/trained/{name}": work / name for name in
                        ("heads.npz", "training.json", "dev_metrics.json")})
    expanded = sum(p.stat().st_size for p in members.values())
    if expanded > 32_000_000_000:
        raise ValueError("Expanded archive exceeds 32 GB")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, name)
        archive.writestr("requirements.txt", REQUIREMENTS)
        if not training:
            archive.writestr("script.py", '''import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model/runtime"))
from run_v6_final import main
if __name__ == "__main__":
    sys.argv.insert(1, "predict")
    sys.argv.extend(["--assets", str(ROOT / "model/assets"), "--work", str(ROOT / "model/trained")])
    main()
''')
    if not training and temporary.stat().st_size > 10_000_000_000:
        raise ValueError("Submission exceeds 10 GB")
    with zipfile.ZipFile(temporary) as archive:
        expanded = sum(info.file_size for info in archive.infolist())
        count = len(archive.infolist())
        if expanded > 32_000_000_000:
            raise ValueError("Expanded archive exceeds 32 GB")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupted archive member: {bad}")
    temporary.replace(output)
    report = dict(kind="training_bundle" if training else "submission", asset_id=identity,
                  sha256=sha256(output), bytes=output.stat().st_size, expanded_bytes=expanded,
                  members=count, trained=not training,
                  target_gpu_runtime="not verified by packaging")
    write_json(output.with_suffix(".json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("v6_run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    print(json.dumps(package(args.assets, args.work, args.output, training=args.training), indent=2))
