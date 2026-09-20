"""Build a selected graph's offline archive without recompressing model tensors.

The evaluator already supplies the dependencies used here, so requirements.txt
is empty. Colab setup belongs to the experiment notebook, not the submission.
"""

import csv
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Literal, TypedDict

from graph_scoring import PredictionRow, SUBMISSION_COLUMNS, validate_predictions

if TYPE_CHECKING:
    from graph_fusion import Fusion


class Selection(TypedDict):
    graph: Literal["G0", "G1", "G2", "G3", "G4"]
    fusion: "Fusion | None"


RUNTIME_FILES: Final = ("graph_features.py", "graph_fusion.py", "graph_scoring.py")
MAX_ZIP_BYTES: Final = 10_000_000_000
MAX_EXPANDED_BYTES: Final = 32_000_000_000
DEPLOY_SCRIPT: Final = '''#!/usr/bin/env python3
"""Run the selected frozen graph independently on each evaluation audio file."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "model" / "runtime"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUNTIME))

from graph_features import Extractor
from graph_fusion import graph_predictions
from graph_scoring import SUBMISSION_COLUMNS, validate_predictions


def main():
    graph = json.loads((RUNTIME / "choice.json").read_text())["graph"]
    if graph not in {"G0", "G1", "G2", "G3", "G4"}:
        raise ValueError("Unsupported graph selection")
    fusion = json.loads((RUNTIME / "fusion.json").read_text())
    extractor = Extractor(RUNTIME / "official_baseline.py", include_raw=graph in {"G3", "G4"})
    try:
        base = extractor.base
        columns, sample = base.read_sample_submission(ROOT / "data" / "sample_submission.csv")
        if columns != list(SUBMISSION_COLUMNS):
            raise ValueError("Sample submission columns differ from the official six-column schema")
        paths = base.order_audio_files(base.find_audio_files(ROOT / "data" / "test"), sample)
        rows = []
        for index, path in enumerate(paths, 1):
            prediction = graph_predictions(extractor.extract(path), fusion)[graph]
            rows.extend(validate_predictions([prediction], [path.stem]))
            print(f"PREDICT {index}/{len(paths)} {path.stem}", flush=True)
        rows = validate_predictions(rows, [row["ID"] for row in sample])
        destination = ROOT / "output" / "submission.csv"
        temporary = destination.with_suffix(".csv.tmp")
        base.save_submission(temporary, columns, rows)
        temporary.replace(destination)
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
'''


def _model_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [member for member in archive.infolist() if member.filename.startswith("model/")]
    names = [member.filename for member in members]
    if not members or len(names) != len(set(names)):
        message = "Baseline model assets must be present with unique archive names"
        raise ValueError(message)
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
            message = f"Unsafe baseline member: {member.filename}"
            raise ValueError(message)
        if path.parts[:2] == ("model", "runtime"):
            message = "Baseline already contains reserved model/runtime paths"
            raise ValueError(message)
    return members


def build_submission(original_zip: Path, output_zip: Path, selection: Selection) -> Path:
    """Keep baseline model assets and package one selected graph atomically."""
    if original_zip.resolve() == output_zip.resolve():
        message = "The official baseline archive must not be overwritten"
        raise ValueError(message)
    graph = selection["graph"]
    if graph not in {"G0", "G1", "G2", "G3", "G4"}:
        message = f"Unsupported graph selection: {graph}"
        raise ValueError(message)
    fusion = selection.get("fusion")
    if graph == "G4" and fusion is None:
        message = "G4 packaging requires its frozen fusion parameters"
        raise ValueError(message)
    # G0-G3 cannot accidentally execute unrelated learned-head parameters.
    fusion_json = json.dumps(fusion if graph == "G4" else None, allow_nan=False)
    source_directory = Path(__file__).resolve().parent
    runtime = {f"model/runtime/{name}": (source_directory / name).read_bytes() for name in RUNTIME_FILES}
    runtime["model/runtime/fusion.json"] = fusion_json.encode()
    runtime["model/runtime/choice.json"] = json.dumps({"graph": graph}).encode()
    compile(DEPLOY_SCRIPT, "script.py", "exec")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_zip.with_name(output_zip.name + ".tmp")
    try:
        with zipfile.ZipFile(original_zip) as source:
            members = _model_members(source)
            baseline = source.read("script.py").decode("utf-8")
            original_base = "BASE_DIR = Path(__file__).resolve().parent"
            if baseline.count(original_base) != 1:
                message = "Official baseline BASE_DIR no longer matches the expected source"
                raise ValueError(message)
            baseline = baseline.replace(original_base, "BASE_DIR = Path(__file__).resolve().parents[2]", 1)
            compile(baseline, "model/runtime/official_baseline.py", "exec")
            runtime["model/runtime/official_baseline.py"] = baseline.encode()
            expanded_bytes = sum(member.file_size for member in members)
            expanded_bytes += sum(map(len, runtime.values())) + len(DEPLOY_SCRIPT.encode())
            if expanded_bytes > MAX_EXPANDED_BYTES or expanded_bytes + 1_000_000 > MAX_ZIP_BYTES:
                message = f"Stored archive would exceed submission limits: {expanded_bytes} bytes"
                raise ValueError(message)
            # Tensor compression costs CPU time without improving the inference result.
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as target:
                for member in members:
                    if member.is_dir():
                        target.writestr(member.filename, b"")
                        continue
                    with source.open(member) as incoming, target.open(member.filename, "w", force_zip64=True) as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
                for name, content in runtime.items():
                    target.writestr(name, content)
                target.writestr("script.py", DEPLOY_SCRIPT)
                target.writestr("requirements.txt", b"")
        if temporary.stat().st_size > MAX_ZIP_BYTES:
            message = "Built archive exceeds the compressed submission size limit"
            raise ValueError(message)
        temporary.replace(output_zip)
    finally:
        temporary.unlink(missing_ok=True)
    return output_zip


def verify_packaged(unpacked_dir: Path, deadline: float) -> list[PredictionRow]:
    """Run the real freshly unpacked script and parse its validated predictions.

    The caller provides data/test and data/sample_submission.csv, and compares
    these rows with the notebook's reference predictions. Deadline is Unix time.
    """
    remaining = deadline - time.time()
    if remaining <= 0:
        message = "No experiment time remains for packaged inference"
        raise TimeoutError(message)
    output = unpacked_dir / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "submission.csv").unlink(missing_ok=True)
    with (output / "packaged_inference.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str((unpacked_dir / "script.py").resolve())],
            cwd=unpacked_dir, stdout=log, stderr=subprocess.STDOUT,
            check=True, timeout=min(remaining, 3600),
        )
    with (unpacked_dir / "data" / "sample_submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        expected_ids = [row["ID"] for row in csv.DictReader(stream)]
    with (output / "submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(SUBMISSION_COLUMNS):
            message = "Packaged output must have the exact official six-column header"
            raise ValueError(message)
        rows = list(reader)
    return validate_predictions(rows, expected_ids)
