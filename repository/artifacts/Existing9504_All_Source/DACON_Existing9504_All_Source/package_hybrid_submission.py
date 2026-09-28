"""Package the dev-locked G0/AST hybrid without changing either model's inference."""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Final

SUBMISSION_COLUMNS: Final = (
    "ID",
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
)
AST_COLUMNS: Final = ("MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
RUNTIME_FILES: Final = ("ast_multitask.py", "v2_inference.py", "package_hybrid_submission.py")
DEPLOY_SCRIPT: Final = '''#!/usr/bin/env python3
"""Run the dev-locked G0/AST hybrid completely offline."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "model" / "runtime"))

from hybrid_runtime import main

if __name__ == "__main__":
    main(ROOT)
'''
HYBRID_RUNTIME: Final = '''"""Execute frozen G0 and AST, then route only same-ID probabilities."""
import csv
from pathlib import Path

import g0_runtime
from package_hybrid_submission import SUBMISSION_COLUMNS, route_predictions
from v2_inference import generate_submission


def _read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(SUBMISSION_COLUMNS):
            raise ValueError("Component submission columns differ from the official schema")
        return list(reader)


def main(root: Path):
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    g0_runtime.main()
    g0_path = output / "submission.csv"
    staged_g0 = output / "g0_submission.csv"
    g0_path.replace(staged_g0)
    ast_path = generate_submission(root)
    rows = route_predictions(_read_rows(staged_g0), _read_rows(ast_path))
    temporary = ast_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(ast_path)
'''


def _validated_rows(rows: list[dict[str, str]], name: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Parse one component CSV once and reject malformed or duplicate evaluator IDs."""
    order: list[str] = []
    parsed: dict[str, dict[str, float]] = {}
    for row in rows:
        if set(row) != set(SUBMISSION_COLUMNS):
            raise ValueError(f"{name} submission columns differ from the official schema")
        identifier = row["ID"]
        if not identifier or identifier in parsed:
            raise ValueError(f"{name} submission IDs must be non-empty and unique")
        try:
            probabilities = {column: float(row[column]) for column in SUBMISSION_COLUMNS[1:]}
        except ValueError as error:
            raise ValueError(f"{name} submission has a non-numeric probability") from error
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities.values()):
            raise ValueError(f"{name} submission has an invalid probability")
        order.append(identifier)
        parsed[identifier] = probabilities
    if not order:
        raise ValueError(f"{name} submission is empty")
    return order, parsed


def route_predictions(g0_rows: list[dict[str, str]], ast_rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    """Route each ID's two frozen G0 values and three frozen AST values without mixing files."""
    ast_order, ast = _validated_rows(ast_rows, "AST")
    _g0_order, g0 = _validated_rows(g0_rows, "G0")
    if set(ast) != set(g0):
        raise ValueError("G0 and AST submission IDs differ")
    return [
        {
            "ID": identifier,
            "FILE_FAKE_PROB": g0[identifier]["FILE_FAKE_PROB"],
            "VOICE_FAKE_PROB": g0[identifier]["VOICE_FAKE_PROB"],
            "MUSIC_FAKE_PROB": ast[identifier]["MUSIC_FAKE_PROB"],
            "VOICE_PRESENT_PROB": ast[identifier]["VOICE_PRESENT_PROB"],
            "MUSIC_PRESENT_PROB": ast[identifier]["MUSIC_PRESENT_PROB"],
        }
        for identifier in ast_order
    ]


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Keep only the already-validated G0 model tree and reject path collisions."""
    members = [member for member in archive.infolist() if member.filename.startswith("model/")]
    names = [member.filename for member in members]
    if len(names) != len(set(names)) or any(name.startswith("model/ast/") for name in names):
        raise ValueError("G0 archive model members are unsafe for hybrid packaging")
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
            raise ValueError(f"Unsafe G0 archive member: {member.filename}")
    return members


def build_hybrid_submission(original_zip: Path, ast_checkpoint_dir: Path, output_zip: Path) -> Path:
    """Build the fixed G0/AST hybrid from the immutable baseline and selected AST checkpoint."""
    from package_submission import MAX_EXPANDED_BYTES, MAX_ZIP_BYTES, build_submission
    from package_v2_submission import MODEL_INFO, _checkpoint_members

    checkpoint = _checkpoint_members(ast_checkpoint_dir.resolve())
    output_zip = output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    g0_archive = output_zip.with_name(output_zip.name + ".g0.tmp")
    temporary = output_zip.with_name(output_zip.name + ".tmp")
    try:
        build_submission(original_zip, g0_archive, {"graph": "G0", "fusion": None})
        with zipfile.ZipFile(g0_archive) as source:
            members = _safe_members(source)
            g0_script = source.read("script.py").decode("utf-8")
            root_line = "ROOT = Path(__file__).resolve().parent"
            if g0_script.count(root_line) != 1:
                raise ValueError("G0 runtime root contract differs from the expected package")
            g0_runtime = g0_script.replace(root_line, "ROOT = Path(__file__).resolve().parents[2]", 1)
            compile(g0_runtime, "model/runtime/g0_runtime.py", "exec")
            source_root = Path(__file__).resolve().parent
            runtime = {name: source_root / name for name in RUNTIME_FILES}
            license_text = (source_root / "AST_LICENSE.txt").read_bytes()
            if any(not path.is_file() for path in runtime.values()):
                raise ValueError("Hybrid runtime source is incomplete")
            expanded = sum(member.file_size for member in members)
            expanded += sum(path.stat().st_size for path in checkpoint.values())
            expanded += sum(path.stat().st_size for path in runtime.values()) + len(g0_runtime.encode())
            expanded += len(DEPLOY_SCRIPT.encode()) + len(HYBRID_RUNTIME.encode()) + len(json.dumps(MODEL_INFO).encode()) + len(license_text)
            if expanded > MAX_EXPANDED_BYTES or expanded > MAX_ZIP_BYTES:
                raise ValueError(f"Hybrid archive would exceed submission limits: {expanded} bytes")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as target:
                for member in members:
                    if member.is_dir():
                        target.writestr(member.filename, b"")
                    else:
                        with source.open(member) as incoming, target.open(member.filename, "w", force_zip64=True) as outgoing:
                            shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
                for name, path in checkpoint.items():
                    target.write(path, f"model/ast/{name}")
                target.writestr("model/ast/MODEL_INFO.json", json.dumps(MODEL_INFO, sort_keys=True))
                target.writestr("model/ast/LICENSE.txt", license_text)
                for name, path in runtime.items():
                    target.write(path, f"model/runtime/{name}")
                target.writestr("model/runtime/g0_runtime.py", g0_runtime)
                target.writestr("model/runtime/hybrid_runtime.py", HYBRID_RUNTIME)
                target.writestr("script.py", DEPLOY_SCRIPT)
                target.writestr("requirements.txt", "")
        if temporary.stat().st_size > MAX_ZIP_BYTES:
            raise ValueError("Built hybrid archive exceeds compressed submission size limit")
        temporary.replace(output_zip)
    finally:
        temporary.unlink(missing_ok=True)
        g0_archive.unlink(missing_ok=True)
    return output_zip
