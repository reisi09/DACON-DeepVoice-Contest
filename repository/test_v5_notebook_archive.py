"""Protect extraction of already compressed official ZIPs without weakening hashes."""

import ast
import hashlib
import os
import shutil
import stat
import zipfile
from pathlib import Path

import nbformat
import pytest


def test_nested_zip_can_expand_under_deflate_but_must_match_receipt(tmp_path: Path) -> None:
    notebook = nbformat.read(Path(__file__).with_name("DACON_236749_compare_v5.ipynb"), as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.cell_type == "code" and "def extract_baseline_member(" in cell.source)
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "extract_baseline_member")
    payload = os.urandom(65536)
    archive, destination = tmp_path / "open.zip", tmp_path / "inner.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as packed:
        packed.writestr("baseline_submit.zip", payload)
    namespace = dict(zipfile=zipfile, Path=Path, stat=stat, shutil=shutil,
                     BASELINE_BYTES=len(payload), BASELINE_SHA256=hashlib.sha256(payload).hexdigest(),
                     sha256=lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook_extraction", "exec"), namespace)
    namespace["extract_baseline_member"](archive, destination)
    assert destination.read_bytes() == payload
    namespace["BASELINE_SHA256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt mismatch"):
        namespace["extract_baseline_member"](archive, destination)
