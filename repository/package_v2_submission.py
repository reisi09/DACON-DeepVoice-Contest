"""Create a minimal self-contained AST-only submission archive."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Final

from package_submission import MAX_EXPANDED_BYTES, MAX_ZIP_BYTES

CHECKPOINT_FILES: Final = (
    "config.json",
    "preprocessor_config.json",
    "multitask_config.json",
    "model.safetensors",
)
RUNTIME_FILES: Final = ("ast_multitask.py", "v2_inference.py")
MODEL_INFO: Final = {
    "base_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "revision": "f826b80d28226b62986cc218e5cec390b1096902",
    "license": "BSD-3-Clause",
    "license_source": "https://github.com/YuanGongND/ast/blob/31088be8a3f6ef96416145c4b8d43c81f99eba7a/LICENSE",
    "format": "dacon236749-ast-multitask-v1",
}
DEPLOY_SCRIPT: Final = '''#!/usr/bin/env python3
"""Run the self-contained v2 AST model with no network access."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "model" / "runtime"))

from v2_inference import generate_submission

if __name__ == "__main__":
    generate_submission(ROOT)
'''


def _checkpoint_members(directory: Path) -> dict[str, Path]:
    """Require every inference asset before building an archive or touching the network."""
    members = {name: directory / name for name in CHECKPOINT_FILES}
    missing = [name for name, path in members.items() if not path.is_file()]
    if missing:
        raise ValueError(f"AST checkpoint missing required assets: {', '.join(missing)}")
    try:
        metadata = json.loads(members["multitask_config.json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("AST checkpoint metadata is not valid JSON") from error
    if metadata.get("format_version") != 1 or metadata.get("mode") not in {"frozen", "last_block"}:
        raise ValueError("AST checkpoint metadata has an unsupported contract")
    return members


def build_ast_submission(checkpoint_dir: Path, output_zip: Path) -> Path:
    """Package only AST weights and local runtime files, without baseline model copies."""
    checkpoint_dir = checkpoint_dir.resolve()
    output_zip = output_zip.resolve()
    members = _checkpoint_members(checkpoint_dir)
    source = Path(__file__).resolve().parent
    runtime = {name: source / name for name in RUNTIME_FILES}
    license_text = (source / "AST_LICENSE.txt").read_bytes()
    missing_runtime = [name for name, path in runtime.items() if not path.is_file()]
    if missing_runtime:
        raise ValueError(f"Submission runtime missing: {', '.join(missing_runtime)}")
    expanded_size = sum(path.stat().st_size for path in members.values())
    expanded_size += sum(path.stat().st_size for path in runtime.values()) + len(DEPLOY_SCRIPT.encode())
    expanded_size += len(json.dumps(MODEL_INFO, sort_keys=True).encode()) + len(license_text)
    if expanded_size > MAX_EXPANDED_BYTES or expanded_size > MAX_ZIP_BYTES:
        raise ValueError(f"AST archive would exceed submission limits: {expanded_size} bytes")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_zip.with_name(output_zip.name + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, path in members.items():
                archive.write(path, f"model/ast/{name}")
            archive.writestr("model/ast/MODEL_INFO.json", json.dumps(MODEL_INFO, sort_keys=True))
            archive.writestr("model/ast/LICENSE.txt", license_text)
            for name, path in runtime.items():
                archive.write(path, f"model/runtime/{name}")
            archive.writestr("script.py", DEPLOY_SCRIPT)
            archive.writestr("requirements.txt", "")
        if temporary.stat().st_size > MAX_ZIP_BYTES:
            raise ValueError("Built AST archive exceeds compressed submission size limit")
        temporary.replace(output_zip)
    finally:
        temporary.unlink(missing_ok=True)
    return output_zip
