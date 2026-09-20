"""Deterministically package author_code.zip (sorted members, fixed timestamps/permissions).

    python tools/build_author_zip.py --root . --out author_code.zip

Rebuilding from the same tree yields the same SHA-256, so the hash bound into the
notebook can be reproduced by the reviewer.
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

INCLUDE = ("dacon9504a100", "reference", "tests", "tools")
FILES = ("README_REVIEWER.md",)
DATE = (2026, 9, 19, 0, 0, 0)


def members(root: Path) -> list[Path]:
    result = []
    for name in INCLUDE:
        result += [p for p in (root / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and p.suffix not in (".pyc",)]
    result += [root / name for name in FILES if (root / name).is_file()]
    return sorted(result, key=lambda p: p.relative_to(root).as_posix())


def build(root: Path, out: Path) -> str:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in members(root):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=DATE)
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return hashlib.sha256(out.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.root, args.out))
