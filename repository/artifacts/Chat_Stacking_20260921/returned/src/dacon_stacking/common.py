"""Strict JSON, hashes, and path containment; never deserialize arbitrary objects."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

class ContractError(ValueError):
    """An input violates a declared experimental/deployment contract."""

class NotRunError(RuntimeError):
    """A required optional runtime is unavailable, without a substitute."""

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf8'))

def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)

def contained(root: str | Path, name: str) -> Path:
    root = Path(root).resolve(); path = (root/name).resolve()
    if not path.is_relative_to(root):
        raise ContractError(f'Path escapes root: {name}')
    return path

def check_hash(path: str | Path, expected: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ContractError(f'SHA-256 mismatch: {path}')

def finite(values: Any, name: str, probability: bool = False) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ContractError(f'{name}: empty or missing/nonfinite measurements')
    if probability and (np.any(a < 0) or np.any(a > 1)):
        raise ContractError(f'{name}: probability outside [0,1]')
    return a
