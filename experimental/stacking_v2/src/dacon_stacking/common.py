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


# Common *policy*, not per-record observations. Keep the shipped Engine default.
# Historical gain-preserving caches can still be inspected by diagnostic readers,
# but every production training/evaluation/deployment boundary calls these guards.
PRODUCTION_PREPROCESSING = {
    'sr': 16000, 'mono': True, 'default_peak_cap': .6, 'preserve_gain': False,
    'demucs': 'shifts=0;split=True;overlap=.25;trim/right-pad; source snapshot hashed',
}
_PREPROCESSING_MEASUREMENTS = frozenset(('source_peak', 'normalization_gain', 'normalized_rms'))


def preprocessing_policy(preprocessing: Any) -> dict:
    """Validate the production policy and return only shared policy fields.

    Cache sidecars retain per-file peak/gain/RMS in their own identity hashes.
    Those measured values must NOT participate in cross-file policy equality.
    Missing policy is not inferred from measurements or silently backfilled.
    """
    if not isinstance(preprocessing, dict):
        raise ContractError('Explicit preprocessing policy is required; legacy locks must be rebuilt')
    missing = PRODUCTION_PREPROCESSING.keys() - preprocessing.keys()
    extra = preprocessing.keys() - (PRODUCTION_PREPROCESSING.keys() | _PREPROCESSING_MEASUREMENTS)
    if missing or extra:
        raise ContractError(f'Incomplete/unknown preprocessing policy: missing={sorted(missing)}, extra={sorted(extra)}')
    policy = {key: preprocessing[key] for key in PRODUCTION_PREPROCESSING}
    if type(policy['preserve_gain']) is not bool:
        raise ContractError('preprocessing preserve_gain must be a boolean, not a string or integer')
    if canonical(policy) != canonical(PRODUCTION_PREPROCESSING):
        raise ContractError('Unsupported production preprocessing; require preserve_gain=False, '
                            'mono 16000 Hz, peak cap 0.6 and the pinned Demucs policy')
    for name in _PREPROCESSING_MEASUREMENTS & preprocessing.keys():
        value = finite([preprocessing[name]], 'preprocessing measurement ' + name)[0]
        if isinstance(preprocessing[name], bool) or value < 0 or (name == 'normalization_gain' and value == 0):
            raise ContractError('Invalid per-file preprocessing measurement: ' + name)
    return policy


def manifest_preprocessing(row: dict) -> dict:
    """An omitted manifest preserve_gain means the existing deployment default.

    Explicit true/non-boolean flags are rejected before GPU loading or fitting.
    An optional full manifest declaration is checked too; it cannot override the
    fixed policy. File-specific observations in such a declaration may differ.
    """
    preserve = row.get('preserve_gain', False)
    if type(preserve) is not bool or preserve:
        raise ContractError('Production manifest preserve_gain must be False (boolean); '
                            'gain-preserving diagnostics cannot enter a production candidate')
    return preprocessing_policy(row.get('preprocessing', dict(PRODUCTION_PREPROCESSING)))
