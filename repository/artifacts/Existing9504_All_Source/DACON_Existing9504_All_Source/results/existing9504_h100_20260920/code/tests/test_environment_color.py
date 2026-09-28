"""Reproduce the VESSL notebook's forced-color environment against real uv."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacon9504a100.environment import _normalize, freeze, uv_env


def test_freeze_matches_installed_metadata_with_forced_color():
    uv = shutil.which("uv")
    assert uv, "Run with the real uv binary on PATH"
    original = {key: os.environ.get(key) for key in ("FORCE_COLOR", "CLICOLOR_FORCE", "NO_COLOR")}
    try:
        os.environ.update(FORCE_COLOR="1", CLICOLOR_FORCE="1")
        os.environ.pop("NO_COLOR", None)
        with tempfile.TemporaryDirectory() as directory:
            metadata = subprocess.check_output(
                [sys.executable, "-c", "import importlib.metadata as m,json,sysconfig;print(json.dumps("
                 "{d.metadata['Name']:d.version for d in m.distributions("
                 "path=[sysconfig.get_paths()['purelib']])}))"],
                env=uv_env(Path(directory)), text=True)
            expected = {_normalize(k): v for k, v in json.loads(metadata).items()}
            actual = freeze(Path(uv), Path(sys.executable), Path(directory))
        assert actual == expected, f"Parsed {len(actual)} installed packages; expected {len(expected)}"
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    test_freeze_matches_installed_metadata_with_forced_color()
    print("PASS: forced-color uv freeze matches actual installed package metadata")
