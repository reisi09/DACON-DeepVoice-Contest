"""Reject changed staged executable content before invoking the pinned preflight."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('fault', ['changed_code', 'unexpected_code', 'changed_manifest'])
def test_retry_rejects_changed_bundle_before_entrypoint(tmp_path, fault):
    prepared, audio, assets = (tmp_path / name for name in ('prepared', 'audio', 'assets'))
    for root in (prepared, audio, assets):
        root.mkdir()
    code = prepared / 'code/data_v4/gpu_experiments.py'
    code.parent.mkdir(parents=True)
    marker = tmp_path / 'entrypoint_ran'
    code.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
    hashes = {'code/data_v4/gpu_experiments.py': hashlib.sha256(code.read_bytes()).hexdigest()}
    manifest = prepared / 'SHA256SUMS.json'
    manifest.write_text(json.dumps(hashes))
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if fault == 'changed_code':
        code.write_text(code.read_text() + '# changed\n')
    elif fault == 'unexpected_code':
        (prepared / 'code/extra.py').write_text('raise RuntimeError("unexpected executable")\n')
    else:
        manifest.write_text('{}')
    # Execute the script's real integrity payload with the local existing Python;
    # the deployment shell separately requires /root/envs/data/bin/python.
    script = Path(__file__).parent / 'data_v4/verify_gpu_cpu.sh'
    payload = script.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY\n', 1)[0]
    attempt = tmp_path / 'fresh_attempt'
    result = subprocess.run([sys.executable, '-', str(prepared), str(audio), str(assets), str(attempt), expected],
                            input=payload, text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert not marker.exists() and not attempt.exists()
    assert ('hash mismatch' if fault == 'changed_code' else 'unexpected files' if fault == 'unexpected_code'
            else 'identity mismatch') in result.stderr
