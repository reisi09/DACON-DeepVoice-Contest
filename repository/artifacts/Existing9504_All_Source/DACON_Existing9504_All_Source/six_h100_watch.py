"""Reuse verified export/download/shutdown logic at the requested five-minute interval."""
from pathlib import Path
import json
import subprocess

import redesign_h100_watch as watch

if __name__ == '__main__':
    watch.OUTPUT = Path(__file__).resolve().parent/'results/six_h100_20260917'
    watch.RUN_METADATA = watch.OUTPUT/'workspace_run.json'
    watch.EXPECTED_PREFIX = 'six-h100-20260917/'
    watch.POLL_SECONDS = 300
    watch.monitor()
    stopped = json.loads((watch.OUTPUT/'resource_stop.json').read_text())
    if stopped['pause_verified']:
        original = Path(__file__).resolve().parent/'_workspace/six_h100_20260917/original_bootstrap'
        scripts = list(original.rglob('redesign_full_refit_run.sh'))
        if len(scripts) != 1:
            raise RuntimeError('Original bootstrap backup is ambiguous')
        subprocess.run([*watch.CLI_PREFIX, 'volume', 'upload', watch.EXPECTED_VOLUME, str(scripts[0]),
                        '--remote-prefix', 'redesign-full-20260917/', '--overwrite'], check=True)
        (watch.OUTPUT/'bootstrap_restored.json').write_text(json.dumps({'restored':True, 'sha256':watch._sha256(scripts[0])}))
