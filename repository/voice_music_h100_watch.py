"""Five-minute monitoring using the verified download and shutdown implementation."""
from pathlib import Path
import json
import subprocess
import sys

import redesign_h100_watch as watch


def main(run_name='voice_music_h100_20260917'):
    root=Path(__file__).resolve().parent
    if run_name not in ('voice_music_h100_20260917','voice_music_h100_20260917_retry1'):
        raise ValueError('Unknown authorized run')
    watch.OUTPUT=root/'results'/run_name
    watch.RUN_METADATA=watch.OUTPUT/'workspace_run.json'
    watch.EXPECTED_PREFIX=run_name.replace('_','-')+'/'
    watch.POLL_SECONDS=300
    watch.MAX_SECONDS=3600
    stopped=watch.OUTPUT/'resource_stop.json'
    if not stopped.exists() or not json.loads(stopped.read_text()).get('pause_verified'):
        watch.monitor()
    if json.loads(stopped.read_text())['pause_verified']:
        run=json.loads(watch.RUN_METADATA.read_text())
        run.update(status='stopped',phase=json.loads(stopped.read_text())['phase'])
        watch.RUN_METADATA.write_text(json.dumps(run,indent=2))
        if (watch.OUTPUT/'bootstrap_restored.json').exists():
            return
        original=root/'_workspace/voice_music_h100_20260917/original_bootstrap/redesign_full_refit_run.sh'
        subprocess.run([*watch.CLI_PREFIX,'volume','upload',watch.EXPECTED_VOLUME,str(original),
            '--remote-prefix','redesign-full-20260917/','--overwrite'],check=True)
        (watch.OUTPUT/'bootstrap_restored.json').write_text(json.dumps(dict(restored=True,sha256=watch._sha256(original))))


if __name__=='__main__':
    main(*sys.argv[1:])
