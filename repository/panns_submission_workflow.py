# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def run_jobs(jobs: list[tuple[str, list[str], float]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    pending, running, records = list(jobs), {}, {}
    stopped: list[int] = []
    handlers = {sig: signal.signal(sig, lambda number, _: stopped.append(number))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or running:
            if stopped:
                raise InterruptedError('Workflow termination requested')
            while pending and len(running) < 2:
                name, command, deadline = pending.pop(0)
                if not math.isfinite(deadline) or time.time() >= deadline:
                    raise TimeoutError('Job deadline already exhausted: ' + name)
                log = (output / (name + '.log')).open('ab')
                try:
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError:
                    log.close()
                    raise
                running[name] = process, log, deadline
                records[name] = dict(pid=process.pid, started_epoch=time.time(),
                                     deadline_epoch=deadline, reaped=False, exit_code=None)
                save(output / 'jobs.json', records)
            for name, (process, log, deadline) in list(running.items()):
                code = process.poll()
                if code is None:
                    if time.time() >= deadline:
                        raise TimeoutError('Job deadline exhausted: ' + name)
                    continue
                log.close()
                records[name].update(exit_code=code, reaped=True, finished_epoch=time.time())
                del running[name]
                save(output / 'jobs.json', records)
                if code:
                    raise RuntimeError(f'Job failed: {name}, exit={code}')
            time.sleep(.2)
    finally:
        for process, _, _ in running.values():
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        grace = time.monotonic() + 45
        for name, (process, log, _) in running.items():
            try:
                process.wait(timeout=max(.01, grace-time.monotonic()))
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
            finally:
                log.close()
            records[name].update(exit_code=process.returncode, reaped=True, finished_epoch=time.time())
        save(output / 'jobs.json', records)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def completed_training(config: dict, request: dict, arm: str) -> bool:
    out = Path(request['output'])/f'train_{arm}_41'
    terminal = out/'terminal.json'
    if not terminal.exists():
        return False
    result = json.loads(terminal.read_text())
    if result.get('export_ready') is not True:
        return False
    from panns_submission_train import condition
    from panns_submission_selection import Selection
    _, run_id, _ = condition(config, request, arm)
    selected = Selection.read(out/'export/selection.json')
    if (result['arm'] != arm or result['seed'] != 41
            or result['run_identity'] != run_id or selected.run_identity != run_id
            or selected.arm != arm or selected.config_identity != config['identity']
            or selected.best_epoch != result['best_epoch'] or selected.stop_reason != result['stop_reason']
            or result['deadline_epoch'] != request['deadlines'][arm]
            or selected.checkpoint_sha256 != result['checkpoint_sha256']
            or digest(out/'checkpoint.pt') != selected.checkpoint_sha256
            or digest(out/'export/best_model.pt') != selected.weights_sha256
            or digest(Path(config['reference'])) != selected.reference_sha256):
        raise ValueError('Completed training identity or artifact changed: '+arm)
    return True


def workflow(path: Path) -> None:
    request = json.loads(path.read_text())
    began, deadline = request['started_epoch'], request['deadline_epoch']
    if (request['workspace'] != 'wsp-nvxqw1jf9zdv'
            or not all(math.isfinite(v) for v in (began, deadline))
            or deadline != began + 12*3600 or not began <= time.time() < deadline):
        raise ValueError('Expected the approved existing A100 and original twelve-hour deadline')
    for name, expected in request['inputs'].items():
        if digest(Path(name)) != expected:
            raise ValueError('Workflow input changed: ' + Path(name).name)
    root, out = Path(__file__).parent, Path(request['output'])
    artifacts = Path(request['artifact_root'])
    artifacts.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    identity = digest(path)
    receipt = out / 'workflow_identity.json'
    if receipt.exists() and json.loads(receipt.read_text()) != {'sha256': identity}:
        raise ValueError('Use a fresh output for a changed workflow')
    save(receipt, {'sha256': identity})
    original, train_request = request['original_run'], request['trainer_request']
    config = json.loads(Path(original).read_text())
    train = json.loads(Path(train_request).read_text())
    if train['hard_deadline_epoch'] != deadline:
        raise ValueError('Trainer and workspace deadlines differ')
    def command(script: str, *args: object) -> list[str]:
        return [sys.executable, '-u', str(root/script), *map(str, args)]
    jobs = [(arm, command('panns_submission_train.py', '--run', original, '--request', train_request,
                         '--arm', arm), train['deadlines'][arm] + 90) for arm in ('R1', 'R3')
            if not completed_training(config, train, arm)]
    jobs.append(('fixed', command('panns_cached_fake.py', '--run', original,
        '--legacy-tar', request['legacy_tar'], '--output', out/'fixed', '--role', 'selection',
        '--device', 'cuda', '--deadline-epoch', began+8*3600-90), began+8*3600-45))
    save(out/'stage.json', dict(stage='parallel_training', deadline_epoch=deadline, max_gpu_jobs=2))
    run_jobs(jobs, out/'training_jobs')
    exports = [f'{arm}={Path(train["output"])/f"train_{arm}_41"/"export"}' for arm in ('R1', 'R3')]
    save(out/'stage.json', dict(stage='selection_evaluation', deadline_epoch=deadline))
    run_jobs([('evaluate', command('panns_submission_evaluate.py', '--run', original,
        '--fixed', out/'fixed/fixed.json', '--models', *exports, '--output', out/'evaluation',
        '--deadline-epoch', began+8*3600-30), began+8*3600)], out/'evaluation_jobs')
    chosen = json.loads((out/'evaluation/chosen.json').read_text())
    candidate = chosen.get('selected_export') or chosen.get('preferred_trial_export')
    if not candidate:
        raise ValueError('No validated trained candidate available for packaging')
    save(out/'stage.json', dict(stage='packaging', deadline_epoch=deadline))
    run_jobs([('package', command('package_panns_submission.py', '--reference', config['reference'],
        '--export', candidate, '--output', artifacts/'candidate.zip'), began+10*3600)], out/'package_jobs')
    save(out/'stage.json', dict(stage='offline_package_qa', deadline_epoch=deadline))
    run_jobs([('verify', command('panns_submission_verify.py', '--run', original,
        '--package', artifacts/'candidate.zip', '--output', artifacts/'package_qa',
        '--deadline-epoch', began+11*3600-60), began+11*3600)], out/'verify_jobs')
    publish_code = ('import hashlib,shutil,sys\nfrom pathlib import Path\n'
        'source,target=map(Path,sys.argv[1:])\n'
        'names=["candidate.zip","candidate.json","package_qa/verification.json",'
        '"package_qa/prediction.csv","package_qa/fresh_engine_predictions.json",'
        '"package_qa/entrypoint.log","package_qa/fresh_engine.log"]\n'
        'for name in names:\n'
        ' src,dst=source/name,target/name\n'
        ' dst.parent.mkdir(parents=True,exist_ok=True)\n'
        ' with src.open("rb") as inp,dst.open("xb") as out:\n'
        '  shutil.copyfileobj(inp,out,8<<20)\n'
        ' with src.open("rb") as inp,dst.open("rb") as out:\n'
        '  assert hashlib.file_digest(inp,"sha256").digest()==hashlib.file_digest(out,"sha256").digest()\n')
    run_jobs([('publish', [sys.executable, '-c', publish_code, str(artifacts), str(out)], deadline-120)],
             out/'publish_jobs')
    save(out/'workflow_complete.json', dict(workflow_sha256=identity, finished_epoch=time.time(),
        deadline_epoch=deadline, selected_arm=chosen['selected_arm'], candidate_export=candidate,
        package_sha256=digest(out/'candidate.zip'), official_submission=False,
        quality_claim='selection diagnostics only; leaderboard improvement unverified'))
    save(out/'stage.json', dict(stage='complete', deadline_epoch=deadline))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    workflow(parser.parse_args().request)
