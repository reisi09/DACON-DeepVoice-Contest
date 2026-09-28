"""Keep the existing Colab cache alive across one bounded temporal-stage recovery."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def state(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', type=int)
    parser.add_argument('--kernel', type=int)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    if args.runner is None:
        candidates = []
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                words = path.read_bytes().decode().split('\0')
                if 'v5_compare.py' in words and str(args.output) in words:
                    candidates.append(int(path.parent.name))
            except (FileNotFoundError, PermissionError):
                continue
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one live V5 runner, found {len(candidates)}')
        args.runner = candidates[0]
    command = Path(f'/proc/{args.runner}/cmdline').read_bytes().decode().split('\0')
    parent = int(Path(f'/proc/{args.runner}/status').read_text().split('PPid:\t', 1)[1].splitlines()[0])
    if args.kernel is None:
        args.kernel = parent
    kernel_command = Path(f'/proc/{args.kernel}/cmdline').read_bytes().decode()
    if ('v5_compare.py' not in command or str(args.output) not in command or parent != args.kernel
            or not any(name in kernel_command for name in ('colab_kernel_launcher', 'ipykernel_launcher'))):
        raise RuntimeError('Refusing to control a process outside the verified V5 notebook')
    if args.deadline != 1789315095 or time.time() >= args.deadline - 300:
        raise RuntimeError('Original allocation deadline changed or insufficient recovery time')
    script = args.work / 'v5_resume_temporal.py'
    incoming = Path(__file__).with_name('v5_resume_temporal.py')
    if not script.is_file() and incoming.is_file():
        shutil.copy2(incoming, script)
    if not script.is_file():
        raise FileNotFoundError(script)
    receipt = {'allocation_start': 1789308015, 'safe_deadline': args.deadline,
               'kernel_pid': args.kernel, 'original_runner_pid': args.runner,
               'started_unix': time.time(), 'resume_source_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
               'status': 'waiting_for_original_runner', 'supervisor_pid': os.getpid()}

    def save():
        temporary = args.output / 'temporal_supervisor.json.tmp'
        temporary.write_text(json.dumps(receipt, indent=2))
        temporary.replace(args.output / 'temporal_supervisor.json')

    owner = os.getpid()
    guard = os.fork()
    if guard == 0:
        while time.time() < args.deadline - 30 and os.getppid() == owner:
            time.sleep(2)
        try:
            os.kill(args.kernel, signal.SIGCONT)
        except ProcessLookupError:
            pass
        os._exit(0)

    process = None
    def interrupted(signum, frame):
        raise SystemExit(f'Supervisor interrupted: {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        save()
        os.kill(args.kernel, signal.SIGSTOP)
        while state(args.runner) not in (None, 'Z'):
            if time.time() >= args.deadline - 330:
                raise TimeoutError('Original stage left insufficient time for recovery')
            time.sleep(2)
        if (args.output / 'finished.json').is_file():
            receipt['status'] = 'original_completed_without_recovery'
            return
        receipt['status'] = 'recovery_running'
        receipt['recovery_started_unix'] = time.time()
        save()
        with (args.output / 'temporal_recovery.log').open('w', buffering=1) as log:
            process = subprocess.Popen([sys.executable, '-u', str(script), '--work', str(args.work),
                '--output', str(args.output), '--deadline', str(args.deadline - 60)], cwd=args.work,
                stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
            receipt['recovery_pid'] = process.pid
            save()
            while process.poll() is None:
                if time.time() >= args.deadline - 60:
                    raise TimeoutError('Recovery reached its bounded deadline')
                time.sleep(2)
            receipt['recovery_returncode'] = process.returncode
            receipt['status'] = 'recovery_completed' if process.returncode == 0 else 'recovery_failed'
    except BaseException as error:
        receipt['status'] = 'supervisor_failed'
        receipt['error'] = repr(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        receipt['ended_unix'] = time.time()
        try:
            save()
        finally:
            os.kill(args.kernel, signal.SIGCONT)
            os.kill(guard, signal.SIGTERM)
            os.waitpid(guard, 0)


if __name__ == '__main__':
    main()
