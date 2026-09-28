"""Create exactly one authorized A100 workspace; verify exports and pause by its deadline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redesign_h100_watch import verify_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    cli = request['cli']
    deadline = request['pause_deadline_epoch']
    if not 0 < deadline - time.time() <= 3600:
        raise ValueError('Need a future absolute deadline of at most one hour')
    args.output.mkdir(parents=True, exist_ok=False)
    slug = None

    def call(arguments, timeout=40, budgeted=True):
        if budgeted:
            remaining = deadline - time.time() - 45
            if remaining <= 0:
                raise TimeoutError('Workspace pause reserve reached')
            timeout = min(timeout, remaining)
        result = subprocess.run(cli + arguments, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError('VESSL operation failed: ' + ' '.join(arguments[:2]))
        return result.stdout

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + '\n')

    def download(exclude_zip=True):
        target = args.output / 'downloaded'
        command = ['volume', 'download', request['object_volume'], str(target),
                   '--remote-prefix', request['object_prefix'], '--overwrite',
                   '--exclude', 'bootstrap_bundle.zip']
        if exclude_zip:
            command += ['--exclude', 'results.zip']
        remaining = int(deadline - time.time() - 45)
        if remaining <= 0:
            raise TimeoutError('Stop reserve reached')
        call(command, min(180, remaining))
        return target

    try:
        # If create has an ambiguous outcome, recover its unique name instead of creating twice.
        try:
            response = call(request['create_args'], timeout=90)
            matches = set(re.findall(r'\bwsp-[a-z0-9]+\b', response))
            if len(matches) == 1:
                slug = matches.pop()
        finally:
            if slug is None:
                rows = json.loads(call(['workspace', 'list', '-o', 'json'], budgeted=False))
                matches = [x['workspace']['slug'] for x in rows
                           if x['workspace']['name'] == request['name']]
                if len(matches) == 1:
                    slug = matches[0]
        if slug is None:
            raise RuntimeError('No unique created workspace found; inspect cloud state')
        save('workspace_run.json', dict(workspace=slug, name=request['name'],
             pause_deadline_epoch=deadline, object_volume=request['object_volume'],
             object_prefix=request['object_prefix']))
        print(json.dumps(dict(workspace=slug, pause_deadline_epoch=deadline)), flush=True)
        while time.time() < deadline - 45:
            detail = json.loads(call(['workspace', 'show', slug, '-o', 'json']))['workspace']
            if detail['name'] != request['name'] or detail['resourceSpecSlug'] != 'resourcespec-a100x1':
                raise RuntimeError('Workspace identity/allocation changed')
            state = {key: detail.get(key) for key in ('slug', 'name', 'state', 'phase', 'resourceSpecSlug')}
            state['observed_epoch'] = time.time()
            save('workspace_latest.json', state)
            print(json.dumps(state), flush=True)
            try:
                logs = call(['workspace', 'logs', slug, '--limit', '120'])
                (args.output / 'workspace_logs.txt').write_text(logs)
                print('\n'.join(logs.splitlines()[-6:]), flush=True)
            except RuntimeError:
                print('Workspace logs not available yet', flush=True)
            if state['state'] in ('stopped', 'terminating'):
                break
            target = download()
            receipt_path = target / 'export_receipt.json'
            exit_path = target / 'bootstrap_exit.json'
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                if receipt.get('archive_verified') is not True:
                    raise ValueError('Export does not claim verified archive')
                target = download(exclude_zip=False)
                verified = verify_archive(target / 'results.zip', receipt['archive_sha256'], target / 'unpacked')
                if verified['archive_bytes'] != receipt['archive_bytes']:
                    raise ValueError('Downloaded export size differs')
                save('verified_export.json', {**verified, 'receipt': receipt})
                print('RESULTS_DOWNLOADED_AND_VERIFIED', flush=True)
                break
            if exit_path.is_file():
                print('BOOTSTRAP_ENDED_WITHOUT_VERIFIED_EXPORT', flush=True)
                break
            time.sleep(max(0, min(300, deadline - time.time() - 45)))
    except Exception as error:
        save('controller_error.json', dict(type=type(error).__name__, message=str(error)))
        print(type(error).__name__ + ': ' + str(error), flush=True)
    finally:
        if slug:
            # Budget enforcement takes priority over waiting indefinitely for an export.
            stopped = False
            for attempt in range(3):
                try:
                    current = json.loads(call(['workspace', 'show', slug, '-o', 'json'], budgeted=False))['workspace']
                    if current['name'] != request['name']:
                        raise RuntimeError('Refuse to pause a different workspace')
                    if current['state'] not in ('stopped', 'terminating'):
                        call(['workspace', 'pause', slug], timeout=40, budgeted=False)
                    current = json.loads(call(['workspace', 'show', slug, '-o', 'json'], budgeted=False))['workspace']
                    stopped = current['state'] in ('stopped', 'stopping', 'terminating')
                    save('pause_receipt.json', {key: current.get(key) for key in
                         ('slug', 'name', 'state', 'phase', 'resourceSpecSlug')})
                    if stopped:
                        break
                except Exception as error:
                    print('PAUSE_ATTEMPT_FAILED ' + type(error).__name__, flush=True)
                time.sleep(5)
            print('PAUSE_REQUEST_CONFIRMED' if stopped else 'PAUSE_REQUIRES_IMMEDIATE_ATTENTION', flush=True)


if __name__ == '__main__':
    main()
