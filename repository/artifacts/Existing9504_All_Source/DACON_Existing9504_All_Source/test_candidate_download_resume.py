"""Protect retained partial bytes and checksum-gated promotion after interrupted transfers."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from candidate_data_acquire import SourceFile, acquire_locked


def main():
    job = SourceFile('check', 'check.bin', 'https://zenodo.org/check', 4,
                     'sha256:' + hashlib.sha256(b'good').hexdigest(), 'test', 'unassigned')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'check.bin'
        partial = path.with_name(path.name + '.part')
        calls = []
        def interrupted(command, **kwargs):
            calls.append(command)
            assert command[command.index('--continue-at') + 1] == '-'
            if len(calls) == 1:
                partial.write_bytes(b'go')
                raise subprocess.CalledProcessError(18, command)
            assert partial.read_bytes() == b'go', 'Interrupted bytes were discarded'
            with partial.open('ab') as stream:
                stream.write(b'od')
            return subprocess.CompletedProcess(command, 0)
        with patch('candidate_data_acquire.subprocess.run', side_effect=interrupted), patch('time.sleep'):
            receipt = acquire_locked(job, path)
        assert len(calls) == 2 and path.read_bytes() == b'good' and not partial.exists()
        assert receipt['sha256'] == job.checksum.split(':')[1]
        path.rename(partial)
        with patch('candidate_data_acquire.subprocess.run', side_effect=AssertionError('Complete part requested network')):
            acquire_locked(job, path)
        path.rename(partial)
        partial.write_bytes(b'evil')
        with patch('candidate_data_acquire.subprocess.run', side_effect=AssertionError('Corrupt full part requested network')):
            try:
                acquire_locked(job, path)
            except RuntimeError as error:
                assert 'checksum mismatch' in str(error)
            else:
                raise AssertionError('Corrupt partial promoted to original')
        assert not path.exists() and partial.read_bytes() == b'evil'
    print('Interrupted transfer resumed; complete part promoted only after checksum; corruption retained and rejected.')


if __name__ == '__main__':
    main()
