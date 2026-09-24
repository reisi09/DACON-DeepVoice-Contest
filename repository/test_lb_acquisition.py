"""Unit regression for authorization refusal: pending, no retry, no token in argv/logs."""
from types import SimpleNamespace

from candidate_data_acquire import SourceFile, acquire_locked


def test_access_refusal_is_pending_and_token_never_enters_argv(tmp_path, monkeypatch):
    calls = []
    marker = 'synthetic_test_token'
    monkeypatch.setenv('HF_TOKEN', marker)
    def refused(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout='403', returncode=22)
    monkeypatch.setattr('candidate_data_acquire.subprocess.run', refused)
    job = SourceFile('unit', 'source.bin', 'https://huggingface.co/datasets/unit/source', 4,
                     'size-only:unpublished', 'pending', 'unassigned')
    result = acquire_locked(job, tmp_path / 'source.bin')
    assert result['status'] == 'pending_access' and result['training_admitted'] is False
    assert len(calls) == 1 and all(marker not in arg for arg in calls[0][0])
    assert marker not in str(result) and not (tmp_path / 'source.bin').exists()
