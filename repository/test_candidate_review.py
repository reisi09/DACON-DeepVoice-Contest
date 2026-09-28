"""Protect against reference-speaker leakage and accidental admission during review."""
import json
from pathlib import Path
import subprocess
import sys


def test_reference_speaker_and_pcm_links_restrict_candidates_without_admitting(tmp_path):
    # Given different transcripts cloned from the same reference speaker, and a PCM duplicate.
    prior = dict(ID='mlaad:en/TTS/old', source='mlaad', role='calibration',
                 group_keys=['mailabs_text:old'], path='old.wav', sha256='d'*64,
                 FILE_FAKE=1, VOICE_PRESENT=1, MUSIC_PRESENT=0, VOICE_FAKE=1, MUSIC_FAKE=None)
    prior_train = dict(prior, ID='mlaad:en/TTS/train', role='train', sha256='e'*64,
                       group_keys=['mailabs_text:train'])
    existing = tmp_path / 'existing.jsonl'
    existing.write_text(json.dumps(prior) + '\n' + json.dumps(prior_train) + '\n')
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    (metadata / 'TTS.csv').write_text('path|original_file|reference_speaker\n'
        './fake/en/TTS/old.wav|en_US/by_book/male/text/book/wavs/old.wav|en_US/by_book/female/alice/book/wavs/a.wav\n'
        './fake/en/TTS/train.wav|en_US/by_book/male/text/book/wavs/train.wav|en_US/by_book/female/alice/book/wavs/c.wav\n'
        './fake/en/TTS/new.wav|en_US/by_book/male/text/book/wavs/new.wav|en_US/by_book/female/alice/other/wavs/b.wav\n')
    batch = tmp_path / 'payload' / 'batch'
    batch.mkdir(parents=True)
    rows = [dict(ID='mlaad:en/TTS/new', source='MLAAD', group_keys=['mailabs_text:new'],
                 path='new.wav', sha256='a'*64, pcm_sha256='b'*64, duration_seconds=20.,
                 FILE_FAKE=1, VOICE_PRESENT=1, MUSIC_PRESENT=0, VOICE_FAKE=1, MUSIC_FAKE=None,
                 quality_flags=[], full_audio_decode_verified=True, training_admitted=False),
            dict(ID='duplicate', source='other', group_keys=[], path='duplicate.wav',
                 sha256='c'*64, pcm_sha256='b'*64, duration_seconds=20., FILE_FAKE=1,
                 VOICE_PRESENT=1, MUSIC_PRESENT=0, VOICE_FAKE=1, MUSIC_FAKE=None,
                 quality_flags=[], full_audio_decode_verified=True, training_admitted=False)]
    (batch / 'verified_sources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    output = tmp_path / 'out'
    # When the real CPU review CLI reconstructs lineage across old and new records.
    script = Path(__file__).parent / 'data_v4' / 'prepare_candidate_review.py'
    result = subprocess.run([sys.executable, str(script), str(existing), str(batch.parent),
                             str(metadata), str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    reviewed = [json.loads(l) for l in (output / 'candidate_manifest.jsonl').open()]
    # Then shared speakers/PCM inherit the evaluation restriction, never a training approval.
    assert len({r['component_id'] for r in reviewed}) == 1
    assert all(r['prior_roles'] == ['calibration', 'train'] for r in reviewed)
    assert all(r['role'] == 'unassigned' and r['training_admitted'] is False for r in reviewed)
    assert all('prior_evaluation_component' in r['admission_blockers'] for r in reviewed)
    assert all('within_candidate_pcm_duplicate' in r['admission_blockers'] for r in reviewed)
    assert [json.loads(l) for l in existing.read_text().splitlines()] == [prior, prior_train]
    safe_history = [json.loads(l) for l in (output / 'roles-v4' / 'manifest.jsonl').open()]
    assert all(r['role'] == 'development' for r in safe_history)
    assert all('calibration' in r['historical_exposure'] for r in safe_history)
