"""Unit regression for authorization refusal: pending, no retry, no token in argv/logs."""
from types import SimpleNamespace

from candidate_data_acquire import SourceFile, acquire_locked


def test_long_speech_selection_excludes_prior_audio_and_short_utterances(tmp_path, monkeypatch):
    """Keep original long audio while excluding the prior pilot by content identity."""
    import hashlib
    import io
    import json
    import tarfile

    import numpy as np
    import soundfile as sf
    from data_v4.prepare_librispeech_pilot import main

    # Given: a short clip, a previously collected long clip, and a new long clip.
    archive_path = tmp_path / 'source.tar.gz'
    raw_audio = []
    with tarfile.open(archive_path, 'w:gz') as archive:
        for index, seconds in enumerate((10, 16, 18)):
            stream = io.BytesIO()
            samples = .1 * np.sin(np.arange(seconds * 16000) * (index + 1) * .02)
            sf.write(stream, samples, 16000, format='FLAC')
            raw = stream.getvalue()
            raw_audio.append(raw)
            member = tarfile.TarInfo(f'LibriSpeech/dev-other/1/2/1-2-{index}.flac')
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    receipt = tmp_path / 'receipt.jsonl'
    receipt.write_text(json.dumps(dict(status='original_checksum_verified',
        sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(), url='https://www.openslr.org/12/',
        checksum='test-archive')) + '\n')
    previous = tmp_path / 'prior.jsonl'
    previous.write_text(json.dumps(dict(sha256=hashlib.sha256(raw_audio[1]).hexdigest())) + '\n')
    out = tmp_path / 'selected'
    monkeypatch.setattr('sys.argv', ['prepare', '--archive', str(archive_path), '--receipt', str(receipt),
        '--out', str(out), '--limit', '2', '--min-seconds', '15', '--max-seconds', '60',
        '--exclude-sources', str(previous)])
    # When: selection runs through the actual archive/decode/manifest path.
    main()
    # Then: only the new long recording survives, with unchanged bytes and no training admission.
    rows = [json.loads(line) for line in (out / 'sources.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['duration_seconds'] == 18
    assert (out / rows[0]['path']).read_bytes() == raw_audio[2]
    assert rows[0]['training_admitted'] is False and rows[0]['role'] == 'unassigned'


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


def test_music_selection_keeps_prompt_groups_and_masks_unverified_vocals(tmp_path):
    """Do not turn instrumental prompts into labels or admit held-out/duplicate audio."""
    import json
    from data_v4.prepare_music_arena_pilot import main

    # Given: two released files with identical bytes and one held-out generator.
    metadata = tmp_path / 'metadata'
    battles = metadata / 'battle_data' / 'month'
    battles.mkdir(parents=True)
    (metadata / 'README.md').write_text('Publisher metadata fixture')
    tree = dict(repo_id='fixture/music', revision='revision', files=[])
    for index, generator in enumerate(('musicgen-small', 'musicgen-medium', 'lyria-2')):
        audio = f'audio_files/month/{index}.mp3'
        tree['files'].append(dict(path=audio, size=100, lfs=dict(sha256='a' * 64)))
        (battles / f'{index}.json').write_text(json.dumps(dict(battle_uuid=str(index),
            prompt='Shared prompt', is_instrumental=True, audio_a=audio, audio_b='',
            system_a=generator, duration_a=30)))
    tree_path = tmp_path / 'tree.json'
    tree_path.write_text(json.dumps(tree))
    output = tmp_path / 'selected'
    # When: the production planner selects source records.
    main(tree_path, metadata, output)
    # Then: one checksum identity remains and prompt tags never become voice truth.
    rows = [json.loads(line) for line in (output / 'sources.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and 'lyria' not in rows[0]['generator']
    assert rows[0]['VOICE_PRESENT'] is None and rows[0]['VOICE_FAKE'] is None
    assert 'prompt:music_arena:' + rows[0]['prompt_sha256'] in rows[0]['group_keys']
    assert rows[0]['training_admitted'] is False
