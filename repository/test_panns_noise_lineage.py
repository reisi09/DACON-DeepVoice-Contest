import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from panns_adaptation_data import admit_noise, audit, quarantine_roles, sha256


@pytest.fixture
def noise_recovery(tmp_path: Path):
    audio = tmp_path / 'noise.wav'
    sf.write(audio, np.linspace(-.2, .2, 800, dtype=np.float32), 16000)
    original = dict(ID='noise:a', source='musan_noise', role='development',
                    path='missing/noise.wav', sha256='a' * 64, bytes=1600,
                    VOICE_PRESENT=0, MUSIC_PRESENT=0, VOICE_FAKE=None,
                    MUSIC_FAKE=None, FILE_FAKE=0, group_keys=['original-group'],
                    parent_ids=['noise:raw'], crop_start_s=0., crop_len_s=.05,
                    gain_db=0., channels=1, codec_chain='pcm_wav')
    derived = original | dict(ID=f'{original["ID"]}:derived:{sha256(audio)}',
        original_id=original['ID'], parent_ids=['noise:a', 'noise:raw'],
        path=audio.name, sha256=sha256(audio), bytes=audio.stat().st_size,
        group_keys=original['group_keys'] + [f'bytes:{sha256(audio)}'],
        original_processed_sha256=original['sha256'])
    manifest = tmp_path / 'recovered.jsonl'
    manifest.write_text(json.dumps(derived) + '\n')
    return original, derived, manifest


def test_admission_keeps_historical_identity_missing_and_links_role_closure(noise_recovery):
    # Given a recovered file and another-role child of its historical identity.
    original, derived, manifest = noise_recovery
    sibling = dict(ID='child', source='fixture', role='train', group_keys=[],
                   sha256=original['sha256'])
    verified = {}
    # When the real audio admission path accepts the derived file.
    rows = admit_noise([original, sibling], verified, manifest, sha256(manifest))
    # Then old bytes are not marked present, and every connected role is quarantined.
    assert {r['ID']: r for r in rows}[original['ID']] == original
    assert set(verified) == {derived['ID']}
    assert verified[derived['ID']]['sha256'] == sha256(manifest.parent / derived['path'])
    closure = quarantine_roles(rows)
    assert all(closure[rid]['quarantined'] for rid in (original['ID'], derived['ID'], 'child'))
    assert len({closure[rid]['component_id'] for rid in closure}) == 1


@pytest.mark.parametrize('mutation', [
    {'ID': 'noise:a'}, {'ID': 'unbound-derived-id'}, {'original_id': 'unknown'},
    {'ID': 'noise:a', 'parent_ids': ['noise:raw']},
    {'parent_ids': ['noise:raw']}, {'parent_ids': ['noise:a']},
    {'group_keys': []}, {'role': 'train'}, {'MUSIC_PRESENT': 1},
    {'original_processed_sha256': 'b' * 64}, {'crop_start_s': 1.},
])
def test_admission_rejects_broken_identity_or_inheritance(noise_recovery, mutation):
    # Given changed bytes with forged identity, provenance, role, labels or preprocessing.
    original, derived, manifest = noise_recovery
    manifest.write_text(json.dumps(derived | mutation) + '\n')
    # When admission reads the real local audio and manifest, then it rejects the row.
    with pytest.raises(ValueError):
        admit_noise([original], {}, manifest, sha256(manifest))


def test_audit_counts_derived_audio_without_erasing_missing_historical_row(noise_recovery):
    # Given real archive audio plus a separately recovered missing historical file.
    original, derived, recovered = noise_recovery
    root = recovered.parent
    rows = [original]
    archive = root / 'audio.tar'
    with tarfile.open(archive, 'w') as stream:
        for label in (0, 1):
            row = original | dict(ID=f'train:{label}', source='fixture', role='train',
                parent_ids=[], group_keys=[], path=f'audio/{label}.wav',
                VOICE_PRESENT=label, MUSIC_PRESENT=label)
            audio = io.BytesIO()
            sf.write(audio, np.linspace(-.1, .1, 900 + label, dtype=np.float32), 16000, format='WAV')
            body = audio.getvalue()
            row.update(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
            member = tarfile.TarInfo(row['path'])
            member.size = len(body)
            stream.addfile(member, io.BytesIO(body))
            rows.append(row)
    manifest = root / 'historical.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    roles = root / 'roles.csv'
    with roles.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['ID', 'role'])
        writer.writeheader()
        writer.writerows({key: row[key] for key in ('ID', 'role')} for row in rows)
    # When the full audit hashes and decodes both input sources.
    report = audit(dict(output=str(root / 'audit'), manifest=str(manifest), roles=str(roles),
        archive=str(archive), archive_sha256=sha256(archive),
        recovered_noise_manifest=str(recovered), recovered_noise_sha256=sha256(recovered)))
    # Then the derived row is admitted while its historical bytes stay explicitly missing.
    selected = [json.loads(line) for line in (root / 'audit/presence_rows.jsonl').read_text().splitlines()]
    assert report['rows'] == 4 and report['verified'] == 3 and report['missing'] == 1
    assert report['recovered_noise'] == 1 and report['missing_by_source'] == {'musan_noise': 1}
    assert {r['ID'] for r in selected} == {derived['ID'], 'train:0', 'train:1'}
    assert report['excluded']['missing_audio|musan_noise|development'] == 1
