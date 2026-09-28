"""Real CPU crop/export integration: immutable audio, quantization, admission and masks.

Synthetic waves exercise transformation arithmetic, not perceptual audio quality.
AST score records are contract inputs here; model execution is verified separately.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from data_v4.admit_candidates_cpu import AST_MODEL_SHA256, TOLERANCE, finalize, prepare, read_rows, write_rows
from data_v4.build_manifest import components
from v2_data import decode_audio, pcm_sha256, sha256_file, write_wav


def test_derivation_and_hash_bound_train_only_admission(tmp_path):
    payload, pilot = tmp_path / 'payload', tmp_path / 'pilot'
    batch = payload / 'batch'
    batch.mkdir(parents=True)
    pilot.mkdir()
    rows = []
    for i, name in enumerate(('speech', 'song', 'low_confidence', 'evaluation', 'rights_blocked', 'short')):
        path = batch / (name + '.wav')
        time = np.arange((8 if name == 'short' else 20) * 16000) / 16000
        wav = .7 * np.sin(2 * np.pi * ((173 + 11 * i) * time + 2 * time**2))
        write_wav(path, wav)
        decoded = decode_audio(path, limit_seconds=None)
        rows.append(dict(ID=name, source='Blocked' if name == 'rights_blocked' else 'MusicArena' if name == 'song' else 'LibriSpeech',
            path=path.name, collection_batch='batch', group_keys=['speaker:' + name], component_id='speaker:' + name,
            sha256=sha256_file(path), bytes=path.stat().st_size, pcm_sha256=pcm_sha256(decoded),
            duration_seconds=len(decoded)/16000, full_audio_decode_verified=True, quality_flags=[], prior_roles=[],
            admission_blockers=['listening_review_pending', 'rights_review_pending', 'role_assignment_pending'],
            VOICE_PRESENT=None if name == 'song' else 1, MUSIC_PRESENT=1 if name == 'song' else 0,
            VOICE_FAKE=None if name == 'song' else 0, MUSIC_FAKE=1 if name == 'song' else None,
            FILE_FAKE=int(name == 'song'), license='test fixture', training_admitted=False, listening_verified=False))
    history = [dict(rows[0], ID='old', group_keys=['speaker:evaluation'], component_id='speaker:evaluation',
                    role='calibration', sha256='a'*64, training_admitted=True)]
    history_path, candidates_path, rights_path = (tmp_path / name for name in ('history.jsonl', 'candidates.jsonl', 'rights.json'))
    write_rows(history_path, history)
    write_rows(candidates_path, rows)
    rights_path.write_text(json.dumps({'sources': {'LibriSpeech': {'training_allowed': True}, 'MusicArena': {'training_allowed': True}}}))
    raw_hashes = {str(p): sha256_file(p) for p in batch.glob('*.wav')}
    prepared = tmp_path / 'prepared'
    receipt = prepare(history_path, candidates_path, payload, pilot, prepared, rights_path)
    normalized = read_rows(prepared / 'normalized_candidates.jsonl')
    assert {r['ID'] for r in normalized} == {'speech', 'song', 'low_confidence'}
    decisions = {r['ID']: r for r in read_rows(prepared / 'dispositions.jsonl')}
    assert 'prior_evaluation_component' in decisions['evaluation']['excluded_reason'], 'actual source graph must override missing prior_roles'
    assert 'rights_not_allowed' in decisions['rights_blocked']['excluded_reason']
    assert 'shorter_than_15_seconds' in decisions['short']['excluded_reason']
    for row in normalized:
        original = decode_audio(batch / row['original_source_path'], limit_seconds=None)
        derived = decode_audio(prepared / row['path'], limit_seconds=None)
        start = row['crop_start_sample']
        assert np.max(np.abs(derived - original[start:start+240000]*row['total_gain'])) <= TOLERANCE
        assert row['original_source_sha256'] == raw_hashes[str(batch / row['original_source_path'])]
        assert row['training_admitted'] is False and row['listening_verified'] is False
    reviews = [dict(ID=r['ID'], audio_sha256=r['sha256'], review_pcm_sha256=r['pcm_sha256'],
        speech_score=.1 if r['ID'] == 'low_confidence' else .9, music_score=.9,
        review_model='AST-AudioSet', device='cpu', gpu_used=False, model_sha256=AST_MODEL_SHA256,
        code_sha256=sha256_file(Path(__file__).parent / 'data_v4/review_audio_cpu.py'),
        evidence_kind='automatic_audio_event_screening', input_duration_seconds=15, review_start_seconds=0,
        review_seconds=15, window_starts_seconds=[0, 5], window_durations_seconds=[10, 10],
        reviewed_crop_coverage_fraction=1, class_by_window={'Speech':[.1, .1] if r['ID']=='low_confidence' else [.9, .9], 'Music':[.9, .9]},
        input_sha256=receipt['output_sha256']['normalized_candidates.jsonl'],
        listening_verified=False, authenticity_inferred=False) for r in normalized]
    reviews_path = tmp_path / 'reviews.jsonl'
    write_rows(reviews_path, [dict(r, audio_sha256='0'*64) if r['ID']=='speech' else r for r in reviews])
    with pytest.raises(ValueError, match='audio hash mismatch'):
        finalize(prepared, reviews_path, tmp_path / 'mismatched')
    write_rows(reviews_path, [dict(r, model_sha256='0'*64) for r in reviews])
    with pytest.raises(ValueError, match='evidence contract'):
        finalize(prepared, reviews_path, tmp_path / 'wrong_model')
    write_rows(reviews_path, [dict(r, window_durations_seconds=[10, 1]) for r in reviews])
    with pytest.raises(ValueError, match='evidence contract'):
        finalize(prepared, reviews_path, tmp_path / 'truncated_review')
    write_rows(reviews_path, reviews[:-1])
    with pytest.raises(ValueError, match='coverage'):
        finalize(prepared, reviews_path, tmp_path / 'incomplete')
    write_rows(reviews_path, reviews)
    final = tmp_path / 'final'
    report = finalize(prepared, reviews_path, final)
    assert report['admitted_count'] == 2 and report['candidate_evaluation_rows'] == 0
    combined = read_rows(final / 'manifest.jsonl')
    assert {r['ID'] for r in combined if r['role']=='train'} == {'speech', 'song'}
    grouped = components(combined)
    assert next(r for r in combined if r['ID']=='old')['component_id'] == history[0]['component_id']
    assert grouped['old'] not in {grouped['speech'], grouped['song']}
    with (final / 'bank_trainval.csv').open() as stream:
        exported = {r['orig_id']: r for r in csv.DictReader(stream)}
    assert exported['song']['VOICE_FAKE'] == '' and exported['song']['VOICE_FAKE_mask'] == '0'
    assert exported['speech']['MUSIC_FAKE_mask'] == '0'
    assert all(sha256_file(Path(path)) == digest for path, digest in raw_hashes.items())
    assert read_rows(history_path) == history, 'historical manifest must remain unchanged'
    assert not (tmp_path / 'mismatched').exists() and not (tmp_path / 'incomplete').exists()
