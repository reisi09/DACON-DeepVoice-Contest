"""Regressions for observed metadata errors and train/evaluation leakage."""
import json
import random

import pytest

from data_v4.build_manifest import components, ing_echoes, lab, pair_components
from data_v4.export_bank import export


def test_echoes_generator_after_mode_and_shared_reference(tmp_path):
    for gen in ('musicgen', 'suno'):
        path = tmp_path / 'Echoes' / 'TTA' / gen / f'Artist_Title_{gen}_TTA_001.wav'
        path.parent.mkdir(parents=True)
        path.touch()
    rows = list(ing_echoes(tmp_path))
    assert {r['generator'] for r in rows} == {'musicgen', 'suno'}
    assert {r['generation_mode'] for r in rows} == {'TTA'}
    assert len(set(components(rows).values())) == 1, 'same reference across generators must stay together'


def test_parent_pairs_never_cross_roles():
    voices = [dict(ID=f'v{i}', role=role, group_keys=[f'v{i}']) for i, role in enumerate(('train', 'development'))]
    music = [dict(ID=f'm{i}', role=role, group_keys=[f'm{i}']) for i, role in enumerate(('development', 'train'))]
    pairs = pair_components(voices, music, components(voices + music), random.Random(3), 5)
    assert pairs and all(a['role'] == b['role'] for a, b in pairs)
    missing = [{k: v for k, v in row.items() if k != 'role'} for row in voices]
    with pytest.raises(ValueError, match='role'):
        pair_components(missing, music, components(missing + music), random.Random(3), 5)


def test_exact_bytes_and_explicit_aliases_share_components():
    rows = [dict(ID='full', group_keys=[], sha256='a' * 64, canonical_source_id='song:1'),
            dict(ID='crop', group_keys=[], sha256='b' * 64, canonical_source_id='song:1'),
            dict(ID='duplicate', group_keys=[], sha256='b' * 64)]
    assert len(set(components(rows).values())) == 1


def test_voice_splice_truth_survives_export_without_inventing_real_regions():
    base = dict(path='a.wav', sha256='c' * 64, group_keys=['speaker:1'], crop_len_s=8,
                source='constructed_splice', fake_voice_segments=[[2, 3]], **lab(1, 0, 1, None))
    rows = [dict(base, ID='aligned', role='train', voice_intervals_complete=True,
                 interval_alignment_verified=True),
            dict(base, ID='unverified', role='train')]
    out, _, _ = export(rows, {r['ID']: 'train' for r in rows})
    aligned, unverified = out
    assert aligned['role'] == 'train'
    events = json.loads(aligned['event_intervals'])
    assert events == [dict(component='voice', start=0.0, end=2.0, label=0),
                      dict(component='voice', start=2.0, end=3.0, label=1),
                      dict(component='voice', start=3.0, end=8.0, label=0)]
    assert json.loads(unverified['event_intervals']) == [], 'unverified codec timing stays masked'


def test_role_quarantine_propagates_history_and_is_order_independent():
    from data_v4.prepare_roles import prepare
    rows = [dict(ID='old', source='existing:fma', group_keys=['track:1'], prior_seen='presence_fit'),
            dict(ID='new', source='fma', group_keys=['track:1']),
            dict(ID='separate', source='fma', group_keys=['track:2'])]
    split = {'old': 'train', 'new': 'test', 'separate': 'val'}
    prepared, report = prepare(rows, split, {}, {})
    by = {r['ID']: r for r in prepared}
    assert by['old']['role'] == by['new']['role'] == 'development'
    assert {'presence_fit', 'head_optimizer', 'development_analysis', 'test', 'train'} <= set(by['new']['historical_exposure'])
    reversed_rows, _ = prepare(list(reversed(rows)), split, {}, {})
    assert {r['ID']: r['role'] for r in reversed_rows} == {r['ID']: r['role'] for r in prepared}
    assert report['sealed'] == 'unavailable'
    assert all('role' not in row for row in rows), 'historical inputs must stay untouched'


def test_file_mask_cannot_remove_other_four_predictions():
    from data_v4.file_ensemble import select_rows, known_file_panel
    rows = [dict(ID='both', split='test', role='development', FILE_FAKE='1', FILE_FAKE_mask='1'),
            dict(ID='haim', split='test', role='development', FILE_FAKE='', FILE_FAKE_mask='0')]
    selected = select_rows(rows, 'test', 'development')
    assert [r['ID'] for r in selected] == ['both', 'haim']
    scores = [dict(ID=r['ID'], Fm=.7) for r in selected]
    fitted, labels = known_file_panel(selected, scores)
    assert [r['ID'] for r in fitted] == ['both'] and labels.tolist() == [1]
    assert select_rows([dict(rows[0], split='val', role='selection')], 'val', 'calibration') == []


def test_explicit_masks_override_populated_labels_without_changing_other_metrics():
    from graph_scoring import official_metrics, PREDICTION_COLUMNS
    rows = [dict(ID=str(i), FILE_FAKE=str(i % 2), VOICE_FAKE=str(i % 2), MUSIC_FAKE=str(i % 2),
                 VOICE_PRESENT='1', MUSIC_PRESENT='1', FILE_FAKE_mask='0' if i == 0 else '1') for i in range(4)]
    preds = [dict(ID=str(i), **{key: .9 if i % 2 else .1 for key in PREDICTION_COLUMNS}) for i in range(4)]
    report = official_metrics(rows, preds)
    assert report['counts']['file_eer'] == 3 and report['unknown_counts']['file_eer'] == 1
    assert report['counts']['music_eer'] == report['counts']['voice_eer'] == 4
    assert report['score'] is None


def test_legacy_ids_with_colons_restore_the_original_group():
    from data_v4.prepare_roles import prepare
    rows = [dict(ID='existing:libri:libri:speaker-file', source='existing:libri', group_keys=['old-hash']),
            dict(ID='new', source='libri', group_keys=['speaker:1'])]
    prepared, report = prepare(rows, {rows[0]['ID']: 'train', 'new': 'test'},
                               {'libri:speaker-file': {'group_id': 'speaker:1'}}, {})
    assert {r['role'] for r in prepared} == {'development'}
    assert len({r['component_id'] for r in prepared}) == 1
    assert report['legacy_catalog_matched'] == 1
