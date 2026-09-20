# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: imported by prepare_full_bank.py with the existing project Python.
"""Admit every qualified source while protecting connected evaluation identities."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from expanded_data import ROOT, BANK, PILOT, Row, csv_rows, prompt


def music_rows() -> list[Row]:
    rows: list[Row] = []
    for name in ('haim_sources.csv', 'fma_bank_final_manifest.csv', 'jamendo_sources.csv', 'fakemusiccaps_sources.csv'):
        rows.extend(dict(row, catalogue_file=name) for row in csv_rows(name))
    for line in (BANK / 'sonics_retained_sources.jsonl').read_text().splitlines():
        row = {k: str(v) if v is not None else '' for k, v in json.loads(line).items()}
        rows.append(dict(row, catalogue_file='sonics_retained_sources.jsonl', generator=row['provider'], pcm_sha256=row['pcm16k_sha256']))
    for row in rows:
        if row['catalogue_file'] == 'jamendo_sources.csv':
            row.update(ID='jamendo_' + row['track_id'], source='JamendoMaxCaps', generator='REAL', group_id='artist:jamendo:' + row['artist_id'])
        if row['source'] == 'FMA':
            row['generator'] = 'REAL'
        row.setdefault('parent_id', row['ID'])
        row['group_id'] = '|'.join(sorted(atoms(row)))
    return rows


def atoms(row: Row) -> set[str]:
    keys = set(filter(None, row.get('group_id', '').split('|')))
    keys.add('id:' + row['ID'])
    for key in ('source_sha256', 'audio_sha256'):
        if row.get(key):
            keys.add('bytes:' + row[key])
    if row.get('pcm_sha256'):
        keys.add('pcm:' + row['pcm_sha256'])
    if row.get('parent_id'):
        keys.add('id:' + row['parent_id'])
    for parent in json.loads(row.get('parent_ids') or '[]'):
        keys.add('id:' + parent)
    if row.get('source') == 'HAIM' and row.get('generator') == 'acestep' and row.get('metadata') and prompt(row):
        keys.add('prompt:haim:acestep:' + hashlib.sha256(prompt(row).encode()).hexdigest())
    if row.get('source') == 'FMA' and row.get('track_id'):
        keys.add('reference:fma:' + row['track_id'])
    if row.get('source') == 'JamendoMaxCaps' and row.get('track_id'):
        keys.add('reference:jamendo:' + row['track_id'])
    if (row.get('source'), row.get('artist_id')) in {('FMA', '16670'), ('JamendoMaxCaps', '443873')}:
        keys.add('artist_name_candidate:mikrobenkrieg')
    return keys


def technical_reason(row: Row) -> str:
    match row['source']:
        case 'HAIM':
            return '' if row['status'] == 'integrity_checked' and not json.loads(row['quality_flags']) else 'quality_or_integrity_review'
        case 'FMA':
            return '' if row['train_eligible'] == 'True' else 'source_admission_gate'
        case 'JamendoMaxCaps':
            return '' if row['training_candidate_after_technical_checks'] == 'True' else 'source_admission_gate'
        case 'FakeMusicCaps':
            return '' if row['future_training_eligible'] == 'True' else 'source_admission_gate'
        case 'SONICS':
            return '' if row['status'] == 'retained' else 'source_admission_gate'
        case _:
            raise ValueError('Unexpected music source: ' + row['source'])


def selection() -> tuple[list[Row], list[Row]]:
    rows = music_rows()
    frozen = json.loads((ROOT / 'results/expanded_h100_20260918/selection.json').read_text())
    with (PILOT / 'manifest.csv').open() as stream:
        old = list(csv.DictReader(stream))
    registry = json.loads((BANK / 'locked_groups.json').read_text())
    locked: set[str] = set()
    # Prior train remains usable; all_prior_seen is a collection deduplication list.
    for row in old + frozen['new_originals']:
        if row['split'] != 'train':
            locked.update(atoms(row))
    for field, prefix in (('fma_artist_ids', 'artist:fma:'), ('jamendo_artist_ids', 'artist:jamendo:'),
                          ('sonics_ids', 'reference:sonics:'), ('musiccaps_ids', 'reference:musiccaps:')):
        locked.update(prefix + v for v in registry['current_locked_eval'][field])
    # Use audit's exact hash/native identity matches too, without blocking prior train reuse.
    audit = json.loads((BANK / 'cross_bank_audit.json').read_text())
    for check in audit['bank_checks'].values():
        for hit in check['lock_matches']['current_locked_eval']:
            locked.add('id:' + hit['ID'])
    # Transitive closure prevents prompt siblings / alternate encodings bridging partitions.
    connected = [atoms(row) for row in rows + frozen['legacy_rows']]
    changed = True
    while changed:
        previous = len(locked)
        for keys in connected:
            if keys & locked:
                locked.update(keys)
        changed = len(locked) != previous
    admitted, ledger = [], []
    seen: set[str] = set()
    for row in rows:
        reason = technical_reason(row)
        overlaps = sorted(atoms(row) & locked)
        if overlaps:
            reason = 'protected_evaluation_component'
        elif row.get('generator', '').casefold() == 'mustango':
            reason = 'heldout_generator_mustango'
        elif row['source_sha256'] in seen:
            reason = 'exact_duplicate'
        row = dict(row, admission='excluded' if reason else 'train_music', reason=reason,
                   protected_keys=json.dumps(overlaps), input_kind='original_full_duration')
        ledger.append(row)
        if not reason:
            seen.add(row['source_sha256'])
            admitted.append(row)
    for row in frozen['legacy_rows']:
        overlaps = sorted(atoms(row) & locked)
        if overlaps:
            raise ValueError(f'Legacy training intersects evaluation: {row["ID"]}')
        original = dict(row, path=str((PILOT / row['path']).relative_to(ROOT)),
                        source_sha256=row['audio_sha256'], input_kind='legacy_constructed' if row['ID'].startswith('localization_') else 'legacy_crop',
                        admission='train_music', reason='', catalogue_file='frozen_legacy_train')
        if row.get('MUSIC_PRESENT') != '1' or row.get('MUSIC_FAKE') not in ('0', '1'):
            original.update(admission='diagnostic', reason='music_label_unknown_or_absent')
        elif original['source_sha256'] in seen:
            original.update(admission='excluded', reason='exact_duplicate')
        else:
            seen.add(original['source_sha256'])
            admitted.append(original)
        ledger.append(original)
    return admitted, ledger
