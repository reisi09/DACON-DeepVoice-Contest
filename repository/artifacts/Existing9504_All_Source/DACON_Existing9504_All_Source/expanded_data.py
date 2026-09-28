# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: .venv/bin/python expanded_data.py
"""Freeze a fresh group split before any prediction; original bank files stay untouched."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

from eight_contract import read_manifest, sha256_file

ROOT = Path(__file__).resolve().parent
BANK = ROOT / 'results/data_expansion_20260918'
OUTPUT = ROOT / 'results/expanded_h100_20260918'
PILOT = ROOT / '_workspace/localization_a100_20260918/dataset'
SPLITS = ('train', 'dev', 'calibration', 'test', 'confirmation_external')
Row = dict[str, str]


class SplitError(ValueError):
    """Source identity or split membership violates the frozen experiment."""


def csv_rows(name: str) -> list[Row]:
    with (BANK / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def rank(value: str) -> str:
    return hashlib.sha256(('newbank-20260918-v1:' + value).encode()).hexdigest()


def prompt(row: Row) -> str:
    value = json.loads(row['metadata']).get('payload_prompt', '')
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def assign(rows: list[Row], quotas: tuple[int, ...], *, force_test: str = '') -> list[Row]:
    """Rank groups independently of scores, assigning only complete single-row groups."""
    if len({row['group_id'] for row in rows}) != len(rows):
        raise SplitError('Select one representative per group before assigning quotas')
    forced = [row for row in rows if force_test and force_test in row['group_id']]
    ordered = sorted((row for row in rows if row not in forced), key=lambda row: rank(row['group_id']))
    if sum(quotas) > len(rows):
        raise SplitError('Insufficient eligible groups for registered quotas')
    result, offset = [], 0
    for split, count in zip(SPLITS, quotas, strict=True):
        current = forced if split == 'test' else []
        needed = count - len(current)
        if needed < 0:
            raise SplitError('Forced group exceeds quota')
        result.extend(dict(row, split=split) for row in current + ordered[offset:offset + needed])
        offset += needed
    return result


def audit_groups(rows: list[Row]) -> None:
    """Reject contributors or exact audio appearing on both sides of a split."""
    seen: dict[str, str] = {}
    identifiers: set[str] = set()
    for row in rows:
        if row['ID'] in identifiers:
            raise SplitError('Duplicate row identifier')
        identifiers.add(row['ID'])
        keys = row['group_id'].split('|') + ['parent:' + p for p in json.loads(row.get('parent_ids') or json.dumps([row['ID']]))]
        keys += [key + ':' + row[key] for key in ('source_sha256', 'pcm_sha256', 'audio_sha256') if row.get(key)]
        for key in keys:
            if seen.setdefault(key, row['split']) != row['split']:
                raise SplitError(f'Cross-split identity: {key}')


def select_new() -> list[Row]:
    """Use all 835 new real tracks and a larger H100 synthetic panel."""
    result: list[Row] = []
    fma = [r for r in csv_rows('fma_bank_final_manifest.csv') if r['train_eligible'] == 'True' and r['origin'] == 'new_download']
    jam = [dict(r, ID='jamendo_' + r['track_id'], source='JamendoMaxCaps', generator='REAL',
                group_id='artist:jamendo:' + r['artist_id']) for r in csv_rows('jamendo_sources.csv')
           if r['training_candidate_after_technical_checks'] == 'True' and r['reused_pilot_original'] == 'False']
    alias = 'artist_name_candidate:mikrobenkrieg'
    for rows, artist, quotas in ((fma, '16670', (209, 35, 23, 39, 20)), (jam, '443873', (326, 55, 37, 61, 30))):
        normalized = [dict(r, generator='REAL', group_id=r['group_id'] + ('|' + alias if r['artist_id'] == artist else '')) for r in rows]
        result += assign(normalized, quotas, force_test=alias)
    haim = csv_rows('haim_sources.csv')
    prior_prompts = {prompt(r) for r in haim if r['generator'] == 'acestep' and r['pilot_original_reused'] == 'True'}
    eligible = [r for r in haim if r['status'] == 'integrity_checked' and not json.loads(r['quality_flags']) and r['pilot_original_reused'] == 'False']
    ace: dict[str, Row] = {}
    for row in sorted(eligible, key=lambda r: rank(r['ID'])):
        if row['generator'] == 'acestep' and prompt(row) and prompt(row) not in prior_prompts:
            group = 'prompt:haim:acestep:' + hashlib.sha256(prompt(row).encode()).hexdigest()
            ace.setdefault(group, dict(row, group_id=group + '|' + row['group_id']))
    result += assign(list(ace.values()), (300, 31, 20, 28, 0))
    for variant, quotas in (('small', (120, 12, 8, 10, 0)), ('medium', (120, 12, 8, 12, 0)), ('large', (60, 7, 4, 6, 0))):
        candidates = [r for r in eligible if r['generator'] == 'musicgen' and r['model_variant'] == variant]
        result += assign(candidates, quotas)
    sonics = [json.loads(line) for line in (BANK / 'sonics_retained_sources.jsonl').read_text().splitlines()]
    for provider, count in (('suno', 551), ('udio', 550)):
        candidates = [dict(r, generator=provider, pcm_sha256=r['pcm16k_sha256']) for r in sonics if r['provider'] == provider and r['status'] == 'retained']
        result += assign(candidates, (count, 32, 22, 30, 0))
    caps: dict[str, list[Row]] = defaultdict(list)
    for row in csv_rows('fakemusiccaps_sources.csv'):
        if row['future_training_eligible'] == 'True':
            caps[row['group_id']].append(row)
    groups = sorted(caps, key=rank)
    offset = 0
    for split, count in zip(SPLITS, (116, 6, 4, 6, 150), strict=True):
        for group in groups[offset:offset + count]:
            result += [dict(r, split=split) for r in caps[group] if (r['generator'] == 'mustango') == (split == 'confirmation_external')]
        offset += count
    components = json.loads((BANK / 'speech_parent_groups.json').read_text())['component_members']
    speech = csv_rows('speech_bank_asv_selected.csv')
    for number, members in enumerate(components):
        for label in ('0', '1'):
            candidates = [r for r in speech if r['target_speaker_id'] in members and r['FILE_FAKE'] == label and float(r['duration_seconds']) >= 2]
            for row in sorted(candidates, key=lambda r: rank(r['ID']))[:16]:
                if row['vc_input_speaker_id'] and row['vc_input_speaker_id'] not in members:
                    raise SplitError('ASV target/input component changed')
                result.append(dict(row, group_id=f'asv_component:{number}|' + row['group_id'], split='confirmation_presence'))
    return result


def freeze() -> None:
    selected = select_new()
    locks = json.loads((BANK / 'locked_groups.json').read_text())['all_prior_seen']
    for row in selected:
        for key in ('source_sha256', 'pcm_sha256'):
            if row.get(key) in set(locks[key]):
                raise SplitError(f'New recording already seen: {row["ID"]}')
        for source, field in (('FMA', 'fma_artist_ids'), ('JamendoMaxCaps', 'jamendo_artist_ids'), ('SONICS', 'sonics_ids'), ('FakeMusicCaps', 'musiccaps_ids')):
            identifier = row.get('artist_id') if 'artist' in field else row.get('source_id') if source == 'SONICS' else row.get('reference_id')
            if row['source'] == source and identifier in locks[field]:
                raise SplitError(f'Prior source identity: {row["ID"]}')
    legacy = [dict(r) for r in read_manifest(PILOT / 'manifest.csv') if r['split'] == 'train' and (r['ID'].startswith('core_') or r['ID'].startswith('localization_'))]
    if len(legacy) != 959 or sum(r['ID'].startswith('core_') for r in legacy) != 831:
        raise SplitError('Original control and matched partial population changed')
    if len(selected) != 3668:
        raise SplitError(f'New H100 panel count changed: {len(selected)}')
    audit_groups(selected + legacy)
    counts = Counter(r['split'] for r in selected)
    if counts != dict(train=2700, dev=240, calibration=160, test=240, confirmation_external=200, confirmation_presence=128):
        raise SplitError(f'Registered split quotas changed: {counts}')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {'split_seed': 'newbank-20260918-v1', 'selection_uses_predictions': False,
               'new_originals': selected, 'legacy_rows': legacy,
               'bank_audit_sha256': sha256_file(BANK / 'cross_bank_audit.json'),
               'locked_groups_sha256': sha256_file(BANK / 'locked_groups.json')}
    path = OUTPUT / 'selection.json'
    content = json.dumps(payload, indent=2, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text() != content:
        raise SplitError('Refusing to replace a frozen selection')
    path.write_text(content)
    summary = {'total_rows': len(selected) + len(legacy), 'new_originals': len(selected),
               'new_split_counts': dict(counts), 'legacy_train': len(legacy),
               'new_music_labels_by_split': {s: dict(Counter(r['MUSIC_FAKE'] for r in selected if r['split'] == s)) for s in SPLITS},
               'selection_sha256': sha256_file(path), 'group_overlap': False,
               'new_generator_holdout': 'mustango', 'pretraining_independence_claimed': False,
               'speech_scope': '128 ASV diagnostic rows, fixed voice expert; not independent validation',
               'unresolved': ['HAIM MusicGen prompt identity', 'Echoes to MusicCaps crosswalk', 'unknown aliases and near duplicates']}
    (OUTPUT / 'split_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    freeze()
