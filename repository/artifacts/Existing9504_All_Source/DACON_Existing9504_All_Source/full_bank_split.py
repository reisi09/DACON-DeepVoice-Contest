# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: .venv/bin/python full_bank_split.py
"""Account for the complete bank, then freeze connected-source 80:20 partitions."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from expanded_data import BANK, PILOT, ROOT, Row, SplitError, csv_rows
from full_bank_catalog import atoms, music_rows
from eight_contract import sha256_file

OUTPUT = ROOT / 'results/full_bank_mil_8020_20260918'
SEED = 20260918


def identities(row: Row) -> set[str]:
    keys = atoms(row)
    if row.get('native_pcm_sha256'):
        keys.add('native_pcm:' + row['native_pcm_sha256'])
    if row['source'] == 'asvspoof2019_la':
        for field in ('target_speaker_id', 'vc_input_speaker_id'):
            if row.get(field):
                keys.add('vctk_speaker:' + row[field])
        for field in ('source_reference_id', 'vc_input_reference_id'):
            if row.get(field):
                keys.add('vctk_recording:' + row[field])
    return keys


def catalog() -> list[Row]:
    """Preserve unknown labels and distinguish source files from tar members."""
    rows = [dict(r, route='music', input_kind='original_full_duration') for r in music_rows()]
    for name in ('speech_bank_asv_selected.csv', 'speech_bank_test-clean.csv'):
        quality = {r['ID']: r for r in csv_rows('speech_quality_asv.csv' if 'asv' in name else 'speech_quality_libri.csv')}
        for row in csv_rows(name):
            q = quality[row['ID']]
            if row['source_sha256'] != q['source_sha256'] or q['source_hash_matches_bank'] != 'True':
                raise SplitError('Speech source QC does not bind to source: ' + row['ID'])
            rows.append(dict(row, catalogue_file=name, route='voice', input_kind='original_full_duration',
                             native_pcm_sha256=q['native_pcm_sha256'],
                             quality_flags=json.dumps([k for k in ('near_full_scale_fraction_ge0p001',) if q[k] == 'True'])))
    frozen = json.loads((ROOT / 'results/expanded_h100_20260918/selection.json').read_text())
    for row in frozen['legacy_rows']:
        rows.append(dict(row, path=str((PILOT / row['path']).relative_to(ROOT)),
                         source_sha256=row['audio_sha256'], route='music',
                         input_kind='legacy_constructed' if row['ID'].startswith('localization_') else 'legacy_crop',
                         catalogue_file='frozen_legacy_train'))
    if len(rows) != 15169 + 959 or len({r['ID'] for r in rows}) != len(rows):
        raise SplitError('Collected bank or legacy identities changed')
    return rows


def components(rows: list[Row]) -> None:
    """Union all contributors, including duplicates excluded from optimization."""
    parents = list(range(len(rows)))
    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    owner: dict[str, int] = {}
    for i, row in enumerate(rows):
        for key in sorted(identities(row)):
            if key in owner:
                parents[find(i)] = find(owner[key])
            else:
                owner[key] = i
    names: dict[int, list[str]] = defaultdict(list)
    for i, row in enumerate(rows):
        names[find(i)].append(row['ID'])
    hashes = {i: hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest() for i, ids in names.items()}
    for i, row in enumerate(rows):
        row['component_id'] = hashes[find(i)]


def selection() -> tuple[list[Row], list[Row]]:
    rows = catalog()
    registry = json.loads((BANK / 'locked_groups.json').read_text())['all_prior_seen']
    seen_prior = set(registry['native_group_atoms']) | {'id:' + v for v in registry['parent_ids']}
    for field, prefix in (('source_sha256', 'bytes:'), ('audio_sha256', 'bytes:'), ('pcm_sha256', 'pcm:')):
        seen_prior.update(prefix + v for v in registry[field])
    previous = json.loads((ROOT / 'results/expanded_h100_20260918/selection.json').read_text())
    prior_splits = {r['ID']: r['split'] for r in previous['new_originals'] + previous['legacy_rows']}
    originals = {key for row in rows if row['input_kind'] == 'original_full_duration'
                 for key in identities(row) if key.startswith(('reference:fma:', 'reference:sonics:'))}
    admitted: list[Row] = []
    seen: dict[str, str] = {}
    for row in rows:
        reason = ''
        row.update(prior_split=prior_splits.get(row['ID'], row.get('split', '')),
                   prior_seen=str(row['ID'] in prior_splits or bool(identities(row) & seen_prior)),
                   review_flags=row.get('quality_flags', ''), split='')
        original = row['input_kind'] == 'original_full_duration'
        license_url = row.get('license_url', '')
        if original and not (('creativecommons.org/licenses/' in license_url and '/by-nd/' not in license_url and '/by-nc-nd/' not in license_url)
                             or row.get('license') in ('ODC-By-1.0', 'CC-BY-4.0')):
            reason = 'license_not_admitted'
        if float(row['duration_seconds']) < .25:
            reason = 'unusable_duration_under_0.25_seconds'
        if row.get('decode_status') in ('failed', 'error') or row.get('finite') == 'False':
            reason = 'decode_failure'
        # The full original replaces only its own FMA/SONICS crop, never an Echoes derivative.
        if row['input_kind'] == 'legacy_crop' and row['source'] in ('FMA', 'FMA/FMA', 'suno', 'udio') and identities(row) & originals:
            reason = 'represented_by_full_original'
        hashes = ['bytes:' + row['source_sha256']]
        hashes += [key + ':' + row[key] for key in ('native_pcm_sha256', 'pcm_sha256') if row.get(key)]
        duplicate = next((seen[key] for key in hashes if key in seen), '')
        if not reason and duplicate:
            reason = 'exact_duplicate'
            row['duplicate_of'] = duplicate
        row.update(admission='excluded' if reason else 'admitted', reason=reason)
        if not reason:
            if row[row['route'].upper() + '_FAKE'] not in ('0', '1'):
                raise SplitError('Missing supervised branch label: ' + row['ID'])
            for key in hashes:
                seen[key] = row['ID']
            admitted.append(row)
    components(rows)
    assign(admitted)
    return admitted, rows


def strata(row: Row) -> tuple[str, ...]:
    branch = row['route']
    return ('total', branch, branch + ':' + row[branch.upper() + '_FAKE'],
            row['source'] + ':' + row.get('generator', ''))


def assign(rows: list[Row]) -> None:
    """Minimize 20% deviations without splitting source components or reading scores."""
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[row['component_id']].append(row)
    vectors = {g: Counter(k for r in rs for k in strata(r)) for g, rs in grouped.items()}
    totals = Counter(k for row in rows for k in strata(row))
    targets = {k: n * .2 for k, n in totals.items()}
    weights = {k: (100 if k == 'total' else 50 if k in ('music', 'voice') else 30 if k.startswith(('music:', 'voice:')) else .05) / max(10, n) ** 2 for k, n in totals.items()}
    def delta(counts: Counter[str], vector: Counter[str], sign: int) -> float:
        return sum(weights[k] * ((counts[k] + sign * n - targets[k]) ** 2 - (counts[k] - targets[k]) ** 2) for k, n in vector.items())
    best_score, best = float('inf'), set()
    for attempt in range(32):
        rng = random.Random(SEED + attempt)
        order = sorted(grouped, key=lambda g: (-len(grouped[g]), rng.random()))
        counts: Counter[str] = Counter()
        chosen: set[str] = set()
        large = sorted(g for g, rs in grouped.items() if len(rs) >= 500)
        for bit, group in enumerate(large):
            if attempt & (1 << bit):
                chosen.add(group)
                counts.update(vectors[group])
        for _ in range(3):
            for group in order:
                if group in large:
                    continue
                sign = -1 if group in chosen else 1
                if delta(counts, vectors[group], sign) < -1e-12:
                    chosen.symmetric_difference_update((group,))
                    for k, n in vectors[group].items():
                        counts[k] += sign * n
        score = sum(weights[k] * (counts[k] - targets[k]) ** 2 for k in totals)
        if score < best_score:
            best_score, best = score, chosen
    for row in rows:
        row['split'] = 'val' if row['component_id'] in best else 'train'


def verify(rows: list[Row], ledger: list[Row]) -> None:
    """Protect total accounting, unknown labels, branch coverage and source independence."""
    assert len(ledger) == 16128 and len({r['ID'] for r in ledger}) == len(ledger)
    assert {r['ID'] for r in rows} == {r['ID'] for r in ledger if r['admission'] == 'admitted'}
    observed: dict[str, str] = {}
    for row in rows:
        for key in identities(row) | {row['component_id']}:
            assert observed.setdefault(key, row['split']) == row['split'], key
        if row['route'] == 'voice':
            assert row['MUSIC_FAKE'] == '' and row['MUSIC_FAKE_mask'] == '0'
    for split in ('train', 'val'):
        for route in ('music', 'voice'):
            assert {r[route.upper() + '_FAKE'] for r in rows if r['split'] == split and r['route'] == route} == {'0', '1'}
    assert .18 <= sum(r['split'] == 'val' for r in rows) / len(rows) <= .22


def write_csv(path: Path, rows: list[Row]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, ledger = selection()
    verify(rows, ledger)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, selected in [('all_sources', rows), ('admission_ledger', ledger)] + [(s, [r for r in rows if r['split'] == s]) for s in ('train', 'val')]:
        write_csv(OUTPUT / (name + '.csv'), selected)
    report = dict(seed=SEED, target_train_fraction=.8, bank_originals=15169, legacy_considered=959,
                  admitted=len(rows), excluded=len(ledger)-len(rows),
                  counts=dict(Counter(r['split'] + ':' + r['route'] for r in rows)),
                  source_counts=dict(Counter(r['source'] for r in rows)),
                  label_counts=dict(Counter(r['split'] + ':' + r['route'] + ':' + r[r['route'].upper()+'_FAKE'] for r in rows)),
                  reasons=dict(Counter(r['reason'] or 'admitted' for r in ledger)),
                  components=len({r['component_id'] for r in rows}), group_overlap=False,
                  prior_seen_rows=sum(r['prior_seen']=='True' for r in rows),
                  validation_is_historically_independent=False, official_dacon_evaluation_used=False,
                  quality_policy='Retain soft clipping warnings; reject corrupt/unusable audio, license denial and exact duplicates only. Legacy crops replaced when their full original is admitted.',
                  supervision='Music labels only for music; voice labels only for speech; unknown targets remain masked.',
                  gpu_started=False, csv_sha256={p.name:sha256_file(p) for p in OUTPUT.glob('*.csv')})
    (OUTPUT/'split_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
