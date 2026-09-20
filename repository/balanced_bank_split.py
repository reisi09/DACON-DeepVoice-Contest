# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
# Run: .venv/bin/python balanced_bank_split.py
"""Select a source-balanced, connected-group 7,144-row bank without starting compute."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import tarfile
from typing import Final

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from expanded_data import ROOT, Row, SplitError
from full_bank_split import components, identities, write_csv

INPUT: Final = ROOT / 'results/full_bank_mil_8020_20260918/all_sources.csv'
OUTPUT: Final = ROOT / 'results/balanced7144_20260919'
SEED: Final = 'balanced7144-20260919-v1'
Cell = tuple[str, str, str, str]
Pattern = tuple[tuple[Cell, int], ...]
Quotas = dict[Cell, float]
AudioReceipt = dict[str, int]


def cell(row: Row) -> Cell:
    return row['route'], row[row['route'].upper() + '_FAKE'], row['source'], row['generator']


def rank(value: str) -> str:
    return hashlib.sha256((SEED + ':' + value).encode()).hexdigest()


def targets(rows: list[Row]) -> Quotas:
    """Retain scarce music-real/rare generators; balance the three large music pools."""
    capacity = Counter(cell(row) for row in rows)
    desired: Quotas = {}
    music_real = sum(n for c, n in capacity.items() if c[:2] == ('music', '0'))
    voice_real = [c for c in capacity if c[:2] == ('voice', '0')]
    major = {'HAIM', 'SONICS', 'FakeMusicCaps'}
    for c, n in capacity.items():
        if c[:2] == ('music', '0') or (c[:2] == ('music', '1') and c[2] not in major):
            desired[c] = n
        elif c in voice_real:
            desired[c] = (3572 - music_real) / len(voice_real)
        elif c[:2] == ('voice', '1'):
            desired[c] = min(n, 70)
    remaining = 3572 - sum(n for c, n in desired.items() if c[1] == '1')
    for source in sorted(major):
        eligible = [c for c in capacity if c[:2] == ('music', '1') and c[2] == source]
        for c in eligible:
            desired[c] = remaining / len(major) / len(eligible)
    return desired


def select(rows: list[Row]) -> list[Row]:
    """Return new split rows without mutating the original source manifest."""
    if len({r['ID'] for r in rows}) != len(rows):
        raise SplitError('Duplicate input IDs')
    eligible = [dict(r, previous_split=r['split'], previous_component_id=r['component_id'])
                for r in rows if r['source'] != 'constructed_music']
    components(eligible)
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in eligible:
        grouped[row['component_id']].append(row)
    patterns: dict[Pattern, list[str]] = defaultdict(list)
    for group, members in grouped.items():
        patterns[tuple(sorted(Counter(cell(r) for r in members).items()))].append(group)
    ordered = sorted(patterns)
    edges = [(p, c, n) for p, pattern in enumerate(ordered) for c, n in pattern]
    desired = targets(eligible)
    cells = sorted(desired)
    prefix, count = len(ordered), len(edges)
    size = prefix + 2 * count + 2 * len(cells)
    lower, upper, costs = np.zeros(size), np.full(size, np.inf), np.zeros(size)
    integrality = np.ones(size)
    integrality[prefix + 2 * count:] = 0
    matrix_rows, matrix_cols, matrix_values, lbs, ubs = [], [], [], [], []

    def constraint(terms: list[tuple[int, float]], low: float, high: float) -> None:
        index = len(lbs)
        for column, value in terms:
            matrix_rows.append(index)
            matrix_cols.append(column)
            matrix_values.append(value)
        lbs.append(low)
        ubs.append(high)

    for p, pattern in enumerate(ordered):
        upper[p] = len(patterns[pattern])
        costs[p] = (p + 1) * 1e-10
    for i, (p, _, n) in enumerate(edges):
        train, val = prefix + i, prefix + count + i
        capacity = n * len(patterns[ordered[p]])
        upper[train] = upper[val] = capacity
        constraint([(train, 1), (p, n)], -np.inf, capacity)
        constraint([(val, 1), (p, -n)], -np.inf, 0)
    for k, current in enumerate(cells):
        indices = [i for i, (_, c, _) in enumerate(edges) if c == current]
        train_terms = [(prefix + i, 1) for i in indices]
        val_terms = [(prefix + count + i, 1) for i in indices]
        constraint(train_terms + [(j, -4) for j, _ in val_terms], -2, 2)
        constraint(train_terms, 1, np.inf)
        constraint(val_terms, 1, np.inf)
        plus, minus = prefix + 2 * count + 2 * k, prefix + 2 * count + 2 * k + 1
        constraint(train_terms + val_terms + [(plus, -1), (minus, 1)], desired[current], desired[current])
        costs[plus] = costs[minus] = 1 / max(5, desired[current])
    for label in ('0', '1'):
        for split_index, target in enumerate((2858, 714)):
            constraint([(prefix + split_index * count + i, 1)
                        for i, (_, c, _) in enumerate(edges) if c[1] == label], target, target)
    matrix = coo_matrix((matrix_values, (matrix_rows, matrix_cols)), shape=(len(lbs), size)).tocsc()
    result = milp(costs, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, lbs, ubs),
                  options={'time_limit': 120, 'mip_rel_gap': 0})
    if not result.success or result.x is None:
        raise SplitError(f'No optimal feasible split: {result.message}')
    integer = np.rint(result.x[:prefix + 2 * count]).astype(int)
    selected = []
    for p, pattern in enumerate(ordered):
        members = sorted(patterns[pattern], key=rank)
        val_groups = set(members[:integer[p]])
        for i, (pattern_index, current, _) in enumerate(edges):
            if pattern_index != p:
                continue
            for split_index, split in enumerate(('train', 'val')):
                candidates = [r for g in members if (g in val_groups) == bool(split_index)
                              for r in grouped[g] if cell(r) == current]
                candidates.sort(key=lambda r: rank(r['ID']))
                needed = integer[prefix + split_index * count + i]
                selected.extend(dict(r, split=split) for r in candidates[:needed])
    return sorted(selected, key=lambda r: r['ID'])


def verify(selected: list[Row], original: list[Row]) -> None:
    """Reject source imbalance, label changes, or any connected contributor crossing."""
    reference = {r['ID']: r for r in original}
    if len(selected) != 7144 or len({r['ID'] for r in selected}) != 7144:
        raise SplitError('Selected row accounting failed')
    counts = Counter((r['split'], cell(r)[1]) for r in selected)
    if counts != {('train', '0'): 2858, ('train', '1'): 2858, ('val', '0'): 714, ('val', '1'): 714}:
        raise SplitError('Requested split/class balance failed')
    observed: dict[str, str] = {}
    for row in selected:
        for key, value in reference[row['ID']].items():
            if key not in ('split', 'component_id') and row[key] != value:
                raise SplitError('Source metadata changed: ' + row['ID'] + ':' + key)
        for key in identities(row) | {row['component_id']}:
            if observed.setdefault(key, row['split']) != row['split']:
                raise SplitError('Contributor overlaps splits: ' + key)
    cells = Counter((*cell(r), r['split']) for r in selected)
    expected = {cell(r) for r in original if r['source'] != 'constructed_music'}
    if {key[:-1] for key in cells} != expected:
        raise SplitError('A source/generator disappeared')
    for current in expected:
        train, val = cells[(*current, 'train')], cells[(*current, 'val')]
        if not train or not val or abs(train - 4 * val) > 2:
            raise SplitError('Source/generator ratio failed: ' + str(current))


def check_audio(rows: list[Row]) -> AudioReceipt:
    """Hash every selected local source, including selected members of LibriSpeech tar."""
    checked, total_bytes = 0, 0
    archives: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        if not row['path']:
            archives[row['archive_path']][row['archive_member']] = row['source_sha256']
            continue
        path = ROOT / row['path']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != row['source_sha256']:
            raise SplitError('Source audio hash mismatch: ' + row['ID'])
        checked += 1
        total_bytes += path.stat().st_size
    for archive, members in archives.items():
        remaining = dict(members)
        with tarfile.open(ROOT / archive, 'r|gz') as source:
            for entry in source:
                if entry.name not in remaining:
                    continue
                stream = source.extractfile(entry)
                if stream is None or hashlib.file_digest(stream, 'sha256').hexdigest() != remaining.pop(entry.name):
                    raise SplitError('Archive audio mismatch: ' + entry.name)
                checked += 1
                total_bytes += entry.size
        if remaining:
            raise SplitError('Missing archive members: ' + archive)
    if checked != len(rows):
        raise SplitError('Audio verification did not cover every selected row')
    return {'files_verified': checked, 'audio_bytes': total_bytes}


def main() -> None:
    """Write only new local manifests and an auditable selection ledger."""
    raw = INPUT.read_bytes()
    with INPUT.open(newline='') as stream:
        original = list(csv.DictReader(stream))
    selected = select(original)
    verify(selected, original)
    audio_receipt = check_audio(selected)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    by_id = {r['ID']: r for r in selected}
    write_csv(OUTPUT / 'selected.csv', selected)
    for split in ('train', 'val'):
        write_csv(OUTPUT / (split + '.csv'), [r for r in selected if r['split'] == split])
    ledger = [dict(ID=r['ID'], source=r['source'], generator=r['generator'], route=r['route'],
                   label=cell(r)[1], selected=str(r['ID'] in by_id), split=by_id.get(r['ID'], {}).get('split', ''),
                   reason='selected' if r['ID'] in by_id else 'legacy_cross_parent_mixture_reserved'
                   if r['source'] == 'constructed_music' else 'capacity_and_group_balanced_subsample') for r in original]
    write_csv(OUTPUT / 'selection_ledger.csv', ledger)
    capacities = Counter(cell(r) for r in original)
    counts = Counter((*cell(r), r['split']) for r in selected)
    summary = [dict(route=c[0], label=c[1], source=c[2], generator=c[3], available=str(n),
                    train=str(counts[(*c, 'train')]), val=str(counts[(*c, 'val')]),
                    selected=str(counts[(*c, 'train')] + counts[(*c, 'val')])) for c, n in sorted(capacities.items())]
    write_csv(OUTPUT / 'source_summary.csv', summary)
    report = dict(input=str(INPUT), input_sha256=hashlib.sha256(raw).hexdigest(), seed=SEED,
                  rows=len(selected), train=5716, val=1428, class_counts={'real': 3572, 'fake': 3572},
                  branch_label_counts=dict(Counter(':'.join((r['split'], *cell(r)[:2])) for r in selected)),
                  policy='Global real/fake 1:1; every retained source/generator/label cell 80:20 with at most 0.4 validation-row rounding deviation; connected source groups never cross.',
                  eligible_groups_exclude='128 legacy constructed_music rows; originals preserved; parent groups recomputed before selection.',
                  selection_preferences='Retain scarce real music and rare music generators; ASV fake target 70 per attack; voice-real sources equal; three large music-fake sources equal.',
                  solver='scipy.optimize.milp; optimal feasible solution required; integer group assignments and row counts',
                  historically_independent=False, prior_seen_selected=sum(r['prior_seen'] == 'True' for r in selected),
                  prior_seen_validation=sum(r['prior_seen'] == 'True' and r['split'] == 'val' for r in selected),
                  gpu_started=False, uploaded=False, source_audio_modified=False, train_val_identity_overlap=0,
                  audio_verification=audio_receipt,
                  csv_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.glob('*.csv')})
    if INPUT.read_bytes() != raw:
        raise SplitError('Original manifest changed during selection')
    (OUTPUT / 'split_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
