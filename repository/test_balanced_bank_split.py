"""Protect real-bank source balance, parent isolation, and reproducible selection."""
from collections import Counter
import csv
from pathlib import Path

from balanced_bank_split import select
from full_bank_split import identities


def test_real_bank_selection_preserves_groups_labels_and_source_ratios():
    # Given: actual admitted sources, including ASV speaker graphs and mixed parents.
    path = Path(__file__).parent / 'results/full_bank_mil_8020_20260918/all_sources.csv'
    with path.open(newline='') as stream:
        original = list(csv.DictReader(stream))
    reference = {r['ID']: dict(r) for r in original}

    # When: select the requested bank without using predictions or changing originals.
    selected = select(original)

    # Then: exact user totals and class balance coexist with group-safe source balance.
    label = lambda r: r[r['route'].upper() + '_FAKE']
    assert Counter((r['split'], label(r)) for r in selected) == {
        ('train', '0'): 2858, ('train', '1'): 2858,
        ('val', '0'): 714, ('val', '1'): 714,
    }
    assert len({r['ID'] for r in selected}) == 7144
    observed = {}
    cells = Counter()
    for row in selected:
        source = reference[row['ID']]
        assert row['source'] != 'constructed_music'
        for name in ('path', 'source_sha256', 'VOICE_PRESENT', 'MUSIC_PRESENT',
                     'VOICE_FAKE', 'MUSIC_FAKE', 'VOICE_FAKE_mask', 'MUSIC_FAKE_mask'):
            assert row[name] == source[name], (row['ID'], name)
        for key in identities(row) | {row['component_id']}:
            assert observed.setdefault(key, row['split']) == row['split'], key
        cells[(row['route'], label(row), row['source'], row['generator'], row['split'])] += 1
    strata = {key[:-1] for key in cells}
    expected = {(r['route'], label(r), r['source'], r['generator'])
                for r in original if r['source'] != 'constructed_music'}
    assert strata == expected
    for cell in strata:
        train, val = cells[(*cell, 'train')], cells[(*cell, 'val')]
        assert train > 0 and val > 0, cell
        assert abs(train - 4 * val) <= 2, (cell, train, val)
    assert original == list(reference.values())
    assert [(r['ID'], r['split']) for r in select(list(reversed(original)))] == [
        (r['ID'], r['split']) for r in selected]
