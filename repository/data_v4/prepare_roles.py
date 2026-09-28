# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: python -m data_v4.prepare_roles --help (existing project environment).
"""Create a new metadata/role namespace without changing audio or historical manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from data_v4.build_manifest import components


def prepare(rows: list[dict], old_split: dict[str, str], legacy: dict[str, dict],
            aliases: dict[str, str], seed: int = 20260924) -> tuple[list[dict], dict]:
    """Restore known identities; quarantine unresolved Echoes aliases and role conflicts."""
    ids = [r['ID'] for r in rows]
    if len(set(ids)) != len(ids) or set(old_split) != set(ids) or set(aliases) - set(ids):
        raise ValueError('Duplicate IDs or unmatched split/alias IDs')
    result = []
    unresolved = set()
    for original in rows:
        row = {**original, 'group_keys': list(original['group_keys'])}
        rid = row['ID']
        if rid not in old_split:
            raise ValueError(f'Historical split missing: {rid}')
        if row['source'] == 'echoes':
            parts = Path(rid.split(':', 1)[1]).parts
            positions = [i for i, x in enumerate(parts[:-1]) if x.upper() in ('TTA', 'ATA')]
            if len(positions) != 1 or positions[0] + 2 >= len(parts):
                raise ValueError(f'Unknown Echoes path: {rid}')
            i = positions[0]
            mode, generator = parts[i].upper(), parts[i + 1]
            reference = Path(parts[-1]).stem.rsplit(f'_{generator}_{mode}_', 1)[0]
            row.update(generator=generator, generation_mode=mode)
            row['group_keys'].append('echoes_reference:' + reference.casefold())
        old = legacy.get(rid.removeprefix(row['source'] + ':'), {}) if row['source'].startswith('existing:') else {}
        row['legacy_catalog_matched'] = bool(old) if row['source'].startswith('existing:') else None
        row['group_keys'].extend(x for x in old.get('group_id', '').split('|') if x)
        if rid in aliases:
            row['canonical_source_id'] = aliases[rid]
        elif 'echoes' in row['source'].casefold():
            unresolved.add(rid)
        # Existing codec output needs waveform alignment verification before local supervision.
        if row.get('fake_voice_segments'):
            row.setdefault('interval_alignment_verified', False)
            row.setdefault('voice_intervals_complete', False)
        row['group_keys'] = sorted(set(row['group_keys']))
        result.append(row)
    comp = components(result)
    groups = defaultdict(list)
    for row in result:
        groups[comp[row['ID']]].append(row)
    conflicts, held_aliases = 0, 0
    for component, members in groups.items():
        splits = {old_split[row['ID']] for row in members}
        alias_hold = any(row['ID'] in unresolved for row in members)
        if len(splits) > 1 or alias_hold:
            role = 'development'
            conflicts += len(splits) > 1
            held_aliases += alias_hold
        elif splits == {'train'}:
            role = 'train'
        elif splits == {'test'}:
            role = 'development'
        elif splits == {'val'}:
            bit = int(hashlib.sha256(f'{seed}|{component}'.encode()).hexdigest(), 16) % 2
            role = 'selection' if bit == 0 else 'calibration'
        else:
            raise ValueError(f'Unknown historical splits: {splits}')
        usage = set()
        for split in splits:
            usage.update({'train': ('head_optimizer', 'presence_fit'),
                          'val': ('head_selection', 'cdf_fit', 'presence_analysis'),
                          'test': ('development_analysis',)}[split])
        exposure = sorted(usage | splits | {str(row.get('prior_seen')) for row in members if row.get('prior_seen')})
        for row in members:
            row.update(role=role, component_id=component, historical_exposure=exposure,
                       historical_split=old_split[row['ID']],
                       role_reason='unresolved_echoes_alias' if alias_hold else 'cross_split_component' if len(splits) > 1 else 'historical_assignment')
    report = {'legacy_catalog_matched': sum(r['legacy_catalog_matched'] is True for r in result),
              'legacy_catalog_unmatched': sum(r['legacy_catalog_matched'] is False for r in result), 'rows': len(result), 'components': len(groups), 'role_counts': dict(Counter(r['role'] for r in result)),
              'role_source_counts': dict(Counter(f"{r['role']}|{r['source']}" for r in result)),
              'conflicting_components_quarantined': conflicts, 'alias_components_quarantined': held_aliases,
              'unresolved_echoes_rows': len(unresolved), 'sealed': 'unavailable',
              'voice_interval_rows': sum(bool(r.get('fake_voice_segments')) for r in result),
              'voice_alignment_pending': sum(bool(r.get('fake_voice_segments')) and not r['interval_alignment_verified'] for r in result),
              'echoes_generators': dict(Counter(r['generator'] for r in result if r['source'] == 'echoes')),
              'seed': seed, 'audio_changed': False, 'cache_promoted': False}
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'splits', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--legacy-catalog', type=Path)
    parser.add_argument('--aliases', type=Path, help='Reviewed JSON mapping original ID -> canonical source ID')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output namespace')
    with args.manifest.open() as stream:
        rows = [json.loads(line) for line in stream]
    with args.splits.open() as stream:
        split = {r['ID']: r['split'] for r in csv.DictReader(stream)}
    legacy = {}
    if args.legacy_catalog:
        with args.legacy_catalog.open() as stream:
            legacy = {r['ID']: r for r in csv.DictReader(stream)}
    aliases = json.loads(args.aliases.read_text()) if args.aliases else {}
    rows, report = prepare(rows, split, legacy, aliases)
    args.out.mkdir(parents=True)
    (args.out / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    with (args.out / 'roles.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ID', 'component_id', 'role', 'historical_split', 'historical_exposure', 'fold'])
        writer.writerows((r['ID'], r['component_id'], r['role'], r['historical_split'], json.dumps(r['historical_exposure']), '') for r in rows)
    with (args.out / 'splits.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ID', 'split'])
        writer.writerows((r['ID'], 'train' if r['role'] == 'train' else 'val' if r['role'] in ('selection', 'calibration') else 'test') for r in rows)
    report['inputs'] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                        [('manifest', args.manifest), ('splits', args.splits)] +
                        ([('legacy_catalog', args.legacy_catalog)] if args.legacy_catalog else []) +
                        ([('aliases', args.aliases)] if args.aliases else [])}
    (args.out / 'roles_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'role_source_counts'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
