#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: python data_v4/prepare_candidate_review.py EXISTING_MANIFEST PAYLOAD_ROOT MLAAD_METADATA OUT [PILOT_SOURCES]
"""Build CPU review records and conservative lineage constraints, never training approval."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

from build_manifest import LABELS, components
from export_bank import COLUMNS, export


def enrich(row, metadata):
    """Retain existing keys and connect known reference clips, speakers, and PCM identities."""
    result = {**row, 'group_keys': list(row.get('group_keys', []))}
    keys = set(result['group_keys'])
    matched = metadata.get(row['ID'], {})
    for field in ('original_file', 'reference_speaker'):
        value = matched.get(field) or row.get(field)
        if value and str(value).casefold() not in ('none', 'nan', 'unknown'):
            result[field] = value
    for key in list(keys):
        if key.startswith(('librispeech:', 'speaker:librispeech_test-clean:')):
            keys.add('speaker:librispeech:' + key.rsplit(':', 1)[1])
    for field, prefix in (('pcm_sha256', 'pcm:'), ('sha256', 'bytes:')):
        if row.get(field):
            keys.add(prefix + row[field])
    original = result.get('original_file')
    if original:
        keys.add('mailabs_text:' + PurePosixPath(original).stem)
    reference = result.get('reference_speaker')
    if reference:
        parts = PurePosixPath(reference.replace('\\', '/')).parts
        if len(parts) >= 6 and parts[1] == 'by_book' and parts[-2] == 'wavs':
            keys.add('mailabs_text:' + PurePosixPath(reference).stem)
            keys.add('reference_speaker:mailabs:' + '/'.join(parts[:4]))
        elif len(parts) == 1 and row.get('generator'):
            keys.add('reference_voice:mlaad:' + row['generator'] + ':' + reference.casefold())
            result['reference_voice_cross_system_identity'] = 'unknown'
        else:
            result['reference_speaker_unresolved'] = True
    result['group_keys'] = sorted(keys)
    result['reference_metadata_matched'] = bool(matched)
    return result


def main(existing: Path, payload: Path, metadata_dir: Path, output: Path, pilot: Path | None = None) -> None:
    """Join verified candidates with historical components and write actionable review queues."""
    if output.exists():
        raise FileExistsError(output)
    inputs = [existing, *sorted(payload.glob('*/verified_sources.jsonl'))]
    if pilot:
        inputs.append(pilot)
    history = [json.loads(line) for line in existing.open()]
    candidates = []
    for path in inputs[1:]:
        for line in path.open():
            row = json.loads(line)
            member = PurePosixPath(row['path'])
            if member.is_absolute() or '..' in member.parts:
                raise ValueError(f'Candidate path escapes payload: {row["ID"]}')
            candidates.append({**row, 'collection_batch': path.parent.name,
                               'source_manifest': path.parent.name + '/' + path.name})
    ids = [r['ID'] for r in history + candidates]
    if len(set(ids)) != len(ids):
        raise ValueError('Historical and candidate IDs must be unique')
    wanted = set(ids)
    metadata = {}
    for path in sorted(metadata_dir.glob('*.csv')):
        inputs.append(path)
        with path.open() as stream:
            for row in csv.DictReader(stream, delimiter='|'):
                member = PurePosixPath(row['path'].removeprefix('./'))
                rid = 'mlaad:' + '/'.join(member.parts[1:]).removesuffix('.wav')
                if rid in wanted:
                    if rid in metadata and metadata[rid] != row:
                        raise ValueError(f'Conflicting source metadata: {rid}')
                    metadata[rid] = row
    history = [enrich(row, metadata) for row in history]
    candidates = [enrich(row, metadata) for row in candidates]
    comp = components(history + candidates)
    prior_roles, prior_ids = defaultdict(set), defaultdict(list)
    for row in history:
        prior_roles[comp[row['ID']]].add(row['role'])
        prior_ids[comp[row['ID']]].append(row['ID'])
    candidate_pcm = Counter(r.get('pcm_sha256') for r in candidates if r.get('pcm_sha256'))
    candidate_bytes = Counter(r['sha256'] for r in candidates)
    for row in candidates:
        cid = comp[row['ID']]
        roles = sorted(prior_roles[cid])
        blockers = ['listening_review_pending', 'rights_review_pending', 'role_assignment_pending']
        blockers.extend(row.get('quality_flags', []))
        if row.get('full_audio_decode_verified') is not True:
            blockers.append('full_decode_receipt_pending')
        if roles:
            blockers.append('prior_evaluation_component' if set(roles) - {'train'} else 'prior_training_component')
        if len(roles) > 1:
            blockers.append('historical_component_crosses_roles')
        if candidate_pcm[row.get('pcm_sha256')] > 1:
            blockers.append('within_candidate_pcm_duplicate')
        if candidate_bytes[row['sha256']] > 1:
            blockers.append('within_candidate_byte_duplicate')
        if row.get('reference_speaker_unresolved'):
            blockers.append('reference_speaker_unresolved')
        if str(row.get('generator', '')).startswith('preview-'):
            blockers.append('generator_identity_unresolved')
        if row.get('VOICE_PRESENT') is None or row.get('MUSIC_PRESENT') is None:
            blockers.append('component_presence_unknown')
        row.update(component_id=cid, prior_roles=roles, prior_linked_rows=len(prior_ids[cid]),
                   role='unassigned', training_admitted=False, sealed=False,
                   admission_blockers=sorted(set(blockers)))
    conflicts = [{'component_id': cid, 'prior_roles': sorted(roles),
                  'prior_rows': prior_ids[cid]}
                 for cid, roles in sorted(prior_roles.items()) if len(roles) > 1]
    changed = 0
    for row in history:
        cid = comp[row['ID']]
        row['component_id'] = cid
        if len(prior_roles[cid]) > 1:
            changed += row['role'] != 'development'
            row['role_before_reference_review'] = row['role']
            row['role'] = 'development'
            row['role_reason'] = 'reference_lineage_role_conflict'
            row['historical_exposure'] = sorted(set(row.get('historical_exposure', [])) | prior_roles[cid])
    conflict_exposure = defaultdict(set)
    for row in history:
        conflict_exposure[row['component_id']].update(row.get('historical_exposure', []))
    for row in history:
        if len(prior_roles[row['component_id']]) > 1:
            row['historical_exposure'] = sorted(conflict_exposure[row['component_id']])
    source_stats = {}
    for source in sorted({r['source'] for r in candidates}):
        rows = [r for r in candidates if r['source'] == source]
        duration = sorted(r['duration_seconds'] for r in rows)
        source_stats[source] = dict(files=len(rows), components=len({r['component_id'] for r in rows}),
            duration_quantiles_seconds=[duration[round(q*(len(duration)-1))] for q in (0, .25, .5, .75, 1)],
            codecs=dict(Counter(PurePosixPath(r['path']).suffix for r in rows)),
            linked_to_history=sum(bool(r['prior_roles']) for r in rows),
            reference_metadata_matched=sum(r['reference_metadata_matched'] for r in rows))
    report = dict(candidate_rows=len(candidates), prior_rows=len(history),
        candidate_components=len({r['component_id'] for r in candidates}),
        candidate_prior_role_counts=dict(Counter('|'.join(r['prior_roles']) or 'none' for r in candidates)),
        blockers=dict(Counter(b for r in candidates for b in r['admission_blockers'])),
        prior_reference_metadata_matched=sum(r['reference_metadata_matched'] for r in history),
        historical_conflicting_components=len(conflicts),
        historical_conflicting_rows=sum(len(r['prior_rows']) for r in conflicts),
        historical_roles_changed=changed, historical_safe_role_counts=dict(Counter(r['role'] for r in history)),
        sources=source_stats, label_unknown_counts={k: sum(r.get(k) is None for r in candidates) for k in LABELS},
        training_admitted=0, sealed=0, prior_manifest_modified=False,
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limitations=['Reference speaker grouping is conservative source provenance, not measured output identity.',
                     'English MLAAD metadata only; unmatched historical systems/languages remain unresolved.',
                     'No listening, rights approval, derived audio, model inference, or independent OOD admission.'])
    output.mkdir(parents=True)
    safe = output / 'roles-v4'
    safe.mkdir()
    (safe / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in history))
    splits = {r['ID']: 'train' if r['role'] == 'train' else 'val' if r['role'] in ('selection', 'calibration') else 'test'
              for r in history}
    with (safe / 'roles.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ID', 'component_id', 'role', 'historical_exposure'])
        writer.writerows((r['ID'], r['component_id'], r['role'], json.dumps(r.get('historical_exposure', []))) for r in history)
    with (safe / 'splits.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ID', 'split'])
        writer.writerows(splits.items())
    train, test, skipped = export(history, splits)
    for name, rows in [('bank_trainval.csv', train), ('bank_test.csv', test)]:
        with (safe / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    (safe / 'skipped.json').write_text(json.dumps(skipped, indent=2)+'\n')
    for name, rows in [('candidate_manifest.jsonl', candidates), ('historical_conflicts.jsonl', conflicts)]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    columns = ['ID', 'collection_batch', 'path', 'source', 'component_id', 'prior_roles',
               'duration_seconds', 'quality_flags', 'admission_blockers', 'review_decision']
    with (output / 'review_queue.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in sorted(candidates, key=lambda r: (-len(r['admission_blockers']), r['ID'])):
            item = {k: row.get(k, '') for k in columns}
            for k in ('prior_roles', 'quality_flags', 'admission_blockers'):
                item[k] = json.dumps(row.get(k, []))
            writer.writerow(item)
    (output / 'review_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256', 'sources')}))


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
