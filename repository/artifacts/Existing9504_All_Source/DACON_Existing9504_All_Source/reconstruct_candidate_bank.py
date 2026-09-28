"""Rebuild a provisional group-disjoint bank from verified acquired originals; CPU only.

Run: .venv/bin/python reconstruct_candidate_bank.py
Completed inputs are snapshotted. Original manifests and raw audio are never changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from candidate_data_acquire import BANK, OUT, SourceFile, verify_file
from full_bank_split import ROOT, components, identities, write_csv
from expanded_data import SplitError

OUTPUT = ROOT / 'results/reconstructed_bank_20260919'
OLD = ROOT / 'results/full_bank_mil_8020_20260918/all_sources.csv'
LABELS = ('FILE_FAKE', 'VOICE_FAKE', 'MUSIC_FAKE', 'VOICE_PRESENT', 'MUSIC_PRESENT')
SPLITS = ('train', 'val', 'test')
RATIOS = (.7, .15, .15)
SEED = 'reconstructed-acquired-20260919-v1'


def rank(value):
    return hashlib.sha256((SEED + ':' + value).encode()).hexdigest()


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def receipts():
    result = {}
    for path in [OUT / 'receipts.jsonl', *OUT.glob('*_receipts.jsonl')]:
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('status') == 'original_checksum_verified':
                    result[row['path']] = row
    return result


def audio_quality(path):
    """Decode every frame, reject corrupt/non-finite/silent sources, record native PCM."""
    frames, energy, peak, digest = 0, 0., 0., hashlib.sha256()
    with sf.SoundFile(path) as stream:
        sr, channels = stream.samplerate, stream.channels
        digest.update(f'soundfile-f32le-interleaved:{sr}:{channels}:'.encode())
        for block in stream.blocks(blocksize=262144, dtype='float32', always_2d=True):
            if not np.isfinite(block).all():
                raise SplitError('Non-finite audio: ' + str(path))
            frames += len(block)
            energy += float(np.sum(block.astype(np.float64) ** 2))
            peak = max(peak, float(np.abs(block).max(initial=0)))
            digest.update(block.astype('<f4', copy=False).tobytes())
    if frames / sr < .25 or energy <= 1e-12:
        raise SplitError('Empty/too short/silent audio: ' + str(path))
    return dict(duration_seconds=str(frames / sr), sample_rate=str(sr), channels=str(channels),
                frames=str(frames), peak=str(peak), rms=str((energy / (frames * channels)) ** .5),
                group_pcm='pcm_soundfile:' + digest.hexdigest(), decode_status='full_decode_pass', finite='True')


def haim_rows(completed):
    """Join source parents before grouping; no acoustic presence labels are guessed."""
    mtg = {}
    with (OUT / 'metadata/mtg_raw.tsv').open() as stream:
        for fields in csv.reader(stream, delimiter='\t'):
            if fields and fields[0].startswith('track_'):
                mtg[str(int(fields[0][6:]))] = str(int(fields[1][7:]))
    licenses = {}
    for block in (OUT / 'metadata/mtg_audio_licenses.txt').read_text().strip().split('\n\n'):
        lines = block.splitlines()
        if len(lines) >= 3:
            licenses[str(int(Path(lines[0]).stem))] = lines[-1].rsplit(' ', 1)[-1].strip()
    jobs, ledger = [], []
    for name, row in sorted(completed.items()):
        if name.startswith('HAIM/') and name.endswith('.mp3'):
            meta = str(Path(name).with_suffix('.json'))
            if meta not in completed:
                ledger.append(dict(path=name, admission='pending', reason='verified_sidecar_missing'))
            else:
                jobs.append((row, completed[meta]))

    def convert(pair):
        receipt, meta_receipt = pair
        name = receipt['path']
        path, meta_path = BANK / name, BANK / meta_receipt['path']
        for rec, file in ((receipt, path), (meta_receipt, meta_path)):
            job = SourceFile(**{key: rec[key] for key in SourceFile.__dataclass_fields__})
            if verify_file(file, job) != rec['sha256']:
                raise SplitError('Receipt hash changed: ' + name)
        metadata = json.loads(meta_path.read_text())
        quality = audio_quality(path)
        groups, parents, flags, intervals = {quality.pop('group_pcm')}, [], [], []
        if '/A2_fake/lyria-pro3/' in name:
            generator, label, kind = 'lyria-pro3', '1', 'generated_music'
            prompt = ' '.join(metadata.get('prompt', '').casefold().split())
            if prompt:
                groups.add('prompt:haim:lyria-pro3:' + hashlib.sha256(prompt.encode()).hexdigest())
        elif '/B1_ai_mastered_human__MTG_sonicmaster/' in name:
            generator, label, kind = 'sonicmaster_real', '0', 'mastered_real_music'
            parents = [('human', metadata['input_source'])]
        elif '/C1_mixset_concat/' in name or '/C2_mixset_crossfade/' in name:
            generator = 'concat' if '/C1_' in name else 'crossfade'
            label, kind = '1', 'partial_music'
            parents = [(s['type'], s['source']) for s in metadata['segments']]
            for segment in metadata['segments']:
                a, b = float(segment['output_start']), float(segment['output_end'])
                if not 0 <= a < b <= float(quality['duration_seconds']) + .15:
                    flags.append('segment_outside_decoded_timeline')
                intervals.append(dict(start=a, end=b, fake=int(segment['type'] == 'ai')))
            flags.append('matched_real_sham_not_ready')
        else:
            raise SplitError('Unreviewed HAIM category: ' + name)
        original_licenses = []
        for origin, filename in parents:
            number = str(int(Path(filename).stem))
            if origin == 'human':
                groups.add('reference:jamendo:' + number)
                if number not in mtg or number not in licenses:
                    flags.append('missing_mtg_parent_identity_or_license')
                else:
                    groups.add('artist:jamendo:' + mtg[number])
                    original_licenses.append(licenses[number])
            elif origin == 'ai':
                groups.add('reference:haim:acestep_numeric:' + number)
            else:
                raise SplitError('Unknown parent type: ' + origin)
        result = dict(ID='new_haim_' + generator + '_' + path.stem, source='HAIM', generator=generator,
                      path=str(path.relative_to(ROOT)), source_sha256=receipt['sha256'],
                      source_bytes=str(receipt['size']), metadata_sha256=meta_receipt['sha256'],
                      metadata=json.dumps(metadata, ensure_ascii=False), source_url=receipt['url'],
                      source_revision=receipt['url'].split('/resolve/')[1].split('/')[0],
                      group_id='|'.join(sorted(groups)), parent_ids='[]', route='music', kind=kind,
                      FILE_FAKE=label, MUSIC_FAKE=label, MUSIC_PRESENT='1', VOICE_FAKE='', VOICE_PRESENT='',
                      license=receipt['license'], license_url='https://creativecommons.org/licenses/by-nc/4.0/',
                      parent_licenses=json.dumps(sorted(set(original_licenses))),
                      input_kind='original_full_duration', new_collection='True', split='',
                      prior_seen='False', review_flags=json.dumps(sorted(set(flags))),
                      admission='diagnostic' if flags else 'admitted',
                      reason=';'.join(sorted(set(flags))), event_intervals=json.dumps(intervals), **quality)
        for field in LABELS:
            result[field + '_mask'] = str(int(result[field] in ('0', '1')))
        return result

    def checked(pair):
        try:
            return convert(pair)
        except (SplitError, RuntimeError, sf.LibsndfileError, KeyError, ValueError) as error:
            receipt = pair[0]
            return dict(ID='rejected_' + rank(receipt['path']), path=str((BANK / receipt['path']).relative_to(ROOT)),
                        source='HAIM', generator='unverified', source_sha256=receipt['sha256'],
                        group_id='', new_collection='True', admission='excluded', reason='quality_or_metadata_error:' + str(error))

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, row in enumerate(pool.map(checked, jobs), 1):
            ledger.append(row)
            if index % 100 == 0:
                print('decoded and parent-linked', index, '/', len(jobs), flush=True)
    return ledger


def checked_extra_catalog(path, completed):
    """Bind a full-decode catalog to its audit and verified archive/protocol receipts."""
    audit = json.loads(path.with_name('audit.json').read_text())
    if sha(path) != audit['catalog_sha256']:
        raise SplitError('Additional catalog changed after audit: ' + str(path))
    rows = list(csv.DictReader(path.open()))
    if len(rows) != audit['actual_audio_rows'] or len({r['ID'] for r in rows}) != len(rows):
        raise SplitError('Additional catalog row accounting mismatch')
    for suffix, field in (('_set.zip', 'archive_sha256'), ('.txt', 'protocol_sha256')):
        name = 'CtrSVDD/' + audit['source_split'] + suffix
        receipt = completed[name]
        job = SourceFile(**{key: receipt[key] for key in SourceFile.__dataclass_fields__})
        if verify_file(BANK / name, job) != receipt['sha256'] or any(r[field] != receipt['sha256'] for r in rows):
            raise SplitError('Additional catalog original binding mismatch: ' + name)
    for row in rows:
        if row['source'] != 'CtrSVDD' or row['new_collection'] != 'True':
            raise SplitError('Unreviewed additional corpus')
        if row['admission'] == 'admitted':
            if (row['decode_status'] != 'full_decode_pass' or row['zip_crc_verified'] != 'True'
                    or row['generator'] not in ('REAL', 'A01', 'A02', 'A03', 'A04', 'A05')
                    or not row['path'] or 'singer:ctrsvdd:' + row['target_singer_id'] not in row['group_id'].split('|')
                    or row['license'] != 'CC-BY-NC-ND-4.0'):
                raise SplitError('Additional corpus admission gate failed: ' + row['ID'])
    return rows


def assemble(extra_catalogs=()):
    completed = receipts()
    new = haim_rows(completed)
    for path in extra_catalogs:
        new.extend(checked_extra_catalog(path, completed))
    old = list(csv.DictReader(OLD.open()))
    previous = {r['ID'] for r in csv.DictReader((ROOT / 'results/balanced7144_20260919/selected.csv').open())}
    for row in old:
        row.update(previous_split=row['split'], split='', new_collection='False')
        row['historical_development_exposure'] = str(row['ID'] in previous or row.get('prior_seen') == 'True')
        if row['source'] == 'constructed_music':
            row.update(admission='diagnostic', reason='legacy_mixture_parent_review')
        if row['source'] == 'HAIM' and row['generator'] == 'acestep':
            track = json.loads(row['metadata'])['track_id']
            row['group_id'] += '|reference:haim:acestep_numeric:' + str(int(track))
        for field in LABELS:
            row[field + '_mask'] = str(int(row.get(field) in ('0', '1') and row.get(field + '_mask') != '0'))
    ledger = old + [r for r in new if r.get('ID')]
    if len({r['ID'] for r in ledger}) != len(ledger):
        raise SplitError('Combined catalog repeats IDs')
    components(ledger)  # Diagnostic derivatives also connect all contributing originals.
    exposed = {r['component_id'] for r in ledger if r.get('historical_development_exposure') == 'True'}
    seen, admitted = {}, []
    for row in ledger:
        row['historical_development_exposure'] = str(row['component_id'] in exposed)
        if row['admission'] != 'admitted':
            continue
        digest = row['source_sha256']
        if digest in seen:
            row.update(admission='excluded', reason='exact_bytes_duplicate', duplicate_of=seen[digest])
            continue
        seen[digest] = row['ID']
        admitted.append(row)
    return admitted, ledger, completed


def cell(row):
    return row['route'], row['FILE_FAKE'], row['source'], row['generator']


def select(rows):
    """Generalize the existing pattern-count MILP to three group-disjoint partitions."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['component_id']].append(row)
    patterns = defaultdict(list)
    for group, members in groups.items():
        patterns[tuple(sorted(Counter(cell(r) for r in members).items()))].append(group)
    ordered = sorted(patterns)
    edges = [(p, c, n) for p, pattern in enumerate(ordered) for c, n in pattern]
    cells = sorted({cell(r) for r in rows})
    capacity = Counter(cell(r) for r in rows)
    labels = Counter(r['FILE_FAKE'] for r in rows)
    per_class = min(labels['0'], labels['1'])
    quotas = (round(per_class * .7), round(per_class * .15), 0)
    quotas = (*quotas[:2], per_class - sum(quotas[:2]))
    P, E, C = len(ordered), len(edges), len(cells)
    size = 3 * (P + E) + 6 * C + 24
    upper, cost = np.full(size, np.inf), np.zeros(size)
    integer = np.ones(size)
    integer[3 * (P + E):] = 0
    rr, cc, vv, low, high = [], [], [], [], []
    def constraint(terms, lower, higher):
        for column, value in terms:
            rr.append(len(low)); cc.append(column); vv.append(value)
        low.append(lower); high.append(higher)
    def group_var(s, p): return s * P + p
    def row_var(s, e): return 3 * P + s * E + e
    route_capacity = Counter((r['route'], r['FILE_FAKE']) for r in rows)
    limiting_label = min(('0', '1'), key=lambda label: labels[label])
    route_targets = {}
    for label in ('0', '1'):
        voice = min(route_capacity['voice', label], max(per_class - route_capacity['music', label],
                    route_capacity['voice', limiting_label]))
        route_targets['voice', label] = voice
        route_targets['music', label] = per_class - voice
    for p, pattern in enumerate(ordered):
        count = len(patterns[pattern])
        for s in range(3):
            upper[group_var(s, p)] = count
        constraint([(group_var(s, p), 1) for s in range(3)], count, count)
    for e, (p, c, count) in enumerate(edges):
        for s in range(3):
            constraint([(row_var(s, e), 1), (group_var(s, p), -count)], -np.inf, 0)
    # Preserve scarce route/class examples before subsampling abundant fake music.
    for route_label, target in route_targets.items():
        constraint([(row_var(s, e), 1) for s in range(3) for e, (_, c, _) in enumerate(edges)
                    if c[:2] == route_label], target, target)
    for s in range(3):
        for label in ('0', '1'):
            constraint([(row_var(s, e), 1) for e, (_, c, _) in enumerate(edges) if c[1] == label], quotas[s], quotas[s])
        for k, current in enumerate(cells):
            terms = [(row_var(s, e), 1) for e, (_, c, _) in enumerate(edges) if c == current]
            # Proportional source retention, not equal weight for tiny source cells.
            desired = capacity[current] * quotas[s] / labels[current[1]]
            plus = 3 * (P + E) + 2 * (s * C + k)
            constraint(terms + [(plus, -1), (plus + 1, 1)], desired, desired)
            cost[plus:plus + 2] = 1 / max(10, desired)
            if sum(len(gs) for pattern, gs in patterns.items() if any(c == current for c, n in pattern)) >= 3:
                constraint(terms, 1, np.inf)
        for route in ('voice', 'music'):
            for label in ('0', '1'):
                terms = [(row_var(s, e), 1) for e, (_, c, _) in enumerate(edges) if c[:2] == (route, label)]
                constraint(terms, 1, np.inf)
                desired = route_targets[route, label] * quotas[s] / per_class
                k = s * 4 + ('voice', 'music').index(route) * 2 + int(label)
                plus = 3 * (P + E) + 6 * C + 2 * k
                constraint(terms + [(plus, -1), (plus+1, 1)], desired, desired)
                cost[plus:plus+2] = 50 / max(10, desired)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(low), size)).tocsc()
    result = milp(cost, integrality=integer, bounds=Bounds(np.zeros(size), upper),
                  constraints=LinearConstraint(matrix, low, high), options={'time_limit': 180, 'mip_rel_gap': .005})
    if result.x is None:
        raise SplitError('Requested balanced split is infeasible: ' + result.message)
    counts = np.rint(result.x[:3 * (P + E)]).astype(int)
    selected, assignments = [], {}
    for p, pattern in enumerate(ordered):
        available = sorted(patterns[pattern], key=rank)
        offset = 0
        for s, split in enumerate(SPLITS):
            chosen = available[offset:offset + counts[group_var(s, p)]]
            offset += len(chosen)
            assignments.update({g: split for g in chosen})
            for e, (pi, current, _) in enumerate(edges):
                if pi != p:
                    continue
                candidates = sorted((r for g in chosen for r in groups[g] if cell(r) == current), key=lambda r:rank(r['ID']))
                selected.extend(dict(r, split=split) for r in candidates[:counts[row_var(s, e)]] )
    return sorted(selected, key=lambda r:r['ID']), assignments, dict(per_class=per_class, per_split_per_class=dict(zip(SPLITS, quotas)),
        route_class_targets={r+'_'+c:n for (r,c),n in route_targets.items()}, solver_message=result.message, solver_gap=float(result.mip_gap))


def verify(selected, ledger, accounting):
    reference = {r['ID']: r for r in ledger}
    if len({r['ID'] for r in selected}) != len(selected):
        raise SplitError('Duplicate selected IDs')
    wanted = {(s, c): n for s, n in accounting['per_split_per_class'].items() for c in ('0', '1')}
    if Counter((r['split'], r['FILE_FAKE']) for r in selected) != wanted:
        raise SplitError('Incorrect split or class counts')
    if 'route_class_targets' in accounting and Counter(r['route']+'_'+r['FILE_FAKE'] for r in selected) != accounting['route_class_targets']:
        raise SplitError('Scarce route/class examples lost during balancing')
    observed = {}
    for row in selected:
        for key in identities(row) | {row['component_id']}:
            if observed.setdefault(key, row['split']) != row['split']:
                raise SplitError('Cross-partition parent/group: ' + key)
        if row['admission'] != 'admitted':
            raise SplitError('Diagnostic-only source selected')
        for field in LABELS:
            if row.get(field) != reference[row['ID']].get(field):
                raise SplitError('Source label changed')
            if row[field + '_mask'] == '1' and row.get(field) not in ('0', '1'):
                raise SplitError('Unknown label unmasked')


def main(extra_catalogs=()):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'selected.csv').exists():
        raise SplitError('Snapshot exists; preserve it and choose a new OUTPUT before rebuilding')
    rows, ledger, completed = assemble(extra_catalogs)
    write_csv(OUTPUT / 'combined_catalog.csv', ledger)
    selected, assignments, accounting = select(rows)
    verify(selected, ledger, accounting)
    from balanced_bank_split import check_audio
    receipt = check_audio(selected)
    chosen = {r['ID']:r['split'] for r in selected}
    for row in ledger:
        row['assigned_group_split'] = assignments.get(row['component_id'], '')
        row['split'] = chosen.get(row['ID'], '')
        row['selected'] = str(row['ID'] in chosen)
        if row['admission'] == 'admitted' and row['ID'] not in chosen:
            row['reason'] = 'majority_class_balance_reserve'
    write_csv(OUTPUT / 'combined_catalog.csv', ledger)
    write_csv(OUTPUT / 'selected.csv', selected)
    for split in SPLITS:
        write_csv(OUTPUT / (split + '.csv'), [r for r in selected if r['split'] == split])
    counts = Counter((*cell(r), r['split'], r['new_collection']) for r in selected)
    write_csv(OUTPUT / 'composition.csv', [dict(route=k[0], label=k[1], source=k[2], generator=k[3], split=k[4], new_collection=k[5], rows=str(n)) for k,n in sorted(counts.items())])
    plans = [row for plan in ('haim_plan.json','zenodo_plan.json') for row in json.loads((OUT / plan).read_text())]
    pending = [dict(path=r['path'], reason='download_or_upstream_checksum_pending') for r in plans if r['path'] not in completed]
    write_csv(OUTPUT / 'pending_collection.csv', pending)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='provisional_cpu_reconstruction',
                  gpu_started=False, training_ready=False, total_cap=None, requested_ratios=RATIOS,
                  original_manifest_sha256=sha(OLD), selected_sha256=sha(OUTPUT / 'selected.csv'),
                  extra_catalogs={str(p):sha(p) for p in extra_catalogs},
                  combined_rows=len(ledger), admitted_pool_rows=len(rows), selected_rows=len(selected),
                  new_acquired_catalog_rows=sum(r['new_collection']=='True' for r in ledger),
                  new_acquired_selected_rows=sum(r['new_collection']=='True' for r in selected),
                  pending_download_files=len(pending), split_class_counts=accounting, selected_audio_integrity=receipt,
                  largest_connected_group=max(Counter(r['component_id'] for r in ledger).values()),
                  historical_exposure_by_split={s:sum(r['historical_development_exposure']=='True' for r in selected if r['split']==s) for s in SPLITS},
                  independent_test_claim=False, gaps=['remaining_corpora_and_svc_parent_review',
                  'cross_singer_song_or_score_identity_not_fully_resolved',
                  'component_voice_labels_unknown_for_music', 'partial_music_requires_matched_real_sham',
                  'A100_16h_runtime_not_measured', 'training_loader_and_notebook_not_switched'],
                  diagnostic_rows_by_reason=dict(Counter(r['reason'] for r in ledger if r['admission']!='admitted')))
    (OUTPUT / 'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT, help='New snapshot directory; existing snapshots are preserved')
    parser.add_argument('--extra-catalog', type=Path, action='append', default=[])
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    main([p.resolve() for p in args.extra_catalog])
