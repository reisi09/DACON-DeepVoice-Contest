"""Prepare CPU-only candidate crops, then admit source labels after hash-bound AST triage.

AST scores are automated semantic evidence, never a human listening verdict.
Final CSV audio paths are relative to the prepared audio root (and the existing bank root).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2_data import decode_audio, pcm_sha256, sha256_file, write_wav
from data_v4.build_manifest import components
from data_v4.export_bank import COLUMNS, export, route

SAMPLES = 15 * 16000
TOLERANCE = .5 / 32768 + 1e-7
QUANTIZATION_SCALE = 32767 / 32768
AST_MODEL_SHA256 = 'ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d'
RESOLVED_BY_POLICY = {'listening_review_pending', 'rights_review_pending',
                      'role_assignment_pending', 'component_presence_unknown', 'prior_training_component'}


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).open()]
    if len({r['ID'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate IDs: {path}')
    return rows


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def inside(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
        raise ValueError('Audio path escapes its declared root')
    return path


def prepare(history, candidates, payload, pilot, output, rights):
    history, candidates, payload, pilot, output, rights = map(Path, (history, candidates, payload, pilot, output, rights))
    old, rows = read_rows(history), read_rows(candidates)
    if {r['ID'] for r in old} & {r['ID'] for r in rows}:
        raise ValueError('Candidate IDs overlap history')
    grants = json.loads(rights.read_text())
    grants = grants.get('sources', grants)
    input_hashes = {str(p.resolve()): sha256_file(p) for p in (history, candidates, rights)}
    comp = components(old + rows)
    previous = defaultdict(set)
    for row in old:
        previous[comp[row['ID']]].add(row['role'])
    byte_counts = Counter(r['sha256'] for r in rows)
    pcm_counts = Counter(r.get('pcm_sha256') for r in rows)
    output.mkdir(parents=True, exist_ok=False)
    normalized, dispositions = [], []
    for row in rows:
        reasons = list(row.get('quality_flags', []))
        reasons += sorted(set(row.get('admission_blockers', [])) - RESOLVED_BY_POLICY)
        if grants.get(row['source'], {}).get('training_allowed') is not True:
            reasons.append('rights_not_allowed')
            grant_reason = grants.get(row['source'], {}).get('reason', grants.get(row['source'], {}).get('status'))
            if grant_reason:
                reasons.append('rights:' + str(grant_reason))
        if (set(row.get('prior_roles', [])) | previous[comp[row['ID']]]) - {'train'}:
            reasons.append('prior_evaluation_component')
        if row.get('reference_speaker_unresolved') or str(row.get('generator', '')).startswith('preview-'):
            reasons.append('source_identity_unresolved')
        if byte_counts[row['sha256']] > 1 or pcm_counts[row.get('pcm_sha256')] > 1:
            reasons.append('candidate_duplicate')
        if row.get('full_audio_decode_verified') is not True or not row.get('pcm_sha256'):
            reasons.append('source_decode_unverified')
        if route(row) is None:
            reasons.append('no_known_head_label')
        if float(row['duration_seconds']) < 15:
            reasons.append('shorter_than_15_seconds')
        disposition = dict(ID=row['ID'], source=row['source'], training_admitted=False,
                           listening_verified=False, excluded_reason=sorted(set(reasons)))
        if reasons:
            disposition['status'] = 'excluded'
            dispositions.append(disposition)
            continue
        root = pilot if row['collection_batch'] == pilot.name else inside(payload, row['collection_batch'])
        source = inside(root, row['path'])
        if sha256_file(source) != row['sha256'] or source.stat().st_size != row['bytes']:
            raise ValueError(f'Original bytes changed: {row["ID"]}')
        audio = decode_audio(source, limit_seconds=None)
        if pcm_sha256(audio) != row['pcm_sha256']:
            raise ValueError(f'Original decoded PCM changed: {row["ID"]}')
        if len(audio) < SAMPLES:
            raise ValueError(f'Original duration differs from verified manifest: {row["ID"]}')
        key = hashlib.sha256(row['ID'].encode()).hexdigest()
        start = int(key, 16) % (len(audio) - SAMPLES + 1)
        crop = audio[start:start + SAMPLES]
        peak = float(np.abs(crop).max())
        gain = min(1., .97 / peak) if peak else 1.
        expected = crop * gain
        relative = Path('audio/additional-v2') / (key + '.wav')
        target = inside(output, relative)
        write_wav(target, expected)
        decoded = decode_audio(target, limit_seconds=None)
        error = float(np.max(np.abs(decoded - expected * QUANTIZATION_SCALE))) if len(decoded) == SAMPLES else float('inf')
        if not np.isfinite(error) or error > TOLERANCE or sha256_file(source) != row['sha256']:
            raise ValueError(f'Derivation changed source or exceeded quantization tolerance: {row["ID"]}')
        result = dict(row, path=str(relative), sha256=sha256_file(target), bytes=target.stat().st_size,
            pcm_sha256=pcm_sha256(decoded), duration_seconds=15., crop_len_s=15., crop_start_s=start / 16000,
            crop_start_sample=start, sample_rate=16000, channels=1, codec_chain='pcm_s16le_16k_mono',
            gain=gain, quantization_scale=QUANTIZATION_SCALE, total_gain=gain * QUANTIZATION_SCALE, peak=float(np.abs(decoded).max()), clipping_fraction=float((np.abs(decoded) >= .999).mean()),
            original_source_sha256=row['sha256'], original_source_pcm_sha256=row['pcm_sha256'],
            original_source_bytes=row['bytes'], original_source_path=row['path'], original_source_root=str(root.resolve()),
            original_source_duration_seconds=len(audio) / 16000, reviewed_component_id=row['component_id'],
            derivation_max_abs_error=error, full_audio_decode_verified=True, quality_flags=[],
            canonical_pcm='ffmpeg mono 16000Hz float32 full duration', role='unassigned', training_admitted=False,
            listening_verified=False, admission_blockers=['automated_semantic_review_pending'],
            label_review='publisher_source_labels; unknown components remain masked', rights_decision=grants[row['source']])
        normalized.append(result)
        disposition['status'] = 'pending_AST'
        dispositions.append(disposition)
        if len(normalized) % 100 == 0:
            print(json.dumps({'prepared': len(normalized), 'processed': len(dispositions)}), flush=True)
    write_rows(output / 'normalized_candidates.jsonl', normalized)
    write_rows(output / 'dispositions.jsonl', dispositions)
    outputs = {name: sha256_file(output / name) for name in ('normalized_candidates.jsonl', 'dispositions.jsonl')}
    receipt = dict(input_sha256=input_hashes, history=str(history.resolve()), output_sha256=outputs,
        candidate_count=len(rows), prepared_count=len(normalized), excluded_count=len(rows)-len(normalized),
        gpu_used=False, listening_verified=False, transform='deterministic 15 s crop; peak cap .97; mono 16 kHz PCM16',
        max_abs_tolerance=TOLERANCE, quantization_scale=QUANTIZATION_SCALE, admission_threshold=.5)
    write_json(output / 'prepare_receipt.json', receipt)
    return receipt


def finalize(prepared, reviews, output):
    prepared, reviews, output = map(Path, (prepared, reviews, output))
    receipt = json.loads((prepared / 'prepare_receipt.json').read_text())
    for path, expected in receipt['input_sha256'].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError('Preparation input changed')
    for name, expected in receipt['output_sha256'].items():
        if sha256_file(inside(prepared, name)) != expected:
            raise ValueError('Prepared record changed')
    history = read_rows(receipt['history'])
    candidates = read_rows(prepared / 'normalized_candidates.jsonl')
    decisions = read_rows(prepared / 'dispositions.jsonl')
    by_id = {r['ID']: r for r in decisions}
    review = {r['ID']: r for r in read_rows(reviews)}
    if set(review) != {r['ID'] for r in candidates}:
        raise ValueError('Review coverage does not match prepared candidates')
    admitted = []
    for row in candidates:
        path = inside(prepared, row['path'])
        source = inside(row['original_source_root'], row['original_source_path'])
        if sha256_file(path) != row['sha256'] or sha256_file(source) != row['original_source_sha256']:
            raise ValueError(f'Prepared/source audio changed: {row["ID"]}')
        result = review.get(row['ID'])
        reasons = []
        if result is None:
            reasons.append('automated_review_missing')
        else:
            if result['audio_sha256'] != row['sha256'] or result.get('review_pcm_sha256') != row['pcm_sha256']:
                raise ValueError(f'Review audio hash mismatch: {row["ID"]}')
            if (result.get('review_model') != 'AST-AudioSet' or result.get('device') != 'cpu'
                    or result.get('gpu_used') is not False
                    or result.get('input_sha256') != receipt['output_sha256']['normalized_candidates.jsonl']
                    or result.get('listening_verified') is not False
                    or result.get('authenticity_inferred') is not False
                    or result.get('model_sha256') != AST_MODEL_SHA256
                    or result.get('code_sha256') != sha256_file(Path(__file__).with_name('review_audio_cpu.py'))
                    or result.get('evidence_kind') != 'automatic_audio_event_screening'
                    or result.get('input_duration_seconds') != 15
                    or result.get('review_start_seconds') != 0 or result.get('review_seconds') != 15
                    or result.get('window_starts_seconds') != [0, 5]
                    or result.get('window_durations_seconds') != [10, 10]
                    or result.get('reviewed_crop_coverage_fraction') != 1):
                raise ValueError(f'Review evidence contract mismatch: {row["ID"]}')
            score_key = 'music_score' if route(row) == 'music' else 'speech_score'
            score = result.get(score_key)
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'Invalid semantic review score: {row["ID"]}')
            windows = result.get('class_by_window', {}).get('Music' if score_key == 'music_score' else 'Speech')
            if (not isinstance(windows, list) or len(windows) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 1 for v in windows)
                    or score != min(windows)):
                raise ValueError(f'Review must aggregate both windows conservatively: {row["ID"]}')
            if score < .5:
                reasons.append('automated_semantic_confidence_below_0.5')
        by_id[row['ID']].update(status='excluded' if reasons else 'admitted_train_only', excluded_reason=reasons,
                                training_admitted=not reasons, automated_review=result)
        if not reasons:
            admitted.append(dict(row, role='train', training_admitted=True, admission_blockers=[],
                automated_review=result, admission_basis='source labels plus CPU AST semantic triage; not human listening'))
    combined = history + admitted
    for row in combined:
        reviewed_id = row.get('reviewed_component_id', row['component_id'])
        row['group_keys'] = sorted(set(row['group_keys']) | {reviewed_id})
    comp = components(combined)
    stable = defaultdict(set)
    for row in combined:
        stable[comp[row['ID']]].add(row.get('reviewed_component_id', row['component_id']))
    if any(len(ids) != 1 for ids in stable.values()):
        raise ValueError('Derived audio merged previously distinct reviewed components')
    role_sets = defaultdict(set)
    for row in combined:
        row['component_id'] = next(iter(stable[comp[row['ID']]]))
        role_sets[row['component_id']].add(row['role'])
    if any(len(roles) != 1 for roles in role_sets.values()):
        raise ValueError('Admitted source component crosses roles')
    splits = {r['ID']: 'train' if r['role'] == 'train' else 'val' if r['role'] in ('selection', 'calibration') else 'test' for r in combined}
    train, test, skipped = export(combined, splits)
    stable_by_id = {r['ID']: r['component_id'] for r in combined}
    for row in train + test:
        row['component_id'] = row['group_id'] = stable_by_id[row['orig_id']]
    exported = {r['orig_id'] for r in train if r['split'] == 'train'}
    if not {r['ID'] for r in admitted} <= exported:
        raise ValueError('Admitted rows missing from training export')
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / 'manifest.jsonl', combined)
    write_rows(output / 'dispositions.jsonl', decisions)
    for name, rows in (('bank_trainval.csv', train), ('bank_test.csv', test)):
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    with (output / 'roles.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ID', 'component_id', 'role', 'historical_exposure'])
        writer.writerows((r['ID'], r['component_id'], r['role'], json.dumps(r.get('historical_exposure', []))) for r in combined)
    with (output / 'splits.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ID', 'split'])
        writer.writerows(splits.items())
    write_json(output / 'skipped.json', skipped)
    report = dict(admitted_count=len(admitted), admitted_sources=dict(Counter(r['source'] for r in admitted)),
        total_candidates=len(decisions), excluded_count=sum(r['training_admitted'] is False for r in decisions),
        prior_rows=len(history), combined_rows=len(combined), cross_role_components=0, candidate_evaluation_rows=0,
        gpu_used=False, listening_verified=False, prepared_audio_root=str(prepared.resolve()),
        review_sha256=sha256_file(reviews), prepare_receipt_sha256=sha256_file(prepared/'prepare_receipt.json'),
        limitations=['AST is automated semantic triage, not human listening or detector quality evidence.',
                     'Common output format does not erase source recording or original codec differences.'])
    write_json(output / 'admission_receipt.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    prep = stages.add_parser('prepare')
    for name in ('history', 'candidates', 'payload', 'pilot', 'output', 'rights'):
        prep.add_argument('--' + name, required=True, type=Path)
    final = stages.add_parser('finalize')
    for name in ('prepared', 'reviews', 'output'):
        final.add_argument('--' + name, required=True, type=Path)
    args = vars(parser.parse_args())
    function = prepare if args.pop('stage') == 'prepare' else finalize
    print(json.dumps(function(**args)))


if __name__ == '__main__':
    main()
