"""Read-only score-regression diagnostics; run from the repository root.

Run: .venv/bin/python results/balanced7144_regression_20260919/analyze.py
Checks protect prediction/label alignment, score arithmetic and fusion integrity.
No model fitting, GPU execution or submission is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from graph_scoring import _roc_metrics, validate_predictions
from full_bank_batches import Item, loss_weights

OUT = Path(__file__).resolve().parent
SELECTED = ROOT / 'results/balanced7144_20260919/selected.csv'
TRAIN = ROOT / 'results/balanced7144_h100_20260919/downloaded/unpacked/output/train'
OLD = ROOT / 'results/expanded_h100_20260918/downloaded/unpacked/output'


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def measure(rows: list[dict[str, str]], score: str) -> dict[str, float | int | None]:
    labels = [int(r['label']) for r in rows]
    values = [float(r[score]) for r in rows]
    real, fake = labels.count(0), labels.count(1)
    eer, auc = _roc_metrics(labels, values) if real and fake else (None, None)
    return {'rows': len(rows), 'real': real, 'fake': fake, 'eer': eer, 'auc': auc,
            'real_fp_at_0.5': sum(y == 0 and p >= .5 for y, p in zip(labels, values)),
            'fake_fn_at_0.5': sum(y == 1 and p < .5 for y, p in zip(labels, values))}


rows = read_rows(SELECTED)
for row in rows:
    row['label'] = row[row['route'].upper() + '_FAKE']
    assert row['label'] in ('0', '1') and row['label'] == row['FILE_FAKE']
val = [r for r in rows if r['split'] == 'val']
predictions = read_rows(TRAIN / 'validation_predictions.csv')
pred_by_id = {r['ID']: r for r in predictions}
assert len(pred_by_id) == len(predictions) == len(val) == 1428
assert set(pred_by_id) == {r['ID'] for r in val}
validate_predictions([pred_by_id[r['ID']] for r in val], [r['ID'] for r in val])
joined = [dict(r, **{k: v for k, v in pred_by_id[r['ID']].items() if k != 'ID'}) for r in val]
for r in joined:
    voice = float(r['VOICE_PRESENT_PROB']) * float(r['VOICE_FAKE_PROB'])
    music = float(r['MUSIC_PRESENT_PROB']) * float(r['MUSIC_FAKE_PROB'])
    assert abs(float(r['FILE_FAKE_PROB']) - max(voice, music)) < 1e-6
    r['oracle_route_score'] = str(voice if r['route'] == 'voice' else music)
    r['oracle_route_fake'] = r['VOICE_FAKE_PROB'] if r['route'] == 'voice' else r['MUSIC_FAKE_PROB']
    r['voice_dominates'] = str(voice > music)
    r['voice_only_fp_at_0.5'] = str(r['label'] == '0' and voice >= .5 and music < .5)

report = {'scope': 'Local diagnostic validation, not an independent test or causal attribution',
          'threshold_note': '0.5 counts are diagnostic and are not official EER operating points',
          'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (SELECTED, TRAIN / 'validation_predictions.csv', OLD / 'dataset_manifest.csv')},
          'counts': dict(Counter(f"{r['split']}:{r['route']}:{r['label']}" for r in rows)),
          'all_file': measure(joined, 'FILE_FAKE_PROB'),
          'oracle_route_only': measure(joined, 'oracle_route_score'),
          'routes': {}, 'source_slices': {}, 'source_only_baseline': {}, 'weight_mass': {},
          'exposure_slices': {}, 'duration_seconds': {}, 'manifest_comparison': {}}
official = json.loads((TRAIN / 'validation_metrics.json').read_text())
assert abs(report['all_file']['eer'] - official['metrics']['file_eer']) < 1e-12
for route in ('music', 'voice'):
    train = [r for r in rows if r['split'] == 'train' and r['route'] == route]
    subset = [r for r in joined if r['route'] == route]
    score = route.upper() + '_FAKE_PROB'
    stats = measure(subset, score)
    assert abs(stats['eer'] - official['metrics'][route + '_eer']) < 1e-12
    report['routes'][route] = {'head': stats, 'file': measure(subset, 'FILE_FAKE_PROB'),
                              'oracle_route': measure(subset, 'oracle_route_score'),
                              'real_voice_only_fp_at_0.5': sum(r['voice_only_fp_at_0.5'] == 'True' for r in subset)}
    by_source = defaultdict(list)
    for r in train:
        by_source[r['source']].append(int(r['label']))
    assert all(r['source'] in by_source for r in subset)
    source_preds = [dict(r, source_probability=str(np.mean(by_source[r['source']]))) for r in subset]
    report['source_only_baseline'][route] = measure(source_preds, 'source_probability')
    real_median = np.median([float(r['duration_seconds']) for r in train if r['label'] == '0'])
    fake_median = np.median([float(r['duration_seconds']) for r in train if r['label'] == '1'])
    duration_sign = 1 if fake_median > real_median else -1
    duration_eer, duration_auc = _roc_metrics([int(r['label']) for r in subset],
                                             [duration_sign * float(r['duration_seconds']) for r in subset])
    report['source_only_baseline'][route + '_duration_only'] = {
        'eer': duration_eer, 'auc': duration_auc, 'sign_selected_from_training_medians': duration_sign,
        'note': 'Metadata diagnostic only; no waveform or fitted classifier'}
    items = [Item(r, np.empty((0, 0)), np.empty(0), float(r['label'])) for r in train]
    weights = loss_weights(items)
    assert abs(sum(weights.values()) - len(train)) < 1e-6
    cells = defaultdict(list)
    for r in train:
        cells[(r['label'], r['source'], r['generator'])].append(r)
    report['weight_mass'][route] = [
        {'label': c[0], 'source': c[1], 'generator': c[2], 'rows': len(members),
         'row_fraction': len(members) / len(train),
         'loss_fraction': sum(weights[r['ID']] for r in members) / len(train)}
        for c, members in sorted(cells.items())]
    for source in sorted({r['source'] for r in subset}):
        report['source_slices'][route + ':' + source] = measure([r for r in subset if r['source'] == source], score)
    for exposure in ('True', 'False'):
        report['exposure_slices'][route + ':' + exposure] = measure([r for r in subset if r['prior_seen'] == exposure], score)
    for label in ('0', '1'):
        durations = [float(r['duration_seconds']) for r in train if r['label'] == label]
        report['duration_seconds'][route + ':' + label] = dict(zip(('min', 'p25', 'median', 'p75', 'max'), np.quantile(durations, [0, .25, .5, .75, 1]).tolist()))

old_rows = read_rows(OLD / 'dataset_manifest.csv')
old_train = [r for r in old_rows if r['split'] == 'train']
new_music = [r for r in rows if r['split'] == 'train' and r['route'] == 'music']
for name, cohort in [('expanded_music_train', old_train), ('balanced_music_train', new_music)]:
    report['manifest_comparison'][name] = {
        'rows': len(cohort), 'labels': dict(Counter(r['MUSIC_FAKE'] for r in cohort)),
        'variants': dict(Counter(r['variant'] for r in cohort)),
        'with_explicit_music_intervals': sum(any(e.get('component') == 'music' for e in json.loads(r['event_intervals'] or '[]')) for r in cohort),
        'known_voice_fake': sum(r['VOICE_FAKE'] in ('0', '1') for r in cohort),
        'duration_median_seconds': float(np.median([float(r['duration_seconds']) for r in cohort]))}
report['historical_exposed_validation_rows'] = sum(r['prior_seen'] == 'True' for r in val)
families = defaultdict(float)
for cell in report['weight_mass']['music']:
    families[cell['source'].split('/')[0]] += cell['loss_fraction']
report['music_loss_mass_by_source_family'] = dict(families)
report['music_learning_curves'] = {}
for seed in (41, 42, 43):
    history = json.loads((TRAIN / f'music_{seed}_history.json').read_text())
    selected_epoch = json.loads((TRAIN / 'results.json').read_text())[f'music_{seed}']['epoch']
    best = next(r for r in history if r['epoch'] == selected_epoch)
    report['music_learning_curves'][str(seed)] = {
        'selected_epoch': selected_epoch, 'completed_epochs': len(history),
        'selected_bce': best['validation_bce'], 'final_bce': history[-1]['validation_bce'],
        'selected_eer': best['validation_eer'], 'final_eer': history[-1]['validation_eer']}
report['fake_voice_sources'] = dict(Counter(r['source'] for r in rows if r['route'] == 'voice' and r['label'] == '1'))
report['fake_voice_speakers'] = {split: sorted({r['speaker_id'] for r in rows if r['route'] == 'voice' and r['label'] == '1' and r['split'] == split}) for split in ('train', 'val')}
old_pred = read_rows(OLD / 'train/mil_ensemble.csv')
old_by_hash = {r['audio_sha256']: r for r in old_pred if r['audio_sha256']}
overlap = [dict(r, old_music=old_by_hash[r['audio_sha256']]['MUSIC_FAKE_PROB']) for r in joined if r['audio_sha256'] and r['audio_sha256'] in old_by_hash]
report['exact_audio_validation_overlap'] = {'rows': len(overlap), 'note': 'No comparison of different crops or parent-only matches'}
if overlap:
    report['exact_audio_validation_overlap'].update(old=measure(overlap, 'old_music'), new=measure(overlap, 'MUSIC_FAKE_PROB'))
old_total, new_total = Decimal('0.7579036931'), Decimal('0.7559347884')
ads_delta = Decimal('0.7299920635') - Decimal('0.7321825397')
cps_delta = Decimal('0.9894193122') - Decimal('0.9893940741')
assert abs(new_total - old_total - (Decimal('.9') * ads_delta + Decimal('.1') * cps_delta)) < Decimal('0.0000000001')
report['official_change'] = {'previous_submission': 92644, 'current_submission': 93093,
    'old_total': str(old_total), 'new_total': str(new_total), 'delta_total': str(new_total-old_total),
    'ads_contribution': str(Decimal('.9') * ads_delta), 'cps_contribution': str(Decimal('.1') * cps_delta),
    'limitation': 'Rendered submissions do not expose individual EER values or ZIP hashes; prior Expanded package identity unconfirmed'}
columns = ['ID', 'source', 'duration_seconds', 'prior_seen', 'FILE_FAKE_PROB', 'VOICE_FAKE_PROB',
           'MUSIC_FAKE_PROB', 'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB', 'oracle_route_score']
with (OUT / 'music_real_voice_branch_false_positives.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(r for r in joined if r['route'] == 'music' and r['voice_only_fp_at_0.5'] == 'True')
(OUT / 'diagnostics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('weight_mass', 'source_hashes')}, indent=2, ensure_ascii=False))
