"""Compare E00/E04 on one development panel using F0 fitted only on E00 calibration.

Uses cached features and completed heads on CPU; does not train or submit a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_v4.file_ensemble import fit, known_file_panel
from eight_fixed_io import atomic_json
from full_bank_contract import role, validate_feature_contract
from full_bank_evaluate import five_columns, score_rows
from full_bank_features import load_rows
from graph_scoring import LABEL_COLUMNS, SUBMISSION_COLUMNS, official_metrics, validate_predictions
from v6_data import sha256


def fixed_f0_comparison(calibration, baseline_calibration, development, baseline, candidate):
    """One fit call, with neither candidate calibration nor development labels as input."""
    for rows, expected in ((calibration, 'calibration'), (development, 'development')):
        if not rows or any(role(r) != expected for r in rows) or len({r['ID'] for r in rows}) != len(rows):
            raise ValueError('Invalid ' + expected + ' panel')
    if ({r['ID'] for r in calibration} & {r['ID'] for r in development} or
            {r['component_id'] for r in calibration} & {r['component_id'] for r in development}):
        raise ValueError('Calibration and development overlap')
    for rows, scores in ((calibration, baseline_calibration), (development, baseline), (development, candidate)):
        validate_predictions([five_columns(s) for s in scores], [r['ID'] for r in rows])
        if any(s.get('prev_file') is not None for s in scores):
            raise ValueError('A previous package is not part of the data-only comparison')
    if any((b['Pv'], b['Pm']) != (c['Pv'], c['Pm']) for b, c in zip(baseline, candidate, strict=True)):
        raise ValueError('PANNs presence changed between candidates')
    fit_files, labels = known_file_panel(calibration, baseline_calibration)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('E00 calibration requires both known FILE classes')
    ensemble, fit_report = fit(fit_files, labels)
    ensemble['fitted_on'] = 'E00 calibration only'
    predictions = {name: validate_predictions([five_columns(s, ensemble=ensemble) for s in scores],
                    [r['ID'] for r in development]) for name, scores in (('E00', baseline), ('E04', candidate))}
    metrics = {name: official_metrics(development, rows) for name, rows in predictions.items()}
    delta = {name: None if metrics['E00']['metrics'][name] is None or value is None else
             value - metrics['E00']['metrics'][name] for name, value in metrics['E04']['metrics'].items()}
    return ensemble, predictions, dict(fit=fit_report, fitted_ids=[s['ID'] for s in fit_files],
                                      metrics=metrics, metrics_delta_e04_minus_e00=delta)


def checked_run(directory, manifest, feature_manifest, cache, index):
    identity = json.loads((directory / 'run_identity.json').read_text())
    complete = json.loads((directory / 'complete.json').read_text())
    if (identity.get('manifest') != sha256(manifest) or
            identity.get('feature_manifest') != sha256(feature_manifest) or
            identity.get('cache') != sha256(cache / 'index.json') or
            identity.get('voice_initial') != sha256(cache / 'voice_classifier_initial.pt')):
        raise ValueError('Run manifest or cache identity differs: ' + str(directory))
    if complete.get('identity') != identity or complete.get('results_sha256') != sha256(directory / 'results.json'):
        raise ValueError('Run is incomplete or results changed: ' + str(directory))
    validate_feature_contract(identity, index['identity'])
    source_root = Path(__file__).resolve().parents[1]
    for key, name in {'trainer': 'full_bank_train.py', 'model': 'localization_models.py',
                      'head': 'eight_heads.py', 'evaluator': 'full_bank_evaluate.py',
                      'batches': 'full_bank_batches.py', 'contract': 'full_bank_contract.py'}.items():
        if identity.get(key) != sha256(source_root / name):
            raise ValueError('Training implementation changed: ' + name)
    results = json.loads((directory / 'results.json').read_text())
    checkpoints = {}
    for seed in identity['config']['seeds']:
        for route_name in ('music', 'voice'):
            name = f'{route_name}_{seed}'
            checkpoints[name + '.pt'] = sha256(directory / (name + '.pt'))
            if results[name]['checkpoint_sha256'] != checkpoints[name + '.pt']:
                raise ValueError('Selected checkpoint changed: ' + name)
    return identity, checkpoints


def write_csv(path, rows, columns):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, '') for key in columns} for row in rows)


def compare(args):
    union = {r['ID']: r for r in load_rows(args.feature_manifest)}
    banks = {name: load_rows(path) for name, path in (('E00', args.e00_manifest), ('E04', args.e04_manifest))}
    for rows in banks.values():
        if any(union.get(r['ID']) != r for r in rows):
            raise ValueError('Requested row differs from feature manifest')
    by_id = {name: {r['ID']: r for r in rows} for name, rows in banks.items()}
    if any(by_id['E04'].get(key) != row for key, row in by_id['E00'].items()):
        raise ValueError('E04 must preserve every historical E00 row')
    additions = [r for r in banks['E04'] if r['ID'] not in by_id['E00']]
    if not additions or any(role(r) != 'train' for r in additions):
        raise ValueError('E04 must add only training rows')
    nontrain = {name: {r['ID']: r for r in rows if role(r) != 'train'} for name, rows in banks.items()}
    if nontrain['E00'] != nontrain['E04']:
        raise ValueError('E00/E04 non-training panels differ')
    calibration = [r for r in banks['E00'] if role(r) == 'calibration']
    development = [r for r in union.values() if role(r) == 'development']
    if not calibration or not development:
        raise ValueError('Both calibration and development panels are required')
    index = json.loads((args.cache / 'index.json').read_text())
    if index['identity']['manifest_sha256'] != sha256(args.feature_manifest):
        raise ValueError('Cache feature manifest differs')
    if set(index['items']) != set(union):
        raise ValueError('Feature cache coverage differs from union')
    runs = {name: checked_run(directory, manifest, args.feature_manifest, args.cache, index)
            for name, directory, manifest in (('E00', args.e00_dir, args.e00_manifest),
                                              ('E04', args.e04_dir, args.e04_manifest))}
    if ({k: v for k, v in runs['E00'][0].items() if k != 'manifest'} !=
            {k: v for k, v in runs['E04'][0].items() if k != 'manifest'}):
        raise ValueError('E00/E04 differ in training settings or implementation')
    source_root = Path(__file__).resolve().parents[1]
    implementation = {name: sha256(source_root / name) for name in (
        'data_v4/compare_data_admission.py', 'data_v4/file_ensemble.py', 'full_bank_evaluate.py',
        'full_bank_contract.py', 'full_bank_features.py', 'decision_heads.py', 'graph_scoring.py')}
    for name, digest in index['identity']['implementation'].items():
        path = (source_root / name).resolve()
        if not path.is_relative_to(source_root) or sha256(path) != digest:
            raise ValueError('Feature implementation changed: ' + name)
    identity = dict(runs=runs, implementation=implementation, device=args.device,
                    feature_manifest_sha256=sha256(args.feature_manifest),
                    presence='unchanged PANNs', fit='E00 calibration only')
    # Convert tuples to their serialized form so a matching rerun compares identically.
    identity = json.loads(json.dumps(identity))
    if args.output.exists():
        saved = args.output / 'run_identity.json'
        complete = args.output / 'complete.json'
        if not saved.is_file() or json.loads(saved.read_text()) != identity or not complete.is_file():
            raise ValueError('Output belongs to another or incomplete comparison; use a fresh directory')
        outputs = json.loads(complete.read_text())['outputs']
        if set(outputs) != {'run_identity.json', 'F0.json', 'report.json', 'calibration_panel.csv',
                           'development_panel.csv', 'E00_development_predictions.csv', 'E04_development_predictions.csv'}:
            raise ValueError('Completed comparison receipt is incomplete')
        for name, digest in outputs.items():
            path = (args.output / name).resolve()
            if not path.is_relative_to(args.output.resolve()) or sha256(path) != digest:
                raise ValueError('Completed comparison artifact changed: ' + name)
        return json.loads((args.output / 'report.json').read_text())
    baseline_calibration = score_rows(calibration, args.cache, args.e00_dir, args.device)
    baseline = score_rows(development, args.cache, args.e00_dir, args.device)
    candidate = score_rows(development, args.cache, args.e04_dir, args.device)
    ensemble, predictions, report = fixed_f0_comparison(calibration, baseline_calibration, development, baseline, candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'run_identity.json', identity)
    atomic_json(args.output / 'F0.json', ensemble)
    panel_columns = ['ID', 'component_id', 'role', *LABEL_COLUMNS, *(name + '_mask' for name in LABEL_COLUMNS)]
    for name, rows in (('calibration', calibration), ('development', development)):
        write_csv(args.output / (name + '_panel.csv'), rows, panel_columns)
    for name, rows in predictions.items():
        write_csv(args.output / (name + '_development_predictions.csv'), rows, SUBMISSION_COLUMNS)
    report.update(F0_sha256=sha256(args.output / 'F0.json'),
                  same_development_ids_masks_components=True, presence='unchanged PANNs',
                  added_training_rows=len(additions), calibration_rows=len(calibration), development_rows=len(development),
                  development_components=len({r['component_id'] for r in development}),
                  panel_files={name: sha256(args.output / name) for name in ('calibration_panel.csv', 'development_panel.csv')},
                  official_submission=False, official_leaderboard_score=False, independent_test=False,
                  interpretation='Development metrics only; EER decreases and presence AUC increases are better')
    atomic_json(args.output / 'report.json', report)
    atomic_json(args.output / 'complete.json', {'outputs': {path.name: sha256(path) for path in args.output.iterdir()}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('e00-manifest', 'e04-manifest', 'feature-manifest', 'cache', 'e00-dir', 'e04-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    report = compare(parser.parse_args())
    print(json.dumps({'F0_sha256': report['F0_sha256'], 'development_rows': report['development_rows'],
                      'metrics_delta_e04_minus_e00': report['metrics_delta_e04_minus_e00']}, indent=2))


if __name__ == '__main__':
    main()
