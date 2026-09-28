"""Summarize actual V5 artifacts on the original and unique-audio dev panels."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from graph_scoring import SUBMISSION_COLUMNS, official_metrics
from v5_dataset import load_splits
from v5_reporting import labels_as_rows, write_json


def analyze(results: Path, data: Path) -> dict:
    _, dev, audit = load_splits(data)
    labels = labels_as_rows(dev)
    groups = {}
    for row in dev:
        with row.path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        groups.setdefault(digest, []).append(row.id)
    representatives = {min(ids) for ids in groups.values()}
    unique_labels = [row for row in labels if row['ID'] in representatives]
    comparison = json.loads((results / 'comparison.json').read_text())
    report = {'scope': 'Local development; incomplete labels; not a leaderboard score',
              'data_audit': audit, 'row_count': len(dev), 'unique_audio_count': len(groups),
              'selection_panel': 'Original 230 rows; unique audio is a secondary sensitivity analysis',
              'arms': {}, 'jobs': {}}
    log = (results / 'experiment.log').read_text() if (results / 'experiment.log').is_file() else ''
    preparation_failed = 'TimeoutError: Full raw/stem preparation exceeded its stage budget' in log
    report['execution'] = {'status': 'partial_preparation_timeout' if preparation_failed else 'snapshot',
                           'new_submission': False}
    if (results / 'release_requested.json').is_file():
        report['execution']['release_request'] = json.loads((results / 'release_requested.json').read_text())
    for arm in ('B0', 'B1', 'B2', 'B3', 'B4', 'B4_joint'):
        raw = comparison.get(arm, {})
        entry = {'status': raw.get('status', 'not_recorded'),
                 'runtime_status': raw.get('status')}
        path = results / f'{arm}_predictions.csv'
        if path.is_file():
            with path.open(newline='') as stream:
                predictions = list(csv.DictReader(stream))
            by_id = {row['ID']: row for row in predictions}
            if set(by_id) != {row.id for row in dev} or len(predictions) != len(dev):
                raise ValueError(f'{arm}: incomplete or duplicate prediction IDs')
            entry['original'] = official_metrics(labels, predictions)
            entry['unique_audio'] = official_metrics(unique_labels, [by_id[row['ID']] for row in unique_labels])
            deltas = []
            for ids in groups.values():
                if len(ids) > 1:
                    values = np.asarray([[float(by_id[id_][name]) for name in SUBMISSION_COLUMNS[1:]] for id_ in ids])
                    deltas.append(float(np.max(np.ptp(values, axis=0))))
            entry['identical_audio_max_probability_delta'] = max(deltas, default=0.)
        elif preparation_failed and arm in ('B3', 'B4'):
            entry['status'] = 'not_run_preparation_timeout'
        report['arms'][arm] = entry
    for path in sorted((results / 'jobs').glob('*/training.json')):
        job = json.loads(path.read_text())
        candidates = [row for row in job['epoch_metrics'] if row['dev'] is not None]
        best = max(candidates, key=lambda row: row['dev']['selection_metric']) if candidates else None
        report['jobs'][path.parent.name] = {
            'status': job['status'], 'epochs_completed': job['epochs_completed'],
            'updates': job['updates'], 'seconds': job['seconds'],
            'selected_epoch': best['epoch'] if best else None,
            'selected_heads': best['dev']['heads'] if best else None,
            'first_step_audit': job['first_step_audit'],
        }
    for arm in ('B1', 'B2'):
        jobs = {kind: report['jobs'].get(f'{arm}_{kind}') for kind in ('voice', 'music', 'presence')}
        report['arms'][arm]['component_training_status'] = {
            kind: job['status'] if job else 'receipt_unavailable' for kind, job in jobs.items()}
        if 'original' in report['arms'][arm]:
            report['arms'][arm]['status'] = ('evaluated_budget_censored'
                if any(job and job['status'] == 'budget_stop' for job in jobs.values())
                else 'evaluated_training_receipts_incomplete' if not all(jobs.values())
                else report['arms'][arm]['status'])
    joint = comparison.get('B4_joint', {})
    if 'original' in report['arms']['B4_joint'] and not joint.get('status'):
        training = joint.get('training', {})
        report['arms']['B4_joint']['status'] = ('evaluated_budget_censored'
            if training.get('status') == 'budget_stop' else 'evaluated'
            if training.get('status') == 'completed' and training.get('updates', 0) > 0
            else 'evaluation_training_unverified')
    write_json(results / 'analysis.json', report)
    lines = ['# V5 measured comparison', '', report['scope'], '',
             f"Execution: {report['execution']['status']}. Missing arms are not experimental results.", '',
             '| Arm | Status | FILE EER | VOICE EER | MUSIC EER | VP AUC | MP AUC |',
             '|---|---|---:|---:|---:|---:|---:|']
    for arm, row in report['arms'].items():
        metrics = row.get('original', {}).get('metrics', {})
        values = [metrics.get(key) for key in ('file_eer', 'voice_eer', 'music_eer', 'voice_presence_auc', 'music_presence_auc')]
        lines.append('| ' + ' | '.join([arm, row['status'], *['—' if value is None else f'{value:.6f}' for value in values]]) + ' |')
    lines.extend(['', 'EER: lower is better. AUC: higher is better.', '',
                  'The 230-row dev panel contains 210 unique audio files. Duplicate splice rows retain their original selection weight; analysis.json includes unique-audio metrics and identical-input prediction deltas.', '',
                  'Expert targets were 60 epochs. Budget-censored results describe only the actual completed epochs below. Checkpoints were selected among evaluated epochs, not every completed epoch. B1/B2 have different exposure counts, so their difference is not an isolated augmentation effect.', '',
                  '| Job | Completed epochs | Updates | Selected epoch |', '|---|---:|---:|---:|'])
    for name, job in report['jobs'].items():
        lines.append(f"| {name} | {job['epochs_completed']} | {job['updates']} | {job['selected_epoch']} |")
    (results / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    plot_learning_curves(results)
    return report


def plot_learning_curves(results: Path) -> None:
    import matplotlib.pyplot as plt

    baseline = json.loads((results / 'comparison.json').read_text()).get('B0', {}).get('metrics', {}).get('metrics', {})
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, kind in zip(axes, ('voice', 'music', 'presence'), strict=True):
        for arm, color in (('B1', '#2563eb'), ('B2', '#ea580c')):
            path = results / 'jobs' / f'{arm}_{kind}' / 'training.json'
            if not path.is_file():
                continue
            job = json.loads(path.read_text())
            measured = [row for row in job['epoch_metrics'] if row['dev'] is not None]
            epochs = [row['epoch'] for row in measured]
            scores = [row['dev']['selection_metric'] for row in measured]
            values = scores if kind == 'presence' else [1 - score for score in scores]
            if epochs:
                axis.plot(epochs, values, marker='o', color=color, label=arm)
        reference = (sum(baseline[key] for key in ('voice_presence_auc', 'music_presence_auc')) / 2
                     if kind == 'presence' and all(baseline.get(key) is not None for key in ('voice_presence_auc', 'music_presence_auc'))
                     else baseline.get(f'{kind}_eer'))
        if reference is not None:
            axis.axhline(reference, linestyle='--', color='#64748b', label='B0 reference')
        axis.set(title=f'{kind.title()} expert', xlabel='Completed epoch',
                 ylabel='Mean presence AUC (higher better)' if kind == 'presence' else 'EER (lower better)')
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(frameon=False)
    figure.suptitle('Measured development checkpoints — missing epochs were not evaluated')
    figure.savefig(results / 'learning_curves.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('v4_validation'))
    args = parser.parse_args()
    analyze(args.results, args.data)
