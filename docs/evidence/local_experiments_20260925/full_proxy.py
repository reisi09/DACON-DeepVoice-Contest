"""Recalculate historical diagnostic slices; never fit or select on this panel."""
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'repository'))
from decision_heads import file_score
from graph_scoring import _label, _roc_metrics, official_metrics, validate_predictions
from v6_data import sha256


def main(root, output):
    bank = root / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/bank_test.csv'
    run = root / 'results/data_v4_20260923/data_v4_phase_c_20260923T1653Z/train_duration_balanced'
    source = run / 'test_predictions_decision.csv'
    ensemble_path = run / 'file_ensemble.json'
    ensemble = json.loads(ensemble_path.read_text())
    lookup = {r['ID']: r for r in csv.DictReader(bank.open())}
    saved = list(csv.DictReader(source.open()))
    predictions = validate_predictions(saved, [r['ID'] for r in saved])
    labels = [lookup[r['ID']] for r in predictions]

    def combine(p):
        return file_score(ensemble, dict(Pv=p['VOICE_PRESENT_PROB'], Pm=p['MUSIC_PRESENT_PROB'],
                                        Fv=p['VOICE_FAKE_PROB'], Fm=p['MUSIC_FAKE_PROB']))

    replay_error = max(abs(combine(p) - p['FILE_FAKE_PROB']) for p in predictions)
    if replay_error > 1e-12:
        raise ValueError('Saved component scores do not reconstruct the preserved FILE combiner')

    slices = {}
    for name, route, music in [('voice_only', 'VOICE', 0), ('voice_with_music', 'VOICE', 1), ('music', 'MUSIC', None)]:
        indices = [i for i, r in enumerate(labels) if _label(r, route+'_PRESENT') == 1
                   and _label(r, route+'_FAKE') is not None
                   and (music is None or _label(r, 'MUSIC_PRESENT') == music)]
        truth = [_label(labels[i], route+'_FAKE') for i in indices]
        scores = [predictions[i][route+'_FAKE_PROB'] for i in indices]
        eer, auc = _roc_metrics(truth, scores)
        slices[name] = dict(rows=len(indices), real=truth.count(0), fake=truth.count(1), eer=eer, auc=auc)
    oracles = {}
    for route in ('MUSIC', 'VOICE'):
        changed = [dict(p) for p in predictions]
        count = 0
        for row, prediction in zip(labels, changed, strict=True):
            if _label(row, route+'_PRESENT') == 1 and _label(row, route+'_FAKE') is not None:
                prediction[route+'_FAKE_PROB'] = float(_label(row, route+'_FAKE'))
                count += 1
            prediction['FILE_FAKE_PROB'] = combine(prediction)
        oracles[route] = dict(replaced_rows=count, reports=official_metrics(labels, changed))
    report = dict(
        inputs_sha256={p.name: sha256(p) for p in (bank, source, ensemble_path)},
        bank_rows=len(lookup), predicted_rows=len(predictions),
        missing_prediction_rows=len(lookup)-len(predictions),
        missing_known_file_labels=sum(_label(r, 'FILE_FAKE') is None for r in labels),
        file_combiner_max_abs_replay_error=replay_error,
        reference=official_metrics(labels, predictions), slices=slices, oracles=oracles,
        oracle_rule='Replace one known-present component fake probability with its known 0/1 label; keep both presence outputs and the FILE combiner fixed.',
        independent_quality_validated=False,
        limitation='Previously inspected historical proxy, not fresh holdout. Voice-with-music is not a verified singing label. Oracle changes are not attainable improvement forecasts.')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(rows=len(predictions), replay_error=replay_error, slices=slices)))


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:3]))
