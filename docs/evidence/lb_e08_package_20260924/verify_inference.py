"""Check actual packaged CPU outputs and a long-clip counterexample to accidental max pooling."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'repository'))
from graph_scoring import validate_predictions

qa, reference_csv = map(Path, sys.argv[1:3])
reference = {r['ID']: r for r in csv.DictReader(reference_csv.open())}
short_rows = json.loads((qa / 'sample_rows.json').read_text())
case = json.loads((qa / 'long_case.json').read_text())
checks = []
for csv_path, expected_ids, exit_path in (
        (qa / 'extracted/output/submission.csv', [r['ID'] for r in short_rows], qa / 'raw_cpu_exit.json'),
        (qa / 'long_output/submission.csv', [case['check']['ID']], qa / 'long_cpu_exit.json')):
    assert json.loads(exit_path.read_text())['returncode'] == 0, 'Packaged entrypoint failed'
    rows = validate_predictions(list(csv.DictReader(csv_path.open())), expected_ids)
    for row in rows:
        differences = {k: abs(row[k] - float(reference[row['ID']][k])) for k in row if k != 'ID'}
        checks.append({'ID': row['ID'], 'prediction': row, 'absolute_difference_from_saved_a100_prediction': differences})
long_prediction = checks[-1]['prediction']['VOICE_FAKE_PROB']
check = case['check']
difference = abs(long_prediction - check['cpu_cached_topk'])
assert difference <= check['raw_cpu_vs_cached_abs_tolerance'], 'Long-clip packaged prediction differs from cached top-k reference'
assert abs(long_prediction - check['cpu_cached_counterfactual_max']) > .02, 'Fixture no longer distinguishes max from top-k'
formats = sorted({Path(r['path']).suffix.lstrip('.').upper() for r in [*short_rows, case['row']]})
print(json.dumps({'packaged_cpu_entrypoint_passed': True, 'audio_files': len(checks), 'real_audio_formats': formats,
                  'cases': checks, 'long_clip_topk_check': {**check, 'raw_cpu_prediction': long_prediction,
                  'absolute_difference_from_cpu_cached_topk': difference, 'passed': True},
                  'official_l4_validated': False, 'full_test_runtime_validated': False,
                  'scope': 'Three original files; exact schema/order/finite probabilities and actual long-clip pooling behavior. Not an accuracy or full-runtime guarantee.'}, indent=2))
