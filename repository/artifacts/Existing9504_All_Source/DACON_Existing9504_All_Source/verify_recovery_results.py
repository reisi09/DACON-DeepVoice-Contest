"""Recompute Recovery V3 metrics and audit executed evidence without a GPU."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from graph_scoring import validate_predictions
from recovery_selection import _report

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results/recovery_v3/colab_20260913_044621'


def read(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def equivalent(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, float):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
    return left == right


def main():
    rows = read(ROOT / 'v2_validation/development/manifest.csv') + read(ROOT / 'v2_validation/locked_test/manifest.csv')
    rows = [x for x in rows if x['split'] in ('dev_select', 'test_generator', 'test_source')]
    fresh = read(ROOT / 'recovery_validation/manifest.csv')
    selection = json.loads((RUN / 'selection.json').read_text())
    comparison = json.loads((RUN / 'comparison.json').read_text())
    predictions = {name: {x['ID']: x for x in validate_predictions(read(RUN / f'{name}_predictions.csv'), [v['ID'] for v in rows])}
                   for name in selection['candidates']}
    roles, details = {}, {}
    for split in ('dev_select', 'test_generator', 'test_source'):
        subset = [x for x in rows if x['split'] == split]
        roles[split] = {name: _report(subset, values) for name, values in predictions.items()}
        groups = defaultdict(list)
        for row in subset:
            groups[row['source'].split('/')[0]].append(row)
        details[split] = {}
        for source, group in groups.items():
            details[split][source] = {}
            for name, values in predictions.items():
                known = [x for x in group if x['MUSIC_PRESENT'] == '1' and x['MUSIC_FAKE'] in ('0', '1')]
                entry = {'n': len(group), 'music_known_n': len(known), 'music_dominant_file_branch': sum(
                    float(values[x['ID']]['MUSIC_PRESENT_PROB']) * float(values[x['ID']]['MUSIC_FAKE_PROB']) >
                    float(values[x['ID']]['VOICE_PRESENT_PROB']) * float(values[x['ID']]['VOICE_FAKE_PROB']) for x in group)}
                for label in ('0', '1'):
                    scores = [float(values[x['ID']]['MUSIC_FAKE_PROB']) for x in known if x['MUSIC_FAKE'] == label]
                    entry['music_label_' + label] = {'n': len(scores), 'mean': sum(scores) / len(scores) if scores else None,
                                                     'above_0.5': sum(v >= .5 for v in scores)}
                details[split][source][name] = entry
    for name, values in predictions.items():
        assert equivalent(roles['dev_select'][name], selection['candidates'][name]['development'])
        for row in rows:
            g0, candidate = predictions['G0'][row['ID']], values[row['ID']]
            assert all(candidate[k] == g0[k] for k in ('VOICE_FAKE_PROB', 'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB'))
    fresh_predictions = {x['ID']: x for x in validate_predictions(read(RUN / 'G0_fresh_predictions.csv'), [x['ID'] for x in fresh])}
    assert equivalent(_report(fresh, fresh_predictions), comparison['fresh']['G0'])
    features = json.loads((RUN / 'raw_features.json').read_text()) + json.loads((RUN / 'fresh_raw_features.json').read_text())
    assert len(features) == 484 and len({x['ID'] for x in features}) == 484
    for feature in features:
        for key in ('sonics_raw', 'sonics_stem'):
            trace, end = feature[key], 0
            for start in trace['starts']:
                assert start <= end
                end = max(end, min(trace['covered_samples'], start + 80000))
            assert end == trace['covered_samples'] == round(trace['audio_seconds'] * 16000)
            assert abs(trace['audio_seconds'] - feature['duration_seconds']) < .001
    lock_hash = hashlib.sha256((RUN / 'candidate_lock.json').read_bytes()).hexdigest()
    written = json.loads((RUN / 'candidate_lock_written.json').read_text())
    opened = json.loads((RUN / 'fresh_opened.json').read_text())
    assert written['unix'] < opened['unix']
    assert written['candidate_lock_sha256'] == opened['candidate_lock_sha256'] == lock_hash
    hashes = json.loads((RUN / 'source_hashes.json').read_text())
    notebook = json.loads((RUN / 'executed.ipynb').read_text())
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if source.startswith('%%writefile '):
            command, body = source.split('\n', 1)
            name = Path(command.split()[-1]).name
            assert hashlib.sha256(body.encode()).hexdigest() == hashes[name]
            assert body == (ROOT / name).read_text()
    assert any("'status': 'finished'" in ''.join(o.get('text', [])) for o in notebook['cells'][-1]['outputs'])
    receipt = {'old_files': len(rows), 'fresh_files': len(fresh), 'valid_predictions_and_protected_heads': True,
               'recomputed_metrics_tolerance': 1e-12, 'gap_free_audio_files': len(features),
               'executed_source_hashes_match_local': True, 'lock_before_fresh_open': True,
               'candidate_lock_sha256': lock_hash, 'final_candidate': selection['selected']}
    (RUN / 'rescored_roles.json').write_text(json.dumps(roles, indent=2) + '\n')
    (RUN.parent / 'new_candidate_diagnostics.json').write_text(json.dumps(details, indent=2) + '\n')
    (RUN / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
