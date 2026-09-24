"""Recompute metrics and verify unchanged supervision/checkpoint provenance on downloaded outputs."""
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'repository'))
from graph_scoring import official_metrics

root, original_manifest = map(Path, sys.argv[1:3])
rows = list(csv.DictReader((root / 'available_rows.csv').open()))
union = {r['ID']: r for r in csv.DictReader(original_manifest.open())}
assert all(union[r['ID']] == r for r in rows), 'Original labels/roles changed'
roles = {role: {r['ID'] for r in rows if r['role'] == role} for role in ('train', 'selection', 'calibration')}
assert not roles['train'] & roles['selection'] and not roles['train'] & roles['calibration']
selection = [r for r in rows if r['role'] == 'selection']
checks = {}
for name in ('E08_topk', 'E00_control'):
    out = root / name
    pred = list(csv.DictReader((out / 'validation_predictions.csv').open()))
    for p in pred:
        for k in p:
            if k != 'ID': p[k] = float(p[k])
    metrics = official_metrics(selection, pred)
    assert metrics == json.loads((out / 'validation_metrics.json').read_text()), 'Reported metric differs from predictions'
    result = json.loads((out / 'results.json').read_text())
    for route in ('music', 'voice'):
        checkpoint = out / f'{route}_41.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == result[f'{route}_41']['checkpoint_sha256']
        history = json.loads((out / f'{route}_41_history.json').read_text())
        assert all(h['rows_seen'] == h['unique_rows_seen'] for h in history), 'Duplicate training exposure within epoch'
    checks[name] = {'metrics_recomputed_match': True, 'checkpoint_hashes_match': True,
                   'unique_training_rows_per_epoch': True, 'metrics': metrics['metrics']}
print(json.dumps({'original_rows_roles_labels_unchanged': True, 'role_counts': {k: len(v) for k,v in roles.items()},
                  'runs': checks, 'scope': 'Ordered partial-cache selection set; no independent test and no official score'}, indent=2))
