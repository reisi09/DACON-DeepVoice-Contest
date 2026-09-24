"""Bounded CPU diagnostic on preserved features; not independent evaluation or model selection."""
import csv
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'repository'))
import numpy as np
import torch
from full_bank_evaluate import load_selected_heads, score_rows, five_columns
from full_bank_train import predict
from full_bank_batches import Item
from full_bank_contract import saved_config
from graph_scoring import official_metrics

signal.signal(signal.SIGALRM, lambda *_: sys.exit('CPU diagnostic exceeded 180 seconds'))
signal.alarm(180)
torch.set_num_threads(2)
archive, unpacked = map(Path, sys.argv[1:3])
receipt = json.loads(archive.with_suffix('.json').read_text())
cache = unpacked / 'cache'
run = unpacked / 'runs/partial_training'
rows = [r for r in csv.DictReader((run / 'available_rows.csv').open()) if r['role'] in ('selection', 'calibration')]
index = json.loads((cache / 'index.json').read_text())
sha = lambda b: hashlib.sha256(b).hexdigest()

with tempfile.TemporaryDirectory(prefix='e08_regression_') as tmp:
    heads = Path(tmp)
    with zipfile.ZipFile(archive) as z:
        names = [f'{r}_{s}.pt' for r in ('music', 'voice') for s in (41, 42, 43)]
        names += ['run_identity.json', 'presence_head.json', 'file_ensemble.json']
        for name in names:
            member = 'model/trained/' + name
            content = z.read(member)
            assert sha(content) == receipt['member_sha256'][member], member
            (heads / name).write_bytes(content)
    # Compare the prior package's documented encoder/preprocessing contract to the cache.
    expected = {'model/assets/' + n: v for n, v in index['identity']['assets'].items()}
    expected.update({'model/runtime/' + n: v for n, v in index['identity']['implementation'].items()
                     if n not in ('full_bank_features.py', 'full_bank_contract.py')})
    assert all(receipt['member_sha256'].get(n) == v for n, v in expected.items()), 'Incompatible prior package'
    identity = json.loads((heads / 'run_identity.json').read_text())
    for key, name in [('model', 'localization_models.py'), ('head', 'eight_heads.py')]:
        assert sha((Path(__file__).resolve().parents[3] / 'repository' / name).read_bytes()) == identity[key]
    music, voice, fake, _ = load_selected_heads(heads, 'cpu')
    config = saved_config(heads)
    presence = json.loads((heads / 'presence_head.json').read_text())
    ensemble = json.loads((heads / 'file_ensemble.json').read_text())
    current = score_rows(rows, cache, run / 'E08_topk', 'cpu')  # validates each cache payload and identity
    assert all(s['presence_features'] is not None for s in current)
    previous = []
    for row, new in zip(rows, current, strict=True):
        with np.load(cache / index['items'][row['ID']]['path'], allow_pickle=False) as data:
            item = lambda key: Item(row, data[key], np.full(len(data[key]), np.nan), 0., bool(data['voice_silent']))
            m, v = item('mert_windows'), item('voice_embeddings')
            score = lambda models, i, route: float(np.mean([predict(h, [i], route, fake, 1, len(i.features), 'cpu', config)[0] for h in models]))
            previous.append({**new, 'Fm': score(music, m, 'music'), 'Fv': score(voice, v, 'voice')})
    variants = {
        'DataV4_original': [five_columns(s, presence, ensemble) for s in previous],
        'DataV4_heads_legacy_decision': [five_columns(s) for s in previous],
        'E08_original': [five_columns(s) for s in current],
        'E08_with_old_music_legacy_decision': [five_columns({**n, 'Fm': o['Fm']}) for o, n in zip(previous, current)],
        'E08_with_old_voice_legacy_decision': [five_columns({**n, 'Fv': o['Fv']}) for o, n in zip(previous, current)],
    }
    reports = {}
    for role in ('selection', 'calibration'):
        selected = [i for i, r in enumerate(rows) if r['role'] == role]
        reports[role] = {name: official_metrics([rows[i] for i in selected], [p[i] for i in selected]) for name, p in variants.items()}
    print(json.dumps({'scope': 'CPU cached-feature diagnostic only; ordered partial subset, prior-model training overlap not excluded; no official score or independent generalization claim.',
                      'old_archive_sha256_recorded': receipt['sha256'], 'rows': len(rows),
                      'reports': reports}, indent=2))
