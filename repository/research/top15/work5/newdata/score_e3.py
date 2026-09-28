"""E3 (AI-Hub, this Mac only): every-row member predictions of the saved DB-protocol models (SAVE_MODELS of jobs_nd.txt) on
the AI-Hub npz, written as split 'e3' into copies of the run JSONs, so new_eval.py (EVAL_SPLITS=e3) scores E3 with the
same CDF tables and FILE rule as E1/E2. Features are built exactly as the every_items of voice_data.py / run.py.
  python score_e3.py RUNS MODELS AIHUB_ROWS_CSV OUT_RUNS"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path[:0] = [str(ROOT), str(ROOT / 'data_v4'), str(ROOT / 'research/top15/work2/backbones')]
import __main__  # noqa: E402
import run as R  # noqa: E402

for name in ('Standardize', 'ResMLP', 'LayerMixMIL'):   # run.py models were pickled from __main__ (python run.py ...)
    if hasattr(R, name):
        setattr(__main__, name, getattr(R, name))
RUNS, MODELS, ROWS, OUT = map(Path, sys.argv[1:5])
rows = [r for r in csv.DictReader(ROWS.open()) if Path(r['npz']).exists()]
old_fake = torch.load(R.H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)['fake_label_index']
MEMBERS = {'voice_df__L': ('voice', old_fake), 'voice_ad__L': ('voice', 0), 'voice_new__wavlm8': ('voice', 0),
           'music_new__mert6_12+eat16+sslampt5': ('music', 0)}


def items(member):
    route = MEMBERS[member][0]
    out = []
    for r in rows:
        with np.load(r['npz']) as d:
            silent = bool(d['voice_silent'])
            if member == 'voice_df__L':
                f = d['voice_embeddings'].astype(np.float32)
            elif member == 'voice_ad__L':
                f = d['ad_embeddings'].astype(np.float32)
            elif member == 'voice_new__wavlm8':
                f = d['wavlm_voice'][:, [list(d['wavlm_layers']).index(8)]].reshape(len(d['wavlm_voice']), -1).astype(np.float32)
            else:   # mert 6+12 | eat 16 | sslam-pt 5, fp16 as stored (run.py every_items keeps music fp16)
                L, EL, SL = list(d['layers']), list(d['eat_layers']), list(d['sslam_layers'])
                f = np.concatenate([d['mert_layers'][:, [L.index(6), L.index(12)]].reshape(len(d['mert_layers']), -1),
                                    d['eat_backing'][:, [EL.index(16)]].reshape(len(d['eat_backing']), -1),
                                    d['sslam_pt_backing'][:, [SL.index(5)]].reshape(len(d['sslam_pt_backing']), -1)], 1)
        out.append(R.Item({**r, 'route': route}, f, np.full(len(f), np.nan, np.float32), 0., silent and route == 'voice'))
    return out


cache = {m: items(m) for m in MEMBERS}
for cfg in sorted(p.name for p in RUNS.iterdir() if p.is_dir()):
    (OUT / cfg).mkdir(parents=True, exist_ok=True)
    for member, (route, fake_index) in MEMBERS.items():
        res = json.loads((RUNS / cfg / f'{member}.json').read_text())
        preds = []
        for seed in (41, 42, 43):
            model = torch.load(MODELS / cfg / f'{member}__{seed}.pt', map_location='cpu', weights_only=False)
            preds.append(R.T.predict(model, cache[member], route, fake_index, 256, 10 ** 6, 'cpu').tolist())
        res['every']['e3'] = {'ids': [x.row['ID'] for x in cache[member]], 'silent': [bool(x.silent) for x in cache[member]], 'preds': preds}
        (OUT / cfg / f'{member}.json').write_text(json.dumps(res))
    print(cfg, 'e3 rows', len(rows), flush=True)
