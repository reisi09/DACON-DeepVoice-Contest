"""Alternative A5 (s8_lock.json, exploratory): S8 music recipe with 5 seeds (41-45, all included) vs 3 seeds (41-43), VAL RT
ADS only (a held-out look needs a separate pre-registered finalist decision).  python a5_eval.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s8_eval as E  # noqa: E402  (loads the S7 base members, rows, presence, tables for every loaded member)

three = json.loads((Path('/root/s8_runs') / 'music_new__mert6_12+eat16+sslampt5+mixsslam5.json').read_text())
two = json.loads((Path('/root/s8_runs_a5') / 'music_new__mert6_12+eat16+sslampt5+mixsslam5.json').read_text())
five = {}
for s in ('val', 'test'):
    assert three['every'][s]['ids'] == two['every'][s]['ids']
    five.update(zip(three['every'][s]['ids'], np.mean(three['every'][s]['preds'] + two['every'][s]['preds'], 0).tolist()))
E.M['music5'] = five
ns = [r['ID'] for r in E.rows['val'] if not E.silent[r['ID']]]
E.tables['music5'] = E.dh.quantile_table([five[i] for i in ns])
out = {'seeds_3': E.val_ads(E.S7_VOICE, 'music_new:mert6_12+eat16+sslampt5+mixsslam5'), 'seeds_5': E.val_ads(E.S7_VOICE, 'music5'),
       'per_seed_val_eer': {**{k: v['val_eer'] for k, v in three['seeds'].items()}, **{k: v['val_eer'] for k, v in two['seeds'].items()}}}
print(json.dumps(out, indent=1))
(Path(__file__).resolve().parent / 'a5_val.json').write_text(json.dumps(out, indent=1))
