"""Alternative A2 (exploratory, VAL ONLY): the S7 voice fusion with its WavLM-8 member pooled by max (runtime) vs mean over
windows, same GPU heads (a2_pooling.py).  python a2_eval.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s8_eval as E  # noqa: E402

ns = [r['ID'] for r in E.rows['val'] if not E.silent[r['ID']]]
out = {}
for k in ('max', 'mean'):
    r = json.loads((Path('/root/s8_runs_a2') / f'voice_new__wavlm8.{k}.json').read_text())
    name = f'a2_{k}'
    E.M[name] = {}
    for s in ('val', 'test'):
        E.M[name].update(zip(r['every'][s]['ids'], np.mean(r['every'][s]['preds'], 0).tolist()))
    E.tables[name] = E.dh.quantile_table([E.M[name][i] for i in ns])
    out[f'val_rt_ads_{k}'] = E.val_ads({'df_L': 1, 'ad_L': .5, name: 1}, E.S7_MUSIC)
print(json.dumps(out, indent=1))
(Path(__file__).resolve().parent / 'a2_val.json').write_text(json.dumps(out, indent=1))
